from __future__ import annotations

import time
from typing import AsyncIterator, Iterator

import httpx

from inference_router.models import (
    RouterRequest,
    RouterResponse,
    RoutingDecision,
    TokenUsage,
)
from inference_router.providers.base import BaseProvider


class HTTPProvider(BaseProvider):
    """
    Generic provider for any OpenAI-compatible HTTP API.

    Works with: Groq, DeepInfra, Together AI, Fireworks,
                Ollama, Anyscale, or any API following the
                OpenAI chat completions format.

    Usage:
        # Groq
        provider = HTTPProvider(
            base_url="https://api.groq.com/openai/v1",
            api_key="your-key",
            model="mixtral-8x7b-32768"
        )

        # DeepInfra
        provider = HTTPProvider(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key="your-key",
            model="meta-llama/Meta-Llama-3-8B-Instruct"
        )

        # Ollama (local, no auth needed)
        provider = HTTPProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3"
        )

        response = provider.complete(RouterRequest(prompt="hello"))
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,          # request timeout in seconds
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict | None = None,   # any custom headers the API needs
        cost_per_1k_input: float = 0.0,      # optional — set if you know the pricing
        cost_per_1k_output: float = 0.0,
    ):
        """
        Args:
            base_url:            API base URL e.g. "https://api.groq.com/openai/v1"
            model:               model name e.g. "mixtral-8x7b-32768"
            api_key:             API key — leave empty for local APIs like Ollama
            timeout:             request timeout in seconds
            temperature:         optional override for request temperature
            max_tokens:          optional override for request max_tokens
            extra_headers:       any additional headers the API requires
            cost_per_1k_input:   cost per 1000 input tokens in USD (optional)
            cost_per_1k_output:  cost per 1000 output tokens in USD (optional)
        """
        self.base_url = base_url.rstrip("/")   # normalize trailing slash
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._temperature_override = temperature
        self._max_tokens_override = max_tokens
        self._extra_headers = extra_headers or {}
        self._cost_per_1k_input = cost_per_1k_input
        self._cost_per_1k_output = cost_per_1k_output

    @property
    def name(self) -> str:
        # extract provider name from base URL for clean display
        # "https://api.groq.com/openai/v1" → "groq/mixtral-8x7b-32768"
        try:
            host = self.base_url.split("//")[1].split("/")[0]
            parts = host.split(".")
            provider = parts[1] if parts[0] == "api" else parts[0]
        except Exception:
            provider = "http"
        return f"{provider}/{self.model}"

    def _build_headers(self) -> dict:
        """Build request headers including auth."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._extra_headers)
        return headers

    def _build_body(self, request: RouterRequest) -> dict:
        """
        Translate RouterRequest into OpenAI-compatible request format.
        """
        # build messages list
        if request.messages:
            messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]
            # append current prompt if provided alongside messages
            if request.prompt:
                messages.append({"role": "user", "content": request.prompt})
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # prepend system message if provided
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self._max_tokens_override or request.max_tokens,
            "temperature": self._temperature_override or request.temperature,
            "stream": False,
        }

        return body

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD if pricing was provided."""
        return (
            (input_tokens / 1000) * self._cost_per_1k_input +
            (output_tokens / 1000) * self._cost_per_1k_output
        )

    def _parse_response(
        self,
        raw: dict,
        latency_ms: float = 0.0,
    ) -> RouterResponse:
        """
        Translate OpenAI-compatible response into normalized RouterResponse.
        """
        # extract text
        text = raw["choices"][0]["message"]["content"]

        # token usage
        usage = raw.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        cost = self._calculate_cost(input_tokens, output_tokens)

        return RouterResponse(
            text=text,
            model_used=self.name,
            tier_used="",               # router fills this in
            tokens=TokenUsage(
                input=input_tokens,
                output=output_tokens,
                total=total_tokens,
            ),
            latency_ms=latency_ms,
            cost_usd=cost,
            routing=RoutingDecision(
                tier_selected="",       # router fills this in
                strategy_used="",       # router fills this in
                reason="",
            ),
            raw=raw,
        )

    def complete(self, request: RouterRequest) -> RouterResponse:
        """Synchronous completion via HTTP."""
        body = self._build_body(request)
        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        start = time.perf_counter()

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()   # raises on 4xx/5xx

        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.json()

        return self._parse_response(raw, latency_ms=latency_ms)

    async def acomplete(self, request: RouterRequest) -> RouterResponse:
        """Async completion via HTTP."""
        body = self._build_body(request)
        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()

        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.json()

        return self._parse_response(raw, latency_ms=latency_ms)

    def stream(self, request: RouterRequest) -> Iterator[str]:
        """Synchronous streaming via HTTP SSE."""
        import json

        body = self._build_body(request)
        body["stream"] = True           # enable streaming
        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=body, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]         # strip "data: " prefix
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            continue

    async def astream(self, request: RouterRequest) -> AsyncIterator[str]:
        """Async streaming via HTTP SSE."""
        import json

        body = self._build_body(request)
        body["stream"] = True
        headers = self._build_headers()
        url = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            continue

    def health_check(self) -> bool:
        """Check if the API endpoint is reachable."""
        try:
            self.complete(RouterRequest(prompt="hi", max_tokens=5))
            return True
        except Exception:
            return False