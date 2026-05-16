from __future__ import annotations

import time
from typing import AsyncIterator, Iterator

from inference_router.models import (
    RouterRequest,
    RouterResponse,
    RoutingDecision,
    TokenUsage,
)
from inference_router.providers.base import BaseProvider


# Approximate cost per 1000 tokens in USD
OPENAI_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":       {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":       {"input": 0.01,    "output": 0.03},
    "gpt-3.5-turbo":     {"input": 0.0005,  "output": 0.0015},
}

DEFAULT_COST = {"input": 0.001, "output": 0.002}


def _get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = DEFAULT_COST
    for key, cost in OPENAI_COSTS.items():
        if key in model.lower():
            pricing = cost
            break
    return (
        (input_tokens / 1000) * pricing["input"] +
        (output_tokens / 1000) * pricing["output"]
    )


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider. Works with any OpenAI-compatible API
    that has the official openai Python client available.

    Also works with Azure OpenAI by passing a custom base_url.

    Usage:
        from inference_router.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            model="gpt-4o",
            api_key="sk-..."
        )

        # Azure OpenAI
        provider = OpenAIProvider(
            model="gpt-4o",
            api_key="your-azure-key",
            base_url="https://your-resource.openai.azure.com/",
        )
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,        # override for Azure or proxies
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ):
        """
        Args:
            model:       OpenAI model name e.g. "gpt-4o", "gpt-4o-mini"
            api_key:     OpenAI API key
            base_url:    optional base URL override for Azure or proxies
            temperature: optional override for request temperature
            max_tokens:  optional override for request max_tokens
            timeout:     request timeout in seconds
        """
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Run: pip install 'inference-router[openai]'"
            )

        self.model = model
        self.api_key = api_key
        self._temperature_override = temperature
        self._max_tokens_override = max_tokens
        self.timeout = timeout

        # sync and async clients
        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def _build_messages(self, request: RouterRequest) -> list[dict]:
        """Build OpenAI messages list from RouterRequest."""
        messages = []

        # system prompt first
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        # conversation history
        if request.messages:
            for m in request.messages:
                messages.append({"role": m.role, "content": m.content})

        # current prompt
        messages.append({"role": "user", "content": request.prompt})

        return messages

    def _parse_response(
        self,
        raw,
        latency_ms: float = 0.0,
    ) -> RouterResponse:
        """Translate OpenAI response into normalized RouterResponse."""
        text = raw.choices[0].message.content
        usage = raw.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cost = _get_cost(self.model, input_tokens, output_tokens)

        return RouterResponse(
            text=text,
            model_used=self.name,
            tier_used="",
            tokens=TokenUsage(
                input=input_tokens,
                output=output_tokens,
                total=usage.total_tokens,
            ),
            latency_ms=latency_ms,
            cost_usd=cost,
            routing=RoutingDecision(
                tier_selected="",
                strategy_used="",
                reason="",
            ),
            raw=raw.model_dump() if hasattr(raw, "model_dump") else None,
        )

    def complete(self, request: RouterRequest) -> RouterResponse:
        """Synchronous completion via OpenAI."""
        messages = self._build_messages(request)
        start = time.perf_counter()

        raw = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens_override or request.max_tokens,
            temperature=self._temperature_override or request.temperature,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(raw, latency_ms=latency_ms)

    async def acomplete(self, request: RouterRequest) -> RouterResponse:
        """Async completion via OpenAI."""
        messages = self._build_messages(request)
        start = time.perf_counter()

        raw = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens_override or request.max_tokens,
            temperature=self._temperature_override or request.temperature,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(raw, latency_ms=latency_ms)

    def stream(self, request: RouterRequest) -> Iterator[str]:
        """Synchronous streaming via OpenAI."""
        messages = self._build_messages(request)

        with self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens_override or request.max_tokens,
            temperature=self._temperature_override or request.temperature,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    async def astream(self, request: RouterRequest) -> AsyncIterator[str]:
        """Async streaming via OpenAI."""
        messages = self._build_messages(request)

        async with await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens_override or request.max_tokens,
            temperature=self._temperature_override or request.temperature,
            stream=True,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    def health_check(self) -> bool:
        """Check if OpenAI API is reachable."""
        try:
            self.complete(RouterRequest(prompt="hi", max_tokens=5))
            return True
        except Exception:
            return False