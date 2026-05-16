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
ANTHROPIC_COSTS: dict[str, dict[str, float]] = {
    "claude-haiku":  {"input": 0.00025, "output": 0.00125},
    "claude-sonnet": {"input": 0.003,   "output": 0.015},
    "claude-opus":   {"input": 0.015,   "output": 0.075},
}

DEFAULT_COST = {"input": 0.001, "output": 0.002}


def _get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = DEFAULT_COST
    for key, cost in ANTHROPIC_COSTS.items():
        if key in model.lower():
            pricing = cost
            break
    return (
        (input_tokens / 1000) * pricing["input"] +
        (output_tokens / 1000) * pricing["output"]
    )


class AnthropicProvider(BaseProvider):
    """
    Anthropic direct API provider. Uses the official anthropic
    Python client — not Bedrock, direct to api.anthropic.com.

    Use this when you want to call Anthropic's API directly
    without going through AWS.

    Usage:
        from inference_router.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            model="claude-opus-4-6",
            api_key="sk-ant-..."
        )

    Requires:
        pip install 'inference-router[anthropic]'
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ):
        """
        Args:
            model:       Anthropic model name e.g. "claude-opus-4-6",
                         "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
            api_key:     Anthropic API key (starts with sk-ant-)
            temperature: optional override for request temperature
            max_tokens:  optional override for request max_tokens
            timeout:     request timeout in seconds
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install 'inference-router[anthropic]'"
            )

        self.model = model
        self.api_key = api_key
        self._temperature_override = temperature
        self._max_tokens_override = max_tokens
        self.timeout = timeout

        self._client = anthropic.Anthropic(api_key=api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def _build_messages(self, request: RouterRequest) -> list[dict]:
        """Build Anthropic messages list from RouterRequest."""
        messages = []

        if request.messages:
            for m in request.messages:
                messages.append({"role": m.role, "content": m.content})

        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _parse_response(self, raw, latency_ms: float = 0.0) -> RouterResponse:
        """Translate Anthropic response into normalized RouterResponse."""
        text = raw.content[0].text
        input_tokens = raw.usage.input_tokens
        output_tokens = raw.usage.output_tokens
        cost = _get_cost(self.model, input_tokens, output_tokens)

        return RouterResponse(
            text=text,
            model_used=self.name,
            tier_used="",
            tokens=TokenUsage(
                input=input_tokens,
                output=output_tokens,
                total=input_tokens + output_tokens,
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
        """Synchronous completion via Anthropic direct API."""
        messages = self._build_messages(request)
        kwargs = {
            "model": self.model,
            "max_tokens": self._max_tokens_override or request.max_tokens,
            "messages": messages,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if self._temperature_override or request.temperature != 0.7:
            kwargs["temperature"] = self._temperature_override or request.temperature

        start = time.perf_counter()
        raw = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(raw, latency_ms=latency_ms)

    async def acomplete(self, request: RouterRequest) -> RouterResponse:
        """Async completion via Anthropic direct API."""
        messages = self._build_messages(request)
        kwargs = {
            "model": self.model,
            "max_tokens": self._max_tokens_override or request.max_tokens,
            "messages": messages,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt
        if self._temperature_override or request.temperature != 0.7:
            kwargs["temperature"] = self._temperature_override or request.temperature

        start = time.perf_counter()
        raw = await self._async_client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return self._parse_response(raw, latency_ms=latency_ms)

    def stream(self, request: RouterRequest) -> Iterator[str]:
        """Synchronous streaming via Anthropic direct API."""
        messages = self._build_messages(request)
        kwargs = {
            "model": self.model,
            "max_tokens": self._max_tokens_override or request.max_tokens,
            "messages": messages,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    async def astream(self, request: RouterRequest) -> AsyncIterator[str]:
        """Async streaming via Anthropic direct API."""
        messages = self._build_messages(request)
        kwargs = {
            "model": self.model,
            "max_tokens": self._max_tokens_override or request.max_tokens,
            "messages": messages,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    def health_check(self) -> bool:
        """Check if Anthropic API is reachable."""
        try:
            self.complete(RouterRequest(prompt="hi", max_tokens=5))
            return True
        except Exception:
            return False