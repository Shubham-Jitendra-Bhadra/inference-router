from __future__ import annotations

import time
from typing import AsyncIterator, Iterator, Optional

from inference_router.models import (
    RouterRequest,
    RouterResponse,
    RoutingDecision,
    Message,
)
from inference_router.providers.base import BaseProvider
from inference_router.strategies.base import BaseStrategy


class InferenceRouter:
    """
    Core router — wires providers and strategies together.

    Routes every request to the right model tier based on
    the configured strategy. Handles fallback automatically
    if the selected provider fails.

    Usage:
        from inference_router import InferenceRouter
        from inference_router.providers.bedrock import BedrockProvider
        from inference_router.strategies import ComplexityStrategy

        router = InferenceRouter(
            tiers={
                "fast":     BedrockProvider("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
                "balanced": BedrockProvider("us.anthropic.claude-sonnet-4-6"),
            },
            strategy=ComplexityStrategy(),
            fallback="fast"
        )

        response = router.complete("explain recursion")
        print(response.text)
        print(response.model_used)
        print(response.cost_usd)
    """

    def __init__(
        self,
        tiers: dict[str, BaseProvider],
        strategy: BaseStrategy,
        fallback: Optional[str] = None,
    ):
        """
        Args:
            tiers:    dict mapping tier name → provider instance
                      e.g. {"fast": BedrockProvider(...), "balanced": BedrockProvider(...)}
            strategy: routing strategy — ComplexityStrategy, CostStrategy,
                      LatencyStrategy, ChainStrategy, or custom
            fallback: tier name to use if selected provider fails
                      defaults to first tier if not specified
        """
        if not tiers:
            raise ValueError("tiers cannot be empty")

        self.tiers = tiers
        self.strategy = strategy
        self.fallback = fallback or list(tiers.keys())[0]

        if self.fallback not in self.tiers:
            raise ValueError(
                f"fallback tier '{self.fallback}' not found in tiers: {list(tiers.keys())}"
            )

    @property
    def available_tiers(self) -> list[str]:
        """List of configured tier names."""
        return list(self.tiers.keys())

    def _select_tier(self, request: RouterRequest) -> tuple[str, str]:
        """
        Ask strategy to select a tier.
        Returns (tier_name, strategy_name).
        """
        tier = self.strategy.select_tier(request, self.available_tiers)
        strategy_name = self.strategy.__class__.__name__
        return tier, strategy_name

    def _fill_routing(
        self,
        response: RouterResponse,
        tier: str,
        strategy_name: str,
        score: Optional[float] = None,
        reason: str = "",
        was_fallback: bool = False,
    ) -> RouterResponse:
        """Fill in routing metadata on the response."""
        response.tier_used = tier
        response.routing = RoutingDecision(
            tier_selected=tier,
            strategy_used=strategy_name,
            score=score,
            reason="fallback" if was_fallback else reason,
        )
        return response

    def _notify_strategy(
        self,
        tier: str,
        request: RouterRequest,
        response: RouterResponse,
    ) -> None:
        """Notify strategy of completed request so it can update state."""
        self.strategy.on_response(
            tier_used=tier,
            request=request,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )

    def complete(
        self,
        prompt: str,
        messages: Optional[list[Message]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
        tier: Optional[str] = None,    # force a specific tier
    ) -> RouterResponse:
        """
        Synchronous completion.

        Args:
            prompt:        user prompt
            messages:      optional conversation history
            max_tokens:    max tokens to generate
            temperature:   sampling temperature
            system_prompt: optional system prompt
            metadata:      extra info for strategies e.g. {"user_id": "123"}
            tier:          force a specific tier, skip strategy

        Returns:
            RouterResponse — normalized, same shape for any provider
        """
        request = RouterRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        # select tier
        if tier:
            selected_tier = tier
            strategy_name = "manual"
        else:
            selected_tier, strategy_name = self._select_tier(request)

        # try selected tier
        provider = self.tiers[selected_tier]
        was_fallback = False

        try:
            response = provider.complete(request)
        except Exception as e:
            # selected tier failed — try fallback
            if selected_tier != self.fallback:
                fallback_provider = self.tiers[self.fallback]
                response = fallback_provider.complete(request)
                selected_tier = self.fallback
                was_fallback = True
            else:
                raise RuntimeError(
                    f"Provider '{selected_tier}' failed and no fallback available: {e}"
                ) from e

        # fill routing metadata
        response = self._fill_routing(
            response,
            tier=selected_tier,
            strategy_name=strategy_name,
            was_fallback=was_fallback,
        )

        # notify strategy
        self._notify_strategy(selected_tier, request, response)

        return response

    async def acomplete(
        self,
        prompt: str,
        messages: Optional[list[Message]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
        tier: Optional[str] = None,
    ) -> RouterResponse:
        """
        Async completion.

        Same as complete() but non-blocking.
        Use in FastAPI endpoints and async applications.
        """
        request = RouterRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        if tier:
            selected_tier = tier
            strategy_name = "manual"
        else:
            selected_tier, strategy_name = self._select_tier(request)

        provider = self.tiers[selected_tier]
        was_fallback = False

        try:
            response = await provider.acomplete(request)
        except Exception as e:
            if selected_tier != self.fallback:
                fallback_provider = self.tiers[self.fallback]
                response = await fallback_provider.acomplete(request)
                selected_tier = self.fallback
                was_fallback = True
            else:
                raise RuntimeError(
                    f"Provider '{selected_tier}' failed and no fallback available: {e}"
                ) from e

        response = self._fill_routing(
            response,
            tier=selected_tier,
            strategy_name=strategy_name,
            was_fallback=was_fallback,
        )

        self._notify_strategy(selected_tier, request, response)

        return response

    def stream(
        self,
        prompt: str,
        messages: Optional[list[Message]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
        tier: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Synchronous streaming — yields text chunks as they arrive.
        """
        request = RouterRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        if tier:
            selected_tier = tier
        else:
            selected_tier, _ = self._select_tier(request)

        provider = self.tiers[selected_tier]

        try:
            yield from provider.stream(request)
        except Exception:
            if selected_tier != self.fallback:
                yield from self.tiers[self.fallback].stream(request)

    async def astream(
        self,
        prompt: str,
        messages: Optional[list[Message]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
        tier: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming — yields text chunks as they arrive.
        """
        request = RouterRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        if tier:
            selected_tier = tier
        else:
            selected_tier, _ = self._select_tier(request)

        provider = self.tiers[selected_tier]

        try:
            async for chunk in provider.astream(request):
                yield chunk
        except Exception:
            if selected_tier != self.fallback:
                async for chunk in self.tiers[self.fallback].astream(request):
                    yield chunk

    def __repr__(self) -> str:
        return (
            f"InferenceRouter("
            f"tiers={list(self.tiers.keys())}, "
            f"strategy={self.strategy}, "
            f"fallback={self.fallback!r})"
        )