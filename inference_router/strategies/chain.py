from __future__ import annotations

from inference_router.models import RouterRequest
from inference_router.strategies.base import BaseStrategy


class ChainStrategy(BaseStrategy):
    """
    Combines multiple strategies evaluated in order.

    Hard constraint strategies (CostStrategy, LatencyStrategy) go first.
    They override by returning the cheapest/fallback tier.
    The last strategy (usually ComplexityStrategy) makes the
    final call if no hard constraint overrode.

    Usage:
        strategy = ChainStrategy([
            CostStrategy(budget_usd_per_day=10.0),
            LatencyStrategy(sla_ms=800),
            ComplexityStrategy(),
        ])
    """

    def __init__(self, strategies: list[BaseStrategy]):
        if not strategies:
            raise ValueError("ChainStrategy requires at least one strategy")
        self.strategies = strategies

    def select_tier(
    self,
    request: RouterRequest,
    available_tiers: list[str],
) -> str:
        """
        Evaluate strategies in order.

        Non-last strategies are hard constraints — they only
        override if they return the cheapest (first) tier,
        meaning a hard limit was hit (budget exhausted, SLA breached).

        The last strategy always makes the final call.
        """
        cheapest_tier = available_tiers[0]   # "fast"

        for i, strategy in enumerate(self.strategies):
            is_last = i == len(self.strategies) - 1
            decision = strategy.select_tier(request, available_tiers)

            if is_last:
                # last strategy — always final say
                return decision

            # hard constraint fired only if it returned the cheapest tier
            # meaning budget exhausted or SLA badly breached
            if decision == cheapest_tier:
                return decision

            # constraint didn't fire — continue to next strategy

        return available_tiers[0]

    def on_response(
        self,
        tier_used: str,
        request: RouterRequest,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        for strategy in self.strategies:
            strategy.on_response(tier_used, request, latency_ms, cost_usd)

    def reset(self) -> None:
        for strategy in self.strategies:
            strategy.reset()

    def __repr__(self) -> str:
        names = ", ".join(s.__class__.__name__ for s in self.strategies)
        return f"ChainStrategy([{names}])"