from __future__ import annotations

from collections import defaultdict
from datetime import datetime, date
from threading import Lock

from inference_router.models import RouterRequest
from inference_router.strategies.base import BaseStrategy


class CostStrategy(BaseStrategy):
    """
    Routes requests based on a cost budget.
    Downgrades to cheaper tiers as spend approaches the budget limit.

    Tracks spend globally and optionally per-user via request.metadata["user_id"].

    Usage:
        # global daily budget
        strategy = CostStrategy(
            budget_usd_per_day=10.0,
            tiers_by_cost=["powerful", "balanced", "fast"]
        )

        # per-user budget via metadata
        response = router.complete(
            "your prompt",
            metadata={"user_id": "user_123", "budget_usd": 1.0}
        )

        # custom downgrade thresholds
        strategy = CostStrategy(
            budget_usd_per_day=10.0,
            tiers_by_cost=["powerful", "balanced", "fast"],
            downgrade_at=0.8,    # downgrade to balanced at 80% budget used
            floor_at=0.95,       # downgrade to cheapest at 95% budget used
        )
    """

    def __init__(
        self,
        budget_usd_per_day: float,
        tiers_by_cost: list[str] | None = None,
        downgrade_at: float = 0.8,     # downgrade at 80% budget used
        floor_at: float = 0.95,        # use cheapest tier at 95% budget used
    ):
        """
        Args:
            budget_usd_per_day: daily spend limit in USD
            tiers_by_cost:      tiers ordered most expensive → cheapest
                                e.g. ["powerful", "balanced", "fast"]
                                if None, uses available_tiers order as-is
            downgrade_at:       fraction of budget at which to downgrade one tier
            floor_at:           fraction of budget at which to use cheapest tier
        """
        self.budget_usd_per_day = budget_usd_per_day
        self.tiers_by_cost = tiers_by_cost
        self.downgrade_at = downgrade_at
        self.floor_at = floor_at

        # global spend tracking
        self._global_spend: float = 0.0
        self._reset_date: date = datetime.now().date()

        # per-user spend tracking — {user_id: spend}
        self._user_spend: dict[str, float] = defaultdict(float)
        self._user_reset_date: dict[str, date] = {}

        # thread safety
        self._lock = Lock()

    def _reset_if_new_day(self) -> None:
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if today != self._reset_date:
            self._global_spend = 0.0
            self._user_spend.clear()
            self._user_reset_date.clear()
            self._reset_date = today

    def _get_ordered_tiers(self, available_tiers: list[str]) -> list[str]:
        """
        Return tiers ordered most expensive → cheapest.
        Uses self.tiers_by_cost if set, otherwise available_tiers as-is.
        """
        if self.tiers_by_cost:
            # filter to only include tiers that are actually available
            return [t for t in self.tiers_by_cost if t in available_tiers]
        return available_tiers

    def _select_by_budget(
        self,
        spend: float,
        budget: float,
        ordered_tiers: list[str],
    ) -> str:
        """
        Pick a tier based on how much of the budget has been used.

        0% - downgrade_at%  → most expensive tier (full capability)
        downgrade_at% - floor_at% → middle tier (balanced)
        floor_at% - 100%+   → cheapest tier (floor)
        """
        if not ordered_tiers:
            return "fast"

        fraction_used = spend / budget if budget > 0 else 1.0

        if fraction_used >= self.floor_at:
            # budget nearly exhausted — use cheapest tier
            return ordered_tiers[-1]
        elif fraction_used >= self.downgrade_at:
            # budget getting low — use middle tier
            mid = len(ordered_tiers) // 2
            return ordered_tiers[mid]
        else:
            # budget healthy — use best tier
            return ordered_tiers[0]

    def select_tier(
        self,
        request: RouterRequest,
        available_tiers: list[str],
    ) -> str:
        with self._lock:
            self._reset_if_new_day()
            ordered_tiers = self._get_ordered_tiers(available_tiers)

            # check per-user budget first if user_id in metadata
            user_id = request.metadata.get("user_id")
            user_budget = request.metadata.get("budget_usd")

            if user_id and user_budget:
                user_spend = self._user_spend[user_id]
                return self._select_by_budget(
                    user_spend, user_budget, ordered_tiers
                )

            # otherwise use global daily budget
            return self._select_by_budget(
                self._global_spend,
                self.budget_usd_per_day,
                ordered_tiers
            )

    def on_response(
        self,
        tier_used: str,
        request: RouterRequest,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Update spend counters after each completion."""
        with self._lock:
            self._global_spend += cost_usd

            user_id = request.metadata.get("user_id")
            if user_id:
                self._user_spend[user_id] += cost_usd

    def remaining_budget(self) -> float:
        """How much of the daily budget is left."""
        with self._lock:
            self._reset_if_new_day()
            return max(0.0, self.budget_usd_per_day - self._global_spend)

    def spent_today(self) -> float:
        """How much has been spent today."""
        with self._lock:
            self._reset_if_new_day()
            return self._global_spend

    def reset(self) -> None:
        """Reset all spend counters."""
        with self._lock:
            self._global_spend = 0.0
            self._user_spend.clear()
            self._reset_date = datetime.now().date()