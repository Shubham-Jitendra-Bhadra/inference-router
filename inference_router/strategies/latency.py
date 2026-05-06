from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock

import statistics

from inference_router.models import RouterRequest
from inference_router.strategies.base import BaseStrategy


class LatencyStrategy(BaseStrategy):
    """
    Routes requests based on real-time p90 latency per tier.
    Downgrades to a faster tier when latency exceeds SLA threshold.

    Maintains a rolling window of latency measurements per tier
    and computes p90 on each routing decision.

    Usage:
        strategy = LatencyStrategy(
            sla_ms=800,                          # target p90 latency
            preferred_tier="balanced",           # start here
            fallback_tier="fast",                # drop to this if SLA breached
            window_size=50,                      # rolling window size
        )

        # custom per-tier SLAs
        strategy = LatencyStrategy(
            sla_ms=800,
            tier_slas={
                "powerful": 2000,
                "balanced": 800,
                "fast":     400,
            }
        )
    """

    def __init__(
        self,
        sla_ms: float,
        preferred_tier: str | None = None,     # if None uses first available tier
        fallback_tier: str | None = None,      # if None uses last available tier
        window_size: int = 50,                 # number of recent requests to track
        min_samples: int = 5,                  # minimum samples before enforcing SLA
        tier_slas: dict[str, float] | None = None,  # per-tier SLA overrides
    ):
        """
        Args:
            sla_ms:         target p90 latency in milliseconds
            preferred_tier: tier to use when SLA is healthy
            fallback_tier:  tier to use when SLA is breached
            window_size:    rolling window of latency samples per tier
            min_samples:    don't enforce SLA until we have this many samples
            tier_slas:      per-tier SLA thresholds — overrides sla_ms per tier
        """
        self.sla_ms = sla_ms
        self.preferred_tier = preferred_tier
        self.fallback_tier = fallback_tier
        self.window_size = window_size
        self.min_samples = min_samples
        self.tier_slas = tier_slas or {}

        # rolling latency window per tier — {tier: deque of latency_ms values}
        self._latencies: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._lock = Lock()

    def _p90(self, tier: str) -> float | None:
        """
        Calculate p90 latency for a tier.
        Returns None if not enough samples yet.
        """
        samples = list(self._latencies[tier])
        if len(samples) < self.min_samples:
            return None     # not enough data yet
        sorted_samples = sorted(samples)
        idx = int(len(sorted_samples) * 0.9)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def _sla_for_tier(self, tier: str) -> float:
        """Get SLA threshold for a specific tier."""
        return self.tier_slas.get(tier, self.sla_ms)

    def _is_healthy(self, tier: str) -> bool:
        """
        Check if a tier's p90 latency is within its SLA.
        Returns True if not enough samples yet — give it benefit of the doubt.
        """
        p90 = self._p90(tier)
        if p90 is None:
            return True     # no data → assume healthy
        return p90 <= self._sla_for_tier(tier)

    def select_tier(
        self,
        request: RouterRequest,
        available_tiers: list[str],
    ) -> str:
        with self._lock:
            preferred = self.preferred_tier or available_tiers[0]
            fallback = self.fallback_tier or available_tiers[-1]

            # make sure configured tiers are actually available
            if preferred not in available_tiers:
                preferred = available_tiers[0]
            if fallback not in available_tiers:
                fallback = available_tiers[-1]

            # if preferred tier is healthy, use it
            if self._is_healthy(preferred):
                return preferred

            # preferred SLA breached — try intermediate tiers
            for tier in available_tiers:
                if tier == preferred:
                    continue
                if tier == fallback:
                    continue
                if self._is_healthy(tier):
                    return tier

            # all tiers degraded or only fallback left
            return fallback

    def on_response(
        self,
        tier_used: str,
        request: RouterRequest,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Record latency measurement for the tier that was used."""
        with self._lock:
            self._latencies[tier_used].append(latency_ms)

    def stats(self) -> dict:
        """
        Return current p90 latency and health status per tier.
        Useful for debugging and observability.
        """
        with self._lock:
            result = {}
            for tier, samples in self._latencies.items():
                p90 = self._p90(tier)
                result[tier] = {
                    "p90_ms": round(p90, 2) if p90 else None,
                    "samples": len(samples),
                    "sla_ms": self._sla_for_tier(tier),
                    "healthy": self._is_healthy(tier),
                }
            return result

    def reset(self) -> None:
        """Clear all latency measurements."""
        with self._lock:
            self._latencies.clear()