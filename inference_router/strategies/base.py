from __future__ import annotations
from abc import ABC, abstractmethod
from inference_router.models import RouterRequest


class BaseStrategy(ABC):
    """
    Abstract base class for all routing strategies.

    A strategy has one job: look at the incoming request
    and return which tier to route it to.

    To build a custom strategy, subclass this and implement
    select_tier():

        class MyStrategy(BaseStrategy):

            def select_tier(
                self,
                request: RouterRequest,
                available_tiers: list[str]
            ) -> str:
                # your logic here
                # must return one of the available_tiers
                return available_tiers[0]

    The router calls select_tier() before every request.
    available_tiers is the list of tier names the router
    was configured with — your strategy must return one of them.
    """

    @abstractmethod
    def select_tier(
        self,
        request: RouterRequest,
        available_tiers: list[str],
    ) -> str:
        """
        Decide which tier to route this request to.

        Args:
            request:         the incoming RouterRequest
            available_tiers: list of tier names configured
                             in the router e.g. ["fast", "balanced", "powerful"]

        Returns:
            str — one of the available_tiers names
        """
        ...

    def on_response(
        self,
        tier_used: str,
        request: RouterRequest,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """
        Optional callback — called after every successful completion.

        Strategies that need to track state (latency history,
        running cost totals) use this to update their internal state.

        Args:
            tier_used:   which tier was actually used
            request:     the original request
            latency_ms:  how long the completion took
            cost_usd:    how much it cost
        """
        pass  

    def reset(self) -> None:
        """
        Optional — reset any internal state.
        Called by router.reset_stats().
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"