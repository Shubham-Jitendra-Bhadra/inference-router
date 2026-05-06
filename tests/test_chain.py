from inference_router.strategies import (
    ChainStrategy,
    CostStrategy,
    LatencyStrategy,
    ComplexityStrategy,
)
from inference_router.models import RouterRequest
from inference_router.strategies.complexity import score_prompt

c = ComplexityStrategy()
available = ["fast", "balanced", "powerful"]

req1 = RouterRequest(prompt="what is the capital of France?")
req2 = RouterRequest(prompt="explain the tradeoffs between B-trees and LSM trees in detail")

print("complexity simple:", c.select_tier(req1, available))
print("complexity complex:", c.select_tier(req2, available))
print("rules:", c.rules)
strategy = ChainStrategy([
    CostStrategy(
        budget_usd_per_day=1.0,
        tiers_by_cost=["powerful", "balanced", "fast"],  # expensive → cheap
        downgrade_at=0.8,
        floor_at=0.95,
    ),
    LatencyStrategy(sla_ms=800, preferred_tier="balanced", fallback_tier="fast"),
    ComplexityStrategy(),
])

available = ["fast", "balanced", "powerful"]

# simple query — complexity decides
req = RouterRequest(prompt="what is the capital of France?")
print("simple query:", strategy.select_tier(req, available))
print("simple score:", score_prompt("what is the capital of France?"))


# complex query — complexity decides
req = RouterRequest(prompt="explain the tradeoffs between B-trees and LSM trees in detail")
print("complex query:", strategy.select_tier(req, available))
print("complex score:", score_prompt("explain the tradeoffs between B-trees and LSM trees in detail"))

# budget exceeded — cost overrides
strategy.strategies[0]._global_spend = 0.97
req = RouterRequest(prompt="explain the tradeoffs between B-trees and LSM trees in detail")
print("budget exceeded:", strategy.select_tier(req, available))

print("repr:", strategy)