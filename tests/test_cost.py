from inference_router.strategies.cost import CostStrategy
from inference_router.models import RouterRequest

strategy = CostStrategy(
    budget_usd_per_day=1.0,
    tiers_by_cost=["powerful", "balanced", "fast"],
    downgrade_at=0.8,
    floor_at=0.95,
)

available = ["fast", "balanced", "powerful"]

# simulate fresh budget — should get best tier
req = RouterRequest(prompt="explain recursion")
print("fresh budget:", strategy.select_tier(req, available))

# simulate 85% spent
strategy._global_spend = 0.85
print("85% spent:", strategy.select_tier(req, available))

# simulate 96% spent
strategy._global_spend = 0.96
print("96% spent:", strategy.select_tier(req, available))

# per-user budget
req_user = RouterRequest(
    prompt="hello",
    metadata={"user_id": "user_123", "budget_usd": 0.10}
)
strategy._user_spend["user_123"] = 0.09   # 90% of user budget used
print("user 90% spent:", strategy.select_tier(req_user, available))

print("remaining budget:", strategy.remaining_budget())