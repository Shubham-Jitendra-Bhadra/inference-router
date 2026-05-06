from inference_router.strategies.latency import LatencyStrategy
from inference_router.models import RouterRequest

strategy = LatencyStrategy(
    sla_ms=800,
    preferred_tier="balanced",
    fallback_tier="fast",
    min_samples=3,
)

available = ["fast", "balanced", "powerful"]
req = RouterRequest(prompt="explain recursion")

# no samples yet — should use preferred tier
print("no samples:", strategy.select_tier(req, available))

# simulate healthy latencies for balanced
for ms in [300, 400, 350, 420, 380]:
    strategy.on_response("balanced", req, ms, 0.001)

print("healthy p90:", strategy.select_tier(req, available))
print("stats:", strategy.stats())

# simulate slow latencies — p90 breaches SLA
for ms in [900, 1100, 950, 1200, 1050]:
    strategy.on_response("balanced", req, ms, 0.001)

print("breached p90:", strategy.select_tier(req, available))
print("stats after breach:", strategy.stats())