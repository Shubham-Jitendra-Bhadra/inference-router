from dotenv import load_dotenv
load_dotenv()

from inference_router import InferenceRouter
from inference_router.providers.bedrock import BedrockProvider
from inference_router.strategies import ComplexityStrategy, ChainStrategy, CostStrategy, LatencyStrategy

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "balanced": BedrockProvider("us.anthropic.claude-sonnet-4-6"),
    },
    strategy=ComplexityStrategy(),
    fallback="fast"
)

print("router:", router)
print()

# simple query — should route to fast
response = router.complete("what is the capital of France?")
print("--- simple query ---")
print("text:       ", response.text)
print("tier:       ", response.tier_used)
print("model:      ", response.model_used)
print("tokens:     ", response.tokens.total)
print("cost:       ", response.cost_usd)
print("latency:    ", response.latency_ms, "ms")
print("strategy:   ", response.routing.strategy_used)
print()

# complex query — should route to balanced
response = router.complete("explain the tradeoffs between SQL and NoSQL databases in detail")
print("--- complex query ---")
print("text:       ", response.text[:80], "...")
print("tier:       ", response.tier_used)
print("model:      ", response.model_used)
print("tokens:     ", response.tokens.total)
print("cost:       ", response.cost_usd)
print("latency:    ", response.latency_ms, "ms")
print("strategy:   ", response.routing.strategy_used)
print()

# force a tier
response = router.complete("hello", tier="balanced")
print("--- forced tier ---")
print("tier:       ", response.tier_used)
print("strategy:   ", response.routing.strategy_used)