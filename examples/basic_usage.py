"""
inference-router — basic usage examples

Run: python examples/basic_usage.py
"""

from dotenv import load_dotenv
load_dotenv()

from inference_router import InferenceRouter
from inference_router.providers.bedrock import BedrockProvider
from inference_router.strategies import (
    ComplexityStrategy,
    CostStrategy,
    LatencyStrategy,
    ChainStrategy,
)


# ── setup 

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "balanced": BedrockProvider("us.anthropic.claude-sonnet-4-6"),
    },
    strategy=ChainStrategy([
        CostStrategy(
            budget_usd_per_day=5.0,
            tiers_by_cost=["balanced", "fast"],
        ),
        LatencyStrategy(
            sla_ms=3000,
            preferred_tier="balanced",
            fallback_tier="fast",
        ),
        ComplexityStrategy(),
    ]),
    fallback="fast"
)

print("=" * 60)
print("inference-router — basic usage")
print("=" * 60)


# ── example 1: simple query → fast tier ────────────────────────────────────

print("\n[1] simple query")
response = router.complete("what is the capital of France?")
print(f"    answer:   {response.text.strip()}")
print(f"    tier:     {response.tier_used}")
print(f"    model:    {response.model_used}")
print(f"    tokens:   {response.tokens.total}")
print(f"    cost:     ${response.cost_usd:.6f}")
print(f"    latency:  {response.latency_ms:.0f}ms")
print(f"    strategy: {response.routing.strategy_used}")


# ── example 2: complex query → balanced tier ───────────────────────────────

print("\n[2] complex query")
response = router.complete(
    "explain the tradeoffs between microservices and monolithic architecture"
)
print(f"    answer:   {response.text[:80].strip()}...")
print(f"    tier:     {response.tier_used}")
print(f"    model:    {response.model_used}")
print(f"    tokens:   {response.tokens.total}")
print(f"    cost:     ${response.cost_usd:.6f}")
print(f"    latency:  {response.latency_ms:.0f}ms")
print(f"    strategy: {response.routing.strategy_used}")


# ── example 3: force a specific tier ───────────────────────────────────────

print("\n[3] forced tier")
response = router.complete("hello", tier="balanced")
print(f"    answer:   {response.text.strip()}")
print(f"    tier:     {response.tier_used}")
print(f"    strategy: {response.routing.strategy_used}")


# ── example 4: multi-turn conversation ─────────────────────────────────────

from inference_router.models import Message

print("\n[4] multi-turn conversation")
response = router.complete(
    prompt="what did I just tell you?",
    messages=[
        Message(role="user", content="my name is Shubham and I love distributed systems"),
        Message(role="assistant", content="Nice to meet you Shubham! Distributed systems is a fascinating field."),
    ]
)
print(f"    answer:   {response.text.strip()[:100]}...")
print(f"    tier:     {response.tier_used}")


# ── example 5: system prompt ────────────────────────────────────────────────

print("\n[5] system prompt")
response = router.complete(
    prompt="what do you do?",
    system_prompt="You are a concise assistant. Reply in one sentence only."
)
print(f"    answer:   {response.text.strip()}")
print(f"    tier:     {response.tier_used}")


# ── example 6: async completion ─────────────────────────────────────────────

import asyncio

async def run_async():
    print("\n[6] async completion")
    response = await router.acomplete(
        "what is a binary search tree?"
    )
    print(f"    answer:   {response.text[:80].strip()}...")
    print(f"    tier:     {response.tier_used}")
    print(f"    latency:  {response.latency_ms:.0f}ms")

asyncio.run(run_async())


# ── example 7: streaming ────────────────────────────────────────────────────

print("\n[7] streaming")
print("    answer:   ", end="", flush=True)
for chunk in router.stream("write a haiku about distributed systems"):
    print(chunk, end="", flush=True)
print()


# ── example 8: complexity explain ───────────────────────────────────────────

from inference_router.strategies.complexity import ComplexityStrategy as CS

print("\n[8] complexity scorer debug")
scorer = CS()
prompts = [
    "what is 2+2?",
    "explain recursion with examples",
    "design a distributed rate limiter and explain the tradeoffs",
]
for p in prompts:
    result = scorer.explain(p)
    print(f"    score: {result['score']:5.1f} | tier: {result['tier_selected']:10} | {p[:50]}")


print("\n" + "=" * 60)
print("all examples complete")
print("=" * 60)