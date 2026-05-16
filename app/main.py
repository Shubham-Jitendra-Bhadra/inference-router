"""
inference-router — FastAPI layer

Run: uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse

from inference_router import InferenceRouter
from inference_router.providers.bedrock import BedrockProvider
from inference_router.strategies import (
    ComplexityStrategy,
    CostStrategy,
    LatencyStrategy,
    ChainStrategy,
)
from inference_router.models import Message


# ── router setup ────────────────────────────────────────────────────────────

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


# ── app setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="inference-router",
    description="Pluggable LLM inference routing API",
    version="0.1.0",
)


# ── request / response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    prompt: str
    messages: Optional[list[Message]] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    tier: Optional[str] = None          # force a specific tier
    metadata: dict = {}


class QueryResponse(BaseModel):
    text: str
    model_used: str
    tier_used: str
    tokens_total: int
    cost_usd: float
    latency_ms: float
    strategy_used: str
    was_fallback: bool


class ComplexityDebugResponse(BaseModel):
    prompt_preview: str
    score: float
    tier_selected: str
    breakdown: dict


# ── endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "inference-router",
        "version": "0.1.0",
        "status": "running",
        "tiers": router.available_tiers,
        "strategy": str(router.strategy),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Route a prompt to the appropriate model tier.
    Strategy decides the tier automatically unless tier is specified.
    """
    try:
        response = await router.acomplete(
            prompt=req.prompt,
            messages=req.messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system_prompt=req.system_prompt,
            metadata=req.metadata,
            tier=req.tier,
        )
        return QueryResponse(
            text=response.text,
            model_used=response.model_used,
            tier_used=response.tier_used,
            tokens_total=response.tokens.total,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            strategy_used=response.routing.strategy_used,
            was_fallback=response.routing.reason == "fallback",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    Stream a response chunk by chunk.
    """
    async def generate():
        try:
            async for chunk in router.astream(
                prompt=req.prompt,
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system_prompt=req.system_prompt,
                tier=req.tier,
            ):
                yield chunk
        except Exception as e:
            yield f"\n[error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/tiers")
def tiers():
    """List all configured tiers and their providers."""
    return {
        "tiers": {
            name: str(provider)
            for name, provider in router.tiers.items()
        },
        "fallback": router.fallback,
    }


@app.post("/debug/complexity", response_model=ComplexityDebugResponse)
def debug_complexity(req: QueryRequest):
    """
    Debug the complexity scorer for a prompt.
    Shows score, tier selected, and per-dimension breakdown.
    """
    from inference_router.strategies.complexity import ComplexityStrategy as CS
    scorer = CS()
    result = scorer.explain(req.prompt)
    return ComplexityDebugResponse(**result)


@app.delete("/stats/reset")
def reset_stats():
    """Reset strategy state — cost spend, latency history."""
    router.strategy.reset()
    return {"status": "reset"}