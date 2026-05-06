from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""
    role: str
    content: str


class RouterRequest(BaseModel):
    """What the user sends to the router."""
    prompt: str
    messages: Optional[list[Message]] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    """Token breakdown for a completion."""
    input: int
    output: int
    total: int


class RoutingDecision(BaseModel):
    """Why the router picked this tier."""
    tier_selected: str
    strategy_used: str
    score: Optional[float] = None
    reason: str


class RouterResponse(BaseModel):
    """Normalized response — same shape regardless of provider."""
    text: str
    model_used: str
    tier_used: str
    tokens: TokenUsage
    latency_ms: float
    cost_usd: float
    routing: RoutingDecision
    raw: Optional[dict[str, Any]] = None