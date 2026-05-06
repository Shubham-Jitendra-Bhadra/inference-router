from __future__ import annotations

import json
import time
from typing import AsyncIterator, Iterator

import boto3

from inference_router.models import (
    RouterRequest,
    RouterResponse,
    RoutingDecision,
    TokenUsage,
)
from inference_router.providers.base import BaseProvider


# Approximate cost per 1000 tokens in USD
BEDROCK_COSTS: dict[str, dict[str, float]] = {
    "anthropic.claude-haiku":  {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-sonnet": {"input": 0.003,   "output": 0.015},
    "anthropic.claude-opus":   {"input": 0.015,   "output": 0.075},
}

# Default cost if model not in table above
DEFAULT_COST = {"input": 0.001, "output": 0.002}


def _get_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate approximate cost in USD for a completion.
    Strips cross-region prefix (us., eu.) before matching.
    """
    # normalize — remove cross-region prefix if present
    normalized = model_id.lower()
    for prefix in ("us.", "eu.", "ap."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    pricing = DEFAULT_COST
    for key, cost in BEDROCK_COSTS.items():
        if key in normalized:
            pricing = cost
            break

    return (
        (input_tokens / 1000) * pricing["input"] +
        (output_tokens / 1000) * pricing["output"]
    )

class BedrockProvider(BaseProvider):
    """
    AWS Bedrock provider. Supports any model available in your
    Bedrock account — Claude, Titan, Llama, Mistral, etc.

    Credentials are read from environment variables or ~/.aws/credentials.
    Load from .env using python-dotenv before creating this provider.

    Usage:
        from dotenv import load_dotenv
        load_dotenv()

        provider = BedrockProvider(
            model_id="anthropic.claude-haiku-4-5-20251001",
            region="us-east-1"
        )
        response = provider.complete(RouterRequest(prompt="hello"))

    Supports:
        - Sync completions     (complete)
        - Async completions    (acomplete)
        - Sync streaming       (stream)
        - Async streaming      (astream)
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        temperature: float | None = None,   # overrides request temperature if set
        max_tokens: int | None = None,      # overrides request max_tokens if set
    ):
        """
        Args:
            model_id:    Bedrock model ID e.g. 'anthropic.claude-haiku-4-5-20251001'
            region:      AWS region where Bedrock is available
            temperature: Optional override — if set, ignores request.temperature
            max_tokens:  Optional override — if set, ignores request.max_tokens
        """
        self.model_id = model_id
        self.region = region
        self._temperature_override = temperature
        self._max_tokens_override = max_tokens

        # Runtime client — used for actual inference
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region
        )

    @property
    def name(self) -> str:
        return f"bedrock/{self.model_id}"

    def _build_body(self, request: RouterRequest) -> dict:
        """
        Translate RouterRequest into Bedrock's expected request format.

        Handles both single prompt and multi-turn messages.
        """
        # Build messages list
        if request.messages:
            # multi-turn conversation
            messages = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]
        else:
            # single prompt
            messages = [
                {"role": "user", "content": request.prompt}
            ]

        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens_override or request.max_tokens,
            "temperature": self._temperature_override or request.temperature,
            "messages": messages,
        }

        # Add system prompt if provided
        if request.system_prompt:
            body["system"] = request.system_prompt

        return body

    def _parse_response(
        self,
        raw: dict,
        tier_used: str = "",
        latency_ms: float = 0.0,
    ) -> RouterResponse:
        """
        Translate Bedrock's raw response into a normalized RouterResponse.
        """
        # Extract text from content blocks
        text = ""
        for block in raw.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        # Token usage
        usage = raw.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Calculate cost
        cost = _get_cost(self.model_id, input_tokens, output_tokens)

        return RouterResponse(
            text=text,
            model_used=self.name,
            tier_used=tier_used,        # router fills this in
            tokens=TokenUsage(
                input=input_tokens,
                output=output_tokens,
                total=input_tokens + output_tokens,
            ),
            latency_ms=latency_ms,      # router measures this
            cost_usd=cost,
            routing=RoutingDecision(
                tier_selected=tier_used,
                strategy_used="",       # router fills this in
                reason="",              # router fills this in
            ),
            raw=raw,
        )

    def complete(self, request: RouterRequest) -> RouterResponse:
        """
        Synchronous completion via Bedrock.
        """
        body = self._build_body(request)
        start = time.perf_counter()

        raw_response = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        latency_ms = (time.perf_counter() - start) * 1000
        raw = json.loads(raw_response["body"].read())

        return self._parse_response(raw, latency_ms=latency_ms)

    async def acomplete(self, request: RouterRequest) -> RouterResponse:
        """
        Async completion via Bedrock.

        Note: boto3 is synchronous — we run it in a thread pool
        so it doesn't block the event loop.
        """
        import anyio

        # run sync boto3 call in a thread so we don't block async event loop
        return await anyio.to_thread.run_sync(
            lambda: self.complete(request)
        )
    
    def stream(self, request: RouterRequest) -> Iterator[str]:
        """
        Synchronous streaming via Bedrock's invoke_model_with_response_stream.
        Yields text chunks as they arrive.
        """
        body = self._build_body(request)

        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            # Bedrock streams delta events
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")

    async def astream(self, request: RouterRequest) -> AsyncIterator[str]:
        """
        Async streaming — runs sync stream() in a thread,
        yields chunks to async callers.
        """
        import anyio

        # collect chunks from sync stream in a thread
        # then yield them async
        chunks: list[str] = []

        def _collect():
            for chunk in self.stream(request):
                chunks.append(chunk)

        await anyio.to_thread.run_sync(_collect)

        for chunk in chunks:
            yield chunk

    def health_check(self) -> bool:
        """
        Verify Bedrock is reachable with a minimal request.
        """
        try:
            self.complete(RouterRequest(prompt="hi", max_tokens=10))
            return True
        except Exception:
            return False