## Work in progress

# inference-router

A pluggable Python SDK for intelligent LLM inference routing. Route requests across model tiers based on complexity, cost, and latency — without changing your application code.

```python
from inference_router import InferenceRouter
from inference_router.providers.bedrock import BedrockProvider
from inference_router.strategies import ComplexityStrategy

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("anthropic.claude-haiku-4-5-20251001"),
        "balanced": BedrockProvider("anthropic.claude-sonnet-4-6"),
        "powerful": BedrockProvider("anthropic.claude-opus-4-6"),
    },
    strategy=ComplexityStrategy(),
    fallback="fast"
)

response = router.complete("explain recursion in one sentence")
print(response.text)
print(response.model_used)   # "bedrock/anthropic.claude-haiku-4-5-20251001"
print(response.cost_usd)     # 0.000019
print(response.latency_ms)   # 312.4
```

---

## Why inference-router?

Every LLM app today sends every request to the same model — same cost, same latency, regardless of how simple or complex the question is. That's like using a surgeon for a bandaid.

`inference-router` sits between your app and your LLM providers. It classifies each request and dispatches it to the right model automatically:

- **Simple query** → small, fast, cheap model
- **Complex reasoning** → large, powerful model
- **Budget exceeded** → downgrade tier automatically
- **Provider slow or down** → failover to backup instantly

No changes to your application code. Just wrap your LLM calls with the router.

---

## Installation

Core SDK (no providers):
```bash
pip install inference-router
```

With provider extras:
```bash
# AWS Bedrock
pip install "inference-router[bedrock]"

# OpenAI
pip install "inference-router[openai]"

# Anthropic direct API
pip install "inference-router[anthropic]"

# Multiple providers
pip install "inference-router[bedrock,openai]"
```

---

## Providers

### Built-in providers

#### AWS Bedrock
```python
from inference_router.providers.bedrock import BedrockProvider

# Credentials loaded from ~/.aws/credentials or environment variables
provider = BedrockProvider(
    model_id="anthropic.claude-haiku-4-5-20251001",
    region="us-east-1"
)
```

#### OpenAI
```python
from inference_router.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    model="gpt-4o",
    api_key="sk-..."
)
```

#### Anthropic direct API
```python
from inference_router.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(
    model="claude-opus-4-6",
    api_key="sk-ant-..."
)
```

#### Generic HTTP (OpenAI-compatible APIs)
Works with DeepInfra, Groq, Together AI, Fireworks, Anyscale, and any
provider that exposes an OpenAI-compatible endpoint:

```python
from inference_router.providers.http import HTTPProvider

# DeepInfra
provider = HTTPProvider(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key="your-key",
    model="meta-llama/Meta-Llama-3-8B-Instruct"
)

# Groq
provider = HTTPProvider(
    base_url="https://api.groq.com/openai/v1",
    api_key="your-key",
    model="mixtral-8x7b-32768"
)

# Together AI
provider = HTTPProvider(
    base_url="https://api.together.xyz/v1",
    api_key="your-key",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1"
)
```

### Custom providers

Support any API in ~30 lines by subclassing `BaseProvider`:

```python
from inference_router.providers.base import BaseProvider
from inference_router.models import RouterRequest, RouterResponse, TokenUsage, RoutingDecision

class MyCustomProvider(BaseProvider):

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def name(self) -> str:
        return f"mycustom/{self.model}"

    def complete(self, request: RouterRequest) -> RouterResponse:
        import httpx
        response = httpx.post(
            "https://api.mycustom.com/v1/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "prompt": request.prompt}
        )
        data = response.json()
        return RouterResponse(
            text=data["output"],
            model_used=self.name,
            tier_used="",
            tokens=TokenUsage(input=0, output=0, total=0),
            latency_ms=0.0,
            cost_usd=0.0,
            routing=RoutingDecision(tier_selected="", strategy_used="", reason=""),
        )

    async def acomplete(self, request: RouterRequest) -> RouterResponse:
        import anyio
        return await anyio.to_thread.run_sync(lambda: self.complete(request))
```

---

## Routing strategies

### Complexity strategy
Routes based on a heuristic complexity score of the prompt.
Scores prompts by length, question count, code presence, and reasoning keywords.

```python
from inference_router.strategies import ComplexityStrategy

strategy = ComplexityStrategy(
    rules={
        "fast":     (0, 3),   # complexity score 0-3
        "balanced": (3, 7),   # complexity score 3-7
        "powerful": (7, 10),  # complexity score 7-10
    }
)
```

How scoring works:
- Short prompt, no reasoning keywords → score 1-2 → `fast`
- Medium prompt with some analysis → score 4-6 → `balanced`
- Long prompt, code, "explain why", "compare", "tradeoffs" → score 7-9 → `powerful`

### Cost strategy
Routes based on a daily/per-request budget. Downgrades tier when budget is exceeded.

```python
from inference_router.strategies import CostStrategy

strategy = CostStrategy(
    budget_usd_per_day=10.0,
    tier_when_exceeded="fast"   # fallback tier when budget runs out
)

# Per-user budgets via request metadata
response = router.complete(
    "your prompt",
    metadata={"user_id": "user_123", "budget_usd": 1.0}
)
```

### Latency strategy
Routes based on real-time p90 latency per provider. Switches to a faster
tier when latency exceeds your SLA threshold.

```python
from inference_router.strategies import LatencyStrategy

strategy = LatencyStrategy(
    sla_ms=800,              # target p90 latency
    fallback_tier="fast"     # switch to this tier when SLA breached
)
```

### Chaining strategies
Combine multiple strategies — evaluated in order, first match wins:

```python
from inference_router.strategies import ChainStrategy

strategy = ChainStrategy([
    CostStrategy(budget_usd_per_day=10.0),   # check budget first
    LatencyStrategy(sla_ms=800),              # then check latency
    ComplexityStrategy(),                      # finally route by complexity
])
```

---

## Router

### Basic usage

```python
from inference_router import InferenceRouter
from inference_router.providers.bedrock import BedrockProvider
from inference_router.strategies import ComplexityStrategy

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("anthropic.claude-haiku-4-5-20251001"),
        "balanced": BedrockProvider("anthropic.claude-sonnet-4-6"),
    },
    strategy=ComplexityStrategy(),
    fallback="fast"
)
```

### Sync completion
```python
response = router.complete("what is the capital of France?")
print(response.text)
```

### Async completion
```python
response = await router.acomplete("explain transformer architecture")
print(response.text)
```

### Streaming
```python
# sync
for chunk in router.stream("write a short story"):
    print(chunk, end="", flush=True)

# async
async for chunk in router.astream("write a short story"):
    print(chunk, end="", flush=True)
```

### Force a specific tier
```python
response = router.complete("your prompt", tier="powerful")
```

### Multi-turn conversations
```python
from inference_router.models import Message

response = router.complete(
    prompt="what did I just ask?",
    messages=[
        Message(role="user", content="my name is Shubham"),
        Message(role="assistant", content="Nice to meet you, Shubham!"),
    ]
)
```

---

## RouterResponse

Every completion returns the same normalized shape regardless of provider:

```python
response.text           # the generated text
response.model_used     # "bedrock/anthropic.claude-haiku-4-5-20251001"
response.tier_used      # "fast"
response.tokens.input   # 42
response.tokens.output  # 180
response.tokens.total   # 222
response.latency_ms     # 312.4
response.cost_usd       # 0.000019
response.routing        # RoutingDecision object

# Routing decision details
response.routing.tier_selected   # "fast"
response.routing.strategy_used   # "ComplexityStrategy"
response.routing.score           # 2.1  (complexity score)
response.routing.reason          # "short prompt, no reasoning keywords"

# Original provider response (for debugging)
response.raw
```

---

## Observability

```python
stats = router.stats()

print(stats.requests_total)       # 1423
print(stats.cost_total_usd)       # 2.14
print(stats.routing_breakdown)    # {"fast": 0.68, "balanced": 0.27, "powerful": 0.05}
print(stats.avg_latency_ms)       # {"fast": 280.0, "balanced": 640.0}
print(stats.fallbacks_triggered)  # 3
print(stats.errors_total)         # 1
```

Reset stats:
```python
router.reset_stats()
```

---

## FastAPI integration

```python
from fastapi import FastAPI
from dotenv import load_dotenv
from inference_router import InferenceRouter
from inference_router.providers.bedrock import BedrockProvider
from inference_router.strategies import ComplexityStrategy
from pydantic import BaseModel

load_dotenv()
app = FastAPI()

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("anthropic.claude-haiku-4-5-20251001"),
        "balanced": BedrockProvider("anthropic.claude-sonnet-4-6"),
    },
    strategy=ComplexityStrategy(),
    fallback="fast"
)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query(req: QueryRequest):
    response = await router.acomplete(req.prompt)
    return {
        "text": response.text,
        "model_used": response.model_used,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
    }

@app.get("/stats")
def stats():
    return router.stats()
```

---

## Project structure

```
inference_router/
├── __init__.py              # public API
├── router.py                # core InferenceRouter class
├── models.py                # Pydantic request/response models
├── observability.py         # stats tracking
├── providers/
│   ├── base.py              # BaseProvider — implement for any API
│   ├── bedrock.py           # AWS Bedrock
│   ├── anthropic.py         # Anthropic direct API
│   ├── openai.py            # OpenAI / compatible APIs
│   └── http.py              # Generic HTTP for DeepInfra, Groq, etc.
└── strategies/
    ├── base.py              # BaseStrategy interface
    ├── complexity.py        # heuristic complexity scorer
    ├── cost.py              # budget-based routing
    ├── latency.py           # SLA-based routing
    └── chain.py             # combine multiple strategies
```

---

## Roadmap

- [ ] `providers/bedrock.py` — AWS Bedrock
- [ ] `providers/http.py` — Generic OpenAI-compatible HTTP
- [ ] `providers/openai.py` — OpenAI
- [ ] `providers/anthropic.py` — Anthropic direct
- [ ] `strategies/complexity.py` — heuristic scorer
- [ ] `strategies/cost.py` — budget enforcer
- [ ] `strategies/latency.py` — SLA routing
- [ ] `strategies/chain.py` — strategy chaining
- [ ] `router.py` — core router
- [ ] `observability.py` — stats tracking
- [ ] FastAPI example
- [ ] PyPI publish

---

## License

MIT
