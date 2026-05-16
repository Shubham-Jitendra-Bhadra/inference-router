# inference-router

A pluggable Python SDK for intelligent LLM inference routing. Route requests across model tiers based on complexity, cost, and latency — without changing your application code.

```python
from llm-inference-router import InferenceRouter
from llm-inference-router.providers.bedrock import BedrockProvider
from llm-inference-router.strategies import ComplexityStrategy

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "balanced": BedrockProvider("us.anthropic.claude-sonnet-4-6"),
    },
    strategy=ComplexityStrategy(),
    fallback="fast"
)

response = router.complete("explain recursion in one sentence")
print(response.text)
print(response.model_used)   # "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
print(response.cost_usd)     # 0.000016
print(response.latency_ms)   # 1263.4
print(response.tier_used)    # "fast"
```

---

## Why inference-router?

Every LLM app today sends every request to the same model — same cost, same latency, regardless of how simple or complex the question is. That's like using a surgeon for a bandaid.

`inference-router` sits between your app and your LLM providers. It classifies each request and dispatches it to the right model automatically:

- **Simple query** → small, fast, cheap model
- **Complex reasoning** → large, powerful model
- **Budget exceeded** → downgrade tier automatically
- **Provider slow or down** → failover to backup instantly

No changes to your application code. Zero vendor lock-in. Swap providers in one line.

---

## Installation

Core SDK:
```bash
pip install llm-inference-router
```

With provider extras:
```bash
# AWS Bedrock
pip install "llm-inference-router[bedrock]"

# OpenAI / OpenAI-compatible APIs (Groq, DeepInfra, Together AI)
pip install "llm-inference-router[openai]"

# Anthropic direct API
pip install "llm-inference-router[anthropic]"

# Multiple providers
pip install "llm-inference-router[bedrock,openai]"
```

---

## Quickstart

```python
from dotenv import load_dotenv
load_dotenv()

from llm-inference-router import InferenceRouter
from llm-inference-router.providers.bedrock import BedrockProvider
from llm-inference-router.strategies import ChainStrategy, CostStrategy, LatencyStrategy, ComplexityStrategy

router = InferenceRouter(
    tiers={
        "fast":     BedrockProvider("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        "balanced": BedrockProvider("us.anthropic.claude-sonnet-4-6"),
    },
    strategy=ChainStrategy([
        CostStrategy(budget_usd_per_day=5.0, tiers_by_cost=["balanced", "fast"]),
        LatencyStrategy(sla_ms=3000, preferred_tier="balanced", fallback_tier="fast"),
        ComplexityStrategy(),
    ]),
    fallback="fast"
)

# simple — routes to fast automatically
response = router.complete("what is the capital of France?")
print(response.tier_used)   # fast
print(response.cost_usd)    # ~$0.000016

# complex — routes to balanced automatically
response = router.complete("explain the tradeoffs between SQL and NoSQL in detail")
print(response.tier_used)   # balanced

# async
response = await router.acomplete("explain recursion")

# streaming
for chunk in router.stream("write a haiku about distributed systems"):
    print(chunk, end="", flush=True)

# force a specific tier
response = router.complete("hello", tier="balanced")
```

---

## Providers

### AWS Bedrock

Credentials loaded automatically from `~/.aws/credentials` or environment variables.
Newer Claude models require the cross-region inference profile prefix (`us.`).

```python
from llm-inference-router.providers.bedrock import BedrockProvider

provider = BedrockProvider(
    model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    region="us-east-1"
)
```

### Generic HTTP — OpenAI-compatible APIs

One provider covers Groq, DeepInfra, Together AI, Fireworks, Ollama, and any
API following the OpenAI chat completions format:

```python
from llm-inference-router.providers.http import HTTPProvider

# Groq
provider = HTTPProvider(
    base_url="https://api.groq.com/openai/v1",
    api_key="your-key",
    model="mixtral-8x7b-32768"
)

# DeepInfra
provider = HTTPProvider(
    base_url="https://api.deepinfra.com/v1/openai",
    api_key="your-key",
    model="meta-llama/Meta-Llama-3-8B-Instruct"
)

# Ollama (local, no auth needed)
provider = HTTPProvider(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="llama3"
)
```

### Custom providers

Support any API in ~30 lines by subclassing `BaseProvider`:

```python
from llm-inference-router.providers.base import BaseProvider
from llm-inference-router.models import RouterRequest, RouterResponse, TokenUsage, RoutingDecision
import httpx

class MyCustomProvider(BaseProvider):

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    def name(self) -> str:
        return f"mycustom/{self.model}"

    def complete(self, request: RouterRequest) -> RouterResponse:
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

Scores prompts 0–10 across five dimensions — token length, question count,
reasoning keywords, code presence, and simple-query detection. Maps score to a tier.

```python
from llm-inference-router.strategies import ComplexityStrategy

strategy = ComplexityStrategy(
    rules={
        "fast":     (0, 3),    # score 0-3
        "balanced": (3, 6),    # score 3-6
        "powerful": (6, 10),   # score 6-10
    }
)
```

Example scores:

| Prompt | Score | Tier |
|---|---|---|
| "what is the capital of France?" | 0.0 | fast |
| "explain how transformers work" | 3.5 | balanced |
| "compare B-trees vs LSM trees, explain tradeoffs" | 7.5 | powerful |

Debug the scorer:

```python
result = strategy.explain("compare B-trees vs LSM trees")
print(result["score"])          # 7.5
print(result["tier_selected"])  # powerful
print(result["breakdown"])      # per-dimension scores
```

### Cost strategy

Tracks running spend globally and per-user. Downgrades tier as budget is consumed.

```python
from llm-inference-router.strategies import CostStrategy

strategy = CostStrategy(
    budget_usd_per_day=10.0,
    tiers_by_cost=["powerful", "balanced", "fast"],  # expensive → cheap
    downgrade_at=0.8,    # downgrade at 80% budget used
    floor_at=0.95,       # use cheapest tier at 95% budget used
)

# per-user budgets via metadata
response = router.complete(
    "your prompt",
    metadata={"user_id": "user_123", "budget_usd": 1.0}
)
```

### Latency strategy

Tracks rolling p90 latency per tier. Failovers when SLA is breached.

```python
from llm-inference-router.strategies import LatencyStrategy

strategy = LatencyStrategy(
    sla_ms=3000,
    preferred_tier="balanced",
    fallback_tier="fast",
    window_size=50,
    min_samples=5,
)
```

### Chain strategy

Combines strategies in priority order. Hard constraints first, complexity last.

```python
from llm-inference-router.strategies import ChainStrategy

strategy = ChainStrategy([
    CostStrategy(budget_usd_per_day=10.0, tiers_by_cost=["balanced", "fast"]),
    LatencyStrategy(sla_ms=3000, preferred_tier="balanced", fallback_tier="fast"),
    ComplexityStrategy(),
])
```

---

## RouterResponse

Every completion returns the same normalized shape regardless of provider:

```python
response.text                        # generated text
response.model_used                  # "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
response.tier_used                   # "fast"
response.tokens.input                # 42
response.tokens.output               # 180
response.tokens.total                # 222
response.latency_ms                  # 1263.4
response.cost_usd                    # 0.000016
response.routing.tier_selected       # "fast"
response.routing.strategy_used       # "ChainStrategy"
response.routing.reason              # "" or "fallback"
response.raw                         # original provider response
```

---

## FastAPI layer

Run `inference-router` as a REST API — any app in any language can use it over HTTP.

### Start the server

```bash
python -m uvicorn app.main:app --reload
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | health check, server status |
| `POST` | `/query` | route a prompt, get response |
| `POST` | `/query/stream` | streaming response |
| `GET` | `/tiers` | list configured tiers and providers |
| `POST` | `/debug/complexity` | debug complexity scorer for a prompt |
| `DELETE` | `/stats/reset` | reset cost and latency counters |

### Example request

```bash
POST /query
Content-Type: application/json

{
    "prompt": "explain recursion",
    "max_tokens": 512,
    "temperature": 0.7,
    "system_prompt": "Reply concisely.",
    "tier": "balanced"
}
```

### Example response

```json
{
    "text": "Recursion is when a function calls itself...",
    "model_used": "bedrock/us.anthropic.claude-sonnet-4-6",
    "tier_used": "balanced",
    "tokens_total": 180,
    "cost_usd": 0.002340,
    "latency_ms": 3420.1,
    "strategy_used": "ChainStrategy",
    "was_fallback": false
}
```

Interactive docs: `http://localhost:8000/docs`

---

## Multi-turn conversations

```python
from llm-inference-router.models import Message

response = router.complete(
    prompt="what did I just tell you?",
    messages=[
        Message(role="user", content="my name is Shubham"),
        Message(role="assistant", content="Nice to meet you, Shubham!"),
    ]
)
```

---

## Project structure

```
inference-router/
├── llm-inference-router/
│   ├── __init__.py              # public API
│   ├── router.py                # core InferenceRouter class
│   ├── models.py                # Pydantic request/response models
│   ├── providers/
│   │   ├── base.py              # BaseProvider — implement for any API
│   │   ├── bedrock.py           # AWS Bedrock
│   │   └── http.py              # Generic HTTP for Groq, DeepInfra, etc.
│   └── strategies/
│       ├── base.py              # BaseStrategy interface
│       ├── complexity.py        # heuristic complexity scorer
│       ├── cost.py              # budget-based routing
│       ├── latency.py           # SLA-based routing
│       └── chain.py             # combine multiple strategies
├── app/
│   └── main.py                  # FastAPI REST API layer
├── examples/
│   └── basic_usage.py           # end-to-end usage examples
├── tests/
├── .env
├── pyproject.toml
└── README.md
```

