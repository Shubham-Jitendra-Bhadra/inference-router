from __future__ import annotations

import re
from inference_router.models import RouterRequest
from inference_router.strategies.base import BaseStrategy


# Keywords that signal complex reasoning is needed
REASONING_KEYWORDS = [
    "explain", "why", "how does", "compare", "difference between",
    "tradeoffs", "pros and cons", "analyze", "evaluate", "critique",
    "design", "architect", "implement", "optimize", "debug",
    "what would happen", "step by step", "in detail", "walk me through",
]

# Keywords that signal code-related complexity
CODE_KEYWORDS = [
    "code", "function", "class", "algorithm", "implement", "debug",
    "error", "exception", "refactor", "performance", "complexity",
    "big o", "runtime", "memory", "async", "concurrent", "thread",
]

# Keywords that signal simple lookups
SIMPLE_KEYWORDS = [
    "what is", "define", "who is", "when was", "where is",
    "capital of", "how many", "what does", "spell", "translate",
]


def _count_tokens(text: str) -> int:
    """
    Approximate token count — splits on whitespace and punctuation.
    Not exact but good enough for routing decisions.
    Rule of thumb: 1 token ≈ 0.75 words
    """
    words = len(text.split())
    return int(words / 0.75)


def _count_questions(text: str) -> int:
    """Count number of questions in the prompt."""
    return text.count("?")


def _contains_code(text: str) -> bool:
    """Check if prompt contains code blocks or code-like patterns."""
    has_code_block = "```" in text or "`" in text
    has_code_keyword = any(kw in text.lower() for kw in CODE_KEYWORDS)
    return has_code_block or has_code_keyword


def _reasoning_keyword_count(text: str) -> int:
    """Count how many reasoning keywords appear in the prompt."""
    text_lower = text.lower()
    return sum(1 for kw in REASONING_KEYWORDS if kw in text_lower)


def _is_simple(text: str) -> bool:
    """Check if prompt looks like a simple factual lookup."""
    text_lower = text.lower()
    return any(text_lower.startswith(kw) for kw in SIMPLE_KEYWORDS)


def score_prompt(prompt: str) -> float:
    """
    Score a prompt's complexity on a 0-10 scale.

    0-3:  simple   — factual lookups, short questions
    3-6:  moderate — explanations, single-topic analysis
    6-10: complex  — multi-step reasoning, code, comparisons

    Args:
        prompt: the user's prompt text

    Returns:
        float between 0.0 and 10.0
    """
    score = 0.0
    token_count = _count_tokens(prompt)

    # --- length score (0-3 points) ---
    if token_count < 20:
        score += 0.5
    elif token_count < 50:
        score += 1.0
    elif token_count < 150:
        score += 2.0
    elif token_count < 300:
        score += 2.5
    else:
        score += 3.0

    # --- question count score (0-2 points) ---
    questions = _count_questions(prompt)
    if questions == 1:
        score += 0.5
    elif questions == 2:
        score += 1.0
    elif questions >= 3:
        score += 2.0

    # --- reasoning keywords (0-3 points) ---
    reasoning_count = _reasoning_keyword_count(prompt)
    if reasoning_count == 1:
        score += 1.0
    elif reasoning_count == 2:
        score += 2.0
    elif reasoning_count >= 3:
        score += 3.0

    # --- code presence (0-2 points) ---
    if _contains_code(prompt):
        score += 2.0

    # --- simple keyword penalty (-2 points) ---
    # pulls score down for obvious simple lookups
    if _is_simple(prompt):
        score -= 2.0

    # clamp to 0-10
    return round(max(0.0, min(10.0, score)), 2)


class ComplexityStrategy(BaseStrategy):
    """
    Routes requests based on heuristic complexity scoring.

    Scores the prompt across 5 dimensions: length, question count,
    reasoning keywords, code presence, and simple-query detection.
    Maps the score to a tier using user-defined score ranges.

    Default tier names are "fast", "balanced", "powerful" but
    you can use any names that match your router's tiers.

    Usage:
        # default — 3 tiers with standard score ranges
        strategy = ComplexityStrategy()

        # custom tier names and ranges
        strategy = ComplexityStrategy(
            rules={
                "cheap":    (0, 3),
                "standard": (3, 7),
                "premium":  (7, 10),
            }
        )

        # 2 tiers only
        strategy = ComplexityStrategy(
            rules={
                "small": (0, 5),
                "large": (5, 10),
            }
        )
    """

    # default score ranges — covers most use cases
    DEFAULT_RULES = {
        "fast":     (0.0, 3.0),
        "balanced": (3.0, 6.0),
        "powerful": (6.0, 10.0),
    }

    def __init__(
        self,
        rules: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Args:
            rules: mapping of tier_name → (min_score, max_score)
                   score ranges are inclusive on min, exclusive on max
                   except the last range which is fully inclusive
                   if None, uses DEFAULT_RULES
        """
        self.rules = rules or self.DEFAULT_RULES.copy()

    def select_tier(
        self,
        request: RouterRequest,
        available_tiers: list[str],
    ) -> str:
        """
        Score the prompt and return the matching tier.

        Falls back to the first available tier if no rule matches
        or if configured tier names don't match available tiers.
        """
        prompt = request.prompt
        if request.messages:
            # include last user message in scoring if multi-turn
            last_user = next(
                (m.content for m in reversed(request.messages) if m.role == "user"),
                ""
            )
            prompt = f"{prompt} {last_user}".strip()

        complexity_score = score_prompt(prompt)

        # find matching tier from rules
        for tier_name, (min_score, max_score) in self.rules.items():
            if min_score <= complexity_score <= max_score:
                # make sure this tier is actually available in the router
                if tier_name in available_tiers:
                    return tier_name

        # fallback — return first available tier
        return available_tiers[0]

    def explain(self, prompt: str) -> dict:
        """
        Debug helper — shows the full scoring breakdown for a prompt.
        Useful for tuning your rules.

        Returns:
            dict with score, tier selected, and per-dimension breakdown
        """
        score = score_prompt(prompt)
        tier = self.select_tier(
            RouterRequest(prompt=prompt),
            list(self.rules.keys())
        )
        return {
            "prompt_preview": prompt[:80] + "..." if len(prompt) > 80 else prompt,
            "score": score,
            "tier_selected": tier,
            "breakdown": {
                "token_count": _count_tokens(prompt),
                "question_count": _count_questions(prompt),
                "reasoning_keywords": _reasoning_keyword_count(prompt),
                "contains_code": _contains_code(prompt),
                "is_simple": _is_simple(prompt),
            }
        }