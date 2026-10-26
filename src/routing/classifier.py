"""Multi-factor heuristic prompt complexity scorer, S in [0.0, 1.0], runs in <1ms.

Ported 1:1 from the original classifier.py; weights are now named constants.
"""
from enum import StrEnum

from pydantic import BaseModel

REASONING_KEYWORDS: frozenset[str] = frozenset({
    "prove", "derive", "analyze", "trade-off", "tradeoff", "architecture",
    "architect", "concurrency", "thread-safe", "deadlock", "edge case",
    "root cause", "debug", "difference between", "compare", "optimize",
})
CODE_INDICATORS: frozenset[str] = frozenset({
    "def ", "class ", "fn ", "pub fn", "impl", "struct", "async ",
    "import ", "from ", "SELECT ", "WHERE ", "JOIN ", "```", "{", "}",
})
SIMPLE_TASK_KEYWORDS: frozenset[str] = frozenset({
    "fix typo", "fix typos", "grammar", "summarize", "summary",
    "translate", "rephrase", "format as", "bullet points", "polite", "casual",
})

# Feature weights and caps
REASONING_WEIGHT_PER_HIT = 0.18
REASONING_CAP = 0.35
CODE_WEIGHT_PER_HIT = 0.10
CODE_CAP = 0.30
VOLUME_CAP = 0.20
VOLUME_SATURATION_WORDS = 150.0
SIMPLE_TASK_DISCOUNT = 0.25


class RouteTier(StrEnum):
    LOCAL_SMALL = "LOCAL_SMALL"
    CLOUD_FRONTIER = "CLOUD_FRONTIER"


class ComplexityBreakdown(BaseModel):
    reasoning: float
    code_syntax: float
    volume: float
    simple_discount: float


class ComplexityResult(BaseModel):
    score: float
    tier: RouteTier
    breakdown: ComplexityBreakdown


def score_prompt_complexity(prompt: str, threshold: float) -> ComplexityResult:
    """Score prompt difficulty; `score >= threshold` routes to the cloud frontier tier."""
    text_lower = prompt.lower()
    word_count = len(prompt.split())

    reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in text_lower)
    score_reasoning = min(REASONING_CAP, reasoning_hits * REASONING_WEIGHT_PER_HIT)

    syntax_hits = sum(1 for ind in CODE_INDICATORS if ind in prompt)
    score_code = CODE_CAP if "```" in prompt else min(CODE_CAP, syntax_hits * CODE_WEIGHT_PER_HIT)

    score_volume = min(VOLUME_CAP, (word_count / VOLUME_SATURATION_WORDS) * VOLUME_CAP)

    is_simple = any(kw in text_lower for kw in SIMPLE_TASK_KEYWORDS)
    discount = SIMPLE_TASK_DISCOUNT if is_simple else 0.0

    raw = score_reasoning + score_code + score_volume - discount
    final_score = round(max(0.0, min(1.0, raw)), 3)

    return ComplexityResult(
        score=final_score,
        tier=RouteTier.CLOUD_FRONTIER if final_score >= threshold else RouteTier.LOCAL_SMALL,
        breakdown=ComplexityBreakdown(
            reasoning=round(score_reasoning, 2),
            code_syntax=round(score_code, 2),
            volume=round(score_volume, 2),
            simple_discount=-round(discount, 2),
        ),
    )
