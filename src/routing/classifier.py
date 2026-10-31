"""Hybrid lexical + semantic prototype prompt complexity scorer, S in [0.0, 1.0], runs in <1ms."""
from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

REASONING_KEYWORDS: frozenset[str] = frozenset({
    "prove", "derive", "analyze", "trade-off", "tradeoff", "architecture",
    "architect", "concurrency", "thread-safe", "deadlock", "edge case",
    "root cause", "debug", "difference between", "compare", "optimize",
    # Expanded coding, debugging, system design & math signals
    "infinite loop", "memory leak", "race condition", "time complexity",
    "space complexity", "big-o", "algorithm", "binary search",
    "dynamic programming", "recursion", "distributed", "scalability",
    "microservice", "rate limiter", "load balancer", "consensus",
    "b-tree", "lsm", "integral", "derivative", "eigenvalue", "differential",
    "theorem", "probability", "why does my", "stack trace", "segmentation fault",
    "segfault", "nullpointerexception",
})
CODE_INDICATORS: frozenset[str] = frozenset({
    "def ", "class ", "fn ", "pub fn", "impl", "struct", "async ",
    "import ", "from ", "SELECT ", "WHERE ", "JOIN ", "```", "{", "}",
})
CODE_KEYWORDS: frozenset[str] = frozenset({
    "function", "implement", "refactor", "useeffect", "react", "typescript",
    "javascript", "golang", "rust", "c++", "sql query", "regex", "dockerfile",
    "kubernetes", "fastapi", "pytorch", "asyncio", "leetcode", "linked list",
    "hash map", "lru cache", "binary tree", "quicksort", "mergesort",
})
SIMPLE_TASK_KEYWORDS: frozenset[str] = frozenset({
    "fix typo", "fix typos", "grammar", "summarize", "summary",
    "translate", "rephrase", "format as", "bullet points", "polite", "casual",
})

COMPLEX_PROTOTYPES: tuple[str, ...] = (
    "Write a Python function or algorithm to implement a data structure",
    "Debug this code error, stack trace, infinite loop, or exception",
    "Design a distributed system architecture with high scalability and concurrency",
    "Solve this calculus integral, linear algebra, probability, or mathematical proof",
    "Write an optimized SQL query with joins, indexes, and aggregations",
)
SIMPLE_PROTOTYPES: tuple[str, ...] = (
    "Hello, how are you doing today?",
    "What is the capital city of a country or a basic factual question?",
    "Translate this short phrase or sentence into another language",
    "Fix typos, grammar, summarize in bullet points, or rephrase politely",
    "Tell me a short joke or a simple everyday greeting",
)

# Feature weights and caps
REASONING_WEIGHT_PER_HIT = 0.18
REASONING_CAP = 0.45
CODE_WEIGHT_PER_HIT = 0.10
CODE_KEYWORD_WEIGHT_PER_HIT = 0.18
CODE_CAP = 0.35
VOLUME_CAP = 0.20
VOLUME_SATURATION_WORDS = 150.0
SIMPLE_TASK_DISCOUNT = 0.25
SEMANTIC_SCALE = 1.2
SEMANTIC_CAP = 0.35
SEMANTIC_MARGIN_FLOOR = 0.03

_COMPLEX_CENTROID: np.ndarray | None = None
_SIMPLE_CENTROID: np.ndarray | None = None


def _centroid(vectors: Sequence[bytes]) -> np.ndarray:
    mat = np.vstack([np.frombuffer(v, dtype=np.float32) for v in vectors])
    mean = mat.mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean


def register_semantic_prototypes(
    complex_vectors: Sequence[bytes], simple_vectors: Sequence[bytes]
) -> None:
    """Store normalized 384-d Complex and Simple anchor centroids for <0.1ms routing."""
    global _COMPLEX_CENTROID, _SIMPLE_CENTROID
    if complex_vectors and simple_vectors:
        _COMPLEX_CENTROID = _centroid(complex_vectors)
        _SIMPLE_CENTROID = _centroid(simple_vectors)


def _semantic_score(vector: bytes | None) -> float:
    if vector is None or _COMPLEX_CENTROID is None or _SIMPLE_CENTROID is None:
        return 0.0
    vec = np.frombuffer(vector, dtype=np.float32)
    if vec.shape != _COMPLEX_CENTROID.shape:
        return 0.0
    margin = float(vec @ _COMPLEX_CENTROID - vec @ _SIMPLE_CENTROID)
    if margin > SEMANTIC_MARGIN_FLOOR:
        return min(SEMANTIC_CAP, (margin - SEMANTIC_MARGIN_FLOOR) * SEMANTIC_SCALE)
    if margin < -SEMANTIC_MARGIN_FLOOR:
        return -min(0.15, (-margin - SEMANTIC_MARGIN_FLOOR) * 0.6)
    return 0.0


class RouteTier(StrEnum):
    LOCAL_SMALL = "LOCAL_SMALL"
    CLOUD_FRONTIER = "CLOUD_FRONTIER"


class ComplexityBreakdown(BaseModel):
    reasoning: float
    code_syntax: float
    volume: float
    simple_discount: float
    semantic: float = 0.0


class ComplexityResult(BaseModel):
    score: float
    tier: RouteTier
    breakdown: ComplexityBreakdown


def score_prompt_complexity(
    prompt: str, threshold: float, vector: bytes | None = None
) -> ComplexityResult:
    """Score prompt difficulty; `score >= threshold` routes to the cloud frontier tier."""
    text_lower = prompt.lower()
    word_count = len(prompt.split())

    reasoning_hits = sum(1 for kw in REASONING_KEYWORDS if kw in text_lower)
    score_reasoning = min(REASONING_CAP, reasoning_hits * REASONING_WEIGHT_PER_HIT)

    syntax_hits = sum(1 for ind in CODE_INDICATORS if ind in prompt)
    code_kw_hits = sum(1 for kw in CODE_KEYWORDS if kw in text_lower)
    raw_code = syntax_hits * CODE_WEIGHT_PER_HIT + code_kw_hits * CODE_KEYWORD_WEIGHT_PER_HIT
    score_code = 0.30 if "```" in prompt and raw_code < 0.30 else min(CODE_CAP, raw_code)

    score_volume = min(VOLUME_CAP, (word_count / VOLUME_SATURATION_WORDS) * VOLUME_CAP)

    is_simple = any(kw in text_lower for kw in SIMPLE_TASK_KEYWORDS)
    discount = SIMPLE_TASK_DISCOUNT if is_simple else 0.0

    score_semantic = _semantic_score(vector)

    raw = score_reasoning + score_code + score_volume + score_semantic - discount
    final_score = round(max(0.0, min(1.0, raw)), 3)

    return ComplexityResult(
        score=final_score,
        tier=RouteTier.CLOUD_FRONTIER if final_score >= threshold else RouteTier.LOCAL_SMALL,
        breakdown=ComplexityBreakdown(
            reasoning=round(score_reasoning, 2),
            code_syntax=round(score_code, 2),
            volume=round(score_volume, 2),
            simple_discount=-round(discount, 2),
            semantic=round(score_semantic, 2),
        ),
    )
