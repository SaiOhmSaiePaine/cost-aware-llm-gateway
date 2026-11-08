"""Hybrid lexical + semantic prototype prompt complexity scorer, S in [0.0, 1.0], runs in <1ms."""
import re
from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

SemanticCentroids = tuple[np.ndarray, np.ndarray]

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
    "theorem", "probability", "stack trace", "segmentation fault",
    "segfault", "nullpointerexception",
})
CODE_INDICATORS: frozenset[str] = frozenset({
    "def ", "fn ", "pub fn", "impl ", "struct ", "async ",
    "SELECT ", "WHERE ", "JOIN ", "```", "{", "}",
})
_AMBIGUOUS_CODE_INDICATORS: frozenset[str] = frozenset({"class ", "import "})
_SQL_QUERY_RE = re.compile(
    r"\bselect\b.+?\bfrom\b\s+\w+\s+\b(?:where|(?:inner|left|right|full|cross)\s+join|join|group\s+by|order\s+by|limit)\b",
    re.IGNORECASE | re.DOTALL,
)
CODE_KEYWORDS: frozenset[str] = frozenset({
    "search function", "write a function", "implement", "refactor", "useeffect",
    "react", "typescript", "javascript", "golang", "in rust", "c++",
    "sql query", "regex", "dockerfile", "kubernetes", "fastapi", "pytorch",
    "asyncio", "leetcode", "linked list", "hash map", "lru cache",
    "binary tree", "quicksort", "mergesort",
})
def _compile_word_re(keywords: frozenset[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(kw) for kw in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alts})(?=s?(?!\w))")


_MAX_SCAN_CHARS = 4096
SIMPLE_TASK_KEYWORDS: frozenset[str] = frozenset({
    "fix typo", "grammar", "summarize", "summary",
    "translate", "rephrase", "format as", "bullet points", "polite", "casual",
})
_REASONING_KW_RE = _compile_word_re(REASONING_KEYWORDS)
_CODE_KW_RE = _compile_word_re(CODE_KEYWORDS)
_SIMPLE_KW_RE = _compile_word_re(SIMPLE_TASK_KEYWORDS)

COMPLEX_PROTOTYPES: tuple[str, ...] = (
    "Write a function, method, or algorithm in Python, JavaScript, Rust, Go, Java, or C++",
    "Implement a data structure such as a tree, graph, hash table, heap, queue, or cache",
    "Debug this software error, stack trace, crash, exit code, segmentation fault, or memory leak",
    "Troubleshoot a backend server, HTTP status error, API gateway, reverse proxy, or timeout",
    "Design a distributed system architecture with high scalability, replication, and fault tolerance",
    "Write an optimized SQL database query, schema migration, join, or index tuning",
    "Fix security vulnerabilities, authentication, encryption, or injection attacks",
    "Solve this calculus integral, derivative, linear algebra matrix, or differential equation",
    "Prove this mathematical theorem, recurrence relation, probability, or statistical model",
    "Write a shell script, CLI automation, file processor, or regular expression pattern",
    "Configure cloud infrastructure, containers, orchestration, or continuous integration pipelines",
    "Manage version control branches, commits, rebasing, and merge conflicts",
    "Build a frontend web interface, CSS layout, DOM component, or state management",
    "Serialize, parse, or transform structured data formats, schemas, and data frames",
    "Explain operating system internals, memory management, concurrency, threads, or compilers",
    "Write automated unit or integration tests with mocks, fixtures, and assertions",
)
SIMPLE_PROTOTYPES: tuple[str, ...] = (
    "Hello, how are you doing today?",
    "What is the capital city of a country or a basic world geography fact?",
    "Who won a historical event, sports championship, or wrote a famous novel?",
    "Translate this short phrase or everyday greeting into another language",
    "Summarize this short paragraph in bullet points or make it briefer",
    "Fix typos, spelling, grammar, or rewrite a sentence to sound polite",
    "Draft a short thank-you note, out-of-office reply, or casual message",
    "Give me a kitchen recipe or step-by-step instructions to cook food at home",
    "How many cups, ounces, miles, or minutes are in an everyday measurement?",
    "Recommend a few popular movies, novels, songs, or vacation spots",
    "What are common symptoms of a minor illness or general wellness habits?",
    "Explain a basic nature, plant, or animal fact to a young child",
    "What time zone is a city in or what is the local weather forecast?",
    "List a few ideas for breakfast, birthday gifts, or weekend hobbies",
    "Tell me a short family-friendly joke, riddle, or fun animal trivia",
    "What is the everyday meaning of a common word or proverb?",
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
SEMANTIC_SCALE = 6.0
SEMANTIC_CAP = 0.55
SEMANTIC_MARGIN_FLOOR = 0.03


def _normalize_rows(vectors: Sequence[bytes]) -> np.ndarray:
    mat = np.vstack([np.frombuffer(v, dtype=np.float32) for v in vectors])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.divide(mat, norms, out=mat, where=norms > 0)


def build_semantic_centroids(
    complex_vectors: Sequence[bytes], 
    simple_vectors: Sequence[bytes]
) -> SemanticCentroids:
    """Build normalized 384-d Complex and Simple anchor matrices for <0.1ms max-margin routing."""
    return _normalize_rows(complex_vectors), _normalize_rows(simple_vectors)


def _semantic_score(vector: bytes | None, centroids: SemanticCentroids | None) -> float:
    if vector is None or centroids is None:
        return 0.0
    complex_mat, simple_mat = centroids
    vec = np.frombuffer(vector, dtype=np.float32)
    if vec.ndim != 1 or vec.shape[0] != complex_mat.shape[-1]:
        return 0.0
    margin = float(np.max(complex_mat @ vec) - np.max(simple_mat @ vec))
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
    prompt: str,
    threshold: float,
    vector: bytes | None = None,
    centroids: SemanticCentroids | None = None,
) -> ComplexityResult:
    """Score prompt difficulty; `score >= threshold` routes to the cloud frontier tier."""
    sample = prompt[:_MAX_SCAN_CHARS]
    text_lower = sample.lower()
    word_count = len(sample.split())

    reasoning_hits = len(set(_REASONING_KW_RE.findall(text_lower)))
    score_reasoning = min(REASONING_CAP, reasoning_hits * REASONING_WEIGHT_PER_HIT)

    syntax_hits = sum(1 for ind in CODE_INDICATORS if ind in sample)
    code_kw_hits = len(set(_CODE_KW_RE.findall(text_lower)))
    if _SQL_QUERY_RE.search(sample) and syntax_hits < 2:
        syntax_hits = max(syntax_hits, 2)
    if syntax_hits > 0 or code_kw_hits > 0:
        syntax_hits += sum(1 for ind in _AMBIGUOUS_CODE_INDICATORS if ind in sample)
    raw_code = syntax_hits * CODE_WEIGHT_PER_HIT + code_kw_hits * CODE_KEYWORD_WEIGHT_PER_HIT
    score_code = 0.30 if "```" in sample and raw_code < 0.30 else min(CODE_CAP, raw_code)

    score_volume = min(VOLUME_CAP, (word_count / VOLUME_SATURATION_WORDS) * VOLUME_CAP)

    is_simple = _SIMPLE_KW_RE.search(text_lower) is not None
    discount = SIMPLE_TASK_DISCOUNT if is_simple else 0.0

    score_semantic = _semantic_score(vector, centroids)

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
