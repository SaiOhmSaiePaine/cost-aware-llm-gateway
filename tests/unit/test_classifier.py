"""Complexity scorer edge cases and Phase 2.5 hybrid semantic + lexical routing."""
import numpy as np
import pytest

from src.routing.classifier import (
    RouteTier,
    build_semantic_centroids,
    score_prompt_complexity,
)

THRESHOLD = 0.35


def test_long_but_easy_prompt_routes_local() -> None:
    prompt = (
        "Please fix typos and summarize this in bullet points: we deployed the new caching "
        "layer yesterday and latency dropped by 40 percent across all endpoints"
    )
    result = score_prompt_complexity(prompt, THRESHOLD)
    assert result.score == 0.0
    assert result.tier is RouteTier.LOCAL_SMALL
    assert result.breakdown.simple_discount == -0.25


def test_short_but_hard_prompt_routes_cloud() -> None:
    result = score_prompt_complexity("Write a thread-safe singleton class in Python using def and class", THRESHOLD)
    assert result.tier is RouteTier.CLOUD_FRONTIER
    assert result.breakdown.reasoning == 0.18
    assert result.breakdown.code_syntax == 0.2


def test_code_block_saturates_code_score() -> None:
    result = score_prompt_complexity("Debug this:\n```python\nprint(x)\n```", THRESHOLD)
    assert result.breakdown.code_syntax == 0.30


def test_score_is_clamped_to_unit_interval() -> None:
    heavy = "prove derive analyze architecture concurrency deadlock " + "```code``` " + "word " * 500
    assert score_prompt_complexity(heavy, THRESHOLD).score <= 1.0
    assert score_prompt_complexity("translate hi", THRESHOLD).score >= 0.0


def test_empty_prompt_is_local() -> None:
    result = score_prompt_complexity("", THRESHOLD)
    assert result.score == 0.0
    assert result.tier is RouteTier.LOCAL_SMALL


def test_threshold_boundary_is_inclusive() -> None:
    # One reasoning hit (0.18) + small volume; set threshold exactly at the score.
    result = score_prompt_complexity("analyze", 0.0)
    assert result.tier is RouteTier.CLOUD_FRONTIER
    exact = score_prompt_complexity("analyze", THRESHOLD).score
    assert score_prompt_complexity("analyze", exact).tier is RouteTier.CLOUD_FRONTIER


@pytest.mark.parametrize(
    "prompt",
    [
        "Write a Python binary search function",
        "Why does my React useEffect cause an infinite loop?",
        "Design a distributed rate limiter with high scalability",
        "Derive and solve the integral of x^2 * e^x",
        "Implement an LRU cache in TypeScript",
    ],
)
def test_coding_debugging_design_and_math_prompts_route_cloud(prompt: str) -> None:
    result = score_prompt_complexity(prompt, THRESHOLD)
    assert result.score >= THRESHOLD
    assert result.tier is RouteTier.CLOUD_FRONTIER


@pytest.mark.parametrize(
    "prompt",
    [
        "Hello! How are you doing today?",
        "What is the capital of Japan?",
        "Translate 'good morning' into Spanish",
        "Summarize this short note in bullet points",
        "What is the biological function of the liver?",
        "How do I remove rust from iron tools?",
    ],
)
def test_greetings_facts_and_translations_route_local(prompt: str) -> None:
    result = score_prompt_complexity(prompt, THRESHOLD)
    assert result.score < THRESHOLD
    assert result.tier is RouteTier.LOCAL_SMALL


def test_semantic_prototype_boost_routes_complex_vector_to_cloud() -> None:
    dim = 384
    complex_vec = np.zeros(dim, dtype=np.float32)
    complex_vec[0] = 1.0
    simple_vec = np.zeros(dim, dtype=np.float32)
    simple_vec[1] = 1.0
    centroids = build_semantic_centroids([complex_vec.tobytes()], [simple_vec.tobytes()])

    # Prompt with no lexical keywords but whose embedding aligns with the complex prototype centroid
    result = score_prompt_complexity(
        "Build a custom parser for nested expressions",
        THRESHOLD,
        vector=complex_vec.tobytes(),
        centroids=centroids,
    )
    assert result.breakdown.semantic > 0.30
    assert result.tier is RouteTier.CLOUD_FRONTIER
