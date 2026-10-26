"""Guardrails must veto numerically or logically divergent cache hits."""
import pytest

from src.cache.guardrails import extract_negations, extract_numbers, passes_guardrails


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        ("Calculate taxes for 2024", "Calculate taxes for 2025"),
        ("Give me 5 ideas", "Give me 10 ideas"),
        ("Summarize the report", "Summarize the 2024 report"),
        ("Deploy with tests", "Deploy without tests"),
        ("Is this safe to run", "Is this not safe to run"),
        ("I do want logs", "I don't want logs"),
    ],
)
def test_blocks_divergent_prompts(new: str, cached: str) -> None:
    assert passes_guardrails(new, cached) is False


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        ("Translate 'hello friend' into Spanish.", "Can you translate hello friend to Spanish?"),
        ("What happened in 2024?", "what happened in 2024"),  # trailing punctuation ignored
        ("Price is $1,000", "price is 1000"),  # thousands separator ignored
        ("Don't use recursion", "Do not use recursion"),  # contraction == 'not'
    ],
)
def test_allows_equivalent_prompts(new: str, cached: str) -> None:
    assert passes_guardrails(new, cached) is True


def test_extract_numbers_normalizes_punctuation() -> None:
    assert extract_numbers("Is it 2024? Costs $1,250.50 total.") == {"2024", "1250.50"}


def test_extract_negations_handles_contractions() -> None:
    assert extract_negations("It isn't done and never will be") == {"not", "never"}
    assert extract_negations("Nothing here") == {"nothing"}
    assert extract_negations("Notice the note") == set()  # substrings must not match
