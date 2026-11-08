"""Guardrails must veto numerically or logically divergent cache hits."""
import pytest

from src.cache.guardrails import extract_negations, extract_numbers, passes_guardrails


@pytest.mark.parametrize(
    ("new", "cached"),
    [
        ("Calculate taxes for 2024", "Calculate taxes for 2025"),
        ("Give me 5 ideas", "Give me 10 ideas"),
        ("Summarize the report", "Summarize the 2024 report"),
        ("What is 10 divided by 2?", "What is 2 divided by 10?"),
        ("Deploy with tests", "Deploy without tests"),
        ("Is this safe to run", "Is this not safe to run"),
        ("I do want logs", "I don't want logs"),
        ("I do want logs", "I don’t want logs"),
        ("Explain recursion simply", "Explain recursion in detail"),
        ("Explain recursion", "Explain recursion briefly"),
        ("Convert -40C to Fahrenheit", "Convert 40C to Fahrenheit"),
        ("Balance is -500 dollars", "Balance is 500 dollars"),
        ("Explain ipv4", "Explain ipv6"),
        ("Compare sha256 and md5", "Compare sha512 and md5"),
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
        ("Don’t use recursion", "Do not use recursion"),  # curly apostrophe == 'not'
        ("Why can't I connect?", "Why cannot I connect?"),  # can't == cannot == 'not'
    ],
)
def test_allows_equivalent_prompts(new: str, cached: str) -> None:
    assert passes_guardrails(new, cached) is True


def test_extract_numbers_normalizes_punctuation_and_preserves_order() -> None:
    assert extract_numbers("Is it 2024? Costs $1,250.50 total.") == ["2024", "1250.50"]


def test_extract_negations_handles_contractions() -> None:
    assert extract_negations("It isn't done and never will be") == {"not", "never"}
    assert extract_negations("Nothing here") == {"nothing"}
    assert extract_negations("Notice the note") == set()  # substrings must not match
