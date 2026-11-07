"""Deterministic guardrails that veto semantically-similar but logically different cache hits.

Embeddings capture topic, not symbolic logic: "taxes for 2023" vs "taxes for 2024"
or "deploy with tests" vs "deploy without tests" score >0.9 cosine similarity.
"""
import re

_NUMBER_RE = re.compile(r"(?:(?<![a-zA-Z0-9])-)?\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
NEGATION_TOKENS: frozenset[str] = frozenset(
    {"not", "no", "never", "without", "none", "nor", "cannot", "neither", "nothing"}
)
STYLE_TOKENS: frozenset[str] = frozenset({
    "simply", "simple", "briefly", "brief", "short", "concise", "tldr", "eli5",
    "beginner", "detail", "detailed", "deep", "comprehensive", "advanced", "expert",
    "formal", "casual",
})


def extract_numbers(text: str) -> list[str]:
    """Ordered numbers with punctuation-insensitive normalization: '2024?' -> '2024', '1,000' -> '1000'."""
    return [match.replace(",", "") for match in _NUMBER_RE.findall(text)]


def extract_negations(text: str) -> set[str]:
    """Negation markers, treating contractions like "don't" / "isn't" (including curly '’') as 'not'."""
    negations: set[str] = set()
    for word in _WORD_RE.findall(text.lower().replace("’", "'")):
        if word in NEGATION_TOKENS:
            negations.add(word)
        elif word.endswith("n't"):
            negations.add("not")
    return negations


def extract_style_modifiers(text: str) -> set[str]:
    """Depth, length, and tone modifiers ('simply' vs 'in detail') that alter the expected answer."""
    return {w for w in _WORD_RE.findall(text.lower()) if w in STYLE_TOKENS}


def passes_guardrails(new_prompt: str, cached_prompt: str) -> bool:
    """True only if both prompts share identical numbers, negations, and style modifiers."""
    return (
        extract_numbers(new_prompt) == extract_numbers(cached_prompt)
        and extract_negations(new_prompt) == extract_negations(cached_prompt)
        and extract_style_modifiers(new_prompt) == extract_style_modifiers(cached_prompt)
    )
