"""Unit tests for ChatCompletionRequest input normalization & Swagger UI edge cases."""
import pytest
from pydantic import ValidationError

from src.api.schemas import ChatCompletionRequest


def test_prompt_wins_when_messages_has_swagger_string_placeholder() -> None:
    """In /docs, Swagger pre-fills messages=[{role: user, content: 'string'}]. Editing `prompt` must work."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "auto-router",
            "messages": [{"role": "user", "content": "string"}],
            "prompt": "What is the capital of Japan?",
            "stream": False,
            "simulate_cloud_outage": False,
        }
    )
    assert req.last_user_text == "What is the capital of Japan?"


def test_prompt_works_even_when_swagger_messages_has_system_role_placeholder() -> None:
    """If Swagger pre-fills role='system' with content='string', `prompt` must still succeed without 422."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "auto-router",
            "messages": [{"role": "system", "content": "string"}],
            "prompt": "Explain B-tree indexes",
        }
    )
    assert req.last_user_text == "Explain B-tree indexes"
    assert len(req.to_turns()) == 1


def test_messages_works_when_prompt_has_swagger_string_placeholder() -> None:
    """If the user edits `messages[0].content` and leaves `prompt: 'string'`, `messages` must be used."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "auto-router",
            "messages": [{"role": "user", "content": "Hello from messages"}],
            "prompt": "string",
        }
    )
    assert req.last_user_text == "Hello from messages"


def test_both_swagger_placeholders_rejected_with_422() -> None:
    """Clicking Execute without changing either 'string' placeholder should raise a clear 422 error."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "model": "auto-router",
                "messages": [{"role": "user", "content": "string"}],
                "prompt": "string",
            }
        )
