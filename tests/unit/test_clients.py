"""Unit tests for GeminiClient and OpenAICompatibleClient error, timeout, and finish_reason mapping."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from src.client.base import ChatTurn, UpstreamError
from src.client.cloud_client import GeminiClient
from src.client.vllm_client import OpenAICompatibleClient


@pytest.mark.anyio
async def test_openai_compatible_client_success_length_and_errors() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        if "timeout" in request.read().decode():
            raise httpx.ReadTimeout("timed out")
        if "bad_json" in request.read().decode():
            return httpx.Response(200, json={"unexpected": True})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "cut off"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 512},
            },
        )

    client = OpenAICompatibleClient("http://local.test/v1", "llama3.2:1b", 5.0)
    await client.aclose()
    client._http = httpx.AsyncClient(base_url="http://local.test/v1", transport=httpx.MockTransport(_handler))

    res = await client.complete([ChatTurn(role="user", content="hi")])
    assert res.text == "cut off"
    assert res.finish_reason == "length"
    assert res.completion_tokens == 512

    with pytest.raises(UpstreamError, match="timeout"):
        await client.complete([ChatTurn(role="user", content="timeout")])

    with pytest.raises(UpstreamError, match="malformed"):
        await client.complete([ChatTurn(role="user", content="bad_json")])

    await client.aclose()


@pytest.mark.anyio
async def test_openai_compatible_client_handles_null_fields_without_type_error() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        if "null_choices" in body:
            return httpx.Response(200, json={"choices": None})
        if "null_content" in body:
            return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": None, "completion_tokens": None},
            },
        )

    client = OpenAICompatibleClient("http://local.test/v1", "llama3.2:1b", 5.0)
    await client.aclose()
    client._http = httpx.AsyncClient(base_url="http://local.test/v1", transport=httpx.MockTransport(_handler))

    with pytest.raises(UpstreamError):
        await client.complete([ChatTurn(role="user", content="null_choices")])

    with pytest.raises(UpstreamError):
        await client.complete([ChatTurn(role="user", content="null_content")])

    res = await client.complete([ChatTurn(role="user", content="null_usage")])
    assert res.text == "ok"
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0
    await client.aclose()


@pytest.mark.anyio
async def test_gemini_client_role_mapping_finish_reason_and_errors() -> None:
    no_key_client = GeminiClient(None, "gemini-3.5-flash-lite", 1.0)
    with pytest.raises(UpstreamError, match="GEMINI_API_KEY"):
        await no_key_client.complete([ChatTurn(role="user", content="hi")])

    sys_inst, contents = GeminiClient._to_gemini(
        [
            ChatTurn(role="system", content="be brief"),
            ChatTurn(role="user", content="q1"),
            ChatTurn(role="assistant", content="a1"),
        ]
    )
    assert sys_inst == "be brief"
    assert [c.role for c in contents] == ["user", "model"]

    client = GeminiClient("fake-key", "gemini-3.5-flash-lite", 0.05)
    fake_resp = SimpleNamespace(
        text="partial",
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=10),
    )
    client._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=AsyncMock(return_value=fake_resp)))
    )
    res = await client.complete([ChatTurn(role="user", content="hi")])
    assert res.text == "partial"
    assert res.finish_reason == "length"

    # Empty response (e.g. safety block) raises UpstreamError
    client._client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text=""))  # type: ignore[union-attr]
    with pytest.raises(UpstreamError, match="empty response"):
        await client.complete([ChatTurn(role="user", content="hi")])

