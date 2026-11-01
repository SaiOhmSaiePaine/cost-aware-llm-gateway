"""Unit tests for Phase 2.5: KNN 3 semantic cache, system isolation, LRU embed cache, model override, and bounded generation."""
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.api.schemas import ChatCompletionRequest
from src.cache.embedder import OnnxEmbedder
from src.cache.semantic_cache import SemanticCache
from src.client.base import ChatTurn, LLMClient, LLMResult
from src.config import Settings
from src.routing.circuit_breaker import CircuitBreaker
from src.routing.classifier import RouteTier
from src.routing.router import ModelRouter


class _DummyLLM(LLMClient):
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_kwargs: dict[str, int | float | None] = {}

    async def complete(
        self,
        messages: list[ChatTurn],
        *,
        max_tokens: int | None = 512,
        temperature: float | None = None,
    ) -> LLMResult:
        self.last_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
        return LLMResult(text="ok", model=self.name)


def _make_fake_embedder() -> OnnxEmbedder:
    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder._model_name = "fake-onnx"
    embedder._expected_dim = 384
    call_count = 0

    def _fake_embed_list(texts: list[str]):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # simulate 10ms cold ONNX inference
        yield np.ones(384, dtype=np.float32)

    embedder._model = SimpleNamespace(embed=_fake_embed_list)
    from collections import OrderedDict

    embedder._cache = OrderedDict()
    embedder.call_count = lambda: call_count  # type: ignore[attr-defined]
    return embedder


@pytest.mark.anyio
async def test_lru_embedding_cache_sub_1ms_on_repeat() -> None:
    embedder = _make_fake_embedder()
    vec1 = await embedder.aembed("What is the capital of Japan?")
    t0 = time.perf_counter()
    vec2 = await embedder.aembed("  what is the CAPITAL of japan?  ")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert vec1 == vec2
    assert elapsed_ms <= 1.0
    assert embedder.call_count() == 1  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_knn3_returns_second_neighbor_when_first_blocked_by_guardrail() -> None:
    settings = Settings()
    docs = [
        SimpleNamespace(
            prompt="Calculate 15% tip on $60",
            system_prompt="",
            answer="$9.00",
            vector_distance="0.04",  # similarity 0.96, blocked by number mismatch ($60 vs $90)
        ),
        SimpleNamespace(
            prompt="What is a 15% tip on $90?",
            system_prompt="",
            answer="$13.50",
            vector_distance="0.08",  # similarity 0.92, passes guardrails!
        ),
    ]
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=docs))
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    res = await cache.lookup("Calculate 15% tip on $90", b"\x00" * 1536)

    assert res.hit is True
    assert res.answer == "$13.50"
    assert res.matched_prompt == "What is a 15% tip on $90?"
    assert res.guardrail_blocked is False


@pytest.mark.anyio
async def test_system_prompt_isolation_prevents_cache_collision() -> None:
    settings = Settings()
    docs = [
        SimpleNamespace(
            prompt="Explain recursion",
            system_prompt="you are a pirate",
            answer="Arrr, recursion calls itself!",
            vector_distance="0.02",
        ),
        SimpleNamespace(
            prompt="Explain recursion",
            system_prompt="you are a formal professor",
            answer="Recursion is a method of self-referential definition.",
            vector_distance="0.03",
        ),
    ]
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=docs))
    redis_mock = MagicMock()
    redis_mock.ft.return_value = ft_mock

    cache = SemanticCache(redis_mock, settings)
    res_prof = await cache.lookup(
        "Explain recursion", b"\x00" * 1536, system_prompt="You are a formal professor"
    )
    assert res_prof.hit is True
    assert res_prof.answer == "Recursion is a method of self-referential definition."

    res_chef = await cache.lookup(
        "Explain recursion", b"\x00" * 1536, system_prompt="You are a chef"
    )
    assert res_chef.hit is False


@pytest.mark.anyio
async def test_explicit_model_override_and_bounded_generation_forwarding() -> None:
    settings = Settings()
    local = _DummyLLM("llama3.2:1b")
    cloud = _DummyLLM("gemini-3.5-flash-lite")
    router = ModelRouter(local, cloud, CircuitBreaker(3, 60.0), settings)

    # Complex prompt forced to local via model="local"
    c_local = router.classify("Write a Python binary search function", model="local")
    assert c_local.tier is RouteTier.LOCAL_SMALL

    # Simple prompt forced to cloud via model="cloud"
    c_cloud = router.classify("Hello there", model="cloud")
    assert c_cloud.tier is RouteTier.CLOUD_FRONTIER

    # Default max_tokens=None on ChatCompletionRequest so Cloud isn't truncated, while explicit values forward
    req_default = ChatCompletionRequest.model_validate({"prompt": "Hello", "temperature": 0.2})
    assert req_default.max_tokens is None
    assert req_default.temperature == 0.2

    await router.dispatch(
        req_default.to_turns(),
        c_cloud.tier,
        max_tokens=req_default.max_tokens,
        temperature=req_default.temperature,
    )
    assert cloud.last_kwargs == {"max_tokens": None, "temperature": 0.2}

    req_bounded = ChatCompletionRequest.model_validate(
        {"prompt": "Hello", "max_tokens": 256, "temperature": 0.1}
    )
    await router.dispatch(
        req_bounded.to_turns(),
        c_local.tier,
        max_tokens=req_bounded.max_tokens,
        temperature=req_bounded.temperature,
    )
    assert local.last_kwargs == {"max_tokens": 256, "temperature": 0.1}

