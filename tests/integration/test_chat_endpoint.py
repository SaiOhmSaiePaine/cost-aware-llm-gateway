"""Integration tests for FastAPI endpoints (/v1/chat/completions, /health, /v1/admin/stats)."""
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import GatewayServices
from src.cache.embedder import OnnxEmbedder
from src.cache.semantic_cache import SemanticCache
from src.client.base import ChatTurn, LLMClient, LLMResult, UpstreamError
from src.config import Settings
from src.main import create_app
from src.routing.circuit_breaker import CircuitBreaker
from src.routing.router import ModelRouter
from src.telemetry.stats import GatewayStats


class _StubLLM(LLMClient):
    def __init__(self, name: str, reply: str, should_fail: bool = False) -> None:
        self.name = name
        self.reply = reply
        self.should_fail = should_fail

    async def complete(
        self,
        messages: list[ChatTurn],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        if self.should_fail:
            raise UpstreamError(self.name, "backend down")
        return LLMResult(text=self.reply, model=self.name, prompt_tokens=5, completion_tokens=7)


def _build_test_client(
    cached_docs: list[SimpleNamespace] | None = None,
    local_fail: bool = False,
    cloud_fail: bool = False,
) -> tuple[TestClient, GatewayServices]:
    settings = Settings()
    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder._model_name = "stub-onnx"
    embedder._expected_dim = 384
    embedder._model = SimpleNamespace(embed=lambda texts: iter([np.ones(384, dtype=np.float32)]))
    embedder._cache = OrderedDict()

    redis_mock = MagicMock()
    redis_mock.ping = AsyncMock(return_value=True)
    ft_mock = MagicMock()
    ft_mock.search = AsyncMock(return_value=SimpleNamespace(docs=cached_docs or []))
    redis_mock.ft.return_value = ft_mock
    pipe_mock = MagicMock()
    pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
    pipe_mock.__aexit__ = AsyncMock(return_value=None)
    pipe_mock.execute = AsyncMock(return_value=[True, True])
    redis_mock.pipeline.return_value = pipe_mock

    services = GatewayServices(
        settings=settings,
        redis=redis_mock,
        embedder=embedder,
        cache=SemanticCache(redis_mock, settings),
        router=ModelRouter(
            local=_StubLLM("llama3.2:1b", "local answer", should_fail=local_fail),
            cloud=_StubLLM("gemini-3.5-flash-lite", "cloud answer", should_fail=cloud_fail),
            breaker=CircuitBreaker(3, 60.0),
            settings=settings,
        ),
        stats=GatewayStats(),
    )
    app = create_app()
    app.router.lifespan_context = None  # type: ignore[assignment]
    app.state.services = services
    return TestClient(app), services


def test_chat_completions_local_cloud_and_cache_hit() -> None:
    client, services = _build_test_client()

    # 1. Simple prompt -> LOCAL
    r_local = client.post("/v1/chat/completions", json={"prompt": "What is the capital of Japan?"})
    assert r_local.status_code == 200
    assert r_local.headers["X-Gateway-Route"] == "LOCAL"
    assert r_local.json()["gateway_metadata"]["routed_to"] == "LOCAL"

    # 2. Complex coding prompt -> CLOUD
    r_cloud = client.post("/v1/chat/completions", json={"prompt": "Write a Python binary search function"})
    assert r_cloud.status_code == 200
    assert r_cloud.headers["X-Gateway-Route"] == "CLOUD"
    assert r_cloud.json()["gateway_metadata"]["routed_to"] == "CLOUD"

    # 3. Simulated outage -> LOCAL_AUTO_FAILOVER_FROM_CLOUD without tripping shared breaker
    for _ in range(3):
        r_fail = client.post(
            "/v1/chat/completions",
            json={"prompt": "Write a Python binary search function", "simulate_cloud_outage": True},
        )
        assert r_fail.status_code == 200
        assert r_fail.json()["gateway_metadata"]["routed_to"] == "LOCAL_AUTO_FAILOVER_FROM_CLOUD"
    assert services.router.breaker.state == "CLOSED"

    # 4. Health & Admin stats verify local_routed_calls includes failover calls
    health = client.get("/health").json()
    assert health == {"status": "ok", "redis": "up", "circuit_breaker_state": "CLOSED"}
    stats = client.get("/v1/admin/stats").json()
    assert stats["local_routed_calls"] == 4
    assert stats["cloud_routed_calls"] == 1
    assert stats["failover_saves"] == 3


def test_chat_completions_cache_hit_and_stream_rejected() -> None:
    doc = SimpleNamespace(
        prompt="What is the capital of Japan?",
        system_prompt="",
        answer="Tokyo",
        vector_distance="0.01",
    )
    client, _ = _build_test_client(cached_docs=[doc])

    r_hit = client.post("/v1/chat/completions", json={"prompt": "What is the capital of Japan?"})
    assert r_hit.status_code == 200
    assert r_hit.headers["X-Gateway-Cache-Hit"] == "true"
    assert r_hit.json()["choices"][0]["message"]["content"] == "Tokyo"

    r_stream = client.post("/v1/chat/completions", json={"prompt": "Hi", "stream": True})
    assert r_stream.status_code == 400


@pytest.mark.anyio
async def test_all_upstreams_failed_returns_503() -> None:
    client, _ = _build_test_client(local_fail=True, cloud_fail=True)
    r_503 = client.post("/v1/chat/completions", json={"prompt": "Write a Python binary search function"})
    assert r_503.status_code == 503
    assert r_503.json()["error"]["type"] == "upstream_unavailable"

