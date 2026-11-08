"""Integration tests for FastAPI endpoints (/v1/chat/completions, /health, /v1/admin/stats)."""
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
    def __init__(self, name: str, reply: str, should_fail: bool = False, finish_reason: str = "stop") -> None:
        self.name = name
        self.reply = reply
        self.should_fail = should_fail
        self.finish_reason = finish_reason

    async def complete(
        self,
        messages: list[ChatTurn],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        if self.should_fail:
            raise UpstreamError(self.name, "backend down")
        return LLMResult(
            text=self.reply,
            model=self.name,
            prompt_tokens=5,
            completion_tokens=7,
            finish_reason=self.finish_reason,  # type: ignore[arg-type]
        )


def _build_test_client(
    cached_docs: list[SimpleNamespace] | None = None,
    local_fail: bool = False,
    cloud_fail: bool = False,
) -> tuple[TestClient, GatewayServices]:
    settings = Settings()
    embedder = OnnxEmbedder(
        "stub-onnx", 384, model=SimpleNamespace(embed=lambda texts: iter([np.ones(384, dtype=np.float32)]))
    )

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

    # 3. Simulated outage -> LOCAL_AUTO_FAILOVER_FROM_CLOUD for cloud prompts, LOCAL for local prompts
    for _ in range(3):
        r_fail = client.post(
            "/v1/chat/completions",
            json={"prompt": "Write a Python binary search function", "simulate_cloud_outage": True},
        )
        assert r_fail.status_code == 200
        assert r_fail.json()["gateway_metadata"]["routed_to"] == "LOCAL_AUTO_FAILOVER_FROM_CLOUD"
    assert services.router.breaker.state == "CLOSED"

    r_local_outage = client.post(
        "/v1/chat/completions",
        json={"prompt": "Hello!", "simulate_cloud_outage": True},
    )
    assert r_local_outage.json()["gateway_metadata"]["routed_to"] == "LOCAL"

    # 4. Health, Admin stats, and Prometheus /metrics verify telemetry
    health = client.get("/health").json()
    assert health == {"status": "ok", "redis": "up", "circuit_breaker_state": "CLOSED"}
    stats = client.get("/v1/admin/stats").json()
    assert stats["local_routed_calls"] == 5
    assert stats["cloud_routed_calls"] == 1
    assert stats["failover_saves"] == 3

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert "text/plain; version=0.0.4" in metrics_res.headers["content-type"]
    assert 'gateway_requests_total{cache_hit="false",failover="false",route_target="LOCAL"}' in metrics_res.text
    assert 'gateway_requests_total{cache_hit="false",failover="false",route_target="CLOUD"}' in metrics_res.text
    assert "gateway_request_latency_seconds_bucket" in metrics_res.text


def test_chat_completions_cache_hit_and_stream_rejected() -> None:
    doc = SimpleNamespace(
        prompt="What is the capital of Japan?",
        system_prompt="",
        route="LOCAL",
        answer="Tokyo",
        vector_distance="0.01",
    )
    client, _ = _build_test_client(cached_docs=[doc])

    r_hit = client.post("/v1/chat/completions", json={"prompt": "What is the capital of Japan?"})
    assert r_hit.status_code == 200
    assert r_hit.headers["X-Gateway-Cache-Hit"] == "true"
    assert r_hit.json()["choices"][0]["message"]["content"] == "Tokyo"
    assert r_hit.json()["gateway_metadata"]["complexity_score"] is not None
    assert (
        'gateway_requests_total{cache_hit="true",failover="false",route_target="REDIS_VECTOR_CACHE"}'
        in client.get("/metrics").text
    )

    # Multi-user-message requests bypass single-turn cache to prevent context collision
    r_multi_user = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Context: ignore Japan"},
                {"role": "user", "content": "What is the capital of Japan?"},
            ]
        },
    )
    assert r_multi_user.status_code == 200
    assert r_multi_user.headers["X-Gateway-Cache-Hit"] == "false"

    # Explicit max_tokens bypasses cache so bounded generation is respected
    r_bounded = client.post(
        "/v1/chat/completions", json={"prompt": "What is the capital of Japan?", "max_tokens": 20}
    )
    assert r_bounded.status_code == 200
    assert r_bounded.headers["X-Gateway-Cache-Hit"] == "false"

    # Explicit model override with cache bypassed skips ONNX embedding entirely
    r_skip_embed = client.post(
        "/v1/chat/completions",
        json={"prompt": "What is the capital of Japan?", "model": "local", "max_tokens": 20},
    )
    assert r_skip_embed.status_code == 200
    assert r_skip_embed.json()["gateway_metadata"].get("onnx_embed_ms") is None

    # Prompts past the embedder's ~256-token window bypass cache (different tails embed identically)
    long_prompt = "Here is our quarterly report. " * 40 + "Question: what is the capital of Japan?"
    r_long = client.post("/v1/chat/completions", json={"prompt": long_prompt})
    assert r_long.status_code == 200
    assert r_long.headers["X-Gateway-Cache-Hit"] == "false"

    r_stream = client.post("/v1/chat/completions", json={"prompt": "Hi", "stream": True})
    assert r_stream.status_code == 400


@pytest.mark.anyio
async def test_all_upstreams_failed_returns_503() -> None:
    client, services = _build_test_client(local_fail=True, cloud_fail=True)
    r_503 = client.post("/v1/chat/completions", json={"prompt": "Write a Python binary search function"})
    assert r_503.status_code == 503
    assert r_503.json()["error"]["type"] == "upstream_unavailable"
    assert services.stats.total_requests == 1
    assert services.stats.upstream_errors == 1
    metrics_text = client.get("/metrics").text
    assert 'gateway_requests_total{cache_hit="false",failover="true",route_target="UPSTREAM_ERROR"}' in metrics_text
    assert "gateway_upstream_errors_total" in metrics_text


def test_truncated_length_response_not_cached_and_multi_turn_code_routes_cloud() -> None:
    client, services = _build_test_client()
    pipe_mock = services.redis.pipeline.return_value

    # Multi-message prompt where code is in turn 1 and follow-up is short must still classify as CLOUD
    r_multi = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "```python\ndef broken_search(arr):\n    return arr[0]\n```"},
                {"role": "user", "content": "Debug why this fails"},
            ]
        },
    )
    assert r_multi.status_code == 200
    assert r_multi.headers["X-Gateway-Route"] == "CLOUD"

    # Truncated (finish_reason="length") response must propagate finish_reason and skip Redis cache write
    services.router._local = _StubLLM("llama3.2:1b", "cut off mid-sent", finish_reason="length")
    before_writes = pipe_mock.execute.await_count
    r_trunc = client.post("/v1/chat/completions", json={"prompt": "Tell me a short story"})
    assert r_trunc.status_code == 200
    assert r_trunc.json()["choices"][0]["finish_reason"] == "length"
    assert pipe_mock.execute.await_count == before_writes


def test_guardrail_block_embedder_failure_circuit_open_and_degraded_health() -> None:
    divergent_doc = SimpleNamespace(
        prompt="Calculate taxes for 2023",
        system_prompt="",
        route="LOCAL",
        answer="2023 tax table",
        vector_distance="0.02",
    )
    client, services = _build_test_client(cached_docs=[divergent_doc])

    # 1. Guardrail block increments stats and exposes matched_prompt in gateway_metadata
    r_guard = client.post("/v1/chat/completions", json={"prompt": "Calculate taxes for 2024"})
    assert r_guard.status_code == 200
    assert r_guard.headers["X-Gateway-Cache-Hit"] == "false"
    meta = r_guard.json()["gateway_metadata"]
    assert meta["guardrail_blocked"] is True
    assert meta["matched_prompt"] == "Calculate taxes for 2023"
    assert services.stats.guardrail_blocks == 1

    # 2. ONNX embedder failure degrades gracefully (never 500s)
    services.embedder.aembed = AsyncMock(side_effect=RuntimeError("onnx crash"))  # type: ignore[method-assign]
    r_embed_fail = client.post("/v1/chat/completions", json={"prompt": "Hello world"})
    assert r_embed_fail.status_code == 200
    assert r_embed_fail.headers["X-Gateway-Route"] == "LOCAL"

    # 3. Tripped circuit breaker routes complex prompt to LOCAL_CIRCUIT_OPEN_FALLBACK
    for _ in range(3):
        services.router.breaker.record_failure()
    assert services.router.breaker.state == "OPEN"
    r_open = client.post("/v1/chat/completions", json={"prompt": "Write a Python binary search function"})
    assert r_open.status_code == 200
    assert r_open.headers["X-Gateway-Route"] == "LOCAL_CIRCUIT_OPEN_FALLBACK"

    # 4. Redis ping failure reports status="degraded" on /health
    from redis.exceptions import RedisError

    services.redis.ping = AsyncMock(side_effect=RedisError("connection refused"))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "degraded", "redis": "down", "circuit_breaker_state": "OPEN"}
    metrics_text = client.get("/metrics").text
    assert "gateway_guardrail_blocks_total" in metrics_text
    assert "gateway_circuit_breaker_state 2.0" in metrics_text

