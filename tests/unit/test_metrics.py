"""Unit tests for Prometheus metrics and unified GatewayStats recorder."""
from prometheus_client.exposition import CONTENT_TYPE_PLAIN_0_0_4

from src.routing.router import RouteTarget
from src.telemetry.metrics import render_metrics, set_breaker_state
from src.telemetry.stats import GatewayStats


def test_unified_stats_and_prometheus_exposition() -> None:
    stats = GatewayStats()
    stats.record_embed(1.8)
    stats.record_guardrail_block()
    stats.record_cache_hit(RouteTarget.CACHE, latency_ms=2.4, saved_usd=0.005, breaker_state="CLOSED")
    stats.record_dispatch(
        RouteTarget.LOCAL_FAILOVER,
        latency_ms=42.0,
        saved_usd=0.005,
        breaker_state="OPEN",
        failover=True,
    )
    stats.record_upstream_error(breaker_state="HALF_OPEN")

    assert stats.total_requests == 3
    assert stats.cache_hits == 1
    assert stats.guardrail_blocks == 1
    assert stats.local_routed_calls == 1
    assert stats.failover_saves == 1
    assert stats.upstream_errors == 1
    assert stats.money_saved_usd == 0.01

    payload, content_type = render_metrics()
    text = payload.decode("utf-8")
    assert content_type == CONTENT_TYPE_PLAIN_0_0_4
    assert 'gateway_requests_total{cache_hit="true",failover="false",route_target="REDIS_VECTOR_CACHE"}' in text
    assert (
        'gateway_requests_total{cache_hit="false",failover="true",route_target="LOCAL_AUTO_FAILOVER_FROM_CLOUD"}'
        in text
    )
    assert 'gateway_requests_total{cache_hit="false",failover="true",route_target="UPSTREAM_ERROR"}' in text
    assert 'gateway_request_latency_seconds_bucket{le="0.005",route_target="REDIS_VECTOR_CACHE"}' in text
    assert "gateway_onnx_embed_latency_seconds_count" in text
    assert "gateway_guardrail_blocks_total" in text
    assert "gateway_upstream_errors_total" in text
    assert "gateway_cost_saved_usd_total" in text
    assert "gateway_circuit_breaker_state 1.0" in text

    set_breaker_state("CLOSED")
    assert "gateway_circuit_breaker_state 0.0" in render_metrics()[0].decode("utf-8")
