"""Prometheus metrics registry and low-cardinality collectors for the LLM Gateway."""
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_PLAIN_0_0_4

_BREAKER_STATE_MAP: dict[str, int] = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}

REQUEST_LATENCY_BUCKETS = (
    0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 30.0
)
EMBED_LATENCY_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1)

REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total chat completion requests processed by route, cache hit, and failover status.",
    labelnames=("route_target", "cache_hit", "failover"),
)
REQUEST_LATENCY_SECONDS = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end gateway request latency in seconds by route target.",
    labelnames=("route_target",),
    buckets=REQUEST_LATENCY_BUCKETS,
)
ONNX_EMBED_LATENCY_SECONDS = Histogram(
    "gateway_onnx_embed_latency_seconds",
    "Local ONNX prompt embedding latency in seconds.",
    buckets=EMBED_LATENCY_BUCKETS,
)
GUARDRAIL_BLOCKS_TOTAL = Counter(
    "gateway_guardrail_blocks_total",
    "Total semantic cache candidates rejected by deterministic numeric or negation guardrails.",
)
UPSTREAM_ERRORS_TOTAL = Counter(
    "gateway_upstream_errors_total",
    "Total requests where all eligible upstream LLM backends failed (HTTP 503).",
)
COST_SAVED_USD_TOTAL = Counter(
    "gateway_cost_saved_usd_total",
    "Cumulative estimated USD saved via semantic cache hits and local tier routing.",
)
CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Cloud circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN).",
)


def set_breaker_state(state: str) -> None:
    """Update the circuit breaker state gauge (0=CLOSED, 1=HALF_OPEN, 2=OPEN)."""
    CIRCUIT_BREAKER_STATE.set(_BREAKER_STATE_MAP.get(state, 0))


def render_metrics() -> tuple[bytes, str]:
    """Serialize the default Prometheus registry in text exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_PLAIN_0_0_4
