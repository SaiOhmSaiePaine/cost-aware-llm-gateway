"""Circuit breaker state transitions with a deterministic fake clock."""
import pytest

from src.routing.circuit_breaker import BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=60, clock=clock)


def _trip(breaker: CircuitBreaker) -> None:
    for _ in range(3):
        assert breaker.allow_request()
        breaker.record_failure()


def test_starts_closed(breaker: CircuitBreaker) -> None:
    assert breaker.state is BreakerState.CLOSED
    assert breaker.allow_request()


def test_stays_closed_below_threshold(breaker: CircuitBreaker) -> None:
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_success_resets_failure_count(breaker: CircuitBreaker) -> None:
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.failure_count == 1
    assert breaker.state is BreakerState.CLOSED


def test_opens_after_threshold(breaker: CircuitBreaker) -> None:
    _trip(breaker)
    assert breaker.state is BreakerState.OPEN
    assert breaker.allow_request() is False


def test_half_open_after_timeout_allows_single_probe(breaker: CircuitBreaker, clock: FakeClock) -> None:
    _trip(breaker)
    clock.advance(60)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow_request() is True   # the probe
    assert breaker.allow_request() is False  # concurrent requests still bypass


def test_probe_success_closes(breaker: CircuitBreaker, clock: FakeClock) -> None:
    _trip(breaker)
    clock.advance(61)
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.failure_count == 0


def test_probe_failure_reopens_and_restarts_cooldown(breaker: CircuitBreaker, clock: FakeClock) -> None:
    _trip(breaker)
    clock.advance(61)
    assert breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    clock.advance(59)
    assert breaker.state is BreakerState.OPEN
    clock.advance(1)
    assert breaker.state is BreakerState.HALF_OPEN


def test_release_probe_frees_half_open_slot(breaker: CircuitBreaker, clock: FakeClock) -> None:
    _trip(breaker)
    clock.advance(60)
    assert breaker.allow_request()
    breaker.release_probe()  # e.g. client disconnected mid-probe
    assert breaker.allow_request() is True


def test_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0, recovery_timeout_seconds=1)


@pytest.mark.anyio
async def test_failed_half_open_probe_reopens_circuit_in_router(clock: FakeClock) -> None:
    from src.client.base import ChatTurn, LLMClient, LLMResult, UpstreamError
    from src.config import Settings
    from src.routing.classifier import RouteTier
    from src.routing.router import ModelRouter

    class _FailCloud(LLMClient):
        name = "cloud"

        async def complete(self, messages, *, max_tokens=None, temperature=None) -> LLMResult:
            raise UpstreamError("cloud", "503 down")

    class _OkLocal(LLMClient):
        name = "local"

        async def complete(self, messages, *, max_tokens=None, temperature=None) -> LLMResult:
            return LLMResult(text="local fallback", model="local")

    b = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=60, clock=clock)
    router = ModelRouter(_OkLocal(), _FailCloud(), b, Settings())

    # 1st failure trips breaker to OPEN
    await router.dispatch([ChatTurn(role="user", content="hi")], RouteTier.CLOUD_FRONTIER)
    assert b.state is BreakerState.OPEN

    # Cooldown elapses -> HALF_OPEN; failed probe must immediately re-open the circuit
    clock.advance(60)
    assert b.state is BreakerState.HALF_OPEN
    await router.dispatch([ChatTurn(role="user", content="hi")], RouteTier.CLOUD_FRONTIER)
    assert b.state is BreakerState.OPEN
