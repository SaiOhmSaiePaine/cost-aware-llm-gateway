"""Circuit breaker state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN.

Pure in-memory logic with an injectable clock (no I/O) so it is trivially unit-testable.
All methods are synchronous with no awaits, so they are atomic on a single asyncio loop.
"""
import time
from collections.abc import Callable
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "CLOSED"        # Healthy: requests flow to the cloud
    OPEN = "OPEN"            # Tripped: cloud bypassed until the recovery timeout elapses
    HALF_OPEN = "HALF_OPEN"  # Cooldown over: exactly one probe request is allowed through


# shortcut: single-worker in-memory state, move to Redis/Prometheus when running --workers > 1
class CircuitBreaker:
    """Trips after N consecutive failures; lets a single probe through after a cooldown."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._clock = clock
        self._failure_count = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self._recovery_timeout:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allow_request(self) -> bool:
        """Should the next request try the protected upstream?"""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failure_count = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failure_count += 1
        was_probe = self._probe_in_flight
        self._probe_in_flight = False
        if was_probe or self._failure_count >= self._threshold:
            self._opened_at = self._clock()  # (Re-)open and restart the cooldown

    def release_probe(self) -> None:
        """Free the half-open slot if a probe ended without a verdict (e.g. client cancelled)."""
        self._probe_in_flight = False
