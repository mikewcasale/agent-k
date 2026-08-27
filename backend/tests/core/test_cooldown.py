"""Tests for rate-limit cooldown scheduling.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

from agent_k.core.cooldown import (
    DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS,
    DEFAULT_MAX_COOLDOWN_SECONDS,
    DEFAULT_WAIT_BUDGET_SECONDS,
    RateLimitScheduler,
    parse_retry_after,
)


class _Clock:
    """Deterministic monotonic clock stand-in."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _scheduler(
    models: list[str],
    clock: _Clock,
    *,
    max_consecutive_rate_limits: int = 3,
    max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS,
    wait_budget_seconds: float = DEFAULT_WAIT_BUDGET_SECONDS,
) -> RateLimitScheduler:
    return RateLimitScheduler(
        models=models,
        base_cooldown_seconds=10.0,
        max_cooldown_seconds=max_cooldown_seconds,
        max_consecutive_rate_limits=max_consecutive_rate_limits,
        wait_budget_seconds=wait_budget_seconds,
        time_source=clock,
    )


def test_acquire_rotates_across_ready_models() -> None:
    """Consecutive acquisitions spread load across every ready model."""
    clock = _Clock()
    scheduler = _scheduler(["a", "b", "c"], clock)

    assert [scheduler.acquire().model for _ in range(4)] == ["a", "b", "c", "a"]


def test_rate_limited_model_is_parked_not_dropped() -> None:
    """A single rate limit parks a model on cooldown and it returns afterwards."""
    clock = _Clock()
    scheduler = _scheduler(["a", "b"], clock)

    cooldown = scheduler.record_rate_limit("a")
    assert cooldown == 10.0
    assert scheduler.active_models == ["a", "b"]

    assert scheduler.acquire().model == "b"

    clock.advance(cooldown)
    assert scheduler.acquire().model == "a"


def test_all_models_cooling_down_yields_bounded_wait() -> None:
    """When every model is cooling down the scheduler asks the caller to sleep."""
    clock = _Clock()
    scheduler = _scheduler(["a", "b"], clock)
    scheduler.record_rate_limit("a")
    clock.advance(2.0)
    scheduler.record_rate_limit("b")

    lease = scheduler.acquire()
    assert lease.model is None
    assert lease.exhausted is False
    assert lease.wait_seconds == 8.0

    clock.advance(lease.wait_seconds)
    assert scheduler.acquire().model == "a"


def test_cooldown_escalates_and_is_capped() -> None:
    """Repeated rate limits double the cooldown up to the configured cap."""
    clock = _Clock()
    scheduler = _scheduler(["a"], clock, max_consecutive_rate_limits=10, max_cooldown_seconds=25.0)

    assert scheduler.record_rate_limit("a") == 10.0
    assert scheduler.record_rate_limit("a") == 20.0
    assert scheduler.record_rate_limit("a") == 25.0


def test_retry_after_hint_extends_cooldown() -> None:
    """A longer server supplied Retry-After wins over the computed backoff."""
    clock = _Clock()
    scheduler = _scheduler(["a"], clock, max_cooldown_seconds=120.0)

    assert scheduler.record_rate_limit("a", retry_after=45.0) == 45.0


def test_success_resets_escalation() -> None:
    """A successful call clears the consecutive rate-limit streak."""
    clock = _Clock()
    scheduler = _scheduler(["a"], clock, max_consecutive_rate_limits=10)

    scheduler.record_rate_limit("a")
    scheduler.record_success("a")

    assert scheduler.acquire().model == "a"
    assert scheduler.record_rate_limit("a") == 10.0


def test_model_retires_after_max_consecutive_rate_limits() -> None:
    """A model that keeps rate limiting is retired and exhaustion is reported."""
    clock = _Clock()
    scheduler = _scheduler(["a"], clock, max_consecutive_rate_limits=2)

    scheduler.record_rate_limit("a")
    scheduler.record_rate_limit("a")

    assert scheduler.active_models == []
    assert scheduler.acquire().exhausted is True


def test_wait_budget_is_bounded() -> None:
    """The scheduler stops sleeping once the shared wait budget is spent."""
    clock = _Clock()
    scheduler = _scheduler(["a"], clock, max_consecutive_rate_limits=10, wait_budget_seconds=15.0)

    scheduler.record_rate_limit("a")
    first = scheduler.acquire()
    assert first.wait_seconds == 10.0

    clock.advance(1.0)
    second = scheduler.acquire()
    assert second.wait_seconds == 5.0
    assert second.exhausted is False

    clock.advance(1.0)
    assert scheduler.acquire().exhausted is True


def test_duplicate_models_are_collapsed() -> None:
    """Duplicate model specs collapse so cooldowns are tracked once."""
    clock = _Clock()
    scheduler = _scheduler(["a", "a", "b"], clock)

    assert scheduler.models == ["a", "b"]


def test_default_configuration_grants_retries_before_exhaustion() -> None:
    """Production defaults retry a rate-limited single model instead of giving up at once."""
    clock = _Clock()
    scheduler = RateLimitScheduler(models=["primary"], time_source=clock)

    attempts = 0
    while True:
        attempts += 1
        scheduler.record_rate_limit("primary")
        lease = scheduler.acquire()
        while lease.model is None and not lease.exhausted:
            clock.advance(lease.wait_seconds)
            lease = scheduler.acquire()
        if lease.exhausted:
            break

    assert attempts == DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS


def test_parse_retry_after_seconds_and_dates() -> None:
    """Retry-After parsing accepts seconds, HTTP dates, and rejects junk."""
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after(12) == 12.0
    assert parse_retry_after("0") is None
    assert parse_retry_after("not-a-date") is None
    assert parse_retry_after(None) is None

    future = datetime.now(UTC) + timedelta(seconds=45)
    parsed = parse_retry_after(format_datetime(future, usegmt=True))
    assert parsed is not None
    assert 30.0 < parsed <= 46.0


def test_parse_retry_after_reads_response_headers() -> None:
    """Retry-After is recovered from an exception carrying an HTTP response."""

    class _Response:
        headers = {"retry-after": "17"}

    class _Error(Exception):
        response = _Response()

    assert parse_retry_after(_Error("rate limited")) == 17.0


def test_parse_retry_after_reads_attribute() -> None:
    """Adapters that expose retry_after directly are honored."""

    class _Error(Exception):
        retry_after = 23

    assert parse_retry_after(_Error("rate limited")) == 23.0
