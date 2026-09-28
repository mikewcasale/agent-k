"""Tests for the mission recovery loop shared by the AG-UI mission runners.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_k.core.exceptions import AgentKError, AuthenticationError, RateLimitError
from agent_k.ui.agui import EventEmitter, run_mission_with_recovery

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Replace asyncio.sleep in the recovery loop and record the delays it requests."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("agent_k.ui.agui.asyncio.sleep", fake_sleep)
    yield delays


def _drain(emitter: EventEmitter) -> list[tuple[str, dict[str, Any]]]:
    """Drain the emitter queue into (event_type, payload) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    while not emitter._queue.empty():
        event = emitter._queue.get_nowait()
        events.append((event.type, event.data))
    return events


async def test_success_on_first_attempt_emits_no_recovery_events(no_sleep: list[float]) -> None:
    """A clean run must not emit error or recovery events."""
    emitter = EventEmitter()
    attempts: list[int] = []

    async def operation(attempt: int) -> None:
        attempts.append(attempt)

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1")

    assert attempts == [0]
    assert _drain(emitter) == []
    assert no_sleep == []


async def test_transient_error_is_retried_with_backoff(no_sleep: list[float]) -> None:
    """Transient third-party failures must be retried, not treated as fatal.

    Regression guard: the runners previously gated retries on
    ``isinstance(exc, AgentKError)``, so a dropped connection ended the mission
    on the first failure even though the emitted recovery strategy was "retry".
    """
    emitter = EventEmitter()
    attempts: list[int] = []

    async def operation(attempt: int) -> None:
        attempts.append(attempt)
        if attempt == 0:
            raise ConnectionResetError("connection reset by peer")

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1")

    events = _drain(emitter)

    assert attempts == [0, 1]
    assert no_sleep == [2.0]
    assert [event_type for event_type, _data in events] == ["error-occurred", "recovery-attempt", "recovery-complete"]
    assert events[-1][1] == {"errorId": "mission_1", "success": True, "resolution": "mission_completed"}


async def test_rate_limit_retry_after_drives_the_delay(no_sleep: list[float]) -> None:
    """A platform-supplied retry_after must be honoured before retrying."""
    emitter = EventEmitter()

    async def operation(attempt: int) -> None:
        if attempt == 0:
            raise RateLimitError(platform="kaggle", message="429", retry_after=30)

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1")

    assert no_sleep == [30.0]


async def test_fatal_error_is_not_retried(no_sleep: list[float]) -> None:
    """Authentication failures must abort immediately without a recovery attempt."""
    emitter = EventEmitter()
    attempts: list[int] = []

    async def operation(attempt: int) -> None:
        attempts.append(attempt)
        raise AuthenticationError(platform="kaggle")

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1")

    events = _drain(emitter)

    assert attempts == [0]
    assert no_sleep == []
    assert [event_type for event_type, _data in events] == ["error-occurred"]
    assert events[0][1]["recoveryStrategy"] == "abort"


async def test_attempts_are_bounded_and_exhaustion_is_reported(no_sleep: list[float]) -> None:
    """A persistently failing mission stops after max_attempts and reports exhaustion."""
    emitter = EventEmitter()
    attempts: list[int] = []

    async def operation(attempt: int) -> None:
        attempts.append(attempt)
        raise AgentKError("still broken")

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1", max_attempts=3)

    events = _drain(emitter)

    assert attempts == [0, 1, 2]
    assert no_sleep == [2.0, 4.0]
    assert [event_type for event_type, _data in events] == [
        "error-occurred",
        "recovery-attempt",
        "error-occurred",
        "recovery-attempt",
        "error-occurred",
        "recovery-complete",
    ]
    assert events[-1][1] == {"errorId": "mission_1", "success": False, "resolution": "exhausted"}


async def test_emitter_is_closed_even_when_every_attempt_fails(no_sleep: list[float]) -> None:
    """The SSE emitter must be closed so streaming clients are released."""
    emitter = EventEmitter()

    async def operation(_attempt: int) -> None:
        raise AuthenticationError(platform="kaggle")

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1")

    assert emitter._closed is True


async def test_log_context_is_accepted(no_sleep: list[float]) -> None:
    """Callers pass mission identifiers through as logfire context."""
    emitter = EventEmitter()

    async def operation(_attempt: int) -> None:
        raise AuthenticationError(platform="kaggle")

    await run_mission_with_recovery(operation, emitter=emitter, error_id="mission_1", mission_id="m-1")

    assert [event_type for event_type, _data in _drain(emitter)] == ["error-occurred"]
