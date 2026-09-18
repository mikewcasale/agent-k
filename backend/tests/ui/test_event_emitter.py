"""Tests for the AG-UI EventEmitter broadcast semantics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import json
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import pytest

from agent_k.ui import agui
from agent_k.ui.agui import EventEmitter

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

__all__ = ()

pytestmark = pytest.mark.anyio


def _payload(frame: str) -> dict[str, Any]:
    """Decode an SSE frame into its JSON payload."""
    assert frame.startswith("data: ")
    decoded: dict[str, Any] = json.loads(frame[len("data: ") :].strip())
    return decoded


async def _start(emitter: EventEmitter) -> AsyncGenerator[str, None]:
    """Open a stream and advance it far enough to register the subscription."""
    frames = emitter.stream()
    await emitter.emit("task-progress", {"marker": "subscribe"})
    assert _payload(await anext(frames))["data"] == {"marker": "subscribe"}
    return frames


async def _drain(frames: AsyncGenerator[str, None]) -> list[dict[str, Any]]:
    """Consume a closed emitter's stream into decoded payloads."""
    events: list[dict[str, Any]] = []
    async with aclosing(frames) as stream:
        async for frame in stream:
            events.append(_payload(frame))
    return events


class TestEmitWithoutSubscribers:
    """Emission must stay bounded when nobody is streaming."""

    async def test_history_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Events emitted with no subscriber must not accumulate without bound."""
        monkeypatch.setattr(agui, "_EVENT_HISTORY_LIMIT", 4)
        emitter = EventEmitter()

        for index in range(50):
            await emitter.emit("task-progress", {"index": index})

        assert len(emitter._history) == 4
        assert [event.data["index"] for event in emitter._history] == [46, 47, 48, 49]

    async def test_emit_after_close_is_ignored(self) -> None:
        """A closed emitter must drop further emissions."""
        emitter = EventEmitter()
        emitter.close()

        await emitter.emit("task-progress", {"index": 0})

        assert len(emitter._history) == 0


class TestStreamDelivery:
    """Streaming must deliver every event, including terminal ones."""

    async def test_pending_events_survive_close(self) -> None:
        """Events emitted before close must be yielded before the stream ends."""
        emitter = EventEmitter()
        frames = await _start(emitter)

        await emitter.emit_phase_start("discovery", ["find"])
        await emitter.emit("mission-complete", {"success": True})
        emitter.close()

        assert [event["type"] for event in await _drain(frames)] == ["phase-start", "mission-complete"]

    async def test_history_is_replayed_to_late_subscribers(self) -> None:
        """A client connecting mid-mission must receive the retained history."""
        emitter = EventEmitter()
        await emitter.emit("phase-start", {"phase": "discovery"})
        await emitter.emit("phase-complete", {"phase": "discovery"})
        emitter.close()

        events = await _drain(emitter.stream())

        assert [event["type"] for event in events] == ["phase-start", "phase-complete"]

    async def test_concurrent_subscribers_each_receive_every_event(self) -> None:
        """Two clients must not compete for the same events."""
        emitter = EventEmitter()
        first = emitter.stream()
        second = emitter.stream()
        await emitter.emit("task-progress", {"marker": "subscribe"})
        for frames in (first, second):
            assert _payload(await anext(frames))["data"] == {"marker": "subscribe"}

        await emitter.emit("phase-start", {"phase": "discovery"})
        await emitter.emit("phase-complete", {"phase": "discovery"})
        emitter.close()

        assert [event["type"] for event in await _drain(first)] == ["phase-start", "phase-complete"]
        assert [event["type"] for event in await _drain(second)] == ["phase-start", "phase-complete"]


class TestSubscriberLifecycle:
    """Subscriptions must be released when a client stops reading."""

    async def test_subscriber_is_released_on_stream_close(self) -> None:
        """Closing the stream must unregister the subscriber."""
        emitter = EventEmitter()
        frames = await _start(emitter)

        assert len(emitter._subscribers) == 1
        await frames.aclose()

        assert len(emitter._subscribers) == 0

    async def test_emit_does_not_block_on_a_stalled_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stalled client must drop its oldest backlog instead of stalling the mission."""
        monkeypatch.setattr(agui, "_SUBSCRIBER_QUEUE_LIMIT", 3)
        emitter = EventEmitter()
        frames = await _start(emitter)
        subscriber = next(iter(emitter._subscribers))

        async with asyncio.timeout(5):
            for index in range(10):
                await emitter.emit("task-progress", {"index": index})

        assert subscriber.queue.qsize() == 3
        assert subscriber.dropped == 7
        assert _payload(await anext(frames))["data"]["index"] == 7
        await frames.aclose()
