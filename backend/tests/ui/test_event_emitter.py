"""Tests for AG-UI EventEmitter buffering and shutdown semantics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import json

import pytest

from agent_k.ui.agui import EVENT_QUEUE_MAX_SIZE, EventEmitter

__all__ = ()

pytestmark = pytest.mark.anyio


async def _drain(emitter: EventEmitter) -> list[str]:
    """Collect every frame the emitter yields until the stream ends."""
    return [frame async for frame in emitter.stream()]


def _payloads(frames: list[str]) -> list[dict[str, object]]:
    """Decode SSE data frames, ignoring heartbeat comments."""
    return [json.loads(frame.removeprefix("data: ").strip()) for frame in frames if frame.startswith("data: ")]


class TestEventEmitterShutdown:
    """Tests covering close()/stream() interaction."""

    async def test_stream_drains_events_queued_before_close(self) -> None:
        """Events buffered when close() lands must still be delivered."""
        emitter = EventEmitter()
        for index in range(5):
            await emitter.emit("task-progress", {"taskId": str(index), "progress": index / 5})
        await emitter.emit("mission-complete", {"success": True})
        emitter.close()

        payloads = _payloads(await _drain(emitter))

        assert len(payloads) == 6
        assert payloads[-1]["type"] == "mission-complete"

    async def test_stream_ends_promptly_after_close(self) -> None:
        """close() must terminate an idle stream without waiting for a heartbeat."""
        emitter = EventEmitter()
        consumer = asyncio.create_task(_drain(emitter))
        await asyncio.sleep(0)
        emitter.close()

        frames = await asyncio.wait_for(consumer, timeout=1.0)

        assert frames == []

    async def test_close_is_idempotent(self) -> None:
        """A second close() must not enqueue another sentinel or raise."""
        emitter = EventEmitter()
        emitter.close()
        emitter.close()

        assert await _drain(emitter) == []

    async def test_emit_after_close_is_ignored(self) -> None:
        """Post-shutdown emissions must not resurrect the stream."""
        emitter = EventEmitter()
        emitter.close()
        await emitter.emit("task-progress", {"taskId": "late", "progress": 1.0})

        assert await _drain(emitter) == []


class TestEventEmitterBackpressure:
    """Tests covering the bounded buffer used by headless missions."""

    async def test_emit_does_not_block_without_a_consumer(self) -> None:
        """Headless missions have no reader; emit must stay non-blocking."""
        emitter = EventEmitter()
        overflow = EVENT_QUEUE_MAX_SIZE + 100

        for index in range(overflow):
            await asyncio.wait_for(
                emitter.emit("tool-thinking", {"taskId": "t", "toolCallId": "c", "chunk": str(index)}), timeout=5.0
            )

        assert emitter.dropped_events == 100

    async def test_oldest_events_are_dropped_first(self) -> None:
        """Backpressure must keep the newest events, including terminal ones."""
        emitter = EventEmitter()
        for index in range(EVENT_QUEUE_MAX_SIZE + 1):
            await emitter.emit("task-progress", {"taskId": str(index), "progress": 0.0})
        await emitter.emit("mission-complete", {"success": True})
        emitter.close()

        payloads = _payloads(await _drain(emitter))

        # One overflow emit, the terminal event, and the close sentinel each
        # evict the oldest buffered progress event.
        assert emitter.dropped_events == 3
        assert len(payloads) == EVENT_QUEUE_MAX_SIZE - 1
        assert payloads[0]["data"] == {"taskId": "3", "progress": 0.0}
        assert payloads[-1]["type"] == "mission-complete"

    async def test_close_sentinel_survives_a_full_buffer(self) -> None:
        """The stream must still terminate when close() hits a full buffer."""
        emitter = EventEmitter()
        for index in range(EVENT_QUEUE_MAX_SIZE):
            await emitter.emit("task-progress", {"taskId": str(index), "progress": 0.0})
        emitter.close()

        frames = await asyncio.wait_for(_drain(emitter), timeout=5.0)

        assert len(_payloads(frames)) == EVENT_QUEUE_MAX_SIZE - 1
