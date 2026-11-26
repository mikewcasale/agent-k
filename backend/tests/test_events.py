"""Protect browser event names and resumable stream cursors.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from agent_k.events import AGENT_K_EVENT_TYPES, EventType, parse_last_event_id

__all__ = ()

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def test_event_literal_and_runtime_registry_are_identical() -> None:
    """Prevent accepted worker events from diverging from the public registry."""
    assert set(get_args(EventType)) == AGENT_K_EVENT_TYPES


def test_frontend_event_union_matches_backend_event_names() -> None:
    """Verify the checked-in browser contract rather than a duplicated fixture."""
    source = (FRONTEND / "lib/types/events.ts").read_text(encoding="utf-8")
    union = source.split("export type AgentKEventType =", 1)[1].split(
        "export type AgentKEvent<", 1
    )[0]
    assert set(re.findall(r'"([a-z-]+)"', union)) == AGENT_K_EVENT_TYPES


def test_chat_transform_forwards_terminal_events() -> None:
    """Prevent the terminal event from being dropped by the chat stream adapter."""
    source = (FRONTEND / "app/(chat)/api/chat/route.ts").read_text(encoding="utf-8")
    registry = source.split("const agentKEventTypes = new Set([", 1)[1].split("]);", 1)[0]
    assert '"mission-complete"' in registry


@pytest.mark.parametrize(("value", "expected"), [(None, 0), ("", 0), ("0", 0), ("19", 19)])
def test_last_event_id_accepts_only_resumable_cursors(value: str | None, expected: int) -> None:
    """Parse valid SSE cursors without changing their durable sequence."""
    assert parse_last_event_id(value) == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "not-a-sequence"])
def test_last_event_id_rejects_invalid_cursors(value: str) -> None:
    """Reject cursors that could cause skipped or ambiguous replay."""
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_last_event_id(value)
