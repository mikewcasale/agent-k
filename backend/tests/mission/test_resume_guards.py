"""Tests for bounded startup auto-resume of persisted missions.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from agent_k.core.constants import AUTO_RESUME_MAX_AGE_SECONDS, AUTO_RESUME_MAX_ATTEMPTS
from agent_k.core.models import EvolutionState
from agent_k.mission.persistence import RESUME_LEDGER_FILE, MissionPersistence
from agent_k.mission.state import MissionState
from agent_k.ui.agui import _auto_resume_persisted_missions, _auto_resume_skip_reason, _mission_progress_marker

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

__all__ = ()

pytestmark = pytest.mark.anyio


@pytest.fixture
def persistence(tmp_path: Path) -> MissionPersistence:
    """Create persistence for a mission with a stored snapshot file."""
    instance = MissionPersistence("mission-1", checkpoint_dir=tmp_path)
    instance.json_file.write_text("[]", encoding="utf-8")
    return instance


class TestResumeLedger:
    """Tests for the durable resume-attempt ledger."""

    def test_defaults_to_empty_ledger(self, persistence: MissionPersistence) -> None:
        """A mission with no ledger file should report zero attempts."""
        ledger = persistence.load_resume_ledger()

        assert ledger.attempts == 0
        assert ledger.progress_marker is None
        assert ledger.last_attempt_at is None

    def test_attempts_accumulate_for_same_marker(self, persistence: MissionPersistence) -> None:
        """Repeated resumes without progress should increment the counter."""
        persistence.record_resume_attempt("discovery:0:0:0.0")
        ledger = persistence.record_resume_attempt("discovery:0:0:0.0")

        assert ledger.attempts == 2
        assert persistence.load_resume_ledger().attempts == 2

    def test_attempts_reset_when_mission_advances(self, persistence: MissionPersistence) -> None:
        """A new progress marker should restart the attempt budget."""
        persistence.record_resume_attempt("discovery:0:0:0.0")
        persistence.record_resume_attempt("discovery:0:0:0.0")
        ledger = persistence.record_resume_attempt("research:1:0:20.0")

        assert ledger.attempts == 1
        assert ledger.progress_marker == "research:1:0:20.0"

    def test_attempt_is_written_atomically(self, persistence: MissionPersistence) -> None:
        """Recording an attempt should leave no temporary file behind."""
        persistence.record_resume_attempt("discovery:0:0:0.0")

        assert (persistence.mission_dir / RESUME_LEDGER_FILE).exists()
        assert not list(persistence.mission_dir.glob("*.tmp"))

    def test_corrupt_ledger_is_treated_as_empty(self, persistence: MissionPersistence) -> None:
        """A truncated ledger must not block resumption."""
        (persistence.mission_dir / RESUME_LEDGER_FILE).write_text('{"attempts":', encoding="utf-8")

        assert persistence.load_resume_ledger().attempts == 0

    def test_last_activity_tracks_snapshot_file(self, persistence: MissionPersistence) -> None:
        """Last activity should come from the snapshot file timestamp."""
        last_activity = persistence.last_activity_at()

        assert last_activity is not None
        assert (datetime.now(UTC) - last_activity).total_seconds() < 60

    def test_last_activity_is_none_without_snapshots(self, tmp_path: Path) -> None:
        """A mission directory without a snapshot file has no activity."""
        instance = MissionPersistence("mission-2", checkpoint_dir=tmp_path)

        assert instance.last_activity_at() is None


class TestProgressMarker:
    """Tests for the mission progress marker."""

    def test_marker_for_missing_state(self) -> None:
        """A missing state should produce a stable marker."""
        assert _mission_progress_marker(None) == "unknown"

    def test_marker_changes_with_phase_progress(self) -> None:
        """Completing a phase should change the marker."""
        state = MissionState(mission_id="mission-1")
        before = _mission_progress_marker(state)
        state.phases_completed.append("discovery")
        state.current_phase = "research"

        assert _mission_progress_marker(state) != before

    def test_marker_changes_with_evolution_generation(self) -> None:
        """Advancing a generation should change the marker."""
        state = MissionState(mission_id="mission-1", evolution_state=EvolutionState(current_generation=3))
        before = _mission_progress_marker(state)
        state.evolution_state = EvolutionState(current_generation=4)

        assert _mission_progress_marker(state) != before


class TestAutoResumeSkipReason:
    """Tests for the startup auto-resume guards."""

    def test_fresh_mission_is_resumable(self, persistence: MissionPersistence) -> None:
        """A recently active mission with no attempts should be resumed."""
        assert _auto_resume_skip_reason(persistence, "discovery:0:0:0.0") is None

    def test_skips_once_attempts_are_exhausted(self, persistence: MissionPersistence) -> None:
        """A mission that keeps failing at the same point should be left alone."""
        for _ in range(AUTO_RESUME_MAX_ATTEMPTS):
            persistence.record_resume_attempt("discovery:0:0:0.0")

        reason = _auto_resume_skip_reason(persistence, "discovery:0:0:0.0")

        assert reason is not None
        assert "exhausted" in reason

    def test_progress_restores_the_attempt_budget(self, persistence: MissionPersistence) -> None:
        """A mission that advanced since the last attempt should be resumable again."""
        for _ in range(AUTO_RESUME_MAX_ATTEMPTS):
            persistence.record_resume_attempt("discovery:0:0:0.0")

        assert _auto_resume_skip_reason(persistence, "research:1:0:20.0") is None

    def test_skips_stale_missions(self, persistence: MissionPersistence) -> None:
        """A mission untouched beyond the age limit should not be replayed."""
        now = datetime.now(UTC) + timedelta(seconds=AUTO_RESUME_MAX_AGE_SECONDS + 60)

        reason = _auto_resume_skip_reason(persistence, "discovery:0:0:0.0", now=now)

        assert reason is not None
        assert "stale" in reason

    def test_skips_missions_without_activity(self, tmp_path: Path) -> None:
        """A mission directory with no snapshot file should not be replayed."""
        instance = MissionPersistence("mission-3", checkpoint_dir=tmp_path)

        reason = _auto_resume_skip_reason(instance, "discovery:0:0:0.0")

        assert reason == "Mission has no stored activity"


class TestBoundedAutoResume:
    """Tests for the bounded startup auto-resume pass."""

    async def test_resumes_nothing_without_checkpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty checkpoint directory should resume nothing."""
        monkeypatch.setattr("agent_k.ui.agui._list_persisted_mission_ids", lambda: [])

        assert await _auto_resume_persisted_missions(_recording_scheduler([])) == []

    async def test_resume_count_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the newest missions up to the limit should be resumed."""
        mission_ids = [f"mission-{index}" for index in range(5)]
        monkeypatch.setattr("agent_k.ui.agui._list_persisted_mission_ids", lambda: mission_ids)
        calls: list[str] = []

        resumed = await _auto_resume_persisted_missions(_recording_scheduler(calls), limit=2)

        assert resumed == ["mission-0", "mission-1"]
        assert calls == ["mission-0", "mission-1"]

    async def test_skipped_missions_do_not_consume_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missions that report a skip reason should not count against the limit."""
        mission_ids = [f"mission-{index}" for index in range(5)]
        monkeypatch.setattr("agent_k.ui.agui._list_persisted_mission_ids", lambda: mission_ids)
        calls: list[str] = []

        async def schedule(mission_id: str) -> str | None:
            calls.append(mission_id)
            return "Mission already completed" if mission_id in {"mission-0", "mission-1"} else None

        resumed = await _auto_resume_persisted_missions(schedule, limit=2)

        assert resumed == ["mission-2", "mission-3"]
        assert calls == ["mission-0", "mission-1", "mission-2", "mission-3"]

    async def test_failing_mission_does_not_abort_the_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exception for one mission should not stop later missions resuming."""
        mission_ids = ["mission-0", "mission-1"]
        monkeypatch.setattr("agent_k.ui.agui._list_persisted_mission_ids", lambda: mission_ids)

        async def schedule(mission_id: str) -> str | None:
            if mission_id == "mission-0":
                raise RuntimeError("checkpoint unreadable")
            return None

        assert await _auto_resume_persisted_missions(schedule, limit=2) == ["mission-1"]


def _recording_scheduler(calls: list[str]) -> Callable[[str], Awaitable[str | None]]:
    """Build a scheduler that records mission ids and always resumes."""

    async def schedule(mission_id: str) -> str | None:
        calls.append(mission_id)
        return None

    return schedule
