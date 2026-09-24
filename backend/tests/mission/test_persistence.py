"""Tests for mission checkpoint persistence.

These exercise the real filesystem: durability is the whole point of the
code under test, so a mocked file layer would prove nothing.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
import os
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from agent_k.mission.persistence import CHECKPOINT_PREFIX, MissionPersistence
from agent_k.mission.state import MissionState

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

pytestmark = pytest.mark.anyio

MISSION_ID = "mission-checkpoint-test"
ORPHAN_TEMP_NAME = f".{CHECKPOINT_PREFIX}20200101_000000_000000.json.tmp"


def _persistence(tmp_path: Path, *, max_checkpoints: int = 10) -> MissionPersistence:
    return MissionPersistence(MISSION_ID, checkpoint_dir=tmp_path, max_checkpoints=max_checkpoints)


def _state(phase: str = "discovery") -> MissionState:
    return MissionState(mission_id=str(uuid4()), current_phase=phase)  # type: ignore[arg-type]


def _checkpoints(persistence: MissionPersistence) -> list[Path]:
    return sorted(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))


def _temp_files(persistence: MissionPersistence) -> list[Path]:
    return sorted(persistence.mission_dir.glob(f".{CHECKPOINT_PREFIX}*.json.tmp"))


class TestCheckpointWrites:
    """Every saved state must land in its own complete file."""

    async def test_rapid_saves_do_not_overwrite_each_other(self, tmp_path: Path) -> None:
        """Back-to-back saves inside one wall-clock second must produce distinct files."""
        persistence = _persistence(tmp_path)
        phases = ("discovery", "research", "prototype")

        started_at = time.monotonic()
        for phase in phases:
            await persistence._save_checkpoint(_state(phase))
        assert time.monotonic() - started_at < 1.0, "saves must land inside a single wall-clock second"

        files = _checkpoints(persistence)
        assert len(files) == len(phases)
        saved = [json.loads(path.read_text(encoding="utf-8"))["current_phase"] for path in files]
        assert saved == list(phases), "file name order must match write order"

    async def test_saved_checkpoint_round_trips(self, tmp_path: Path) -> None:
        """A checkpoint must deserialize back into an equivalent state."""
        persistence = _persistence(tmp_path)
        state = _state("evolution")

        await persistence._save_checkpoint(state)

        (path,) = _checkpoints(persistence)
        assert MissionState.model_validate_json(path.read_text(encoding="utf-8")) == state

    async def test_successful_save_leaves_no_temp_file(self, tmp_path: Path) -> None:
        """The atomic write must leave nothing but the final checkpoint behind."""
        persistence = _persistence(tmp_path)

        await persistence._save_checkpoint(_state())

        assert len(_checkpoints(persistence)) == 1
        assert _temp_files(persistence) == []


class TestCheckpointRotation:
    """Rotation must keep the newest checkpoints and drop the rest."""

    async def test_retains_only_the_newest_checkpoints(self, tmp_path: Path) -> None:
        """Saving past the limit must prune the oldest checkpoints, not arbitrary ones."""
        persistence = _persistence(tmp_path, max_checkpoints=3)
        for index in range(7):
            state = _state()
            state.overall_progress = float(index)
            await persistence._save_checkpoint(state)

        files = _checkpoints(persistence)
        assert len(files) == 3
        progress = [json.loads(path.read_text(encoding="utf-8"))["overall_progress"] for path in files]
        assert progress == [4.0, 5.0, 6.0]

    async def test_stale_temp_files_are_swept(self, tmp_path: Path) -> None:
        """A temp file orphaned by an earlier crash must eventually be removed."""
        persistence = _persistence(tmp_path)
        orphan = persistence.mission_dir / ORPHAN_TEMP_NAME
        orphan.write_text("{partial", encoding="utf-8")
        stale = time.time() - 7200.0
        os.utime(orphan, (stale, stale))

        await persistence._save_checkpoint(_state())

        assert not orphan.exists()

    async def test_recent_temp_files_are_kept(self, tmp_path: Path) -> None:
        """A temp file from an in-flight write must not be swept."""
        persistence = _persistence(tmp_path)
        in_flight = persistence.mission_dir / ORPHAN_TEMP_NAME
        in_flight.write_text("{partial", encoding="utf-8")

        await persistence._save_checkpoint(_state())

        assert in_flight.exists()

    async def test_orphaned_temp_file_is_never_mistaken_for_a_checkpoint(self, tmp_path: Path) -> None:
        """Retention must count only complete checkpoints."""
        persistence = _persistence(tmp_path, max_checkpoints=1)
        (persistence.mission_dir / ORPHAN_TEMP_NAME).write_text("{partial", encoding="utf-8")

        await persistence._save_checkpoint(_state())

        assert len(_checkpoints(persistence)) == 1


class TestCheckpointFailures:
    """Checkpointing is auxiliary and must never abort a mission."""

    async def test_unwritable_directory_does_not_raise(self, tmp_path: Path) -> None:
        """A failing checkpoint write must be swallowed; the real snapshot is already stored."""
        persistence = _persistence(tmp_path)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        persistence.mission_dir = blocker / "missing"

        await persistence._save_checkpoint(_state())

        assert blocker.read_text(encoding="utf-8") == "not a directory"

    async def test_failed_write_leaves_earlier_checkpoints_intact(self, tmp_path: Path) -> None:
        """A write that cannot complete must not add or remove any checkpoint."""
        persistence = _persistence(tmp_path)
        await persistence._save_checkpoint(_state())
        before = _checkpoints(persistence)

        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        original_dir = persistence.mission_dir
        persistence.mission_dir = blocker / "missing"
        await persistence._save_checkpoint(_state())
        persistence.mission_dir = original_dir

        assert _checkpoints(persistence) == before
