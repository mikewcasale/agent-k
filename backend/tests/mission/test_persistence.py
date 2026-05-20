"""Tests for mission checkpoint persistence.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from agent_k.mission.persistence import CHECKPOINT_PREFIX, MissionPersistence
from agent_k.mission.state import MissionState

__all__ = ()

pytestmark = pytest.mark.anyio


def _build_state(mission_id: str | None = None) -> MissionState:
    return MissionState(mission_id=mission_id or str(uuid4()))


class TestSaveCheckpoint:
    """Tests for atomic checkpoint persistence."""

    async def test_writes_parseable_json(self, tmp_path: Path) -> None:
        """Checkpoint files should contain valid mission-state JSON."""
        state = _build_state()
        persistence = MissionPersistence(state.mission_id, checkpoint_dir=tmp_path)

        await persistence._save_checkpoint(state)

        checkpoints = list(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
        assert len(checkpoints) == 1
        payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
        assert payload["mission_id"] == state.mission_id

    async def test_leaves_no_partial_tmp_files(self, tmp_path: Path) -> None:
        """A successful write must remove the temp file used for atomic rename."""
        state = _build_state()
        persistence = MissionPersistence(state.mission_id, checkpoint_dir=tmp_path)

        await persistence._save_checkpoint(state)

        assert not list(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json.tmp"))

    async def test_preserves_previous_checkpoint_on_overwrite(self, tmp_path: Path) -> None:
        """Pre-existing tmp residue must not corrupt the final checkpoint."""
        state = _build_state()
        persistence = MissionPersistence(state.mission_id, checkpoint_dir=tmp_path)

        # Simulate a leftover tmp file from a previous crashed run.
        residue = persistence.mission_dir / f"{CHECKPOINT_PREFIX}stale.json.tmp"
        residue.write_text("not json", encoding="utf-8")

        await persistence._save_checkpoint(state)

        checkpoints = sorted(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
        assert len(checkpoints) == 1
        payload = json.loads(checkpoints[0].read_text(encoding="utf-8"))
        assert payload["mission_id"] == state.mission_id

    async def test_rapid_successive_writes_create_distinct_checkpoints(self, tmp_path: Path) -> None:
        """Two saves issued in the same second must not collide."""
        state = _build_state()
        persistence = MissionPersistence(state.mission_id, checkpoint_dir=tmp_path)

        await persistence._save_checkpoint(state)
        await persistence._save_checkpoint(state)

        checkpoints = list(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
        assert len(checkpoints) == 2

    async def test_cleanup_respects_max_checkpoints(self, tmp_path: Path) -> None:
        """Older checkpoints beyond the cap must be removed."""
        state = _build_state()
        persistence = MissionPersistence(state.mission_id, checkpoint_dir=tmp_path, max_checkpoints=3)

        for _ in range(5):
            await persistence._save_checkpoint(state)

        checkpoints = list(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
        assert len(checkpoints) == 3
