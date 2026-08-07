"""Tests for MissionPersistence checkpoint handling.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from uuid import uuid4

import pytest

from agent_k.mission.persistence import CHECKPOINT_PREFIX, MissionPersistence
from agent_k.mission.state import MissionState

__all__ = ()

pytestmark = pytest.mark.anyio


def _list_checkpoints(mission_dir: Path) -> list[Path]:
    return sorted(mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"), key=lambda p: p.name)


class TestSaveCheckpoint:
    """Tests for MissionPersistence._save_checkpoint."""

    async def test_rapid_writes_do_not_collide(self, tmp_path: Path) -> None:
        """Two writes triggered in the same wall-clock second must both survive."""
        mission_id = str(uuid4())
        persistence = MissionPersistence(mission_id, checkpoint_dir=tmp_path, max_checkpoints=10)
        state = MissionState(mission_id=mission_id)

        for _ in range(5):
            await persistence._save_checkpoint(state)

        checkpoints = _list_checkpoints(persistence.mission_dir)
        assert len(checkpoints) == 5, f"expected 5 unique checkpoints, got {[p.name for p in checkpoints]}"
        assert len({p.name for p in checkpoints}) == 5

    async def test_cleanup_retains_most_recent(self, tmp_path: Path) -> None:
        """Cleanup should retain exactly max_checkpoints most-recent files."""
        mission_id = str(uuid4())
        persistence = MissionPersistence(mission_id, checkpoint_dir=tmp_path, max_checkpoints=3)
        state = MissionState(mission_id=mission_id)

        for _ in range(7):
            await persistence._save_checkpoint(state)

        checkpoints = _list_checkpoints(persistence.mission_dir)
        assert len(checkpoints) == 3

    async def test_cleanup_ordering_matches_creation(self, tmp_path: Path) -> None:
        """Cleanup must drop the oldest checkpoints even when timestamps collide."""
        mission_id = str(uuid4())
        persistence = MissionPersistence(mission_id, checkpoint_dir=tmp_path, max_checkpoints=2)
        state = MissionState(mission_id=mission_id)

        for _ in range(4):
            await persistence._save_checkpoint(state)

        checkpoints = _list_checkpoints(persistence.mission_dir)
        # Files are named with a monotonic sequence suffix, so the retained pair
        # must be the two lexicographically largest names.
        expected_suffixes = {"000002", "000003"}
        actual_suffixes = {p.stem.rsplit("_", 1)[-1] for p in checkpoints}
        assert actual_suffixes == expected_suffixes

    async def test_checkpoint_payload_is_state_snapshot(self, tmp_path: Path) -> None:
        """Persisted content must be the state JSON dump."""
        mission_id = str(uuid4())
        persistence = MissionPersistence(mission_id, checkpoint_dir=tmp_path)
        state = MissionState(mission_id=mission_id, competition_id="titanic")

        await persistence._save_checkpoint(state)

        checkpoints = _list_checkpoints(persistence.mission_dir)
        assert len(checkpoints) == 1
        content = checkpoints[0].read_text(encoding="utf-8")
        assert f'"mission_id": "{mission_id}"' in content
        assert '"competition_id": "titanic"' in content

    async def test_missing_file_cleanup_is_tolerated(self, tmp_path: Path) -> None:
        """Cleanup should not raise when a candidate file vanishes concurrently."""
        mission_id = str(uuid4())
        persistence = MissionPersistence(mission_id, checkpoint_dir=tmp_path, max_checkpoints=1)
        state = MissionState(mission_id=mission_id)

        await persistence._save_checkpoint(state)
        # Simulate a concurrent deletion by removing every existing checkpoint,
        # then triggering another save. The next cleanup pass should still
        # complete without raising even though the sibling entries disappeared.
        for existing in _list_checkpoints(persistence.mission_dir):
            existing.unlink()

        await persistence._save_checkpoint(state)
        assert len(_list_checkpoints(persistence.mission_dir)) == 1
