"""Tests for MissionPersistence checkpoint handling.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import pytest

from agent_k.mission.persistence import CHECKPOINT_PREFIX, MissionPersistence
from agent_k.mission.state import MissionState

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

pytestmark = pytest.mark.anyio


def _persistence(tmp_path: Path, *, max_checkpoints: int = 10) -> MissionPersistence:
    return MissionPersistence(mission_id=uuid4().hex, checkpoint_dir=tmp_path, max_checkpoints=max_checkpoints)


class TestCheckpointRotation:
    """Behaviour of the ``_save_checkpoint`` helper under repeated writes."""

    async def test_same_instant_snapshots_do_not_collide(self, tmp_path: Path) -> None:
        """Rapid successive snapshots must each produce a distinct checkpoint file."""
        persistence = _persistence(tmp_path)
        state = MissionState(mission_id=persistence.mission_id)

        frozen = "20260101_000000_000000"
        with patch("agent_k.mission.persistence.datetime") as clock:
            clock.now.return_value.strftime.return_value = frozen
            for _ in range(5):
                await persistence._save_checkpoint(state)

        checkpoints = sorted(persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
        assert len(checkpoints) == 5
        assert len({path.name for path in checkpoints}) == 5

    async def test_rotation_prunes_to_max_and_keeps_newest(self, tmp_path: Path) -> None:
        """When ``max_checkpoints`` is exceeded the oldest files are pruned."""
        persistence = _persistence(tmp_path, max_checkpoints=3)
        state = MissionState(mission_id=persistence.mission_id)

        stamps = [
            "20260101_000000_000000",
            "20260101_000000_000001",
            "20260101_000000_000002",
            "20260101_000000_000003",
            "20260101_000000_000004",
        ]
        with patch("agent_k.mission.persistence.datetime") as clock:
            for stamp in stamps:
                clock.now.return_value.strftime.return_value = stamp
                await persistence._save_checkpoint(state)

        remaining = sorted(path.name for path in persistence.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"))
        assert remaining == [
            f"{CHECKPOINT_PREFIX}{stamps[2]}.json",
            f"{CHECKPOINT_PREFIX}{stamps[3]}.json",
            f"{CHECKPOINT_PREFIX}{stamps[4]}.json",
        ]

    async def test_cleanup_tolerates_missing_files(self, tmp_path: Path) -> None:
        """A checkpoint removed between glob and unlink must not crash cleanup."""
        persistence = _persistence(tmp_path, max_checkpoints=1)
        state = MissionState(mission_id=persistence.mission_id)

        stamps = ("20260101_000000_000000", "20260101_000000_000001", "20260101_000000_000002")
        with patch("agent_k.mission.persistence.datetime") as clock:
            for stamp in stamps:
                clock.now.return_value.strftime.return_value = stamp
                await persistence._save_checkpoint(state)

        # Simulate a racing sibling process that removes the target checkpoint
        # after the glob returned but before ``unlink`` runs.
        stale_target = persistence.mission_dir / f"{CHECKPOINT_PREFIX}{stamps[1]}.json"
        stale_target.write_text("{}", encoding="utf-8")

        original_unlink = type(stale_target).unlink

        def racy_unlink(self: Path, *, missing_ok: bool = False) -> None:
            if self == stale_target and self.exists():
                original_unlink(self, missing_ok=False)
            original_unlink(self, missing_ok=missing_ok)

        with patch("pathlib.Path.unlink", autospec=True, side_effect=racy_unlink):
            await persistence._cleanup_old_checkpoints()

        assert not stale_target.exists()
