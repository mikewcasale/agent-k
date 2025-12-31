"""State persistence utilities for the mission graph.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

# Third-party (alphabetical)
import logfire
from pydantic_graph.persistence.file import FileStatePersistence

# Local imports (core first, then alphabetical)
from .state import MissionResult, MissionState

__all__ = ("MissionPersistence", "create_persistence", "CHECKPOINT_DIR")

CHECKPOINT_DIR: Final[Path] = Path("~/.agent_k/checkpoints").expanduser()


class MissionPersistence(FileStatePersistence[MissionState, MissionResult]):
    """Mission-specific persistence with checkpoint rotation."""

    def __init__(
        self,
        mission_id: str,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        max_checkpoints: int = 10,
    ) -> None:
        self.mission_id = mission_id
        self.max_checkpoints = max_checkpoints
        self.mission_dir = checkpoint_dir / mission_id
        self.mission_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(self.mission_dir / "state.json")

    async def save(self, state: MissionState) -> None:
        """Save state with timestamp and clean up old checkpoints."""
        with logfire.span("mission.persistence.save", mission_id=self.mission_id):
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.mission_dir / f"checkpoint_{timestamp}.json"
            checkpoint_path.write_text(
                state.model_dump_json(indent=2),
                encoding="utf-8",
            )
            await self._cleanup_old_checkpoints()

    async def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(
            self.mission_dir.glob("checkpoint_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_checkpoint in checkpoints[self.max_checkpoints :]:
            old_checkpoint.unlink()


def create_persistence(mission_id: str) -> MissionPersistence:
    """Factory for mission persistence."""
    return MissionPersistence(mission_id)
