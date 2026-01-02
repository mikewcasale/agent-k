"""State persistence utilities for the mission graph.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

# Third-party (alphabetical)
import logfire
from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import EndSnapshot, NodeSnapshot, Snapshot
from pydantic_graph.persistence.file import FileStatePersistence

# Local imports (core first, then alphabetical)
from .state import MissionResult, MissionState

__all__ = ('MissionPersistence', 'create_persistence', 'CHECKPOINT_DIR')

CHECKPOINT_DIR: Final[Path] = Path('~/.agent_k/checkpoints').expanduser()
CHECKPOINT_PREFIX: Final[str] = 'checkpoint_'


class MissionPersistence(FileStatePersistence[MissionState, MissionResult]):
    """Mission-specific persistence with checkpoint rotation and resumability."""

    def __init__(self, mission_id: str, checkpoint_dir: Path = CHECKPOINT_DIR, max_checkpoints: int = 10) -> None:
        self.mission_id = mission_id
        self.max_checkpoints = max_checkpoints
        self.mission_dir = checkpoint_dir / mission_id
        self.mission_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(self.mission_dir / 'state.json')

    def has_snapshots(self) -> bool:
        """Return whether persistence already has stored snapshots."""
        return self.json_file.exists()

    async def snapshot_node(self, state: MissionState, next_node: BaseNode[MissionState, Any, MissionResult]) -> None:
        """Persist the next node snapshot and a checkpointed state."""
        self._ensure_types()
        await super().snapshot_node(state, next_node)
        await self._save_checkpoint(state)

    async def snapshot_end(self, state: MissionState, end: End[MissionResult]) -> None:
        """Persist the end snapshot and a checkpointed state."""
        self._ensure_types()
        await super().snapshot_end(state, end)
        await self._save_checkpoint(state)

    async def load_next(self) -> NodeSnapshot[MissionState, MissionResult] | None:
        """Load the next resumable snapshot, falling back to stalled nodes."""
        self._ensure_types()
        async with self._lock():
            snapshots = await self.load_all()
            snapshot = self._select_resumable_snapshot(snapshots)
            if snapshot is None:
                return None
            snapshot.status = 'pending'
            await self._save(snapshots)
            return snapshot

    async def load_latest_snapshot(self) -> Snapshot[MissionState, MissionResult] | None:
        """Load the most recent snapshot."""
        self._ensure_types()
        snapshots = await self.load_all()
        return snapshots[-1] if snapshots else None

    async def load_latest_state(self) -> MissionState | None:
        """Load the most recent mission state from snapshots."""
        snapshot = await self.load_latest_snapshot()
        return snapshot.state if snapshot else None

    async def load_latest_result(self) -> MissionResult | None:
        """Load the mission result if the run already ended."""
        snapshot = await self.load_latest_snapshot()
        if isinstance(snapshot, EndSnapshot):
            return snapshot.result.data
        return None

    def _select_resumable_snapshot(
        self, snapshots: list[Snapshot[MissionState, MissionResult]]
    ) -> NodeSnapshot[MissionState, MissionResult] | None:
        for snapshot in reversed(snapshots):
            if isinstance(snapshot, NodeSnapshot) and snapshot.status in {'created', 'pending'}:
                return snapshot
        for snapshot in reversed(snapshots):
            if isinstance(snapshot, NodeSnapshot) and snapshot.status in {'running', 'error'}:
                return snapshot
        return None

    def _ensure_types(self) -> None:
        if not self.should_set_types():
            return
        from pydantic_graph.persistence import _utils

        from .nodes import DiscoveryNode, EvolutionNode, PrototypeNode, ResearchNode, SubmissionNode

        with _utils.set_nodes_type_context([DiscoveryNode, ResearchNode, PrototypeNode, EvolutionNode, SubmissionNode]):
            self.set_types(MissionState, MissionResult)

    async def _save_checkpoint(self, state: MissionState) -> None:
        """Save state with timestamp and clean up old checkpoints."""
        with logfire.span('mission.persistence.save', mission_id=self.mission_id):
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            checkpoint_path = self.mission_dir / f'{CHECKPOINT_PREFIX}{timestamp}.json'
            checkpoint_path.write_text(state.model_dump_json(indent=2), encoding='utf-8')
            await self._cleanup_old_checkpoints()

    async def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(
            self.mission_dir.glob(f'{CHECKPOINT_PREFIX}*.json'), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for old_checkpoint in checkpoints[self.max_checkpoints :]:
            old_checkpoint.unlink()


def create_persistence(mission_id: str) -> MissionPersistence:
    """Factory for mission persistence."""
    return MissionPersistence(mission_id)
