"""State persistence utilities for the mission graph.

@notice: |
    State persistence utilities for the mission graph.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.mission.persistence
    provides:
        - agent_k.mission.persistence:MissionPersistence
        - agent_k.mission.persistence:create_persistence
        - agent_k.mission.persistence:CHECKPOINT_DIR
        - agent_k.mission.persistence:CHECKPOINT_TIMESTAMP_FORMAT
    pattern: state-persistence

@similar:
    - id: agent_k.core.tracking
        when: "Tracking experiment metrics; this module persists mission state."

@agent-guidance:
    do:
        - "Use agent_k.mission.persistence as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import logfire
from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import EndSnapshot, NodeSnapshot, Snapshot
from pydantic_graph.persistence.file import FileStatePersistence

from agent_k.core.sage import Doc, Range

from .state import MissionResult, MissionState

__all__ = ("MissionPersistence", "create_persistence", "CHECKPOINT_DIR", "CHECKPOINT_TIMESTAMP_FORMAT")

CHECKPOINT_DIR: Final[Path] = Path("~/.agent_k/checkpoints").expanduser()
CHECKPOINT_PREFIX: Final[str] = "checkpoint_"
CHECKPOINT_TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S_%f"
_TEMP_SUFFIX: Final[str] = ".tmp"
_STALE_TEMP_SECONDS: Final[float] = 3600.0


class MissionPersistence(FileStatePersistence[MissionState, MissionResult]):
    """Mission-specific persistence with checkpoint rotation and resumability.

    @notice: |
        Mission-specific persistence with checkpoint rotation and resumability.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: state-persistence
            rationale: "Stores mission snapshots for resumption and auditability."
            violations: "Ad-hoc persistence complicates recovery."

        @collaborators:
            required:
                - pydantic_graph.persistence.file:FileStatePersistence
            injection: constructor
            lifecycle: "Scoped per mission."

        @concurrency:
            model: asyncio
            safe: false
            reason: "Mutates on-disk checkpoints and in-memory snapshots."

        @invariants:
            - "mission_dir exists before persistence operations."
            - "Checkpoint file names sort lexicographically in write order."
            - "A checkpoint file is either absent or complete; partial writes stay in temp files."
    """

    def __init__(
        self,
        mission_id: Annotated[str, Doc("Mission identifier for checkpoint storage.")],
        checkpoint_dir: Annotated[Path, Doc("Root directory for mission checkpoints.")] = CHECKPOINT_DIR,
        max_checkpoints: Annotated[int, Doc("Maximum checkpoints to retain."), Range(1, 1000)] = 10,
    ) -> None:
        self.mission_id = mission_id
        self.max_checkpoints = max_checkpoints
        self.mission_dir = checkpoint_dir / mission_id
        self.mission_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(self.mission_dir / "state.json")

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
            snapshot.status = "pending"
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
            if isinstance(snapshot, NodeSnapshot) and snapshot.status in {"created", "pending"}:
                return snapshot
        for snapshot in reversed(snapshots):
            if isinstance(snapshot, NodeSnapshot) and snapshot.status in {"running", "error"}:
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
        """Write a rotated checkpoint without letting disk failures abort the mission."""
        with logfire.span("mission.persistence.save", mission_id=self.mission_id):
            # Serialize on the event loop: the graph mutates state between awaits,
            # so a thread must never observe it mid-update.
            payload = state.model_dump_json(indent=2)
            try:
                checkpoint_path = await asyncio.to_thread(self._write_checkpoint, payload)
            except OSError as exc:
                # The authoritative snapshot is already on disk via the base class;
                # a rotated checkpoint is auxiliary and must not fail the run.
                logfire.warning("mission_checkpoint_write_failed", mission_id=self.mission_id, error=str(exc))
                return
            logfire.debug("mission_checkpoint_saved", mission_id=self.mission_id, checkpoint=checkpoint_path.name)
            await self._cleanup_old_checkpoints()

    async def _cleanup_old_checkpoints(self) -> None:
        removed = await asyncio.to_thread(self._prune_checkpoints)
        if removed:
            logfire.debug("mission_checkpoints_pruned", mission_id=self.mission_id, removed=removed)

    def _write_checkpoint(self, payload: str) -> Path:
        """Write ``payload`` to a fresh checkpoint file, replacing it atomically."""
        checkpoint_path = self._next_checkpoint_path()
        temp_path = checkpoint_path.with_name(f".{checkpoint_path.name}{_TEMP_SUFFIX}")
        try:
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(checkpoint_path)
        except OSError:
            temp_path.unlink(missing_ok=True)
            raise
        return checkpoint_path

    def _next_checkpoint_path(self) -> Path:
        """Return an unused checkpoint path that sorts after every existing one."""
        timestamp = datetime.now(UTC).strftime(CHECKPOINT_TIMESTAMP_FORMAT)
        checkpoint_path = self.mission_dir / f"{CHECKPOINT_PREFIX}{timestamp}.json"
        collision = 0
        while checkpoint_path.exists():
            collision += 1
            checkpoint_path = self.mission_dir / f"{CHECKPOINT_PREFIX}{timestamp}_{collision:03d}.json"
        return checkpoint_path

    def _prune_checkpoints(self) -> int:
        """Drop checkpoints beyond the retention limit and leftover temp files."""
        try:
            checkpoints = sorted(self.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"), reverse=True)
            stale_temps = self._stale_temp_files()
        except OSError as exc:
            logfire.warning("mission_checkpoint_prune_failed", mission_id=self.mission_id, error=str(exc))
            return 0

        removed = 0
        for path in [*checkpoints[self.max_checkpoints :], *stale_temps]:
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    def _stale_temp_files(self) -> list[Path]:
        cutoff = time.time() - _STALE_TEMP_SECONDS
        stale: list[Path] = []
        for path in self.mission_dir.glob(f".{CHECKPOINT_PREFIX}*.json{_TEMP_SUFFIX}"):
            try:
                modified_at = path.stat().st_mtime
            except OSError:
                continue
            if modified_at < cutoff:
                stale.append(path)
        return stale


def create_persistence(
    mission_id: Annotated[str, Doc("Mission identifier for checkpoint storage.")],
) -> MissionPersistence:
    """Factory for mission persistence.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Creates a MissionPersistence instance for the given mission.

        @factory-for:
            id: agent_k.mission.persistence:MissionPersistence
            rationale: "Centralizes checkpoint path defaults."
            singleton: false
            cache-key: mission_id

        @canonical-home:
            for:
                - "mission persistence construction"
            notes: "Use create_persistence to ensure defaults."
    """
    return MissionPersistence(mission_id)
