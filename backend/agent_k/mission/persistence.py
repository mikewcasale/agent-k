"""State persistence utilities for the mission graph.

@notice: |
    State persistence utilities for the mission graph.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.mission.persistence
    provides:
        - agent_k.mission.persistence:MissionPersistence
        - agent_k.mission.persistence:ResumeLedger
        - agent_k.mission.persistence:create_persistence
        - agent_k.mission.persistence:CHECKPOINT_DIR
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

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import logfire
from pydantic import BaseModel, Field, ValidationError
from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import EndSnapshot, NodeSnapshot, Snapshot
from pydantic_graph.persistence.file import FileStatePersistence

from agent_k.core.sage import Doc, Range

from .state import MissionResult, MissionState

__all__ = ("MissionPersistence", "ResumeLedger", "create_persistence", "CHECKPOINT_DIR", "RESUME_LEDGER_FILE")

SCHEMA_VERSION: Final[str] = "1.0.0"
CHECKPOINT_DIR: Final[Path] = Path("~/.agent_k/checkpoints").expanduser()
CHECKPOINT_PREFIX: Final[str] = "checkpoint_"
RESUME_LEDGER_FILE: Final[str] = "resume.json"


class ResumeLedger(BaseModel):
    """Durable record of unattended resume attempts for a mission.

    @notice: |
        Durable record of unattended resume attempts for a mission.

    @dev: |
        A mission only becomes unresumable once an end snapshot exists, so a mission that
        crashed terminally stays a resume candidate forever. The ledger bounds that replay:
        attempts accumulate per progress marker and reset as soon as the mission advances.

        @pattern:
            name: state-persistence
            rationale: "Bounds unattended resume replays with on-disk bookkeeping."
            violations: "In-memory counters reset on restart, which is when replays happen."
    """

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    attempts: int = Field(default=0, ge=0, description="Resume attempts made at the current progress marker")
    progress_marker: str | None = Field(default=None, description="Mission progress observed at the last attempt")
    last_attempt_at: datetime | None = Field(default=None, description="Timestamp of the last resume attempt")


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

    def last_activity_at(self) -> datetime | None:
        """Return when the mission last wrote a snapshot.

        @notice: |
            Returns the snapshot file modification time, or None when nothing is stored.

        @dev: |
            Used to age out long-abandoned missions before they are auto-resumed.
        """
        try:
            modified = self.json_file.stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(modified, tz=UTC)

    def load_resume_ledger(self) -> ResumeLedger:
        """Load the resume ledger, returning an empty ledger when absent or unreadable.

        @notice: |
            Returns the durable resume-attempt record for this mission.

        @dev: |
            A corrupt or truncated ledger is treated as empty so bookkeeping can never
            block a mission from being resumed.
        """
        path = self.mission_dir / RESUME_LEDGER_FILE
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError:
            return ResumeLedger()
        try:
            return ResumeLedger.model_validate_json(payload)
        except ValidationError:
            logfire.warning("mission_resume_ledger_invalid", mission_id=self.mission_id)
            return ResumeLedger()

    def record_resume_attempt(
        self, progress_marker: Annotated[str, Doc("Marker describing how far the mission has advanced.")]
    ) -> ResumeLedger:
        """Record a resume attempt and return the updated ledger.

        @notice: |
            Increments the attempt counter, resetting it when the mission has advanced.

        @dev: |
            The ledger is written atomically so a crash mid-write cannot corrupt it.

        @effects:
            io:
                - writes the mission resume ledger
        """
        ledger = self.load_resume_ledger()
        attempts = ledger.attempts + 1 if ledger.progress_marker == progress_marker else 1
        updated = ResumeLedger(attempts=attempts, progress_marker=progress_marker, last_attempt_at=datetime.now(UTC))
        path = self.mission_dir / RESUME_LEDGER_FILE
        tmp_path = path.parent / f"{path.name}.tmp"
        tmp_path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
        return updated

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
        """Save state with timestamp and clean up old checkpoints."""
        with logfire.span("mission.persistence.save", mission_id=self.mission_id):
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.mission_dir / f"{CHECKPOINT_PREFIX}{timestamp}.json"
            checkpoint_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            await self._cleanup_old_checkpoints()

    async def _cleanup_old_checkpoints(self) -> None:
        checkpoints = sorted(
            self.mission_dir.glob(f"{CHECKPOINT_PREFIX}*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for old_checkpoint in checkpoints[self.max_checkpoints :]:
            old_checkpoint.unlink()


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
