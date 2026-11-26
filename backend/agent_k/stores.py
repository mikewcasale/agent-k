"""Durable PostgreSQL runtime state and S3 artifact storage.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

__all__ = (
    "APIError",
    "APIErrorEnvelope",
    "AppendEventsReceipt",
    "ArtifactCompleteResponse",
    "ArtifactCompleteSpec",
    "ArtifactReference",
    "ArtifactUploadSpec",
    "ClaimJobsRequest",
    "ClaimJobsResponse",
    "ControlAction",
    "ControlDirective",
    "EventRecord",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "JobCommand",
    "JobCommandType",
    "JobCompletionRequest",
    "JobCompletionResponse",
    "JobEnqueueRequest",
    "JobFailure",
    "JobFailureRequest",
    "JobFailureResponse",
    "JobLease",
    "JobStatus",
    "JobWorkspace",
    "LeaseFenceError",
    "MigrationCreateRequest",
    "MigrationCreateResponse",
    "MigrationRecord",
    "MigrationSource",
    "MigrationSourceKind",
    "MigrationStatus",
    "MissionRecord",
    "MissionInterrupted",
    "MissionRunResult",
    "MissionSnapshot",
    "MissionSnapshotRecord",
    "PostgresRuntimeStore",
    "RuntimeStoreError",
    "S3ArtifactStore",
    "SignedArtifactURL",
    "SnapshotReceipt",
    "StoreConflictError",
    "WorkerEvent",
)

SCHEMA_VERSION = "1.0"
STATE_SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")


class WireModel(BaseModel):
    """Base model for strict, camel-case worker API payloads."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
    )


class JobStatus(StrEnum):
    """Durable runtime job states."""

    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ControlAction(StrEnum):
    """Control actions returned to a leased worker."""

    CONTINUE = "continue"
    PAUSE = "pause"
    CANCEL = "cancel"


class JobCommandType(StrEnum):
    """Mission commands accepted by a worker."""

    START = "start"
    RESUME = "resume"


class MigrationSourceKind(StrEnum):
    """Supported legacy import source families."""

    MISSION_STATE = "mission-state"
    EXPERIMENT_DB = "experiment-db"


class MigrationStatus(StrEnum):
    """Durable migration processing states."""

    AWAITING_UPLOAD = "awaiting-upload"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCommand(WireModel):
    """Command and resume information for a mission lease."""

    type: JobCommandType
    competition_id: str | None = Field(default=None, min_length=1, max_length=255)
    criteria: dict[str, Any] = Field(default_factory=dict)
    checkpoint_id: uuid.UUID | None = None


class JobWorkspace(WireModel):
    """Artifact namespace assigned to one job."""

    artifact_prefix: str = Field(min_length=1, max_length=768)

    @field_validator("artifact_prefix")
    @classmethod
    def validate_artifact_prefix(cls, value: str) -> str:
        """Reject absolute and traversing S3 key prefixes."""
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifactPrefix must be a relative, non-traversing key")
        return value.strip("/")


class JobLease(WireModel):
    """Fenced lease returned by the runtime queue."""

    schema_version: str = SCHEMA_VERSION
    job_id: uuid.UUID
    mission_id: uuid.UUID
    attempt: int = Field(ge=1)
    lease_id: uuid.UUID = Field(repr=False)
    lease_expires_at: AwareDatetime
    command: JobCommand
    workspace: JobWorkspace
    event_sequence: int = Field(default=0, ge=0)


class ClaimJobsRequest(WireModel):
    """Atomic worker claim request."""

    worker_id: str = Field(min_length=1, max_length=255)
    capabilities: tuple[str, ...] = ()
    lease_seconds: int = Field(default=60, ge=15, le=3600)
    max_jobs: int = Field(default=1, ge=1, le=16)


class ClaimJobsResponse(WireModel):
    """Leases returned to a worker poll."""

    jobs: tuple[JobLease, ...] = ()
    server_time: AwareDatetime


class ControlDirective(WireModel):
    """Current server-side control directive."""

    action: ControlAction = ControlAction.CONTINUE
    reason: str | None = Field(default=None, max_length=2000)
    requested_at: AwareDatetime | None = None
    server_time: AwareDatetime | None = None


class HeartbeatRequest(WireModel):
    """Lease renewal and progress update."""

    worker_id: str = Field(min_length=1, max_length=255)
    lease_id: uuid.UUID = Field(repr=False)
    progress: float | None = Field(default=None, ge=0, le=100)
    current_phase: str | None = Field(default=None, max_length=100)
    current_task_id: str | None = Field(default=None, max_length=255)


class HeartbeatResponse(WireModel):
    """Renewed lease and current control directive."""

    job_id: uuid.UUID
    lease_id: uuid.UUID = Field(repr=False)
    lease_expires_at: AwareDatetime
    control: ControlDirective
    server_time: AwareDatetime


class WorkerEvent(WireModel):
    """One durable mission event emitted by a worker."""

    event_id: uuid.UUID
    sequence: int = Field(ge=1)
    type: str = Field(min_length=1, max_length=100)
    timestamp: AwareDatetime
    data: JsonValue = Field(default_factory=dict)


class AppendEventsRequest(WireModel):
    """Ordered, idempotent event append request."""

    worker_id: str = Field(min_length=1, max_length=255)
    lease_id: uuid.UUID = Field(repr=False)
    events: tuple[WorkerEvent, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_order(self) -> AppendEventsRequest:
        """Require strictly increasing event sequences within a batch."""
        sequences = [event.sequence for event in self.events]
        if sequences != sorted(set(sequences)):
            raise ValueError("events must have unique, increasing sequences")
        return self


class AppendEventsReceipt(WireModel):
    """Acknowledgement for a durable event batch."""

    accepted_through: int = Field(ge=0)
    duplicate_sequences: tuple[int, ...] = ()
    control: ControlDirective


class EventRecord(WorkerEvent):
    """Persisted event with a durable replay cursor."""

    stream_sequence: int = Field(ge=1)
    mission_id: uuid.UUID
    job_id: uuid.UUID


class MissionSnapshot(WireModel):
    """Immutable graph snapshot uploaded by a worker."""

    schema_version: str = STATE_SCHEMA_VERSION
    sequence: int = Field(ge=0)
    created_at: AwareDatetime
    content_type: Literal["application/json"] = "application/json"
    payload: dict[str, Any]


class SnapshotReceipt(WireModel):
    """Receipt for an idempotently stored snapshot."""

    snapshot_id: uuid.UUID
    sequence: int = Field(ge=0)
    created_at: AwareDatetime


class ArtifactReference(WireModel):
    """Reference to an immutable uploaded artifact."""

    artifact_id: uuid.UUID
    role: str = Field(min_length=1, max_length=100)


class MissionSnapshotRecord(MissionSnapshot):
    """Fenced snapshot payload and its immutable workspace artifacts."""

    snapshot_id: uuid.UUID
    job_id: uuid.UUID
    mission_id: uuid.UUID
    artifacts: tuple[ArtifactReference, ...] = ()


class ArtifactUploadSpec(WireModel):
    """Metadata required before signing an artifact upload."""

    worker_id: str = Field(min_length=1, max_length=255)
    lease_id: uuid.UUID = Field(repr=False)
    name: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str
    role: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Accept only portable artifact basenames."""
        if not SAFE_ARTIFACT_NAME.fullmatch(value):
            raise ValueError("name must be a portable artifact basename")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require a lowercase SHA-256 digest."""
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        return value


class ArtifactCompleteSpec(WireModel):
    """Worker identity and immutable metadata for an uploaded artifact."""

    worker_id: str = Field(min_length=1, max_length=255)
    lease_id: uuid.UUID = Field(repr=False)
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require a lowercase SHA-256 digest."""
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        return value


class ArtifactCompleteResponse(WireModel):
    """Server-verified artifact completion receipt."""

    artifact_id: uuid.UUID
    status: Literal["available"] = "available"
    verified_at: AwareDatetime


class SignedArtifactURL(WireModel):
    """Short-lived job-scoped S3 operation."""

    artifact_id: uuid.UUID
    method: Literal["GET", "PUT"]
    url: str = Field(min_length=1, repr=False)
    headers: dict[str, str] = Field(default_factory=dict)
    expires_at: AwareDatetime


class MissionRunResult(WireModel):
    """Successful mission result returned by the isolated mission process."""

    schema_version: str = SCHEMA_VERSION
    success: Literal[True] = True
    mission_id: uuid.UUID
    competition_id: str | None = Field(default=None, min_length=1, max_length=255)
    final_rank: int | None = Field(default=None, ge=1)
    final_score: float | None = None
    total_submissions: int = Field(default=0, ge=0)
    evolution_generations: int = Field(default=0, ge=0)
    evolution_iterations: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    phases_completed: tuple[str, ...] = ()
    final_event_sequence: int = Field(default=0, ge=0)
    artifacts: tuple[ArtifactReference, ...] = ()


class JobCompletionRequest(WireModel):
    """Idempotent completion mutation for a fenced job."""

    worker_id: str = Field(min_length=1, max_length=255)
    lease_id: uuid.UUID = Field(repr=False)
    idempotency_key: str = Field(min_length=16, max_length=255)
    result: MissionRunResult
    final_event_sequence: int = Field(ge=0)
    artifacts: tuple[ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def validate_result_envelope(self) -> JobCompletionRequest:
        """Require duplicated terminal metadata to match the signed result body."""
        if self.final_event_sequence != self.result.final_event_sequence:
            raise ValueError("finalEventSequence must match result.finalEventSequence")
        if self.artifacts != self.result.artifacts:
            raise ValueError("artifacts must match result.artifacts")
        return self


class JobCompletionResponse(WireModel):
    """Durable job completion receipt."""

    job_id: uuid.UUID
    mission_id: uuid.UUID
    status: Literal["completed"] = "completed"
    completed_at: AwareDatetime


class JobFailure(WireModel):
    """Structured failure safe for storage and transport."""

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    retryable: bool
    category: str | None = Field(default=None, max_length=100)
    details: dict[str, Any] | None = None


class JobFailureRequest(WireModel):
    """Idempotent failure mutation for a fenced job."""

    worker_id: str = Field(min_length=1, max_length=255)
    lease_id: uuid.UUID = Field(repr=False)
    idempotency_key: str = Field(min_length=16, max_length=255)
    error: JobFailure
    final_event_sequence: int = Field(ge=0)


class JobFailureResponse(WireModel):
    """Failure receipt indicating whether another attempt is queued."""

    job_id: uuid.UUID
    mission_id: uuid.UUID
    status: Literal["failed", "queued"]
    attempt: int = Field(ge=1)
    next_attempt_at: AwareDatetime | None = None
    failed_at: AwareDatetime | None = None


class JobEnqueueRequest(WireModel):
    """Idempotent job creation request used by the API."""

    job_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    mission_id: uuid.UUID
    command: JobCommand
    workspace: JobWorkspace
    requirements: tuple[str, ...] = ()
    idempotency_key: str = Field(min_length=16, max_length=255)
    priority: int = Field(default=0, ge=-1000, le=1000)
    max_attempts: int = Field(default=3, ge=1, le=20)
    available_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class MissionRecord(WireModel):
    """Persisted mission state returned by the runtime store."""

    mission_id: uuid.UUID
    owner_id: str
    schema_version: str
    status: str
    state: dict[str, Any]
    requested_control: ControlAction
    created_at: AwareDatetime
    updated_at: AwareDatetime


class MigrationSource(WireModel):
    """Immutable legacy source metadata."""

    kind: MigrationSourceKind
    schema_version: str = Field(min_length=1, max_length=64)
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        """Require a lowercase SHA-256 digest."""
        if not SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        return value


class MigrationCreateRequest(WireModel):
    """Request a signed upload for a legacy source."""

    worker_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=16, max_length=255)
    source: MigrationSource
    content_type: Literal["application/json", "application/x-sqlite3"]

    @model_validator(mode="after")
    def validate_content_type(self) -> MigrationCreateRequest:
        """Match the content type to the legacy source family."""
        expected = {
            MigrationSourceKind.MISSION_STATE: "application/json",
            MigrationSourceKind.EXPERIMENT_DB: "application/x-sqlite3",
        }[self.source.kind]
        if self.content_type != expected:
            raise ValueError(f"contentType must be {expected!r} for {self.source.kind}")
        return self


class MigrationCreateResponse(WireModel):
    """Signed migration upload and durable identifier."""

    migration_id: uuid.UUID
    upload: SignedArtifactURL
    status: Literal["awaiting-upload"] = "awaiting-upload"


class MigrationRecord(WireModel):
    """Current status of a legacy source import."""

    migration_id: uuid.UUID
    status: MigrationStatus
    counts: dict[str, int] | None = None
    error: JobFailure | None = None


class APIError(WireModel):
    """Stable worker API error body."""

    code: str
    message: str
    retryable: bool
    details: dict[str, Any] | None = None


class APIErrorEnvelope(WireModel):
    """Stable worker API error envelope."""

    error: APIError
    request_id: str


class RuntimeStoreError(RuntimeError):
    """Base error raised by runtime persistence."""

    code = "runtime_store_error"
    retryable = False


class MissionInterrupted(RuntimeError):
    """Signal a durably checkpointed pause or cancellation to the worker."""

    def __init__(
        self,
        action: Literal["pause", "cancel"],
        reason: str | None = None,
        *,
        final_event_sequence: int,
    ) -> None:
        super().__init__(reason or action)
        self.action = action
        self.reason = reason
        self.final_event_sequence = final_event_sequence


class StoreConflictError(RuntimeStoreError):
    """A durable idempotency or sequence invariant was violated."""

    code = "store_conflict"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details


class LeaseFenceError(StoreConflictError):
    """A stale or expired lease attempted a mutation."""

    code = "stale_lease"


class PostgresRuntimeStore:
    """Async PostgreSQL store for mission state, queue leases, and events."""

    def __init__(
        self,
        database_url: str,
        *,
        migrations_dir: Path | None = None,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("database_url must be a PostgreSQL URL")
        self._database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        self._migrations_dir = migrations_dir
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: Any = None
        self._dict_row: Any = None

    async def __aenter__(self) -> PostgresRuntimeStore:
        await self.open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def open(self) -> None:
        """Open and validate the PostgreSQL connection pool."""
        if self._pool is not None:
            return
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool
        except ImportError as error:
            raise RuntimeError("PostgreSQL support requires `psycopg[binary,pool]`.") from error
        self._dict_row = dict_row
        self._pool = AsyncConnectionPool(
            self._database_url,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        await self._pool.open()
        await self._pool.wait()

    async def close(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self._pool is None:
            return
        pool = self._pool
        self._pool = None
        await pool.close()

    async def initialize(self) -> None:
        """Apply checksum-protected SQL migrations in filename order."""
        pool = self._require_pool()
        migration_paths: list[Any]
        if self._migrations_dir is None:
            resource_root = files("agent_k").joinpath("resources")
            migration_paths = sorted(
                (
                    resource
                    for resource in resource_root.iterdir()
                    if re.fullmatch(r"[0-9]{4}_.+\.sql", resource.name)
                ),
                key=lambda resource: resource.name,
            )
            location = "agent_k/resources"
        else:
            migration_paths = sorted(self._migrations_dir.glob("[0-9][0-9][0-9][0-9]_*.sql"))
            location = str(self._migrations_dir)
        if not migration_paths:
            raise RuntimeStoreError(f"No SQL migrations found in {location}")
        async with pool.connection() as connection:
            await connection.execute(
                "SELECT pg_advisory_lock(hashtext('agent-k-runtime-migrations'))"
            )
            try:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_schema_migrations (
                        version TEXT PRIMARY KEY,
                        sha256 TEXT NOT NULL,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                for migration_path in migration_paths:
                    sql = migration_path.read_text(encoding="utf-8")
                    digest = hashlib.sha256(sql.encode()).hexdigest()
                    row = await (
                        await connection.execute(
                            "SELECT sha256 FROM runtime_schema_migrations WHERE version = %s",
                            (migration_path.name,),
                        )
                    ).fetchone()
                    if row is not None:
                        if row["sha256"] != digest:
                            raise StoreConflictError(
                                f"Applied migration changed: {migration_path.name}",
                            )
                        continue
                    async with connection.transaction():
                        await connection.execute(sql, prepare=False)
                        await connection.execute(
                            """
                            INSERT INTO runtime_schema_migrations (version, sha256)
                            VALUES (%s, %s)
                            """,
                            (migration_path.name, digest),
                        )
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock(hashtext('agent-k-runtime-migrations'))"
                )

    async def create_mission(
        self,
        mission_id: uuid.UUID,
        owner_id: str,
        *,
        state: dict[str, Any] | None = None,
        schema_version: str = STATE_SCHEMA_VERSION,
    ) -> MissionRecord:
        """Create a mission idempotently and return its durable state."""
        pool = self._require_pool()
        payload = state or {}
        state_sha256 = _canonical_sha256(payload)
        async with pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO runtime_missions (
                    mission_id, owner_id, schema_version, state, state_sha256
                ) VALUES (%s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (mission_id) DO NOTHING
                """,
                (mission_id, owner_id, schema_version, _json(payload), state_sha256),
            )
            row = await self._fetch_mission(connection, mission_id)
        if row["owner_id"] != owner_id:
            raise StoreConflictError("missionId is already owned by another principal")
        return _mission_record(row)

    async def get_mission(self, mission_id: uuid.UUID) -> MissionRecord | None:
        """Return one mission or ``None`` when it does not exist."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM runtime_missions WHERE mission_id = %s",
                    (mission_id,),
                )
            ).fetchone()
            if row is not None:
                await self._decorate_mission_row(connection, row)
        return _mission_record(row) if row is not None else None

    async def require_owner(self, mission_id: uuid.UUID, owner_id: str) -> MissionRecord:
        """Return a mission only when the caller owns it."""
        mission = await self.get_mission(mission_id)
        if mission is None:
            raise KeyError(str(mission_id))
        if mission.owner_id != owner_id:
            raise PermissionError("mission owner does not match")
        return mission

    async def request_control(
        self,
        mission_id: uuid.UUID,
        action: ControlAction,
        *,
        reason: str | None = None,
    ) -> ControlDirective:
        """Persist a pause, cancel, or continue directive for a mission."""
        pool = self._require_pool()
        requested_at = datetime.now(UTC)
        async with pool.connection() as connection, connection.transaction():
            row = await (
                await connection.execute(
                    """
                    UPDATE runtime_missions
                    SET requested_control = %s,
                        control_reason = %s,
                        control_requested_at = %s,
                        status = CASE
                            WHEN %s = 'continue' AND status IN ('paused', 'migrated')
                            THEN 'queued'
                            ELSE status
                        END,
                        updated_at = NOW()
                    WHERE mission_id = %s
                    RETURNING requested_control, control_reason, control_requested_at
                    """,
                    (action.value, reason, requested_at, action.value, mission_id),
                )
            ).fetchone()
            if action is ControlAction.CONTINUE:
                await connection.execute(
                    """
                    UPDATE runtime_jobs
                    SET status = 'queued', available_at = NOW(), updated_at = NOW()
                    WHERE mission_id = %s AND status = 'paused'
                    """,
                    (mission_id,),
                )
        if row is None:
            raise KeyError(str(mission_id))
        return _control(row)

    async def enqueue_job(self, request: JobEnqueueRequest) -> uuid.UUID:
        """Create one queue job idempotently."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO runtime_jobs (
                    job_id, mission_id, command, workspace, requirements,
                    idempotency_key, priority, max_attempts, available_at
                ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    request.job_id,
                    request.mission_id,
                    request.command.model_dump_json(by_alias=True),
                    request.workspace.model_dump_json(by_alias=True),
                    list(request.requirements),
                    request.idempotency_key,
                    request.priority,
                    request.max_attempts,
                    request.available_at,
                ),
            )
            row = await (
                await connection.execute(
                    "SELECT job_id FROM runtime_jobs WHERE idempotency_key = %s",
                    (request.idempotency_key,),
                )
            ).fetchone()
        if row is None:
            raise RuntimeStoreError("job insert did not return an id")
        return uuid.UUID(str(row["job_id"]))

    async def claim_jobs(self, request: ClaimJobsRequest) -> ClaimJobsResponse:
        """Atomically claim available jobs using row locks and lease fencing."""
        pool = self._require_pool()
        leases: list[JobLease] = []
        async with pool.connection() as connection, connection.transaction():
            for _index in range(request.max_jobs):
                row = await (
                    await connection.execute(
                        """
                        SELECT j.*,
                            (
                                SELECT COALESCE(MAX(e.job_sequence), 0)
                                FROM runtime_events AS e
                                WHERE e.job_id = j.job_id
                            ) AS event_sequence
                        FROM runtime_jobs AS j
                        JOIN runtime_missions AS m USING (mission_id)
                        WHERE (
                            (j.status = 'queued' AND j.available_at <= NOW())
                            OR (
                                j.status IN ('leased', 'running')
                                AND j.lease_expires_at < NOW()
                            )
                        )
                        AND j.attempt < j.max_attempts
                        AND m.requested_control = 'continue'
                        AND %s::text[] @> j.requirements
                        ORDER BY j.priority DESC, j.available_at, j.created_at
                        FOR UPDATE OF j SKIP LOCKED
                        LIMIT 1
                        """,
                        (list(request.capabilities),),
                    )
                ).fetchone()
                if row is None:
                    break
                await connection.execute(
                    """
                    UPDATE runtime_job_attempts
                    SET ended_at = NOW(), outcome = 'fenced'
                    WHERE job_id = %s AND ended_at IS NULL
                    """,
                    (row["job_id"],),
                )
                lease_id = uuid.uuid4()
                attempt = int(row["attempt"]) + 1
                expires_at = datetime.now(UTC) + timedelta(seconds=request.lease_seconds)
                checkpoint = await (
                    await connection.execute(
                        """
                        SELECT snapshot_id FROM runtime_snapshots
                        WHERE job_id = %s ORDER BY sequence DESC LIMIT 1
                        """,
                        (row["job_id"],),
                    )
                ).fetchone()
                command = dict(row["command"])
                if checkpoint is not None:
                    command["type"] = JobCommandType.RESUME.value
                    command["checkpointId"] = str(checkpoint["snapshot_id"])
                claimed = await (
                    await connection.execute(
                        """
                        UPDATE runtime_jobs
                        SET status = 'leased', attempt = %s, worker_id = %s,
                            lease_id = %s, lease_expires_at = %s,
                            heartbeat_at = NOW(), updated_at = NOW(), error = NULL,
                            command = %s::jsonb
                        WHERE job_id = %s
                        RETURNING *
                        """,
                        (
                            attempt,
                            request.worker_id,
                            lease_id,
                            expires_at,
                            _json(command),
                            row["job_id"],
                        ),
                    )
                ).fetchone()
                claimed["event_sequence"] = row["event_sequence"]
                await connection.execute(
                    """
                    INSERT INTO runtime_job_attempts (
                        job_id, attempt, lease_id, worker_id
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (row["job_id"], attempt, lease_id, request.worker_id),
                )
                await connection.execute(
                    """
                    UPDATE runtime_missions
                    SET status = 'running', updated_at = NOW()
                    WHERE mission_id = %s AND status NOT IN ('completed', 'canceled')
                    """,
                    (row["mission_id"],),
                )
                leases.append(_job_lease(claimed))
        return ClaimJobsResponse(jobs=tuple(leases), server_time=datetime.now(UTC))

    async def heartbeat(
        self,
        job_id: uuid.UUID,
        request: HeartbeatRequest,
        *,
        lease_seconds: int | None = None,
    ) -> HeartbeatResponse:
        """Renew an active lease and return its mission control directive."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(
                connection,
                job_id,
                request.worker_id,
                request.lease_id,
            )
            persisted_seconds = int((job["lease_expires_at"] - job["heartbeat_at"]).total_seconds())
            renewal_seconds = lease_seconds or max(15, min(3600, persisted_seconds))
            expires_at = datetime.now(UTC) + timedelta(seconds=renewal_seconds)
            row = await (
                await connection.execute(
                    """
                    UPDATE runtime_jobs
                    SET status = 'running', lease_expires_at = %s,
                        heartbeat_at = NOW(), progress = COALESCE(%s, progress),
                        current_phase = COALESCE(%s, current_phase),
                        current_task_id = COALESCE(%s, current_task_id),
                        updated_at = NOW()
                    WHERE job_id = %s
                    RETURNING lease_expires_at
                    """,
                    (
                        expires_at,
                        request.progress,
                        request.current_phase,
                        request.current_task_id,
                        job_id,
                    ),
                )
            ).fetchone()
            await connection.execute(
                """
                UPDATE runtime_job_attempts
                SET heartbeat_at = NOW()
                WHERE job_id = %s AND attempt = %s
                """,
                (job_id, job["attempt"]),
            )
            control = await self._control_for_mission(connection, job["mission_id"])
        return HeartbeatResponse(
            job_id=job_id,
            lease_id=request.lease_id,
            lease_expires_at=row["lease_expires_at"],
            control=control,
            server_time=datetime.now(UTC),
        )

    async def append_events(
        self,
        job_id: uuid.UUID,
        request: AppendEventsRequest,
    ) -> AppendEventsReceipt:
        """Append a contiguous event batch with event-id deduplication."""
        pool = self._require_pool()
        duplicates: list[int] = []
        accepted_through = 0
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(
                connection,
                job_id,
                request.worker_id,
                request.lease_id,
            )
            sequence_row = await (
                await connection.execute(
                    "SELECT MAX(job_sequence) AS maximum FROM runtime_events WHERE job_id = %s",
                    (job_id,),
                )
            ).fetchone()
            expected = int(sequence_row["maximum"] or 0) + 1
            for event in request.events:
                if event.sequence < expected:
                    existing = await (
                        await connection.execute(
                            """
                            SELECT event_id FROM runtime_events
                            WHERE job_id = %s AND job_sequence = %s
                            """,
                            (job_id, event.sequence),
                        )
                    ).fetchone()
                    if existing is None or existing["event_id"] != event.event_id:
                        raise StoreConflictError(
                            "event sequence already contains different data",
                            details={
                                "sequence": event.sequence,
                                "expectedSequence": expected,
                            },
                        )
                    duplicates.append(event.sequence)
                    accepted_through = max(accepted_through, event.sequence)
                    continue
                if event.sequence != expected:
                    raise StoreConflictError(
                        "event sequence gap",
                        details={"expectedSequence": expected},
                    )
                await connection.execute(
                    """
                    INSERT INTO runtime_events (
                        event_id, mission_id, job_id, job_sequence,
                        event_type, event_timestamp, data
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        event.event_id,
                        job["mission_id"],
                        job_id,
                        event.sequence,
                        event.type,
                        event.timestamp,
                        _json(event.data),
                    ),
                )
                accepted_through = event.sequence
                expected += 1
            control = await self._control_for_mission(connection, job["mission_id"])
        return AppendEventsReceipt(
            accepted_through=accepted_through,
            duplicate_sequences=tuple(duplicates),
            control=control,
        )

    async def save_snapshot(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
        snapshot: MissionSnapshot,
    ) -> SnapshotReceipt:
        """Store a content-addressed mission snapshot idempotently."""
        pool = self._require_pool()
        payload_sha256 = _canonical_sha256(snapshot.payload)
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(connection, job_id, worker_id, lease_id)
            existing = await (
                await connection.execute(
                    """
                    SELECT snapshot_id, payload_sha256, created_at
                    FROM runtime_snapshots WHERE job_id = %s AND sequence = %s
                    """,
                    (job_id, snapshot.sequence),
                )
            ).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != payload_sha256:
                    raise StoreConflictError("snapshot sequence has different content")
                return SnapshotReceipt(
                    snapshot_id=existing["snapshot_id"],
                    sequence=snapshot.sequence,
                    created_at=existing["created_at"],
                )
            snapshot_id = uuid.uuid4()
            await connection.execute(
                """
                INSERT INTO runtime_snapshots (
                    snapshot_id, mission_id, job_id, sequence, schema_version,
                    content_type, payload, payload_sha256, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    snapshot_id,
                    job["mission_id"],
                    job_id,
                    snapshot.sequence,
                    snapshot.schema_version,
                    snapshot.content_type,
                    _json(snapshot.payload),
                    payload_sha256,
                    snapshot.created_at,
                ),
            )
            state = snapshot.payload.get("state")
            if isinstance(state, dict):
                await connection.execute(
                    """
                    UPDATE runtime_missions
                    SET state = %s::jsonb, state_sha256 = %s, updated_at = NOW()
                    WHERE mission_id = %s
                    """,
                    (_json(state), _canonical_sha256(state), job["mission_id"]),
                )
        return SnapshotReceipt(
            snapshot_id=snapshot_id,
            sequence=snapshot.sequence,
            created_at=snapshot.created_at,
        )

    async def get_snapshot(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
        snapshot_id: uuid.UUID,
    ) -> MissionSnapshotRecord | None:
        """Return one snapshot only to the current fenced job lease."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(connection, job_id, worker_id, lease_id)
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_snapshots
                    WHERE snapshot_id = %s AND mission_id = %s
                    """,
                    (snapshot_id, job["mission_id"]),
                )
            ).fetchone()
            if row is None:
                return None
            artifact_rows = await (
                await connection.execute(
                    """
                    SELECT artifact_id, role FROM runtime_artifacts
                    WHERE job_id = %s AND status = 'available'
                    ORDER BY created_at, artifact_id
                    """,
                    (row["job_id"],),
                )
            ).fetchall()
        return MissionSnapshotRecord(
            snapshot_id=row["snapshot_id"],
            job_id=row["job_id"],
            mission_id=row["mission_id"],
            schema_version=row["schema_version"],
            sequence=row["sequence"],
            created_at=row["created_at"],
            content_type=row["content_type"],
            payload=row["payload"],
            artifacts=tuple(
                ArtifactReference(artifact_id=artifact["artifact_id"], role=artifact["role"])
                for artifact in artifact_rows
            ),
        )

    async def complete_job(
        self,
        job_id: uuid.UUID,
        request: JobCompletionRequest,
    ) -> JobCompletionResponse:
        """Complete a fenced job exactly once."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            saved = await self._idempotent_response(
                connection,
                job_id,
                "complete",
                request.idempotency_key,
                JobCompletionResponse,
            )
            if saved is not None:
                return JobCompletionResponse.model_validate(saved)
            job = await self._lock_lease(
                connection,
                job_id,
                request.worker_id,
                request.lease_id,
            )
            if request.result.mission_id != job["mission_id"]:
                raise StoreConflictError("result missionId does not match leased mission")
            await self._require_terminal_event(
                connection,
                job_id,
                request.final_event_sequence,
                success=True,
            )
            await self._require_available_artifacts(
                connection,
                job_id,
                request.artifacts,
            )
            completed_at = datetime.now(UTC)
            response = JobCompletionResponse(
                job_id=job_id,
                mission_id=job["mission_id"],
                completed_at=completed_at,
            )
            await connection.execute(
                """
                UPDATE runtime_jobs
                SET status = 'completed', result = %s::jsonb, completed_at = %s,
                    updated_at = NOW(), lease_expires_at = NULL
                WHERE job_id = %s
                """,
                (request.result.model_dump_json(by_alias=True), completed_at, job_id),
            )
            await connection.execute(
                """
                UPDATE runtime_job_attempts
                SET ended_at = %s, outcome = 'completed'
                WHERE job_id = %s AND attempt = %s
                """,
                (completed_at, job_id, job["attempt"]),
            )
            await connection.execute(
                """
                UPDATE runtime_missions
                SET status = 'completed', state = %s::jsonb,
                    state_sha256 = %s, completed_at = %s, updated_at = NOW()
                WHERE mission_id = %s
                """,
                (
                    request.result.model_dump_json(by_alias=True),
                    _canonical_sha256(request.result.model_dump(mode="json", by_alias=True)),
                    completed_at,
                    job["mission_id"],
                ),
            )
            await self._save_idempotent_response(
                connection,
                job_id,
                "complete",
                request.idempotency_key,
                response,
            )
        return response

    async def fail_job(
        self,
        job_id: uuid.UUID,
        request: JobFailureRequest,
    ) -> JobFailureResponse:
        """Fail or requeue a fenced job according to retry and control state."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            saved = await self._idempotent_response(
                connection,
                job_id,
                "fail",
                request.idempotency_key,
                JobFailureResponse,
            )
            if saved is not None:
                return JobFailureResponse.model_validate(saved)
            job = await self._lock_lease(
                connection,
                job_id,
                request.worker_id,
                request.lease_id,
            )
            await self._require_event_cursor(
                connection,
                job_id,
                request.final_event_sequence,
            )
            control = await self._control_for_mission(connection, job["mission_id"])
            can_retry = request.error.retryable and job["attempt"] < job["max_attempts"]
            queued = can_retry and control.action is not ControlAction.CANCEL
            now = datetime.now(UTC)
            delay = min(300, 5 * (2 ** (int(job["attempt"]) - 1)))
            next_attempt_at = now + timedelta(seconds=delay) if queued else None
            database_status = "queued" if queued else "failed"
            outcome = "requeued" if queued else "failed"
            if control.action is ControlAction.PAUSE:
                queued = True
                database_status = "paused"
                next_attempt_at = now
                outcome = "paused"
            if control.action is ControlAction.CANCEL:
                database_status = "canceled"
                outcome = "canceled"
            if not queued:
                await self._ensure_failure_terminal_event(
                    connection,
                    job,
                    request.final_event_sequence,
                    request.error,
                )
            snapshot_row = await (
                await connection.execute(
                    """
                    SELECT snapshot_id FROM runtime_snapshots
                    WHERE job_id = %s ORDER BY sequence DESC LIMIT 1
                    """,
                    (job_id,),
                )
            ).fetchone()
            checkpoint_id = snapshot_row["snapshot_id"] if snapshot_row is not None else None
            response = JobFailureResponse(
                job_id=job_id,
                mission_id=job["mission_id"],
                status="queued" if queued else "failed",
                attempt=job["attempt"],
                next_attempt_at=next_attempt_at,
                failed_at=None if queued else now,
            )
            await connection.execute(
                """
                UPDATE runtime_jobs
                SET status = %s, error = %s::jsonb, available_at = COALESCE(%s, available_at),
                    failed_at = CASE WHEN %s = 'failed' THEN %s ELSE failed_at END,
                    command = CASE
                        WHEN %s::uuid IS NOT NULL AND %s IN ('queued', 'paused')
                        THEN jsonb_set(
                            jsonb_set(command, '{checkpointId}', to_jsonb(%s::text), TRUE),
                            '{type}', '"resume"'::jsonb, TRUE
                        )
                        ELSE command
                    END,
                    max_attempts = CASE
                        WHEN %s = 'paused' THEN max_attempts + 1
                        ELSE max_attempts
                    END,
                    lease_id = NULL, lease_expires_at = NULL, worker_id = NULL,
                    updated_at = NOW()
                WHERE job_id = %s
                """,
                (
                    database_status,
                    request.error.model_dump_json(by_alias=True),
                    next_attempt_at,
                    database_status,
                    now,
                    checkpoint_id,
                    database_status,
                    checkpoint_id,
                    database_status,
                    job_id,
                ),
            )
            await connection.execute(
                """
                UPDATE runtime_job_attempts
                SET ended_at = %s, outcome = %s, error = %s::jsonb
                WHERE job_id = %s AND attempt = %s
                """,
                (
                    now,
                    outcome,
                    request.error.model_dump_json(by_alias=True),
                    job_id,
                    job["attempt"],
                ),
            )
            if control.action is ControlAction.PAUSE:
                await connection.execute(
                    """
                    UPDATE runtime_missions
                    SET status = 'paused', updated_at = NOW()
                    WHERE mission_id = %s AND status != 'completed'
                    """,
                    (job["mission_id"],),
                )
            elif not queued:
                await connection.execute(
                    """
                    UPDATE runtime_missions
                    SET status = %s, updated_at = NOW()
                    WHERE mission_id = %s AND status != 'completed'
                    """,
                    (
                        "canceled" if control.action is ControlAction.CANCEL else "failed",
                        job["mission_id"],
                    ),
                )
            await self._save_idempotent_response(
                connection,
                job_id,
                "fail",
                request.idempotency_key,
                response,
            )
        return response

    async def get_control(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
    ) -> ControlDirective:
        """Return control only for the current fenced lease."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(connection, job_id, worker_id, lease_id)
            return await self._control_for_mission(connection, job["mission_id"])

    async def list_events(
        self,
        mission_id: uuid.UUID,
        *,
        after_sequence: int = 0,
        limit: int = 1000,
    ) -> tuple[EventRecord, ...]:
        """Replay durable mission events after a global stream cursor."""
        if limit < 1 or limit > 5000:
            raise ValueError("limit must be between 1 and 5000")
        pool = self._require_pool()
        async with pool.connection() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_events
                    WHERE mission_id = %s AND stream_sequence > %s
                    ORDER BY stream_sequence LIMIT %s
                    """,
                    (mission_id, after_sequence, limit),
                )
            ).fetchall()
        return tuple(
            EventRecord(
                event_id=row["event_id"],
                sequence=row["job_sequence"],
                type=row["event_type"],
                timestamp=row["event_timestamp"],
                data=row["data"],
                stream_sequence=row["stream_sequence"],
                mission_id=row["mission_id"],
                job_id=row["job_id"],
            )
            for row in rows
        )

    async def list_best_results(
        self,
        owner_id: str,
        limit: int,
    ) -> tuple[dict[str, Any], ...]:
        """List an owner's completed mission results ordered by best available rank."""
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        pool = self._require_pool()
        async with pool.connection() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT mission_id, state, completed_at
                    FROM runtime_missions
                    WHERE owner_id = %s AND status = 'completed'
                    ORDER BY
                        NULLIF(state->>'finalRank', '')::integer ASC NULLS LAST,
                        NULLIF(state->>'finalScore', '')::double precision DESC NULLS LAST,
                        completed_at DESC
                    LIMIT %s
                    """,
                    (owner_id, limit),
                )
            ).fetchall()
        return tuple(
            {
                **row["state"],
                "missionId": str(row["mission_id"]),
                "completedAt": row["completed_at"].isoformat(),
            }
            for row in rows
        )

    async def reserve_artifact(
        self,
        job_id: uuid.UUID,
        spec: ArtifactUploadSpec,
        *,
        bucket: str,
    ) -> dict[str, Any]:
        """Authorize a lease and reserve immutable artifact metadata."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(connection, job_id, spec.worker_id, spec.lease_id)
            artifact_id = uuid.uuid4()
            workspace = JobWorkspace.model_validate(job["workspace"])
            object_key = (
                f"{workspace.artifact_prefix}/attempt-artifacts/{artifact_id.hex}/{spec.name}"
            )
            await connection.execute(
                """
                INSERT INTO runtime_artifacts (
                    artifact_id, mission_id, job_id, attempt, role, name,
                    content_type, size_bytes, sha256, bucket, object_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    artifact_id,
                    job["mission_id"],
                    job_id,
                    job["attempt"],
                    spec.role,
                    spec.name,
                    spec.content_type,
                    spec.size_bytes,
                    spec.sha256,
                    bucket,
                    object_key,
                ),
            )
        return {
            "artifact_id": artifact_id,
            "bucket": bucket,
            "object_key": object_key,
            "sha256": spec.sha256,
            "size_bytes": spec.size_bytes,
        }

    async def get_artifact_for_verification(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
        artifact_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Authorize access to reserved metadata under the current lease."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            await self._lock_lease(connection, job_id, worker_id, lease_id)
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_artifacts
                    WHERE artifact_id = %s AND job_id = %s
                    """,
                    (artifact_id, job_id),
                )
            ).fetchone()
        if row is None:
            raise KeyError(str(artifact_id))
        return dict(row)

    async def get_artifact_for_download(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
        artifact_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Authorize a job-scoped artifact download."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            job = await self._lock_lease(connection, job_id, worker_id, lease_id)
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_artifacts
                    WHERE artifact_id = %s AND mission_id = %s AND status = 'available'
                    """,
                    (artifact_id, job["mission_id"]),
                )
            ).fetchone()
        if row is None:
            raise KeyError(str(artifact_id))
        return dict(row)

    async def mark_artifact_available(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
        artifact_id: uuid.UUID,
        *,
        actual_size: int,
        actual_sha256: str,
    ) -> ArtifactCompleteResponse:
        """Finalize an upload only when immutable metadata matches."""
        pool = self._require_pool()
        verified_at = datetime.now(UTC)
        async with pool.connection() as connection, connection.transaction():
            await self._lock_lease(connection, job_id, worker_id, lease_id)
            row = await (
                await connection.execute(
                    """
                    UPDATE runtime_artifacts
                    SET status = CASE
                            WHEN size_bytes = %s AND sha256 = %s THEN 'available'
                            ELSE 'invalid'
                        END,
                        available_at = CASE
                            WHEN size_bytes = %s AND sha256 = %s THEN NOW()
                            ELSE NULL
                        END
                    WHERE artifact_id = %s AND job_id = %s
                    RETURNING status
                    """,
                    (
                        actual_size,
                        actual_sha256,
                        actual_size,
                        actual_sha256,
                        artifact_id,
                        job_id,
                    ),
                )
            ).fetchone()
        if row is None:
            raise KeyError(str(artifact_id))
        if row["status"] != "available":
            raise StoreConflictError("uploaded artifact metadata did not match reservation")
        return ArtifactCompleteResponse(artifact_id=artifact_id, verified_at=verified_at)

    async def create_migration_record(
        self,
        request: MigrationCreateRequest,
        *,
        migration_id: uuid.UUID,
        bucket: str,
        object_key: str,
    ) -> dict[str, Any]:
        """Create or reuse an idempotent migration upload record."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO runtime_migrations (
                    migration_id, worker_id, idempotency_key, source_kind,
                    source_schema_version, source_sha256, size_bytes,
                    content_type, bucket, object_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    migration_id,
                    request.worker_id,
                    request.idempotency_key,
                    request.source.kind.value,
                    request.source.schema_version,
                    request.source.sha256,
                    request.source.size_bytes,
                    request.content_type,
                    bucket,
                    object_key,
                ),
            )
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_migrations
                    WHERE idempotency_key = %s
                    """,
                    (request.idempotency_key,),
                )
            ).fetchone()
        if row is None:
            raise RuntimeStoreError("migration insert did not return an id")
        if (
            row["source_kind"] != request.source.kind.value
            or row["source_sha256"] != request.source.sha256
            or row["size_bytes"] != request.source.size_bytes
        ):
            raise StoreConflictError("migration idempotency key has different source metadata")
        if row["status"] != MigrationStatus.AWAITING_UPLOAD.value:
            raise StoreConflictError(
                "migration source was already accepted",
                details={"migrationId": str(row["migration_id"]), "status": row["status"]},
            )
        return dict(row)

    async def complete_migration_upload(
        self,
        migration_id: uuid.UUID,
        worker_id: str,
        sha256: str,
        size_bytes: int,
    ) -> MigrationRecord:
        """Queue a verified legacy upload exactly once."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    UPDATE runtime_migrations
                    SET status = CASE
                            WHEN source_sha256 = %s AND size_bytes = %s THEN 'queued'
                            ELSE 'failed'
                        END,
                        error = CASE
                            WHEN source_sha256 = %s AND size_bytes = %s THEN NULL
                            ELSE '{"code":"source_mismatch","message":"Uploaded source metadata did not match","retryable":false}'::jsonb
                        END,
                        queued_at = CASE
                            WHEN source_sha256 = %s AND size_bytes = %s THEN NOW()
                            ELSE NULL
                        END
                    WHERE migration_id = %s AND worker_id = %s
                        AND status IN ('awaiting-upload', 'queued')
                    RETURNING *
                    """,
                    (
                        sha256,
                        size_bytes,
                        sha256,
                        size_bytes,
                        sha256,
                        size_bytes,
                        migration_id,
                        worker_id,
                    ),
                )
            ).fetchone()
        if row is None:
            raise KeyError(str(migration_id))
        return _migration_record(row)

    async def get_migration_for_verification(
        self,
        migration_id: uuid.UUID,
        worker_id: str,
    ) -> dict[str, Any]:
        """Return private S3 metadata for a worker-owned pending migration."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_migrations
                    WHERE migration_id = %s AND worker_id = %s
                        AND status IN ('awaiting-upload', 'queued')
                    """,
                    (migration_id, worker_id),
                )
            ).fetchone()
        if row is None:
            raise KeyError(str(migration_id))
        return dict(row)

    async def claim_migration(
        self,
        migration_id: uuid.UUID,
        *,
        lease_seconds: int = 900,
    ) -> dict[str, Any]:
        """Claim one queued migration with an expiring processing token."""
        if lease_seconds < 60 or lease_seconds > 3600:
            raise ValueError("lease_seconds must be between 60 and 3600")
        pool = self._require_pool()
        token = uuid.uuid4()
        expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    UPDATE runtime_migrations
                    SET status = 'running', processing_token = %s,
                        processing_expires_at = %s
                    WHERE migration_id = %s
                        AND (
                            status = 'queued'
                            OR (status = 'running' AND processing_expires_at < NOW())
                        )
                    RETURNING *
                    """,
                    (token, expires_at, migration_id),
                )
            ).fetchone()
        if row is None:
            raise StoreConflictError("migration is not queued or its processing lease is active")
        result = dict(row)
        result["processing_token"] = token
        return result

    async def complete_legacy_import(
        self,
        migration_id: uuid.UUID,
        processing_token: uuid.UUID,
        *,
        mission_states: tuple[dict[str, Any], ...] = (),
        experiments: tuple[dict[str, Any], ...] = (),
        hints: tuple[dict[str, Any], ...] = (),
        submissions: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, int]:
        """Import validated legacy records and source ledger in one transaction."""
        pool = self._require_pool()
        async with pool.connection() as connection, connection.transaction():
            migration = await (
                await connection.execute(
                    """
                    SELECT * FROM runtime_migrations
                    WHERE migration_id = %s AND status = 'running'
                        AND processing_token = %s AND processing_expires_at > NOW()
                    FOR UPDATE
                    """,
                    (migration_id, processing_token),
                )
            ).fetchone()
            if migration is None:
                raise StoreConflictError("migration processing lease is stale")
            ledger = await (
                await connection.execute(
                    """
                    SELECT counts FROM legacy_source_ledger
                    WHERE source_kind = %s AND source_sha256 = %s
                    """,
                    (migration["source_kind"], migration["source_sha256"]),
                )
            ).fetchone()
            if ledger is not None:
                counts = {str(key): int(value) for key, value in ledger["counts"].items()}
                await self._finish_migration(connection, migration_id, counts)
                return counts
            for state in mission_states:
                await self._import_legacy_mission(
                    connection,
                    migration["source_sha256"],
                    state,
                )
            await self._insert_legacy_rows(
                connection,
                "legacy_experiments",
                migration["source_sha256"],
                experiments,
            )
            await self._insert_legacy_rows(
                connection,
                "legacy_hint_attempts",
                migration["source_sha256"],
                hints,
            )
            await self._insert_legacy_rows(
                connection,
                "legacy_submissions",
                migration["source_sha256"],
                submissions,
                history_only=True,
            )
            if submissions:
                await self._refresh_legacy_submission_guards(connection)
            counts = {
                "missionStates": len(mission_states),
                "experiments": len(experiments),
                "hintAttempts": len(hints),
                "submissions": len(submissions),
            }
            await connection.execute(
                """
                INSERT INTO legacy_source_ledger (
                    source_kind, source_sha256, source_schema_version,
                    migration_id, counts
                ) VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                (
                    migration["source_kind"],
                    migration["source_sha256"],
                    migration["source_schema_version"],
                    migration_id,
                    _json(counts),
                ),
            )
            await self._finish_migration(connection, migration_id, counts)
        return counts

    async def fail_migration_import(
        self,
        migration_id: uuid.UUID,
        processing_token: uuid.UUID,
        error: JobFailure,
    ) -> None:
        """Persist a safe terminal error under the active processing lease."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    """
                    UPDATE runtime_migrations
                    SET status = 'failed', error = %s::jsonb,
                        processing_token = NULL, processing_expires_at = NULL,
                        completed_at = NOW()
                    WHERE migration_id = %s AND processing_token = %s
                    RETURNING migration_id
                    """,
                    (error.model_dump_json(by_alias=True), migration_id, processing_token),
                )
            ).fetchone()
        if row is None:
            raise StoreConflictError("migration processing lease is stale")

    async def get_migration(self, migration_id: uuid.UUID) -> MigrationRecord | None:
        """Return migration status without exposing its object-store location."""
        pool = self._require_pool()
        async with pool.connection() as connection:
            row = await (
                await connection.execute(
                    "SELECT * FROM runtime_migrations WHERE migration_id = %s",
                    (migration_id,),
                )
            ).fetchone()
        return _migration_record(row) if row is not None else None

    async def _import_legacy_mission(
        self,
        connection: Any,
        source_sha256: str,
        record: dict[str, Any],
    ) -> None:
        mission_id = uuid.UUID(record["legacyMissionId"])
        owner_id = record["ownerId"]
        snapshot_payload = record["snapshotPayload"]
        competition_id = snapshot_payload["migration"].get("competitionId")
        submission_rows = []
        if competition_id:
            submission_rows = await (
                await connection.execute(
                    """
                    SELECT payload FROM legacy_submissions
                    WHERE payload->>'competition_id' = %s
                    ORDER BY imported_at, record_id
                    """,
                    (competition_id,),
                )
            ).fetchall()
        snapshot_payload = _with_submission_guard(
            snapshot_payload,
            tuple(row["payload"] for row in submission_rows),
        )
        job_id = uuid.uuid5(mission_id, f"agent-k-legacy-job:{source_sha256}")
        snapshot_id = uuid.uuid5(mission_id, f"agent-k-legacy-snapshot:{source_sha256}")
        existing = await (
            await connection.execute(
                "SELECT owner_id, state FROM runtime_missions WHERE mission_id = %s",
                (mission_id,),
            )
        ).fetchone()
        if existing is not None and (
            existing["owner_id"] != owner_id
            or existing["state"].get("migration", {}).get("sourceSha256") != source_sha256
        ):
            raise StoreConflictError(f"legacy missionId collides with runtime mission {mission_id}")
        await connection.execute(
            """
            INSERT INTO legacy_mission_states (
                source_sha256, legacy_mission_id, owner_id, snapshot_payload
            ) VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (source_sha256, legacy_mission_id) DO NOTHING
            """,
            (source_sha256, mission_id, owner_id, _json(snapshot_payload)),
        )
        await connection.execute(
            """
            INSERT INTO runtime_missions (
                mission_id, owner_id, schema_version, status, state, state_sha256,
                requested_control, control_reason, control_requested_at
            ) VALUES (%s, %s, '1.0.0', 'paused', %s::jsonb, %s,
                'pause', 'legacy_import_requires_explicit_resume', NOW())
            ON CONFLICT (mission_id) DO NOTHING
            """,
            (
                mission_id,
                owner_id,
                _json(snapshot_payload),
                _canonical_sha256(snapshot_payload),
            ),
        )
        migration = snapshot_payload["migration"]
        command = {
            "type": "resume",
            "competitionId": migration.get("competitionId"),
            "criteria": migration.get("criteria", {}),
            "checkpointId": str(snapshot_id),
        }
        workspace = {"artifactPrefix": f"missions/{mission_id}"}
        await connection.execute(
            """
            INSERT INTO runtime_jobs (
                job_id, mission_id, status, command, workspace,
                idempotency_key, max_attempts
            ) VALUES (%s, %s, 'queued', %s::jsonb, %s::jsonb, %s, 3)
            ON CONFLICT (idempotency_key) DO NOTHING
            """,
            (
                job_id,
                mission_id,
                _json(command),
                _json(workspace),
                f"legacy:{source_sha256}:{mission_id}",
            ),
        )
        await connection.execute(
            """
            INSERT INTO runtime_snapshots (
                snapshot_id, mission_id, job_id, sequence, schema_version,
                content_type, payload, payload_sha256, created_at
            ) VALUES (%s, %s, %s, 0, '1.0.0', 'application/json',
                %s::jsonb, %s, NOW())
            ON CONFLICT (snapshot_id) DO NOTHING
            """,
            (
                snapshot_id,
                mission_id,
                job_id,
                _json(snapshot_payload),
                _canonical_sha256(snapshot_payload),
            ),
        )

    async def _insert_legacy_rows(
        self,
        connection: Any,
        table: str,
        source_sha256: str,
        records: tuple[dict[str, Any], ...],
        *,
        history_only: bool = False,
    ) -> None:
        if table not in {"legacy_experiments", "legacy_hint_attempts", "legacy_submissions"}:
            raise ValueError("unsupported legacy target table")
        for record in records:
            values: tuple[Any, ...]
            columns = "source_sha256, record_id, payload"
            placeholders = "%s, %s, %s::jsonb"
            values = (source_sha256, record["recordId"], _json(record["payload"]))
            if history_only:
                columns += ", history_only"
                placeholders += ", TRUE"
            await connection.execute(
                f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
                "ON CONFLICT (source_sha256, record_id) DO NOTHING",
                values,
            )

    async def _refresh_legacy_submission_guards(self, connection: Any) -> None:
        snapshots = await (
            await connection.execute(
                """
                SELECT snapshot_id, mission_id, payload
                FROM runtime_snapshots
                WHERE payload->'migration'->>'kind' = 'legacy-v1'
                FOR UPDATE
                """
            )
        ).fetchall()
        for snapshot in snapshots:
            competition_id = snapshot["payload"]["migration"].get("competitionId")
            rows = []
            if competition_id:
                rows = await (
                    await connection.execute(
                        """
                        SELECT payload FROM legacy_submissions
                        WHERE payload->>'competition_id' = %s
                        ORDER BY imported_at, record_id
                        """,
                        (competition_id,),
                    )
                ).fetchall()
            payload = _with_submission_guard(
                snapshot["payload"],
                tuple(row["payload"] for row in rows),
            )
            digest = _canonical_sha256(payload)
            await connection.execute(
                """
                UPDATE runtime_snapshots SET payload = %s::jsonb, payload_sha256 = %s
                WHERE snapshot_id = %s
                """,
                (_json(payload), digest, snapshot["snapshot_id"]),
            )
            await connection.execute(
                """
                UPDATE runtime_missions SET state = %s::jsonb, state_sha256 = %s,
                    updated_at = NOW()
                WHERE mission_id = %s AND status IN ('migrated', 'paused')
                """,
                (_json(payload), digest, snapshot["mission_id"]),
            )

    async def _finish_migration(
        self,
        connection: Any,
        migration_id: uuid.UUID,
        counts: dict[str, int],
    ) -> None:
        await connection.execute(
            """
            UPDATE runtime_migrations
            SET status = 'completed', counts = %s::jsonb, error = NULL,
                processing_token = NULL, processing_expires_at = NULL,
                completed_at = NOW()
            WHERE migration_id = %s
            """,
            (_json(counts), migration_id),
        )

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeStoreError("PostgresRuntimeStore.open() must be called first")
        return self._pool

    async def _fetch_mission(self, connection: Any, mission_id: uuid.UUID) -> dict[str, Any]:
        row = await (
            await connection.execute(
                "SELECT * FROM runtime_missions WHERE mission_id = %s",
                (mission_id,),
            )
        ).fetchone()
        if row is None:
            raise RuntimeStoreError("mission insert did not return a row")
        return dict(row)

    async def _decorate_mission_row(self, connection: Any, row: dict[str, Any]) -> None:
        job = await (
            await connection.execute(
                """
                SELECT progress, current_phase FROM runtime_jobs
                WHERE mission_id = %s ORDER BY created_at DESC LIMIT 1
                """,
                (row["mission_id"],),
            )
        ).fetchone()
        event = await (
            await connection.execute(
                """
                SELECT COALESCE(MAX(stream_sequence), 0) AS latest_sequence
                FROM runtime_events WHERE mission_id = %s
                """,
                (row["mission_id"],),
            )
        ).fetchone()
        state = dict(row["state"])
        state["currentPhase"] = (
            job["current_phase"]
            if job is not None and job["current_phase"] is not None
            else state.get("current_phase", state.get("currentPhase"))
        )
        state["progress"] = (
            job["progress"]
            if job is not None and job["progress"] is not None
            else state.get("overall_progress", state.get("progress", 0.0))
        )
        state["latestSequence"] = int(event["latest_sequence"] if event is not None else 0)
        state["competitionId"] = state.get("competition_id", state.get("competitionId"))
        state["evolutionIterations"] = state.get(
            "evolution_iterations",
            state.get("evolutionIterations"),
        )
        row["state"] = state

    async def _lock_lease(
        self,
        connection: Any,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
    ) -> dict[str, Any]:
        row = await (
            await connection.execute(
                "SELECT * FROM runtime_jobs WHERE job_id = %s FOR UPDATE",
                (job_id,),
            )
        ).fetchone()
        if row is None:
            raise KeyError(str(job_id))
        valid = (
            row["worker_id"] == worker_id
            and row["lease_id"] == lease_id
            and row["status"] in {"leased", "running"}
            and row["lease_expires_at"] is not None
            and row["lease_expires_at"] > datetime.now(UTC)
        )
        if not valid:
            raise LeaseFenceError("lease is stale, expired, or owned by another worker")
        return dict(row)

    async def _control_for_mission(
        self,
        connection: Any,
        mission_id: uuid.UUID,
    ) -> ControlDirective:
        row = await (
            await connection.execute(
                """
                SELECT requested_control, control_reason, control_requested_at
                FROM runtime_missions WHERE mission_id = %s
                """,
                (mission_id,),
            )
        ).fetchone()
        if row is None:
            raise KeyError(str(mission_id))
        return _control(row)

    async def _require_terminal_event(
        self,
        connection: Any,
        job_id: uuid.UUID,
        final_sequence: int,
        *,
        success: bool,
    ) -> None:
        row = await (
            await connection.execute(
                """
                SELECT event_type, data,
                    (SELECT COUNT(*) FROM runtime_events WHERE job_id = %s) AS event_count
                FROM runtime_events
                WHERE job_id = %s AND job_sequence = %s
                """,
                (job_id, job_id, final_sequence),
            )
        ).fetchone()
        valid = (
            row is not None
            and row["event_type"] == "mission-complete"
            and row["event_count"] == final_sequence
            and isinstance(row["data"], dict)
            and row["data"].get("success") is success
        )
        if not valid:
            raise StoreConflictError(
                "finalEventSequence must identify the contiguous mission-complete event",
                details={"finalEventSequence": final_sequence},
            )

    async def _require_event_cursor(
        self,
        connection: Any,
        job_id: uuid.UUID,
        final_sequence: int,
    ) -> None:
        row = await (
            await connection.execute(
                """
                SELECT COUNT(*) AS event_count,
                    COUNT(*) FILTER (WHERE job_sequence = %s) AS cursor_count
                FROM runtime_events WHERE job_id = %s
                """,
                (final_sequence, job_id),
            )
        ).fetchone()
        if row is None or row["event_count"] != final_sequence or row["cursor_count"] != 1:
            raise StoreConflictError(
                "finalEventSequence must identify the contiguous durable event cursor",
                details={"finalEventSequence": final_sequence},
            )

    async def _ensure_failure_terminal_event(
        self,
        connection: Any,
        job: dict[str, Any],
        final_sequence: int,
        error: JobFailure,
    ) -> None:
        row = await (
            await connection.execute(
                """
                SELECT event_type, data FROM runtime_events
                WHERE job_id = %s AND job_sequence = %s
                """,
                (job["job_id"], final_sequence),
            )
        ).fetchone()
        if row is not None and row["event_type"] == "mission-complete":
            if not isinstance(row["data"], dict) or row["data"].get("success") is not False:
                raise StoreConflictError("failure terminal event must have success false")
            return
        sequence = final_sequence + 1
        event_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"agent-k:{job['job_id']}:{sequence}",
        )
        await connection.execute(
            """
            INSERT INTO runtime_events (
                event_id, mission_id, job_id, job_sequence,
                event_type, event_timestamp, data
            ) VALUES (%s, %s, %s, %s, 'mission-complete', NOW(), %s::jsonb)
            """,
            (
                event_id,
                job["mission_id"],
                job["job_id"],
                sequence,
                _json(
                    {
                        "success": False,
                        "finalRank": None,
                        "finalScore": None,
                        "errorMessage": error.message,
                        "totalSubmissions": 0,
                        "evolutionGenerations": 0,
                        "evolutionIterations": 0,
                        "evolutionFailureSummary": {},
                        "durationMs": 0,
                        "phasesCompleted": [],
                    }
                ),
            ),
        )

    async def _require_available_artifacts(
        self,
        connection: Any,
        job_id: uuid.UUID,
        references: tuple[ArtifactReference, ...],
    ) -> None:
        if not references:
            return
        expected = {reference.artifact_id: reference.role for reference in references}
        if len(expected) != len(references):
            raise StoreConflictError("completion artifacts contain duplicate artifactIds")
        rows = await (
            await connection.execute(
                """
                SELECT artifact_id, role, status FROM runtime_artifacts
                WHERE job_id = %s AND artifact_id = ANY(%s::uuid[])
                """,
                (job_id, list(expected)),
            )
        ).fetchall()
        actual = {row["artifact_id"]: row["role"] for row in rows if row["status"] == "available"}
        if actual != expected:
            raise StoreConflictError("all completion artifacts must be verified and available")

    async def _idempotent_response(
        self,
        connection: Any,
        job_id: uuid.UUID,
        operation: str,
        key: str,
        model_type: type[WireModel],
    ) -> Any | None:
        row = await (
            await connection.execute(
                """
                SELECT response FROM runtime_job_idempotency
                WHERE job_id = %s AND operation = %s AND idempotency_key = %s
                """,
                (job_id, operation, key),
            )
        ).fetchone()
        return model_type.model_validate(row["response"]) if row is not None else None

    async def _save_idempotent_response(
        self,
        connection: Any,
        job_id: uuid.UUID,
        operation: str,
        key: str,
        response: WireModel,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO runtime_job_idempotency (
                job_id, operation, idempotency_key, response
            ) VALUES (%s, %s, %s, %s::jsonb)
            """,
            (job_id, operation, key, response.model_dump_json(by_alias=True)),
        )


class S3ArtifactStore:
    """Server-side S3 signer that never exposes storage credentials to workers."""

    def __init__(
        self,
        runtime_store: PostgresRuntimeStore,
        *,
        bucket: str,
        region: str | None = None,
        endpoint_url: str | None = None,
        url_ttl_seconds: int = 3600,
    ) -> None:
        if not bucket:
            raise ValueError("bucket is required")
        if url_ttl_seconds < 60 or url_ttl_seconds > 3600:
            raise ValueError("url_ttl_seconds must be between 60 and 3600")
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError("S3 support requires `boto3`.") from error
        self._runtime_store = runtime_store
        self._bucket = bucket
        self._url_ttl_seconds = url_ttl_seconds
        self._client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4"),
        )

    async def sign_upload(
        self,
        job_id: uuid.UUID,
        spec: ArtifactUploadSpec,
    ) -> SignedArtifactURL:
        """Reserve metadata and sign a checksum-bound S3 PUT request."""
        artifact = await self._runtime_store.reserve_artifact(
            job_id,
            spec,
            bucket=self._bucket,
        )
        checksum = base64.b64encode(bytes.fromhex(spec.sha256)).decode()
        params = {
            "Bucket": self._bucket,
            "Key": artifact["object_key"],
            "ContentType": spec.content_type,
            "ChecksumSHA256": checksum,
        }
        url = await asyncio.to_thread(
            self._client.generate_presigned_url,
            "put_object",
            Params=params,
            ExpiresIn=self._url_ttl_seconds,
            HttpMethod="PUT",
        )
        return SignedArtifactURL(
            artifact_id=artifact["artifact_id"],
            method="PUT",
            url=url,
            headers={
                "content-type": spec.content_type,
                "x-amz-checksum-sha256": checksum,
            },
            expires_at=datetime.now(UTC) + timedelta(seconds=self._url_ttl_seconds),
        )

    async def complete_upload(
        self,
        job_id: uuid.UUID,
        artifact_id: uuid.UUID,
        spec: ArtifactCompleteSpec,
    ) -> ArtifactCompleteResponse:
        """Verify one S3 object server-side before making it available."""
        artifact = await self._runtime_store.get_artifact_for_verification(
            job_id,
            spec.worker_id,
            spec.lease_id,
            artifact_id,
        )
        if artifact["sha256"] != spec.sha256 or artifact["size_bytes"] != spec.size_bytes:
            raise StoreConflictError("artifact completion metadata differs from reservation")
        response = await asyncio.to_thread(
            self._client.head_object,
            Bucket=artifact["bucket"],
            Key=artifact["object_key"],
            ChecksumMode="ENABLED",
        )
        actual_checksum = response.get("ChecksumSHA256")
        expected_checksum = base64.b64encode(bytes.fromhex(spec.sha256)).decode()
        actual_size = int(response["ContentLength"])
        if actual_size != spec.size_bytes or actual_checksum != expected_checksum:
            raise StoreConflictError("S3 object does not match reserved size and checksum")
        return await self._runtime_store.mark_artifact_available(
            job_id,
            spec.worker_id,
            spec.lease_id,
            artifact_id,
            actual_size=actual_size,
            actual_sha256=spec.sha256,
        )

    async def sign_download(
        self,
        job_id: uuid.UUID,
        worker_id: str,
        lease_id: uuid.UUID,
        artifact_id: uuid.UUID,
    ) -> SignedArtifactURL:
        """Sign a GET request only for an artifact in the leased mission."""
        artifact = await self._runtime_store.get_artifact_for_download(
            job_id,
            worker_id,
            lease_id,
            artifact_id,
        )
        url = await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": artifact["bucket"], "Key": artifact["object_key"]},
            ExpiresIn=self._url_ttl_seconds,
            HttpMethod="GET",
        )
        return SignedArtifactURL(
            artifact_id=artifact_id,
            method="GET",
            url=url,
            headers={},
            expires_at=datetime.now(UTC) + timedelta(seconds=self._url_ttl_seconds),
        )

    async def create_migration_upload(
        self,
        request: MigrationCreateRequest,
    ) -> MigrationCreateResponse:
        """Create an idempotent, content-addressed legacy upload."""
        migration_id = uuid.uuid4()
        suffix = "json" if request.source.kind is MigrationSourceKind.MISSION_STATE else "sqlite3"
        object_key = (
            f"migrations/{request.source.kind.value}/{request.source.sha256}/"
            f"{migration_id}.{suffix}"
        )
        migration = await self._runtime_store.create_migration_record(
            request,
            migration_id=migration_id,
            bucket=self._bucket,
            object_key=object_key,
        )
        checksum = base64.b64encode(bytes.fromhex(request.source.sha256)).decode()
        url = await asyncio.to_thread(
            self._client.generate_presigned_url,
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": migration["object_key"],
                "ContentType": request.content_type,
                "ChecksumSHA256": checksum,
            },
            ExpiresIn=self._url_ttl_seconds,
            HttpMethod="PUT",
        )
        upload = SignedArtifactURL(
            artifact_id=migration["migration_id"],
            method="PUT",
            url=url,
            headers={
                "content-type": request.content_type,
                "x-amz-checksum-sha256": checksum,
            },
            expires_at=datetime.now(UTC) + timedelta(seconds=self._url_ttl_seconds),
        )
        return MigrationCreateResponse(migration_id=migration["migration_id"], upload=upload)

    async def complete_migration_upload(
        self,
        migration_id: uuid.UUID,
        worker_id: str,
        sha256: str,
        size_bytes: int,
    ) -> MigrationRecord:
        """Verify a legacy source in S3 before queueing its import."""
        migration = await self._runtime_store.get_migration_for_verification(
            migration_id,
            worker_id,
        )
        if migration["source_sha256"] != sha256 or migration["size_bytes"] != size_bytes:
            raise StoreConflictError("migration completion metadata differs from reservation")
        response = await asyncio.to_thread(
            self._client.head_object,
            Bucket=migration["bucket"],
            Key=migration["object_key"],
            ChecksumMode="ENABLED",
        )
        expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode()
        if (
            int(response["ContentLength"]) != size_bytes
            or response.get("ChecksumSHA256") != expected_checksum
        ):
            raise StoreConflictError("S3 migration object failed size or checksum validation")
        return await self._runtime_store.complete_migration_upload(
            migration_id,
            worker_id,
            sha256,
            size_bytes,
        )

    async def download_claimed_migration(
        self,
        migration: dict[str, Any],
        destination: Path,
    ) -> None:
        """Download and verify a migration held by a database processing lease."""
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            await asyncio.to_thread(
                self._client.download_file,
                migration["bucket"],
                migration["object_key"],
                str(temporary),
            )
            actual_size = temporary.stat().st_size
            actual_sha256 = await asyncio.to_thread(_file_sha256, temporary)
            if (
                actual_size != migration["size_bytes"]
                or actual_sha256 != migration["source_sha256"]
            ):
                raise StoreConflictError("downloaded migration source failed integrity validation")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _with_submission_guard(
    payload: dict[str, Any],
    submissions: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    result = json.loads(_json(payload))
    if not isinstance(result, dict):
        raise ValueError("migration snapshot payload must be an object")
    migration = result["migration"]
    legacy_state = migration.get("legacyState", {})
    identifiers = {
        str(value)
        for value in [
            legacy_state.get("final_submission_id"),
            *(submission.get("submission_id") for submission in submissions),
        ]
        if value
    }
    claimed_submission = "submission" in migration.get("phaseClaims", [])
    ambiguous = (claimed_submission or bool(submissions)) and not identifiers
    migration["submissionGuard"] = {
        "historyOnly": True,
        "ambiguous": ambiguous,
        "submissionIds": sorted(identifiers),
        "status": "ambiguous" if ambiguous else "pending-receipt" if identifiers else "none",
    }
    return {str(key): value for key, value in result.items()}


def _mission_record(row: dict[str, Any]) -> MissionRecord:
    return MissionRecord(
        mission_id=row["mission_id"],
        owner_id=row["owner_id"],
        schema_version=row["schema_version"],
        status=row["status"],
        state=row["state"],
        requested_control=row["requested_control"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _job_lease(row: dict[str, Any]) -> JobLease:
    return JobLease(
        job_id=row["job_id"],
        mission_id=row["mission_id"],
        attempt=row["attempt"],
        lease_id=row["lease_id"],
        lease_expires_at=row["lease_expires_at"],
        command=JobCommand.model_validate(row["command"]),
        workspace=JobWorkspace.model_validate(row["workspace"]),
        event_sequence=row.get("event_sequence", 0),
    )


def _control(row: dict[str, Any]) -> ControlDirective:
    return ControlDirective(
        action=row["requested_control"],
        reason=row["control_reason"],
        requested_at=row["control_requested_at"],
        server_time=datetime.now(UTC),
    )


def _migration_record(row: dict[str, Any]) -> MigrationRecord:
    error = JobFailure.model_validate(row["error"]) if row.get("error") else None
    return MigrationRecord(
        migration_id=row["migration_id"],
        status=row["status"],
        counts=row.get("counts"),
        error=error,
    )
