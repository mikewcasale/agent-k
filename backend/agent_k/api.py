"""Expose authenticated mission, worker, artifact, and migration APIs.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import binascii
import hmac
import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal, Protocol, TypeAliasType

import logfire
from fastapi import Depends, FastAPI, Header, Query, Request, Response, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from .events import AgentKEvent, ModelFamily, ModelSize, WorkerEvent, parse_last_event_id
from .exceptions import AuthenticationError, CompetitionNotFoundError, RateLimitError
from .intent import MissionCriteriaInput, parse_mission_intent
from .version import __version__

__all__ = (
    "ApiError",
    "ApiRuntime",
    "CompetitionGateway",
    "GatewayVerifier",
    "RuntimeStore",
    "SignedArtifactStore",
    "WorkerVerifier",
    "app",
    "create_app",
    "main",
)

DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 9000
DEFAULT_EVENT_BATCH_SIZE: Final[int] = 200
SSE_HEARTBEAT_SECONDS: Final[float] = 15.0
SSE_POLL_SECONDS: Final[float] = 1.0
TERMINAL_MISSION_STATUSES: Final[frozenset[str]] = frozenset(
    {"aborted", "canceled", "cancelled", "completed", "failed"}
)

JsonObject = TypeAliasType("JsonObject", dict[str, Any])
ControlAction = Literal["continue", "pause", "cancel"]
MissionControlAction = Literal["resume", "pause", "cancel"]
RuntimeFactory = Callable[[], AbstractAsyncContextManager["ApiRuntime"]]


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class _WireModel(BaseModel):
    """Base model for strict camelCase API payloads."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        extra="forbid",
        populate_by_name=True,
    )


class ErrorDetail(_WireModel):
    """Machine-readable API failure."""

    code: str
    message: str
    retryable: bool = False
    category: str | None = None
    details: JsonObject | None = None


class ErrorResponse(_WireModel):
    """Stable error envelope returned by every JSON endpoint."""

    error: ErrorDetail
    request_id: str


class ApiError(Exception):
    """HTTP-safe application error with a stable wire code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: JsonObject | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details
        super().__init__(message)


class UserIdentity(_WireModel):
    """Verified identity forwarded by the Auth.js frontend."""

    user_id: str = Field(min_length=1, max_length=200)
    token_id: str = Field(min_length=1, max_length=200)


class WorkerIdentity(_WireModel):
    """Verified internal worker identity."""

    worker_id: str = Field(min_length=1, max_length=200)


class GatewayClaims(_WireModel):
    """Signed short-lived frontend identity claims."""

    sub: str = Field(min_length=1, max_length=200)
    iss: str
    aud: str
    iat: int
    exp: int
    jti: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True, slots=True)
class GatewayVerifier:
    """Verify HMAC-signed frontend identity tokens with key rotation."""

    secrets: tuple[SecretStr, ...]
    issuer: str
    audience: str
    max_ttl_seconds: int = 60
    clock_skew_seconds: int = 10

    def verify(self, token: str) -> UserIdentity:
        """Verify one compact HS256 token and return its user identity."""
        try:
            encoded_header, encoded_claims, encoded_signature = token.split(".")
        except ValueError as exc:
            raise ApiError(401, "invalid_identity", "Invalid identity token") from exc

        header = _decode_json_segment(encoded_header)
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise ApiError(401, "invalid_identity", "Invalid identity token")

        signing_input = f"{encoded_header}.{encoded_claims}".encode()
        signature = _decode_segment(encoded_signature)
        valid_signature = any(
            hmac.compare_digest(
                signature,
                hmac.digest(secret.get_secret_value().encode(), signing_input, "sha256"),
            )
            for secret in self.secrets
        )
        if not valid_signature:
            raise ApiError(401, "invalid_identity", "Invalid identity token")

        try:
            claims = GatewayClaims.model_validate(_decode_json_segment(encoded_claims))
        except ValueError as exc:
            raise ApiError(401, "invalid_identity", "Invalid identity token") from exc

        now = int(time.time())
        if claims.iss != self.issuer or claims.aud != self.audience:
            raise ApiError(401, "invalid_identity", "Invalid identity token")
        if claims.iat > now + self.clock_skew_seconds:
            raise ApiError(401, "invalid_identity", "Identity token is not active")
        if claims.exp <= now - self.clock_skew_seconds:
            raise ApiError(401, "expired_identity", "Identity token has expired")
        if claims.exp - claims.iat > self.max_ttl_seconds:
            raise ApiError(401, "invalid_identity", "Identity token lifetime is too long")
        return UserIdentity(user_id=claims.sub, token_id=claims.jti)


@dataclass(frozen=True, slots=True)
class WorkerVerifier:
    """Verify rotatable opaque internal worker bearer tokens."""

    tokens: tuple[SecretStr, ...]

    def verify(self, token: str, worker_id: str) -> WorkerIdentity:
        """Return the worker identity when a configured token matches."""
        presented = token.encode()
        if not any(
            hmac.compare_digest(presented, candidate.get_secret_value().encode())
            for candidate in self.tokens
        ):
            raise ApiError(401, "invalid_worker_token", "Invalid worker token")
        return WorkerIdentity(worker_id=worker_id)


class MissionStartRequest(_WireModel):
    """Create a durable mission job."""

    schema_version: str = "1.0.0"
    criteria: MissionCriteriaInput
    evolution_models: list[str] | None = None
    user_prompt: str | None = Field(default=None, max_length=20_000)
    competition_id: str | None = Field(default=None, min_length=1, max_length=200)
    competition_url: str | None = Field(default=None, min_length=10, max_length=2_000)


class MissionStartResponse(_WireModel):
    """Identity and initial status of a newly queued mission."""

    mission_id: str
    status: str = "queued"


class MissionStatusResponse(_WireModel):
    """Current durable mission status."""

    mission_id: str
    status: str
    current_phase: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    competition_id: str | None = None
    error_message: str | None = None
    latest_sequence: int = Field(default=0, ge=0)
    iteration: int | None = Field(default=None, ge=0)
    evolution_iterations: int | None = Field(default=None, ge=0)
    model_family: ModelFamily | None = None
    model_size: ModelSize | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MissionControlResponse(_WireModel):
    """Accepted user control request."""

    mission_id: str
    status: str
    action: MissionControlAction


class CompetitionSearchRequest(_WireModel):
    """Search filters for the real competition platform."""

    paid_only: bool = False
    domains: list[str] = Field(default_factory=list)
    competition_types: list[str] = Field(default_factory=list)
    min_prize: int | None = Field(default=None, ge=0)
    min_days_remaining: int = Field(default=7, ge=1)


class CompetitionFetchRequest(_WireModel):
    """Fetch one competition by its platform URL."""

    url: str = Field(min_length=10, max_length=2_000)


class ControlState(_WireModel):
    """Control action visible to a leased worker."""

    action: ControlAction = "continue"
    reason: str | None = None
    requested_at: datetime | None = None


class JobCommand(_WireModel):
    """Mission command assigned to a worker."""

    type: Literal["start", "resume"]
    competition_id: str | None = None
    criteria: JsonObject
    checkpoint_id: str | None = None


class JobWorkspace(_WireModel):
    """Artifact namespace reserved for one job attempt."""

    artifact_prefix: str


class ClaimedJob(_WireModel):
    """Fenced lease and mission command returned by a claim."""

    schema_version: str = "1.0.0"
    job_id: str
    mission_id: str
    attempt: int = Field(ge=1)
    lease_id: str
    lease_expires_at: datetime
    command: JobCommand
    workspace: JobWorkspace
    event_sequence: int = Field(default=0, ge=0)


class ClaimJobsRequest(_WireModel):
    """Worker request for available jobs."""

    worker_id: str = Field(min_length=1, max_length=200)
    capabilities: list[str] = Field(default_factory=list)
    lease_seconds: int = Field(default=60, ge=15, le=3_600)
    max_jobs: int = Field(default=1, ge=1, le=16)


class ClaimJobsResponse(_WireModel):
    """Atomically claimed jobs and server clock."""

    jobs: list[ClaimedJob] = Field(default_factory=list)
    server_time: datetime


class HeartbeatRequest(_WireModel):
    """Renew a fenced job lease and report progress."""

    worker_id: str
    lease_id: str
    progress: float | None = Field(default=None, ge=0.0, le=100.0)
    current_phase: str | None = None
    current_task_id: str | None = None


class HeartbeatResponse(_WireModel):
    """Renewed lease and current control action."""

    job_id: str
    lease_id: str
    lease_expires_at: datetime
    control: ControlState
    server_time: datetime


class AppendEventsRequest(_WireModel):
    """Append one contiguous ordered batch of mission events."""

    worker_id: str
    lease_id: str
    events: list[WorkerEvent] = Field(min_length=1, max_length=256)


class AppendEventsResponse(_WireModel):
    """Event append acknowledgement."""

    accepted_through: int = Field(ge=0)
    duplicate_sequences: list[int] = Field(default_factory=list)
    control: ControlState


class SnapshotPayload(_WireModel):
    """Versioned mission snapshot supplied by a worker."""

    schema_version: str
    sequence: int = Field(ge=0)
    created_at: datetime
    content_type: Literal["application/json"] = "application/json"
    payload: JsonObject


class SaveSnapshotRequest(_WireModel):
    """Save an idempotent mission snapshot."""

    worker_id: str
    lease_id: str
    snapshot: SnapshotPayload


class SaveSnapshotResponse(_WireModel):
    """Stored snapshot identity."""

    snapshot_id: str
    sequence: int = Field(ge=0)
    created_at: datetime


class ArtifactReference(_WireModel):
    """Artifact attached to a terminal mission result."""

    artifact_id: str
    role: str


class MissionSnapshotResponse(_WireModel):
    """Durable resume snapshot and its externally stored artifacts."""

    snapshot_id: str
    job_id: str
    mission_id: str
    schema_version: str
    sequence: int = Field(ge=0)
    created_at: datetime
    content_type: Literal["application/json"] = "application/json"
    payload: JsonObject
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class MissionResultPayload(_WireModel):
    """Successful worker result for a mission."""

    schema_version: str
    success: Literal[True]
    mission_id: str
    competition_id: str | None = None
    final_rank: int | None = Field(default=None, ge=1)
    final_score: float | None = None
    total_submissions: int = Field(ge=0)
    evolution_generations: int = Field(ge=0)
    evolution_iterations: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    phases_completed: list[str]
    final_event_sequence: int = Field(default=0, ge=0)
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class CompleteJobRequest(_WireModel):
    """Commit a terminal successful job result."""

    worker_id: str
    lease_id: str
    idempotency_key: str = Field(min_length=16, max_length=200)
    result: MissionResultPayload
    final_event_sequence: int = Field(ge=0)
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class CompleteJobResponse(_WireModel):
    """Committed successful job status."""

    job_id: str
    mission_id: str
    status: Literal["completed"]
    completed_at: datetime


class WorkerFailure(_WireModel):
    """Sanitized worker failure safe for persistence and operators."""

    code: str
    message: str
    retryable: bool
    category: str | None = None
    details: JsonObject | None = None


class FailJobRequest(_WireModel):
    """Commit or retry a terminal worker failure."""

    worker_id: str
    lease_id: str
    idempotency_key: str = Field(min_length=16, max_length=200)
    error: WorkerFailure
    final_event_sequence: int = Field(ge=0)


class FailJobResponse(_WireModel):
    """Failure disposition selected by the runtime store."""

    job_id: str
    mission_id: str
    status: Literal["failed", "queued"]
    attempt: int = Field(ge=1)
    next_attempt_at: datetime | None = None
    failed_at: datetime | None = None


class WorkerControlResponse(ControlState):
    """Current worker control state and server clock."""

    server_time: datetime


class ArtifactUploadRequest(_WireModel):
    """Request a constrained signed artifact upload URL."""

    worker_id: str
    lease_id: str
    name: str = Field(min_length=1, max_length=500)
    content_type: str = Field(min_length=1, max_length=200)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    role: str = Field(min_length=1, max_length=100)


class ArtifactDownloadRequest(_WireModel):
    """Request a constrained signed artifact download URL."""

    worker_id: str
    lease_id: str


class CompleteArtifactRequest(_WireModel):
    """Finalize an uploaded artifact after server-side object verification."""

    worker_id: str
    lease_id: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class CompleteArtifactResponse(_WireModel):
    """Verified immutable artifact status."""

    artifact_id: str
    status: Literal["available"]
    verified_at: datetime


class SignedArtifactResponse(_WireModel):
    """Short-lived signed object storage operation."""

    artifact_id: str
    method: Literal["GET", "PUT"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    expires_at: datetime


class MigrationSource(_WireModel):
    """Source metadata for one legacy persistence upload."""

    kind: Literal["mission-state", "experiment-db"]
    schema_version: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class CreateMigrationRequest(_WireModel):
    """Create a signed upload for a legacy state migration."""

    worker_id: str
    idempotency_key: str = Field(min_length=16, max_length=200)
    source: MigrationSource
    content_type: Literal["application/json", "application/x-sqlite3"]


class SignedOperation(_WireModel):
    """Signed storage operation embedded in another response."""

    artifact_id: str
    method: Literal["PUT"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    expires_at: datetime


class CreateMigrationResponse(_WireModel):
    """Migration identity and its signed upload operation."""

    migration_id: str
    upload: SignedOperation
    status: Literal["awaiting-upload"]


class CompleteMigrationRequest(_WireModel):
    """Confirm a completed legacy state upload."""

    worker_id: str
    idempotency_key: str = Field(min_length=16, max_length=200)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)


class MigrationStatusResponse(_WireModel):
    """Current migration processing status."""

    migration_id: str
    status: Literal[
        "awaiting-upload",
        "queued",
        "running",
        "completed",
        "failed",
    ]
    counts: dict[str, int] | None = None
    error: ErrorDetail | None = None


class ChatRequest(_WireModel):
    """Vercel AI-style request forwarded by the authenticated frontend."""

    trigger: str | None = None
    id: str = Field(min_length=1, max_length=200)
    messages: list[JsonObject]


class RuntimeStore(Protocol):
    """Durable mission and worker operations required by the API."""

    async def create_mission(self, *, owner_id: str, request: JsonObject) -> JsonObject:
        """Create and enqueue one owned mission."""
        ...

    async def get_mission(self, mission_id: str) -> JsonObject | None:
        """Return one mission without applying an ownership check."""
        ...

    async def require_owner(self, mission_id: str, owner_id: str) -> JsonObject:
        """Return a mission only when the principal owns it."""
        ...

    async def request_control(
        self,
        mission_id: str,
        owner_id: str,
        action: MissionControlAction,
        reason: str | None,
    ) -> JsonObject:
        """Persist an owner-authorized mission control request."""
        ...

    async def list_events(
        self,
        mission_id: str,
        after_sequence: int,
        limit: int = DEFAULT_EVENT_BATCH_SIZE,
    ) -> list[JsonObject]:
        """Replay mission events after a durable global cursor."""
        ...

    async def list_best_results(self, owner_id: str, limit: int) -> list[JsonObject]:
        """Return ranked results restricted to one owner."""
        ...

    async def claim_jobs(self, request: JsonObject) -> JsonObject:
        """Atomically claim available jobs for a worker."""
        ...

    async def heartbeat(self, job_id: str, request: JsonObject) -> JsonObject:
        """Renew a fenced job lease."""
        ...

    async def append_events(self, job_id: str, request: JsonObject) -> JsonObject:
        """Append an ordered worker event batch."""
        ...

    async def save_snapshot(self, job_id: str, request: JsonObject) -> JsonObject:
        """Persist an idempotent mission snapshot."""
        ...

    async def get_snapshot(
        self,
        job_id: str,
        snapshot_id: str,
        worker_id: str,
        lease_id: str,
    ) -> JsonObject | None:
        """Load a snapshot through a current fenced lease."""
        ...

    async def complete_job(self, job_id: str, request: JsonObject) -> JsonObject:
        """Commit a successful fenced job."""
        ...

    async def fail_job(self, job_id: str, request: JsonObject) -> JsonObject:
        """Commit or retry a failed fenced job."""
        ...

    async def get_control(
        self,
        job_id: str,
        worker_id: str,
        lease_id: str,
    ) -> JsonObject:
        """Return control state through a current fenced lease."""
        ...

    async def get_migration(self, migration_id: str) -> JsonObject | None:
        """Return one durable migration record."""
        ...


class SignedArtifactStore(Protocol):
    """Sign constrained S3 artifact transfers without proxying file bytes."""

    async def sign_upload(self, job_id: str, request: JsonObject) -> JsonObject:
        """Reserve and sign a checksum-bound artifact upload."""
        ...

    async def sign_download(
        self,
        job_id: str,
        artifact_id: str,
        request: JsonObject,
    ) -> JsonObject:
        """Sign a fenced artifact download."""
        ...

    async def complete_upload(
        self,
        job_id: str,
        artifact_id: str,
        request: JsonObject,
    ) -> JsonObject:
        """Verify and finalize one artifact upload."""
        ...

    async def create_migration(self, request: JsonObject) -> JsonObject:
        """Create a signed legacy-state upload."""
        ...

    async def complete_migration(
        self,
        migration_id: str,
        request: JsonObject,
    ) -> JsonObject:
        """Verify and enqueue one legacy-state migration."""
        ...


class CompetitionGateway(Protocol):
    """Real competition platform operations exposed to authenticated users."""

    async def search(self, request: JsonObject) -> list[JsonObject]:
        """Search real competitions with validated filters."""
        ...

    async def fetch(self, url: str) -> JsonObject:
        """Fetch one exact competition by URL."""
        ...


@dataclass(slots=True)
class ApiRuntime:
    """Lifespan-owned API dependencies."""

    store: RuntimeStore
    artifacts: SignedArtifactStore | None
    competitions: CompetitionGateway | None
    gateway_verifier: GatewayVerifier
    worker_verifier: WorkerVerifier


def create_app(runtime_factory: RuntimeFactory | None = None) -> FastAPI:
    """Create the FastAPI application without import-time resource access.

    Args:
        runtime_factory: Lifespan context that owns stores and external clients.

    Returns:
        A configured FastAPI application.
    """
    factory = runtime_factory or _default_runtime_factory

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with factory() as runtime:
            application.state.runtime = runtime
            with logfire.instrument_fastapi(
                application,
                capture_headers=False,
                request_attributes_mapper=_safe_request_attributes,
            ):
                yield

    application = FastAPI(
        title="AGENT-K API",
        version=__version__,
        lifespan=lifespan,
    )
    _install_error_handlers(application)
    _install_request_id_middleware(application)
    _install_public_routes(application)
    _install_worker_routes(application)
    return application


def _install_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            request,
            exc.status_code,
            exc.code,
            exc.message,
            retryable=exc.retryable,
            details=exc.details,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        fields = [".".join(str(item) for item in error["loc"]) for error in exc.errors()]
        return _error_response(
            request,
            422,
            "invalid_request",
            "Request validation failed",
            details={"fields": fields},
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, KeyError):
            return _error_response(request, 404, "not_found", "Resource not found")
        if isinstance(exc, PermissionError):
            return _error_response(request, 403, "forbidden", "Access denied")
        if isinstance(exc, CompetitionNotFoundError):
            return _error_response(request, 404, "competition_not_found", str(exc))
        if isinstance(exc, RateLimitError):
            return _error_response(
                request,
                429,
                "competition_rate_limited",
                "The competition service rate limit was reached",
                retryable=True,
            )
        if isinstance(exc, AuthenticationError):
            return _error_response(
                request,
                502,
                "competition_authentication_failed",
                "The competition service rejected its configured credentials",
            )
        error_code = getattr(exc, "code", None)
        if error_code in {"stale_lease", "store_conflict"}:
            return _error_response(
                request,
                409,
                str(error_code),
                str(exc),
                details=getattr(exc, "details", None),
            )
        if error_code == "runtime_store_error":
            return _error_response(
                request,
                503,
                "runtime_store_unavailable",
                "The durable runtime store is unavailable",
                retryable=True,
            )
        status_code = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        if isinstance(status_code, int) and isinstance(code, str):
            return _error_response(
                request,
                status_code,
                code,
                str(exc),
                retryable=bool(getattr(exc, "retryable", False)),
                details=getattr(exc, "details", None),
            )
        logfire.exception(
            "api_request_failed",
            request_id=_request_id(request),
            path=request.url.path,
        )
        return _error_response(
            request,
            500,
            "internal_error",
            "The request could not be completed",
            retryable=True,
        )


def _install_request_id_middleware(application: FastAPI) -> None:
    @application.middleware("http")
    async def add_request_id(request: Request, call_next: Callable[[Request], Any]) -> Any:
        candidate = request.headers.get("X-Request-ID", "")
        request_id = candidate if _valid_request_id(candidate) else str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def _install_public_routes(application: FastAPI) -> None:  # noqa: C901
    @application.get("/health")
    async def health(response: Response) -> dict[str, str]:
        revision = os.environ.get("RENDER_GIT_COMMIT")
        if revision:
            response.headers["X-Revision"] = revision
        response.headers["Cache-Control"] = "no-store"
        return {"status": "healthy"}

    @application.get("/ready")
    async def ready(runtime: ApiRuntime = Depends(_get_runtime)) -> dict[str, str]:
        _require_artifact_store(runtime)
        return {"status": "ready"}

    @application.post(
        "/api/mission/start",
        response_model=MissionStartResponse,
        status_code=202,
    )
    async def start_mission(
        payload: MissionStartRequest,
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MissionStartResponse:
        _require_artifact_store(runtime)
        request_data = payload.model_dump(mode="json", by_alias=True, exclude_none=True)
        if payload.evolution_models:
            request_data["criteria"]["evolutionModels"] = payload.evolution_models
        if payload.competition_id is None and payload.competition_url is not None:
            competition = await _require_competition_gateway(runtime).fetch(payload.competition_url)
            request_data["competitionId"] = competition["id"]
        result = await runtime.store.create_mission(
            owner_id=identity.user_id,
            request=request_data,
        )
        return MissionStartResponse.model_validate(result)

    @application.get(
        "/api/mission/{mission_id}/status",
        response_model=MissionStatusResponse,
    )
    async def get_mission_status(
        mission_id: str,
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MissionStatusResponse:
        record = await runtime.store.require_owner(mission_id, identity.user_id)
        return MissionStatusResponse.model_validate(record)

    @application.post(
        "/api/mission/{mission_id}/resume",
        response_model=MissionControlResponse,
        status_code=202,
    )
    async def resume_mission(
        mission_id: str,
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MissionControlResponse:
        result = await runtime.store.request_control(
            mission_id,
            identity.user_id,
            "resume",
            None,
        )
        return MissionControlResponse.model_validate(result)

    @application.post(
        "/api/mission/{mission_id}/pause",
        response_model=MissionControlResponse,
        status_code=202,
    )
    async def pause_mission(
        mission_id: str,
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MissionControlResponse:
        result = await runtime.store.request_control(
            mission_id,
            identity.user_id,
            "pause",
            "paused_via_api",
        )
        return MissionControlResponse.model_validate(result)

    @application.post(
        "/api/mission/{mission_id}/abort",
        response_model=MissionControlResponse,
        status_code=202,
    )
    async def abort_mission(
        mission_id: str,
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MissionControlResponse:
        result = await runtime.store.request_control(
            mission_id,
            identity.user_id,
            "cancel",
            "aborted_via_api",
        )
        return MissionControlResponse.model_validate(result)

    @application.get("/api/mission/{mission_id}/stream")
    async def stream_mission(
        mission_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> StreamingResponse:
        await runtime.store.require_owner(mission_id, identity.user_id)
        after_sequence = _parse_last_event_id(last_event_id)
        return StreamingResponse(
            _stream_mission_events(
                request,
                runtime.store,
                mission_id,
                after_sequence,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/api/mission/best-results")
    async def best_results(
        limit: int = Query(default=5, ge=1, le=100),
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> JsonObject:
        results = await runtime.store.list_best_results(identity.user_id, limit)
        return {"results": results}

    @application.post("/api/competitions/search")
    async def search_competitions(
        payload: CompetitionSearchRequest,
        _: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> JsonObject:
        gateway = _require_competition_gateway(runtime)
        results = await gateway.search(
            payload.model_dump(mode="json", by_alias=True, exclude_none=True)
        )
        return {"competitions": results, "count": len(results)}

    @application.post("/api/competitions/fetch")
    async def fetch_competition(
        payload: CompetitionFetchRequest,
        _: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> JsonObject:
        gateway = _require_competition_gateway(runtime)
        return {"competition": await gateway.fetch(payload.url)}

    @application.post("/agentic_generative_ui/")
    async def handle_chat(
        payload: ChatRequest,
        request: Request,
        identity: UserIdentity = Depends(_require_user),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> StreamingResponse:
        intent = parse_mission_intent(payload.messages)
        if not intent.is_mission or intent.criteria is None:
            return StreamingResponse(
                _stream_product_help(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        _require_artifact_store(runtime)
        mission_request = {
            "schemaVersion": "1.0.0",
            "criteria": intent.criteria.model_dump(mode="json", by_alias=True),
        }
        result = await runtime.store.create_mission(
            owner_id=identity.user_id,
            request=mission_request,
        )
        mission_id = MissionStartResponse.model_validate(result).mission_id
        return StreamingResponse(
            _stream_mission_events(request, runtime.store, mission_id, 0),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def _install_worker_routes(application: FastAPI) -> None:  # noqa: C901
    @application.post(
        "/api/worker/v1/jobs/claim",
        response_model=ClaimJobsResponse,
    )
    async def claim_jobs(
        payload: ClaimJobsRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> ClaimJobsResponse:
        _ensure_worker(identity, payload.worker_id)
        result = await runtime.store.claim_jobs(_wire_dict(payload))
        return ClaimJobsResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/heartbeat",
        response_model=HeartbeatResponse,
    )
    async def heartbeat(
        job_id: str,
        payload: HeartbeatRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> HeartbeatResponse:
        _ensure_worker(identity, payload.worker_id)
        result = await runtime.store.heartbeat(job_id, _wire_dict(payload))
        return HeartbeatResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/events",
        response_model=AppendEventsResponse,
    )
    async def append_events(
        job_id: str,
        payload: AppendEventsRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> AppendEventsResponse:
        _ensure_worker(identity, payload.worker_id)
        _validate_event_batch(payload.events)
        result = await runtime.store.append_events(job_id, _wire_dict(payload))
        return AppendEventsResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/snapshots",
        response_model=SaveSnapshotResponse,
    )
    async def save_snapshot(
        job_id: str,
        payload: SaveSnapshotRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> SaveSnapshotResponse:
        _ensure_worker(identity, payload.worker_id)
        result = await runtime.store.save_snapshot(job_id, _wire_dict(payload))
        return SaveSnapshotResponse.model_validate(result)

    @application.get(
        "/api/worker/v1/jobs/{job_id}/snapshots/{snapshot_id}",
        response_model=MissionSnapshotResponse,
    )
    async def get_snapshot(
        job_id: str,
        snapshot_id: str,
        lease_id: str = Query(alias="leaseId", min_length=1),
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MissionSnapshotResponse:
        result = await runtime.store.get_snapshot(
            job_id,
            snapshot_id,
            identity.worker_id,
            lease_id,
        )
        if result is None:
            raise ApiError(404, "snapshot_not_found", "Snapshot not found")
        return MissionSnapshotResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/complete",
        response_model=CompleteJobResponse,
    )
    async def complete_job(
        job_id: str,
        payload: CompleteJobRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> CompleteJobResponse:
        _ensure_worker(identity, payload.worker_id)
        result = await runtime.store.complete_job(job_id, _wire_dict(payload))
        return CompleteJobResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/fail",
        response_model=FailJobResponse,
    )
    async def fail_job(
        job_id: str,
        payload: FailJobRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> FailJobResponse:
        _ensure_worker(identity, payload.worker_id)
        result = await runtime.store.fail_job(job_id, _wire_dict(payload))
        return FailJobResponse.model_validate(result)

    @application.get(
        "/api/worker/v1/jobs/{job_id}/control",
        response_model=WorkerControlResponse,
    )
    async def get_control(
        job_id: str,
        lease_id: str = Query(alias="leaseId", min_length=1),
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> WorkerControlResponse:
        result = await runtime.store.get_control(job_id, identity.worker_id, lease_id)
        return WorkerControlResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/artifacts/upload-url",
        response_model=SignedArtifactResponse,
    )
    async def sign_artifact_upload(
        job_id: str,
        payload: ArtifactUploadRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> SignedArtifactResponse:
        _ensure_worker(identity, payload.worker_id)
        artifacts = _require_artifact_store(runtime)
        result = await artifacts.sign_upload(job_id, _wire_dict(payload))
        return SignedArtifactResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/artifacts/{artifact_id}/download-url",
        response_model=SignedArtifactResponse,
    )
    async def sign_artifact_download(
        job_id: str,
        artifact_id: str,
        payload: ArtifactDownloadRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> SignedArtifactResponse:
        _ensure_worker(identity, payload.worker_id)
        artifacts = _require_artifact_store(runtime)
        result = await artifacts.sign_download(
            job_id,
            artifact_id,
            _wire_dict(payload),
        )
        return SignedArtifactResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/jobs/{job_id}/artifacts/{artifact_id}/complete",
        response_model=CompleteArtifactResponse,
    )
    async def complete_artifact_upload(
        job_id: str,
        artifact_id: str,
        payload: CompleteArtifactRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> CompleteArtifactResponse:
        _ensure_worker(identity, payload.worker_id)
        artifacts = _require_artifact_store(runtime)
        result = await artifacts.complete_upload(
            job_id,
            artifact_id,
            _wire_dict(payload),
        )
        return CompleteArtifactResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/migrations",
        response_model=CreateMigrationResponse,
        status_code=201,
    )
    async def create_migration(
        payload: CreateMigrationRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> CreateMigrationResponse:
        _ensure_worker(identity, payload.worker_id)
        artifacts = _require_artifact_store(runtime)
        result = await artifacts.create_migration(_wire_dict(payload))
        return CreateMigrationResponse.model_validate(result)

    @application.post(
        "/api/worker/v1/migrations/{migration_id}/complete",
        response_model=MigrationStatusResponse,
        status_code=202,
    )
    async def complete_migration(
        migration_id: str,
        payload: CompleteMigrationRequest,
        identity: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MigrationStatusResponse:
        _ensure_worker(identity, payload.worker_id)
        artifacts = _require_artifact_store(runtime)
        result = await artifacts.complete_migration(
            migration_id,
            _wire_dict(payload),
        )
        return MigrationStatusResponse.model_validate(result)

    @application.get(
        "/api/worker/v1/migrations/{migration_id}",
        response_model=MigrationStatusResponse,
    )
    async def get_migration(
        migration_id: str,
        _: WorkerIdentity = Depends(_require_worker),
        runtime: ApiRuntime = Depends(_get_runtime),
    ) -> MigrationStatusResponse:
        result = await runtime.store.get_migration(migration_id)
        if result is None:
            raise ApiError(404, "migration_not_found", "Migration not found")
        return MigrationStatusResponse.model_validate(result)


async def _get_runtime(request: Request) -> ApiRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, ApiRuntime):
        raise ApiError(
            503,
            "runtime_unavailable",
            "API runtime is unavailable",
            retryable=True,
        )
    return runtime


async def _require_user(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> UserIdentity:
    runtime = await _get_runtime(request)
    token = _bearer_token(authorization, code="missing_identity")
    return runtime.gateway_verifier.verify(token)


async def _require_worker(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    worker_id: str | None = Header(default=None, alias="X-Worker-Id"),
) -> WorkerIdentity:
    if not worker_id:
        raise ApiError(401, "missing_worker_id", "X-Worker-Id is required")
    runtime = await _get_runtime(request)
    token = _bearer_token(authorization, code="missing_worker_token")
    return runtime.worker_verifier.verify(token, worker_id)


def _bearer_token(value: str | None, *, code: str) -> str:
    if value is None:
        raise ApiError(401, code, "Bearer authentication is required")
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token:
        raise ApiError(401, code, "Bearer authentication is required")
    return token


def _ensure_worker(identity: WorkerIdentity, worker_id: str) -> None:
    if not hmac.compare_digest(identity.worker_id, worker_id):
        raise ApiError(403, "worker_identity_mismatch", "Worker identity mismatch")


def _validate_event_batch(events: list[WorkerEvent]) -> None:
    sequences = [event.sequence for event in events]
    if sequences != sorted(sequences):
        raise ApiError(
            422,
            "invalid_event_order",
            "Events must be ordered by ascending sequence",
        )
    if len(sequences) != len(set(sequences)):
        raise ApiError(
            422,
            "duplicate_event_sequence",
            "An event batch cannot repeat a sequence",
        )
    if any(right != left + 1 for left, right in zip(sequences, sequences[1:])):
        raise ApiError(
            422,
            "noncontiguous_event_batch",
            "Events in a batch must have contiguous sequences",
        )


def _wire_dict(model: BaseModel) -> JsonObject:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


class _PostgresRuntimeAdapter:
    """Adapt typed PostgreSQL operations to the API's wire-oriented boundary."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def create_mission(self, *, owner_id: str, request: JsonObject) -> JsonObject:
        from .stores import JobCommand, JobEnqueueRequest, JobWorkspace

        mission_id = uuid.uuid4()
        initial_state = {
            key: request[key]
            for key in ("schemaVersion", "criteria", "competitionId", "competitionUrl")
            if key in request
        }
        await self._store.create_mission(
            mission_id,
            owner_id,
            state=initial_state,
            schema_version=str(request.get("schemaVersion", "1.0.0")),
        )
        command = JobCommand(
            type="start",
            competition_id=request.get("competitionId"),
            criteria=dict(request.get("criteria", {})),
        )
        await self._store.enqueue_job(
            JobEnqueueRequest(
                mission_id=mission_id,
                command=command,
                workspace=JobWorkspace(artifact_prefix=f"missions/{mission_id}"),
                idempotency_key=f"mission-start:{mission_id}",
            )
        )
        return {"missionId": str(mission_id), "status": "queued"}

    async def get_mission(self, mission_id: str) -> JsonObject | None:
        record = await self._store.get_mission(_uuid(mission_id, "mission_id"))
        return _mission_status(record) if record is not None else None

    async def require_owner(self, mission_id: str, owner_id: str) -> JsonObject:
        record = await self._store.require_owner(
            _uuid(mission_id, "mission_id"),
            owner_id,
        )
        return _mission_status(record)

    async def request_control(
        self,
        mission_id: str,
        owner_id: str,
        action: MissionControlAction,
        reason: str | None,
    ) -> JsonObject:
        from .stores import ControlAction

        mission_uuid = _uuid(mission_id, "mission_id")
        record = await self._store.require_owner(mission_uuid, owner_id)
        if record.status in TERMINAL_MISSION_STATUSES:
            raise ApiError(409, "mission_terminal", "Mission is already terminal")
        control = {
            "resume": ControlAction.CONTINUE,
            "pause": ControlAction.PAUSE,
            "cancel": ControlAction.CANCEL,
        }[action]
        await self._store.request_control(mission_uuid, control, reason=reason)
        status = {"resume": "queued", "pause": "pausing", "cancel": "cancelling"}[action]
        return {"missionId": mission_id, "status": status, "action": action}

    async def list_events(
        self,
        mission_id: str,
        after_sequence: int,
        limit: int = DEFAULT_EVENT_BATCH_SIZE,
    ) -> list[JsonObject]:
        records = await self._store.list_events(
            _uuid(mission_id, "mission_id"),
            after_sequence=after_sequence,
            limit=limit,
        )
        return [
            {
                "eventId": str(record.event_id),
                "missionId": str(record.mission_id),
                "sequence": record.stream_sequence,
                "type": record.type,
                "timestamp": record.timestamp,
                "data": record.data,
            }
            for record in records
        ]

    async def list_best_results(self, owner_id: str, limit: int) -> list[JsonObject]:
        records = await self._store.list_best_results(owner_id, limit)
        return [_model_dict(record) for record in records]

    async def claim_jobs(self, request: JsonObject) -> JsonObject:
        from .stores import ClaimJobsRequest

        result = await self._store.claim_jobs(ClaimJobsRequest.model_validate(request))
        return _model_dict(result)

    async def heartbeat(self, job_id: str, request: JsonObject) -> JsonObject:
        from .stores import HeartbeatRequest

        result = await self._store.heartbeat(
            _uuid(job_id, "job_id"),
            HeartbeatRequest.model_validate(request),
        )
        return _model_dict(result)

    async def append_events(self, job_id: str, request: JsonObject) -> JsonObject:
        from .stores import AppendEventsRequest

        result = await self._store.append_events(
            _uuid(job_id, "job_id"),
            AppendEventsRequest.model_validate(request),
        )
        return _model_dict(result)

    async def save_snapshot(self, job_id: str, request: JsonObject) -> JsonObject:
        from .stores import MissionSnapshot

        result = await self._store.save_snapshot(
            _uuid(job_id, "job_id"),
            str(request["workerId"]),
            _uuid(str(request["leaseId"]), "lease_id"),
            MissionSnapshot.model_validate(request["snapshot"]),
        )
        return _model_dict(result)

    async def get_snapshot(
        self,
        job_id: str,
        snapshot_id: str,
        worker_id: str,
        lease_id: str,
    ) -> JsonObject | None:
        result = await self._store.get_snapshot(
            _uuid(job_id, "job_id"),
            worker_id,
            _uuid(lease_id, "lease_id"),
            _uuid(snapshot_id, "snapshot_id"),
        )
        return _model_dict(result) if result is not None else None

    async def complete_job(self, job_id: str, request: JsonObject) -> JsonObject:
        from .stores import JobCompletionRequest

        result = await self._store.complete_job(
            _uuid(job_id, "job_id"),
            JobCompletionRequest.model_validate(request),
        )
        return _model_dict(result)

    async def fail_job(self, job_id: str, request: JsonObject) -> JsonObject:
        from .stores import JobFailureRequest

        result = await self._store.fail_job(
            _uuid(job_id, "job_id"),
            JobFailureRequest.model_validate(request),
        )
        return _model_dict(result)

    async def get_control(
        self,
        job_id: str,
        worker_id: str,
        lease_id: str,
    ) -> JsonObject:
        result = await self._store.get_control(
            _uuid(job_id, "job_id"),
            worker_id,
            _uuid(lease_id, "lease_id"),
        )
        data = _model_dict(result)
        data["serverTime"] = datetime.now(UTC).isoformat()
        return data

    async def get_migration(self, migration_id: str) -> JsonObject | None:
        result = await self._store.get_migration(_uuid(migration_id, "migration_id"))
        return _model_dict(result) if result is not None else None


class _S3ArtifactAdapter:
    """Adapt the S3 signer to constrained API wire requests."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def sign_upload(self, job_id: str, request: JsonObject) -> JsonObject:
        from .stores import ArtifactUploadSpec

        result = await self._store.sign_upload(
            _uuid(job_id, "job_id"),
            ArtifactUploadSpec.model_validate(request),
        )
        return _model_dict(result)

    async def sign_download(
        self,
        job_id: str,
        artifact_id: str,
        request: JsonObject,
    ) -> JsonObject:
        result = await self._store.sign_download(
            _uuid(job_id, "job_id"),
            str(request["workerId"]),
            _uuid(str(request["leaseId"]), "lease_id"),
            _uuid(artifact_id, "artifact_id"),
        )
        return _model_dict(result)

    async def complete_upload(
        self,
        job_id: str,
        artifact_id: str,
        request: JsonObject,
    ) -> JsonObject:
        from .stores import ArtifactCompleteSpec

        result = await self._store.complete_upload(
            _uuid(job_id, "job_id"),
            _uuid(artifact_id, "artifact_id"),
            ArtifactCompleteSpec.model_validate(request),
        )
        return _model_dict(result)

    async def create_migration(self, request: JsonObject) -> JsonObject:
        from .stores import MigrationCreateRequest

        result = await self._store.create_migration_upload(
            MigrationCreateRequest.model_validate(request)
        )
        return _model_dict(result)

    async def complete_migration(
        self,
        migration_id: str,
        request: JsonObject,
    ) -> JsonObject:
        result = await self._store.complete_migration_upload(
            _uuid(migration_id, "migration_id"),
            str(request["workerId"]),
            str(request["sha256"]),
            int(request["sizeBytes"]),
        )
        return _model_dict(result)


class _KaggleGateway:
    """Expose bounded Kaggle discovery through API wire dictionaries."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    async def search(self, request: JsonObject) -> list[JsonObject]:
        domains = [str(value) for value in request.get("domains", [])]
        categories = [str(value) for value in request.get("competitionTypes", [])]
        minimum_prize = request.get("minPrize")
        if request.get("paidOnly") and minimum_prize is None:
            minimum_prize = 1
        minimum_days = int(request.get("minDaysRemaining", 7))
        results: list[JsonObject] = []
        async for competition in self._adapter.search_competitions(
            categories=categories or None,
            min_prize=int(minimum_prize) if minimum_prize is not None else None,
            active_only=True,
        ):
            if competition.days_remaining < minimum_days:
                continue
            if not _competition_matches_domains(competition, domains):
                continue
            results.append(_serialize_competition(competition))
        return results

    async def fetch(self, url: str) -> JsonObject:
        competition = await self._adapter.get_competition_by_url(url)
        return _serialize_competition(competition)


@asynccontextmanager
async def _default_runtime_factory() -> AsyncIterator[ApiRuntime]:
    import httpx

    from .config import load_settings
    from .kaggle import KaggleAdapter, KaggleSettings
    from .observability import configure_observability
    from .stores import PostgresRuntimeStore, S3ArtifactStore

    settings = load_settings()
    if settings.database_url is None:
        raise RuntimeError("AGENT_K_DATABASE_URL is required")
    if settings.gateway_secret is None:
        raise RuntimeError("AGENT_K_GATEWAY_SECRET is required")
    if settings.worker_token is None:
        raise RuntimeError("AGENT_K_WORKER_TOKEN is required")

    configure_observability(
        service_name="agent-k-api",
        environment=settings.environment,
    )
    store = PostgresRuntimeStore(settings.database_url.get_secret_value())
    await store.open()
    client = httpx.AsyncClient()
    try:
        await store.initialize()
        try:
            kaggle_config = KaggleSettings()
        except ValidationError:
            competitions = None
        else:
            competitions = _KaggleGateway(KaggleAdapter(kaggle_config, client))

        artifact_store = None
        if settings.s3_bucket:
            artifact_store = _S3ArtifactAdapter(
                S3ArtifactStore(
                    store,
                    bucket=settings.s3_bucket,
                    region=settings.s3_region,
                )
            )

        gateway_secrets = tuple(
            secret
            for secret in (settings.gateway_secret, settings.gateway_previous_secret)
            if secret is not None
        )
        worker_tokens = tuple(
            token
            for token in (settings.worker_token, settings.worker_previous_token)
            if token is not None
        )
        yield ApiRuntime(
            store=_PostgresRuntimeAdapter(store),
            artifacts=artifact_store,
            competitions=competitions,
            gateway_verifier=GatewayVerifier(
                gateway_secrets,
                issuer=settings.gateway_issuer,
                audience=settings.gateway_audience,
                max_ttl_seconds=settings.gateway_token_ttl_seconds,
            ),
            worker_verifier=WorkerVerifier(worker_tokens),
        )
    finally:
        await client.aclose()
        await store.close()


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: JsonObject | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            details=details,
        ),
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json", by_alias=True, exclude_none=True),
        headers={"X-Request-ID": request_id},
    )


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid.uuid4())


def _valid_request_id(value: str) -> bool:
    return bool(value) and len(value) <= 200 and value.isascii() and value.isprintable()


def _safe_request_attributes(
    _request: Request | WebSocket,
    _attributes: dict[str, Any],
) -> dict[str, Any]:
    """Prevent automatic request-model, header, and token capture."""
    return {}


def _decode_segment(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(401, "invalid_identity", "Invalid identity token") from exc


def _decode_json_segment(value: str) -> JsonObject:
    try:
        decoded = json.loads(_decode_segment(value))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(401, "invalid_identity", "Invalid identity token") from exc
    if not isinstance(decoded, dict):
        raise ApiError(401, "invalid_identity", "Invalid identity token")
    return decoded


def _uuid(value: str, name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ApiError(422, f"invalid_{name}", f"Invalid {name.replace('_', ' ')}") from exc


def _model_dict(value: Any) -> JsonObject:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _mission_status(record: Any) -> JsonObject:
    state = record.state if isinstance(record.state, dict) else {}
    return {
        "missionId": str(record.mission_id),
        "status": record.status,
        "currentPhase": state.get("currentPhase"),
        "progress": state.get("progress", 0.0),
        "competitionId": state.get("competitionId"),
        "errorMessage": state.get("errorMessage"),
        "latestSequence": state.get("latestSequence", 0),
        "iteration": state.get("iteration"),
        "evolutionIterations": state.get("evolutionIterations"),
        "modelFamily": state.get("modelFamily"),
        "modelSize": state.get("modelSize"),
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
    }


def _parse_last_event_id(value: str | None) -> int:
    try:
        return parse_last_event_id(value)
    except ValueError as exc:
        raise ApiError(400, "invalid_event_cursor", str(exc)) from exc


async def _stream_mission_events(
    request: Request,
    store: RuntimeStore,
    mission_id: str,
    after_sequence: int,
) -> AsyncIterator[str]:
    cursor = after_sequence
    heartbeat_at = time.monotonic() + SSE_HEARTBEAT_SECONDS
    while not await request.is_disconnected():
        events = await store.list_events(mission_id, cursor, DEFAULT_EVENT_BATCH_SIZE)
        if events:
            for raw_event in events:
                event = AgentKEvent.model_validate(raw_event)
                cursor = event.sequence
                yield event.to_sse()
                if event.type == "mission-complete":
                    return
            heartbeat_at = time.monotonic() + SSE_HEARTBEAT_SECONDS
            continue

        mission = await store.get_mission(mission_id)
        if mission is None:
            raise ApiError(404, "mission_not_found", "Mission not found")
        if str(mission.get("status")) in TERMINAL_MISSION_STATUSES:
            return
        now = time.monotonic()
        if now >= heartbeat_at:
            yield ": keep-alive\n\n"
            heartbeat_at = now + SSE_HEARTBEAT_SECONDS
        await asyncio.sleep(SSE_POLL_SECONDS)


async def _stream_product_help() -> AsyncIterator[str]:
    message = (
        "I'm Agent-K. I can discover, research, prototype, evolve, and submit generic "
        "Kaggle solutions. To start, ask me to find or enter a Kaggle competition and "
        "include any prize, domain, deadline, or iteration constraints."
    )
    yield f"data: {json.dumps({'type': 'text-delta', 'textDelta': message})}\n\n"
    yield "data: [DONE]\n\n"


def _require_artifact_store(runtime: ApiRuntime) -> SignedArtifactStore:
    if runtime.artifacts is None:
        raise ApiError(
            503,
            "artifact_store_unavailable",
            "Artifact storage is unavailable",
            retryable=True,
        )
    return runtime.artifacts


def _require_competition_gateway(runtime: ApiRuntime) -> CompetitionGateway:
    if runtime.competitions is None:
        raise ApiError(
            503,
            "competition_gateway_unavailable",
            "Kaggle credentials are not configured",
        )
    return runtime.competitions


def _competition_matches_domains(competition: Any, domains: list[str]) -> bool:
    if not domains:
        return True
    normalized = {value.strip().lower().replace("-", "_").replace(" ", "_") for value in domains}
    tags = {str(tag).lower().replace("-", "_").replace(" ", "_") for tag in competition.tags}
    text = f"{competition.title} {competition.description or ''}".lower().replace("-", "_")
    return bool(normalized & tags) or any(domain in text for domain in normalized)


def _serialize_competition(competition: Any) -> JsonObject:
    return {
        "id": competition.id,
        "title": competition.title,
        "description": competition.description,
        "competitionType": competition.competition_type.value,
        "metric": competition.metric.value,
        "metricDirection": competition.metric_direction,
        "deadline": competition.deadline.isoformat(),
        "daysRemaining": competition.days_remaining,
        "prizePool": competition.prize_pool,
        "maxTeamSize": competition.max_team_size,
        "maxDailySubmissions": competition.max_daily_submissions,
        "tags": sorted(competition.tags),
        "url": competition.url,
    }


app = create_app()


def main() -> None:
    """Run the production API entry point."""
    import uvicorn

    uvicorn.run("agent_k.api:app", host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
