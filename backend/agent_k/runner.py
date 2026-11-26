"""Worker API client and isolated process entrypoints.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from .evaluator import CandidateResult, CandidateSpec, CandidateStatus
from .observability import configure_observability
from .stores import (
    AppendEventsReceipt,
    ArtifactCompleteResponse,
    ArtifactReference,
    ArtifactUploadSpec,
    ClaimJobsRequest,
    ClaimJobsResponse,
    ControlDirective,
    HeartbeatRequest,
    HeartbeatResponse,
    JobCompletionRequest,
    JobCompletionResponse,
    JobFailureRequest,
    JobFailureResponse,
    JobLease,
    MigrationCreateRequest,
    MigrationCreateResponse,
    MigrationRecord,
    MissionInterrupted,
    MissionRunResult,
    MissionSnapshot,
    MissionSnapshotRecord,
    SignedArtifactURL,
    SnapshotReceipt,
    WorkerEvent,
)

__all__ = (
    "WorkerApiClient",
    "WorkerApiError",
    "main",
)

_CHUNK_SIZE = 1024 * 1024


class WorkerApiError(RuntimeError):
    """Structured failure returned by the authenticated worker API."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.details = details


class _MissionProcessOutcome(BaseModel):
    """Private child-process envelope for success or checkpointed control."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "interrupted"]
    result: MissionRunResult | None = None
    action: Literal["pause", "cancel"] | None = None
    reason: str | None = None
    final_event_sequence: int | None = None


class WorkerApiClient:
    """Authenticated async client for fenced worker operations."""

    def __init__(
        self,
        base_url: str,
        worker_id: str,
        token: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be HTTP or HTTPS")
        if not worker_id or not token:
            raise ValueError("worker_id and token are required")
        self._worker_id = worker_id
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {token}",
                "X-Worker-Id": worker_id,
                "User-Agent": "agent-k-worker/1.0",
            },
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._transfer_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=3600.0, write=3600.0, pool=30.0)
        )

    @property
    def worker_id(self) -> str:
        """Return the identity bound to this client and its bearer token."""
        return self._worker_id

    async def __aenter__(self) -> WorkerApiClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Release network connections held by the worker client."""
        await self._client.aclose()
        await self._transfer_client.aclose()

    async def claim_jobs(
        self,
        capabilities: tuple[str, ...],
        *,
        lease_seconds: int = 60,
        max_jobs: int = 1,
    ) -> ClaimJobsResponse:
        """Atomically claim available work from the runtime queue."""
        request = ClaimJobsRequest(
            worker_id=self.worker_id,
            capabilities=capabilities,
            lease_seconds=lease_seconds,
            max_jobs=max_jobs,
        )
        payload = await self._request_json(
            "POST",
            "/api/worker/v1/jobs/claim",
            json=request.model_dump(mode="json", by_alias=True),
        )
        return ClaimJobsResponse.model_validate(payload)

    async def heartbeat(
        self,
        lease: JobLease,
        *,
        progress: float | None = None,
        current_phase: str | None = None,
        current_task_id: str | None = None,
    ) -> HeartbeatResponse:
        """Renew a lease and read its current control directive."""
        request = HeartbeatRequest(
            worker_id=self.worker_id,
            lease_id=lease.lease_id,
            progress=progress,
            current_phase=current_phase,
            current_task_id=current_task_id,
        )
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/heartbeat",
            json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
        )
        return HeartbeatResponse.model_validate(payload)

    async def append_events(
        self,
        lease: JobLease,
        events: tuple[WorkerEvent, ...],
    ) -> AppendEventsReceipt:
        """Append a contiguous, idempotent event batch."""
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/events",
            json={
                "workerId": self.worker_id,
                "leaseId": str(lease.lease_id),
                "events": [event.model_dump(mode="json", by_alias=True) for event in events],
            },
        )
        return AppendEventsReceipt.model_validate(payload)

    async def save_snapshot(
        self,
        lease: JobLease,
        snapshot: MissionSnapshot,
    ) -> SnapshotReceipt:
        """Save an immutable checkpoint under the current lease."""
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/snapshots",
            json={
                "workerId": self.worker_id,
                "leaseId": str(lease.lease_id),
                "snapshot": snapshot.model_dump(mode="json", by_alias=True),
            },
        )
        return SnapshotReceipt.model_validate(payload)

    async def get_snapshot(
        self,
        lease: JobLease,
        snapshot_id: uuid.UUID,
    ) -> MissionSnapshotRecord:
        """Fetch a checkpoint and artifact references through the fenced lease."""
        payload = await self._request_json(
            "GET",
            f"/api/worker/v1/jobs/{lease.job_id}/snapshots/{snapshot_id}",
            params={"leaseId": str(lease.lease_id)},
        )
        return MissionSnapshotRecord.model_validate(payload)

    async def get_control(self, lease: JobLease) -> ControlDirective:
        """Read a pause or cancellation request without renewing the lease."""
        payload = await self._request_json(
            "GET",
            f"/api/worker/v1/jobs/{lease.job_id}/control",
            params={"leaseId": str(lease.lease_id)},
        )
        return ControlDirective.model_validate(payload)

    async def complete_job(
        self,
        lease: JobLease,
        result: MissionRunResult,
        *,
        idempotency_key: str,
    ) -> JobCompletionResponse:
        """Commit a successful mission after its durable terminal event."""
        request = JobCompletionRequest(
            worker_id=self.worker_id,
            lease_id=lease.lease_id,
            idempotency_key=idempotency_key,
            result=result,
            final_event_sequence=result.final_event_sequence,
            artifacts=result.artifacts,
        )
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/complete",
            json=request.model_dump(mode="json", by_alias=True),
        )
        return JobCompletionResponse.model_validate(payload)

    async def fail_job(
        self,
        lease: JobLease,
        request: JobFailureRequest,
    ) -> JobFailureResponse:
        """Commit a structured failure or server-authorized retry."""
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/fail",
            json=request.model_dump(mode="json", by_alias=True),
        )
        return JobFailureResponse.model_validate(payload)

    async def request_upload(
        self,
        lease: JobLease,
        path: Path,
        *,
        content_type: str,
        role: str,
    ) -> SignedArtifactURL:
        """Reserve an immutable object and obtain a checksum-bound S3 PUT URL."""
        size_bytes, sha256 = await asyncio.to_thread(_file_metadata, path)
        request = ArtifactUploadSpec(
            worker_id=self.worker_id,
            lease_id=lease.lease_id,
            name=path.name,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            role=role,
        )
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/artifacts/upload-url",
            json=request.model_dump(mode="json", by_alias=True),
        )
        return SignedArtifactURL.model_validate(payload)

    async def upload_file(
        self,
        lease: JobLease,
        signed: SignedArtifactURL,
        path: Path,
        *,
        role: str,
    ) -> ArtifactReference:
        """Upload and server-verify a file without exposing S3 credentials."""
        size_bytes, sha256 = await asyncio.to_thread(_file_metadata, path)
        response = await self._transfer_client.put(
            signed.url,
            headers=signed.headers,
            content=_iter_file(path),
        )
        response.raise_for_status()
        payload = await self._request_json(
            "POST",
            (f"/api/worker/v1/jobs/{lease.job_id}/artifacts/{signed.artifact_id}/complete"),
            json={
                "workerId": self.worker_id,
                "leaseId": str(lease.lease_id),
                "sha256": sha256,
                "sizeBytes": size_bytes,
            },
        )
        response_id = ArtifactCompleteResponse.model_validate(payload).artifact_id
        if response_id != signed.artifact_id:
            raise WorkerApiError(
                "artifact_id_mismatch",
                "Artifact completion returned a different identifier",
                retryable=False,
                status_code=502,
            )
        return ArtifactReference(artifact_id=response_id, role=role)

    async def upload_signed_file(self, signed: SignedArtifactURL, path: Path) -> None:
        """Upload a checksum-bound source whose completion uses a separate endpoint."""
        response = await self._transfer_client.put(
            signed.url,
            headers=signed.headers,
            content=_iter_file(path),
        )
        response.raise_for_status()

    async def request_download(
        self,
        lease: JobLease,
        artifact_id: uuid.UUID,
    ) -> SignedArtifactURL:
        """Obtain a short-lived GET URL for an available mission artifact."""
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/jobs/{lease.job_id}/artifacts/{artifact_id}/download-url",
            json={"workerId": self.worker_id, "leaseId": str(lease.lease_id)},
        )
        return SignedArtifactURL.model_validate(payload)

    async def download_file(
        self,
        signed: SignedArtifactURL,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> None:
        """Download an artifact atomically and verify its immutable metadata."""
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        digest = hashlib.sha256()
        size = 0
        try:
            async with self._transfer_client.stream(
                "GET",
                signed.url,
                headers=signed.headers,
            ) as response:
                response.raise_for_status()
                with temporary.open("xb") as target:
                    os.chmod(temporary, 0o600)
                    async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                        size += len(chunk)
                        if size > expected_size:
                            raise WorkerApiError(
                                "artifact_size_mismatch",
                                "Downloaded artifact exceeded its expected size",
                                retryable=False,
                                status_code=502,
                            )
                        digest.update(chunk)
                        target.write(chunk)
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise WorkerApiError(
                    "artifact_integrity_mismatch",
                    "Downloaded artifact failed size or checksum validation",
                    retryable=False,
                    status_code=502,
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    async def create_migration_upload(
        self,
        request: MigrationCreateRequest,
    ) -> MigrationCreateResponse:
        """Create a content-addressed legacy migration upload."""
        payload = await self._request_json(
            "POST",
            "/api/worker/v1/migrations",
            json=request.model_dump(mode="json", by_alias=True),
        )
        return MigrationCreateResponse.model_validate(payload)

    async def complete_migration_upload(
        self,
        migration_id: uuid.UUID,
        *,
        idempotency_key: str,
        sha256: str,
        size_bytes: int,
    ) -> MigrationRecord:
        """Confirm an uploaded migration source for asynchronous import."""
        payload = await self._request_json(
            "POST",
            f"/api/worker/v1/migrations/{migration_id}/complete",
            json={
                "workerId": self.worker_id,
                "idempotencyKey": idempotency_key,
                "sha256": sha256,
                "sizeBytes": size_bytes,
            },
        )
        return MigrationRecord.model_validate(payload)

    async def get_migration(self, migration_id: uuid.UUID) -> MigrationRecord:
        """Read one migration's asynchronous processing status."""
        payload = await self._request_json(
            "GET",
            f"/api/worker/v1/migrations/{migration_id}",
        )
        return MigrationRecord.model_validate(payload)

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._client.request(method, path, **kwargs)
        try:
            payload = response.json()
        except ValueError as error:
            raise WorkerApiError(
                "invalid_worker_response",
                "Worker API returned a non-JSON response",
                retryable=response.status_code >= 500,
                status_code=response.status_code,
            ) from error
        if response.is_error:
            detail = payload.get("error", {}) if isinstance(payload, dict) else {}
            raise WorkerApiError(
                str(detail.get("code", "worker_api_error")),
                str(detail.get("message", "Worker API request failed")),
                retryable=bool(detail.get("retryable", response.status_code >= 500)),
                status_code=response.status_code,
                details=detail.get("details"),
            )
        if not isinstance(payload, dict):
            raise WorkerApiError(
                "invalid_worker_response",
                "Worker API returned an invalid JSON envelope",
                retryable=False,
                status_code=response.status_code,
            )
        return payload


async def _run_candidate(spec_path: Path, output_path: Path) -> int:
    spec = CandidateSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    input_root = spec_path.resolve().parent
    output_root = output_path.resolve().parent
    started = asyncio.get_running_loop().time()
    try:
        modality = spec.problem_type.modality.value
        if modality == "tabular":
            from .tabular import run_candidate
        elif modality == "vision":
            from .vision import run_candidate
        elif modality == "text":
            from .text import run_candidate
        else:
            raise ValueError(f"Unsupported candidate modality: {modality}")
        outcome = run_candidate(spec, input_root=input_root, output_root=output_root)
        result = await outcome if inspect.isawaitable(outcome) else outcome
    except Exception as error:
        result = CandidateResult(
            candidate_id=spec.candidate_id,
            status=CandidateStatus.FAILED,
            duration_seconds=asyncio.get_running_loop().time() - started,
            error=f"{type(error).__name__}: {error}",
        )
    _write_private_json(output_path, result.model_dump(mode="json"))
    return 0 if result.status is CandidateStatus.SUCCEEDED else 1


async def _run_mission(job_path: Path, output_path: Path) -> int:
    configure_observability(
        service_name="agent-k-mission",
        environment=os.environ.get("AGENT_K_ENVIRONMENT", "production"),
    )
    lease = JobLease.model_validate_json(job_path.read_text(encoding="utf-8"))
    base_url = os.environ["AGENT_K_API_URL"]
    worker_id = os.environ["AGENT_K_WORKER_ID"]
    token = os.environ["AGENT_K_WORKER_TOKEN"]
    workdir = output_path.resolve().parent
    async with WorkerApiClient(base_url, worker_id, token) as client:
        try:
            from .mission import run_mission

            result = await run_mission(lease, client, workdir)
            outcome = _MissionProcessOutcome(status="completed", result=result)
            code = 0
        except MissionInterrupted as error:
            outcome = _MissionProcessOutcome(
                status="interrupted",
                action=error.action,
                reason=error.reason,
                final_event_sequence=error.final_event_sequence,
            )
            code = 3
    _write_private_json(output_path, outcome.model_dump(mode="json", by_alias=True))
    return code


def _file_metadata(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


async def _iter_file(path: Path) -> AsyncIterator[bytes]:
    with path.open("rb") as source:
        while chunk := await asyncio.to_thread(source.read, _CHUNK_SIZE):
            yield chunk


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as target:
        os.chmod(temporary, 0o600)
        json.dump(payload, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_k.runner")
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--spec", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    mission = commands.add_parser("mission")
    mission.add_argument("--job", type=Path, required=True)
    mission.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the candidate or mission child-process entrypoint."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "candidate":
        return asyncio.run(_run_candidate(arguments.spec, arguments.output))
    return asyncio.run(_run_mission(arguments.job, arguments.output))


if __name__ == "__main__":
    sys.exit(main())
