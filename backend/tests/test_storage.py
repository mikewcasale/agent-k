"""Exercise real S3 checksums and artifact resume with disposable PostgreSQL state.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import contextlib
import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import boto3
import httpx
import pytest
from botocore.exceptions import ClientError

from agent_k.stores import (
    ArtifactCompleteSpec,
    ArtifactUploadSpec,
    ClaimJobsRequest,
    ControlAction,
    JobCommand,
    JobEnqueueRequest,
    JobFailure,
    JobFailureRequest,
    JobLease,
    JobWorkspace,
    LeaseFenceError,
    MissionSnapshot,
    PostgresRuntimeStore,
    S3ArtifactStore,
)

__all__ = ()

pytestmark = pytest.mark.skipif(
    not all(
        os.environ.get(name) for name in ("AGENT_K_TEST_DATABASE_URL", "AGENT_K_TEST_S3_BUCKET")
    ),
    reason="requires disposable PostgreSQL and an explicitly designated real S3 acceptance bucket",
)


async def _delete_acceptance_object(
    bucket: str,
    region: str,
    object_key: str,
    version_id: str | None,
) -> None:
    """Delete a possibly committed object even when the PUT response was lost."""
    client = boto3.client("s3", region_name=region)
    recovered_version = version_id
    if recovered_version is None:
        try:
            response = await asyncio.to_thread(
                client.head_object,
                Bucket=bucket,
                Key=object_key,
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return
            raise
        candidate = response.get("VersionId")
        recovered_version = candidate if isinstance(candidate, str) else None
    if recovered_version is None:
        await asyncio.to_thread(
            client.delete_object,
            Bucket=bucket,
            Key=object_key,
        )
        return
    await asyncio.to_thread(
        client.delete_object,
        Bucket=bucket,
        Key=object_key,
        VersionId=recovered_version,
    )


async def test_s3_verified_artifact_survives_pause_and_reclaim() -> None:
    """Upload real repository license bytes, then restore them under a new fenced lease."""
    bucket = os.environ["AGENT_K_TEST_S3_BUCKET"]
    region = os.environ.get("AGENT_K_TEST_S3_REGION", "us-east-1")
    payload = (Path(__file__).resolve().parents[2] / "LICENSE").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    mission_id = uuid.uuid4()
    job_id = uuid.uuid4()
    first_worker = f"storage-acceptance-{uuid.uuid4().hex}"
    second_worker = f"storage-resume-{uuid.uuid4().hex}"
    store = PostgresRuntimeStore(os.environ["AGENT_K_TEST_DATABASE_URL"])
    await store.open()
    object_key: str | None = None
    uploaded_version: str | None = None
    mission_created = False
    active_lease: JobLease | None = None
    active_worker = first_worker
    try:
        await store.initialize()
        await store.create_mission(mission_id, first_worker, state={})
        mission_created = True
        await store.enqueue_job(
            JobEnqueueRequest(
                job_id=job_id,
                mission_id=mission_id,
                command=JobCommand(type="start", criteria={}),
                workspace=JobWorkspace(artifact_prefix=f"agent-k-acceptance/{mission_id}"),
                idempotency_key=f"storage-acceptance:{job_id}",
            )
        )
        claim = await store.claim_jobs(
            ClaimJobsRequest(worker_id=first_worker, capabilities=("cpu",), lease_seconds=300)
        )
        assert len(claim.jobs) == 1
        lease = claim.jobs[0]
        assert lease.job_id == job_id, "Use an isolated acceptance database without other jobs"
        active_lease = lease
        artifacts = S3ArtifactStore(store, bucket=bucket, region=region, url_ttl_seconds=300)
        signed = await artifacts.sign_upload(
            job_id,
            ArtifactUploadSpec(
                worker_id=first_worker,
                lease_id=lease.lease_id,
                role="diagnostic",
                name="LICENSE.txt",
                content_type="text/plain",
                size_bytes=len(payload),
                sha256=digest,
            ),
        )
        query = parse_qs(urlsplit(signed.url).query)
        assert query.get("X-Amz-Algorithm") == ["AWS4-HMAC-SHA256"]
        assert "x-amz-checksum-sha256" in query["X-Amz-SignedHeaders"][0].split(";")
        metadata = await store.get_artifact_for_verification(
            job_id, first_worker, lease.lease_id, signed.artifact_id
        )
        object_key = metadata["object_key"]
        assert object_key.startswith(f"agent-k-acceptance/{mission_id}/")
        with pytest.raises(KeyError):
            await artifacts.sign_download(job_id, first_worker, lease.lease_id, signed.artifact_id)

        async with httpx.AsyncClient(timeout=30) as client:
            rejected = await client.put(signed.url, headers=signed.headers, content=payload + b"\n")
            assert rejected.status_code == 400, "S3 must reject a body with the wrong SHA-256"
            assert "BadDigest" in rejected.text, "Require checksum rejection, not an auth error"
            response = await client.put(signed.url, headers=signed.headers, content=payload)
            assert response.status_code == 200, f"S3 PUT failed with HTTP {response.status_code}"
            uploaded_version = response.headers.get("x-amz-version-id")

        completion = await artifacts.complete_upload(
            job_id,
            signed.artifact_id,
            ArtifactCompleteSpec(
                worker_id=first_worker,
                lease_id=lease.lease_id,
                sha256=digest,
                size_bytes=len(payload),
            ),
        )
        assert completion.status == "available"
        snapshot = await store.save_snapshot(
            job_id,
            first_worker,
            lease.lease_id,
            MissionSnapshot(
                sequence=0,
                created_at=datetime.now(UTC),
                payload={"artifactId": str(signed.artifact_id), "sha256": digest},
            ),
        )
        await store.request_control(mission_id, ControlAction.PAUSE)
        await store.fail_job(
            job_id,
            JobFailureRequest(
                worker_id=first_worker,
                lease_id=lease.lease_id,
                idempotency_key=f"storage-pause:{lease.lease_id}",
                error=JobFailure(code="mission_pause", message="Acceptance pause", retryable=True),
                final_event_sequence=0,
            ),
        )
        await store.request_control(mission_id, ControlAction.CONTINUE)
        resumed = await store.claim_jobs(
            ClaimJobsRequest(worker_id=second_worker, capabilities=("cpu",), lease_seconds=300)
        )
        assert len(resumed.jobs) == 1
        new_lease = resumed.jobs[0]
        active_lease = new_lease
        active_worker = second_worker
        assert new_lease.command.checkpoint_id == snapshot.snapshot_id
        restored = await store.get_snapshot(
            job_id, second_worker, new_lease.lease_id, snapshot.snapshot_id
        )
        assert restored is not None and restored.payload["sha256"] == digest
        with pytest.raises(LeaseFenceError):
            await artifacts.sign_download(job_id, first_worker, lease.lease_id, signed.artifact_id)
        download = await artifacts.sign_download(
            job_id, second_worker, new_lease.lease_id, signed.artifact_id
        )
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(download.url, headers=download.headers)
            assert response.status_code == 200, f"S3 GET failed with HTTP {response.status_code}"
            assert len(response.content) == len(payload)
            assert hashlib.sha256(response.content).hexdigest() == digest
    finally:
        try:
            if object_key is not None:
                await _delete_acceptance_object(
                    bucket,
                    region,
                    object_key,
                    uploaded_version,
                )
        finally:
            try:
                if mission_created:
                    await store.request_control(mission_id, ControlAction.CANCEL)
                if active_lease is not None:
                    with contextlib.suppress(LeaseFenceError):
                        await store.fail_job(
                            job_id,
                            JobFailureRequest(
                                worker_id=active_worker,
                                lease_id=active_lease.lease_id,
                                idempotency_key=f"storage-cleanup:{active_lease.lease_id}",
                                error=JobFailure(
                                    code="acceptance_cleanup",
                                    message="Acceptance test resources are being removed",
                                    retryable=False,
                                ),
                                final_event_sequence=0,
                            ),
                        )
            finally:
                await store.close()
