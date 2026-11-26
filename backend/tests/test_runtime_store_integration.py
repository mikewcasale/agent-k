"""Real PostgreSQL contract tests for runtime leases and durable resume."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest

from agent_k.stores import (
    AppendEventsRequest,
    ClaimJobsRequest,
    ControlAction,
    HeartbeatRequest,
    JobCommand,
    JobCompletionRequest,
    JobEnqueueRequest,
    JobFailure,
    JobFailureRequest,
    JobWorkspace,
    LeaseFenceError,
    MissionRunResult,
    MissionSnapshot,
    PostgresRuntimeStore,
    WorkerEvent,
)

pytestmark = pytest.mark.skipif(
    "AGENT_K_TEST_DATABASE_URL" not in os.environ,
    reason="requires a real disposable PostgreSQL URL in AGENT_K_TEST_DATABASE_URL",
)


@pytest.fixture
async def runtime_store() -> PostgresRuntimeStore:
    store = PostgresRuntimeStore(os.environ["AGENT_K_TEST_DATABASE_URL"])
    await store.open()
    await store.initialize()
    yield store
    await store.close()


async def test_atomic_claim_and_pause_resume_checkpoint(
    runtime_store: PostgresRuntimeStore,
) -> None:
    mission_id = uuid.uuid4()
    job_id = uuid.uuid4()
    await runtime_store.create_mission(mission_id, "owner-1", state={"criteria": {}})
    await runtime_store.enqueue_job(
        JobEnqueueRequest(
            job_id=job_id,
            mission_id=mission_id,
            command=JobCommand(type="start", criteria={}),
            workspace=JobWorkspace(artifact_prefix=f"missions/{mission_id}"),
            idempotency_key=f"integration:{job_id}",
        )
    )
    requests = (
        ClaimJobsRequest(worker_id="worker-a", capabilities=("cpu",)),
        ClaimJobsRequest(worker_id="worker-b", capabilities=("cpu",)),
    )
    claims = await asyncio.gather(*(runtime_store.claim_jobs(request) for request in requests))
    leases = [lease for response in claims for lease in response.jobs]
    assert len(leases) == 1
    lease = leases[0]
    assert lease.event_sequence == 0
    worker_id = "worker-a" if claims[0].jobs else "worker-b"

    checkpoint_event = WorkerEvent(
        event_id=uuid.uuid5(job_id, "1"),
        sequence=1,
        type="checkpoint-created",
        timestamp=datetime.now(UTC),
        data={"phase": "discovery"},
    )
    await runtime_store.append_events(
        job_id,
        AppendEventsRequest(
            worker_id=worker_id,
            lease_id=lease.lease_id,
            events=(checkpoint_event,),
        ),
    )
    snapshot = await runtime_store.save_snapshot(
        job_id,
        worker_id,
        lease.lease_id,
        MissionSnapshot(
            sequence=1,
            created_at=datetime.now(UTC),
            payload={"workspaceRoot": "/work", "state": {"mission_id": str(mission_id)}},
        ),
    )
    await runtime_store.request_control(mission_id, ControlAction.PAUSE, reason="test pause")
    await runtime_store.append_events(
        job_id,
        AppendEventsRequest(
            worker_id=worker_id,
            lease_id=lease.lease_id,
            events=(
                WorkerEvent(
                    event_id=uuid.uuid5(job_id, "2"),
                    sequence=2,
                    type="state-delta",
                    timestamp=datetime.now(UTC),
                    data={"status": "paused"},
                ),
            ),
        ),
    )
    paused = await runtime_store.fail_job(
        job_id,
        JobFailureRequest(
            worker_id=worker_id,
            lease_id=lease.lease_id,
            idempotency_key=f"pause:{lease.lease_id}",
            error=JobFailure(code="mission_pause", message="paused", retryable=True),
            final_event_sequence=2,
        ),
    )
    assert paused.status == "queued"
    assert not (await runtime_store.claim_jobs(requests[0])).jobs

    await runtime_store.request_control(mission_id, ControlAction.CONTINUE)
    resumed = await runtime_store.claim_jobs(
        ClaimJobsRequest(worker_id="worker-c", capabilities=("cpu",))
    )
    assert len(resumed.jobs) == 1
    resumed_lease = resumed.jobs[0]
    assert resumed_lease.command.type == "resume"
    assert resumed_lease.command.checkpoint_id == snapshot.snapshot_id
    assert resumed_lease.event_sequence == 2
    restored = await runtime_store.get_snapshot(
        job_id,
        "worker-c",
        resumed_lease.lease_id,
        snapshot.snapshot_id,
    )
    assert restored is not None
    assert restored.payload["workspaceRoot"] == "/work"
    with pytest.raises(LeaseFenceError):
        await runtime_store.heartbeat(
            job_id,
            HeartbeatRequest(worker_id=worker_id, lease_id=lease.lease_id),
        )

    await runtime_store.append_events(
        job_id,
        AppendEventsRequest(
            worker_id="worker-c",
            lease_id=resumed_lease.lease_id,
            events=(
                WorkerEvent(
                    event_id=uuid.uuid5(job_id, "3"),
                    sequence=3,
                    type="mission-complete",
                    timestamp=datetime.now(UTC),
                    data={"success": True},
                ),
            ),
        ),
    )
    result = MissionRunResult(mission_id=mission_id, final_event_sequence=3)
    completed = await runtime_store.complete_job(
        job_id,
        JobCompletionRequest(
            worker_id="worker-c",
            lease_id=resumed_lease.lease_id,
            idempotency_key=f"complete:{resumed_lease.lease_id}",
            result=result,
            final_event_sequence=3,
        ),
    )
    assert completed.status == "completed"


async def test_terminal_failure_is_appended_atomically(
    runtime_store: PostgresRuntimeStore,
) -> None:
    mission_id = uuid.uuid4()
    job_id = uuid.uuid4()
    await runtime_store.create_mission(mission_id, "owner-2", state={"criteria": {}})
    await runtime_store.enqueue_job(
        JobEnqueueRequest(
            job_id=job_id,
            mission_id=mission_id,
            command=JobCommand(type="start", criteria={}),
            workspace=JobWorkspace(artifact_prefix=f"missions/{mission_id}"),
            idempotency_key=f"terminal-integration:{job_id}",
        )
    )
    claimed = await runtime_store.claim_jobs(
        ClaimJobsRequest(worker_id="worker-terminal", capabilities=("cpu",))
    )
    lease = claimed.jobs[0]
    failure = JobFailure(
        code="nonretryable",
        message="A terminal integration failure",
        retryable=False,
    )
    await runtime_store.append_events(
        job_id,
        AppendEventsRequest(
            worker_id="worker-terminal",
            lease_id=lease.lease_id,
            events=(
                WorkerEvent(
                    event_id=uuid.uuid5(job_id, "1"),
                    sequence=1,
                    type="error-occurred",
                    timestamp=datetime.now(UTC),
                    data={"message": failure.message},
                ),
            ),
        ),
    )
    response = await runtime_store.fail_job(
        job_id,
        JobFailureRequest(
            worker_id="worker-terminal",
            lease_id=lease.lease_id,
            idempotency_key=f"terminal:{lease.lease_id}",
            error=failure,
            final_event_sequence=1,
        ),
    )
    assert response.status == "failed"
    events = await runtime_store.list_events(mission_id)
    assert [event.type for event in events] == ["error-occurred", "mission-complete"]
    assert isinstance(events[-1].data, dict)
    assert events[-1].data["success"] is False
