"""Portable worker that executes fenced missions in fresh subprocesses.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import logfire
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .observability import configure_observability
from .runner import WorkerApiClient, WorkerApiError
from .sandbox import remove_owned_sandbox_containers, validate_sandbox_runtime
from .stores import (
    ControlAction,
    JobFailure,
    JobFailureRequest,
    JobLease,
    MissionRunResult,
    WorkerEvent,
)

__all__ = ("WorkerSettings", "main", "run_worker")

_MISSION_ENVIRONMENT = frozenset(
    {
        "AGENT_K_ALLOW_SUBMISSIONS",
        "AGENT_K_ENVIRONMENT",
        "AGENT_K_EVALUATION_PARALLELISM",
        "AGENT_K_EVALUATION_TIMEOUT_SECONDS",
        "AGENT_K_EVOLUTION_ITERATIONS",
        "AGENT_K_GPU_DEVICES",
        "AGENT_K_KUMO_CACHE",
        "AGENT_K_MEMORY_ROOT",
        "AGENT_K_MODEL",
        "AGENT_K_PHASE_TIMEOUT_SECONDS",
        "AGENT_K_SANDBOX_IMAGE",
        "AGENT_K_TRANSFORMER_CACHE",
        "AGENT_K_VISION_WEIGHTS",
        "ANTHROPIC_API_KEY",
        "DEVSTRAL_BASE_URL",
        "DEVSTRAL_API_KEY",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "HF_HOME",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "LANG",
        "LC_ALL",
        "LOGFIRE_TOKEN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TRANSFORMERS_CACHE",
        "XDG_RUNTIME_DIR",
    }
)
_MAX_RESULT_BYTES = 10 * 1024 * 1024
_MAX_PROCESS_LOG_BYTES = 256 * 1024


class WorkerSettings(BaseModel):
    """Validated worker process configuration sourced from environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_url: str
    worker_id: str = Field(min_length=1, max_length=200)
    worker_token: str = Field(min_length=16, repr=False)
    capabilities: tuple[str, ...] = ()
    work_root: Path
    sandbox_image: str
    gpu_devices: tuple[str, ...] = ()
    lease_seconds: int = Field(default=60, ge=30, le=3600)
    heartbeat_seconds: float = Field(default=15.0, ge=2.0, le=300.0)
    poll_seconds: float = Field(default=5.0, ge=0.5, le=60.0)
    control_grace_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    shutdown_grace_seconds: float = Field(default=60.0, ge=5.0, le=600.0)
    environment: str = "production"

    @field_validator("api_url")
    @classmethod
    def validate_api_url(cls, value: str) -> str:
        """Require an explicit HTTP worker API endpoint."""
        if not value.startswith(("http://", "https://")):
            raise ValueError("api_url must be HTTP or HTTPS")
        return value.rstrip("/")

    @field_validator("work_root")
    @classmethod
    def validate_work_root(cls, value: Path) -> Path:
        """Reject broad or implicit worker workspace roots."""
        resolved = value.expanduser().resolve()
        if resolved in {Path("/"), Path.home().resolve()}:
            raise ValueError("work_root must be a dedicated subdirectory")
        return resolved

    @model_validator(mode="after")
    def validate_heartbeat_interval(self) -> WorkerSettings:
        """Keep two heartbeat opportunities inside every configured lease."""
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("heartbeat_seconds must be less than half of lease_seconds")
        return self

    @classmethod
    def from_environment(cls) -> WorkerSettings:
        """Load worker settings without accepting storage or database credentials."""
        capabilities = tuple(
            item.strip()
            for item in os.environ.get("AGENT_K_WORKER_CAPABILITIES", "cpu").split(",")
            if item.strip()
        )
        gpu_value = json.loads(os.environ.get("AGENT_K_GPU_DEVICES", "[]"))
        if not isinstance(gpu_value, list) or any(not isinstance(item, str) for item in gpu_value):
            raise ValueError("AGENT_K_GPU_DEVICES must be a JSON array of device identifiers")
        return cls(
            api_url=os.environ["AGENT_K_API_URL"],
            worker_id=os.environ["AGENT_K_WORKER_ID"],
            worker_token=os.environ["AGENT_K_WORKER_TOKEN"],
            capabilities=capabilities,
            work_root=Path(os.environ["AGENT_K_WORK_ROOT"]),
            sandbox_image=os.environ["AGENT_K_SANDBOX_IMAGE"],
            gpu_devices=tuple(gpu_value),
            lease_seconds=int(os.environ.get("AGENT_K_LEASE_SECONDS", "60")),
            heartbeat_seconds=float(os.environ.get("AGENT_K_HEARTBEAT_SECONDS", "15")),
            poll_seconds=float(os.environ.get("AGENT_K_POLL_SECONDS", "5")),
            control_grace_seconds=float(os.environ.get("AGENT_K_CONTROL_GRACE_SECONDS", "300")),
            shutdown_grace_seconds=float(os.environ.get("AGENT_K_SHUTDOWN_GRACE_SECONDS", "60")),
            environment=os.environ.get("AGENT_K_ENVIRONMENT", "production"),
        )


class _LeaseMonitor:
    """Shared heartbeat outcome for one child process."""

    def __init__(self) -> None:
        self.stop = asyncio.Event()
        self.fenced = asyncio.Event()
        self.control_action: Literal["pause", "cancel"] | None = None
        self.control_reason: str | None = None
        self.control_requested_at: float | None = None


async def run_worker(settings: WorkerSettings) -> None:
    """Poll the authenticated queue and run one fenced mission at a time."""
    settings.work_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(settings.work_root, 0o700)
    lock_descriptor = _acquire_worker_lock(settings.work_root)
    try:
        await _run_locked_worker(settings)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


async def _run_locked_worker(settings: WorkerSettings) -> None:
    """Run one exclusive worker owner after reconciling its abandoned containers."""
    _validate_offline_resources()
    await validate_sandbox_runtime(
        settings.sandbox_image,
        gpu_devices=settings.gpu_devices,
    )
    worker_owner = _worker_owner(settings)
    await remove_owned_sandbox_containers(worker_owner)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(handled_signal, stopping.set)
    async with WorkerApiClient(
        settings.api_url,
        settings.worker_id,
        settings.worker_token,
    ) as client:
        while not stopping.is_set():
            try:
                claimed = await client.claim_jobs(
                    settings.capabilities,
                    lease_seconds=settings.lease_seconds,
                )
            except WorkerApiError as error:
                logfire.warn(
                    "Worker claim failed",
                    code=error.code,
                    retryable=error.retryable,
                    status_code=error.status_code,
                )
                await _interruptible_sleep(stopping, settings.poll_seconds)
                continue
            if not claimed.jobs:
                jitter = random.uniform(0.0, min(1.0, settings.poll_seconds / 4))
                await _interruptible_sleep(stopping, settings.poll_seconds + jitter)
                continue
            for lease in claimed.jobs:
                if stopping.is_set():
                    break
                await _run_lease(client, lease, settings, stopping, worker_owner)


async def _run_lease(
    client: WorkerApiClient,
    lease: JobLease,
    settings: WorkerSettings,
    stopping: asyncio.Event,
    worker_owner: str,
) -> None:
    workdir = settings.work_root / f"{lease.job_id}-{lease.attempt}-{lease.lease_id}"
    workdir.mkdir(mode=0o700)
    job_path = workdir / "job.json"
    result_path = workdir / "result.json"
    _write_private(job_path, lease.model_dump(mode="json", by_alias=True))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "agent_k.runner",
        "mission",
        "--job",
        str(job_path),
        "--output",
        str(result_path),
        cwd=workdir,
        env=_mission_environment(settings, lease, worker_owner),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdout_log = asyncio.create_task(_capture_process_log(process.stdout, workdir / "stdout.log"))
    stderr_log = asyncio.create_task(_capture_process_log(process.stderr, workdir / "stderr.log"))
    monitor = _LeaseMonitor()
    heartbeat = asyncio.create_task(_heartbeat_loop(client, lease, settings, monitor))
    wait = asyncio.create_task(process.wait())
    stop_wait = asyncio.create_task(stopping.wait())
    try:
        while process.returncode is None:
            done, _pending = await asyncio.wait(
                {wait, stop_wait},
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait in done:
                break
            if monitor.fenced.is_set():
                await _terminate_process(process, 5.0)
                return
            if stop_wait in done:
                await _terminate_process(process, settings.shutdown_grace_seconds)
                return
            if (
                monitor.control_requested_at is not None
                and loop_time() - monitor.control_requested_at > settings.control_grace_seconds
            ):
                await _terminate_process(process, 5.0)
                break
        await process.wait()
        if monitor.fenced.is_set():
            return
        await _commit_process_outcome(client, lease, process.returncode or 0, result_path, monitor)
    finally:
        monitor.stop.set()
        heartbeat.cancel()
        stop_wait.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        with contextlib.suppress(asyncio.CancelledError):
            await stop_wait
        await remove_owned_sandbox_containers(
            worker_owner,
            lease_owner=lease.lease_id,
        )
        await stdout_log
        await stderr_log
        shutil.rmtree(workdir)


async def _heartbeat_loop(
    client: WorkerApiClient,
    lease: JobLease,
    settings: WorkerSettings,
    monitor: _LeaseMonitor,
) -> None:
    while not monitor.stop.is_set():
        try:
            response = await client.heartbeat(lease)
        except WorkerApiError as error:
            if error.code == "stale_lease" or not error.retryable:
                monitor.fenced.set()
                return
            logfire.warn("Worker heartbeat failed", code=error.code, job_id=str(lease.job_id))
        else:
            if response.control.action in {ControlAction.PAUSE, ControlAction.CANCEL}:
                if monitor.control_requested_at is None:
                    monitor.control_requested_at = loop_time()
                monitor.control_action = (
                    "pause" if response.control.action is ControlAction.PAUSE else "cancel"
                )
                monitor.control_reason = response.control.reason
        try:
            await asyncio.wait_for(monitor.stop.wait(), timeout=settings.heartbeat_seconds)
        except TimeoutError:
            pass


async def _commit_process_outcome(
    client: WorkerApiClient,
    lease: JobLease,
    exit_code: int,
    result_path: Path,
    monitor: _LeaseMonitor,
) -> None:
    outcome: dict[str, object] | None = None
    if result_path.is_file() and result_path.stat().st_size <= _MAX_RESULT_BYTES:
        with contextlib.suppress(json.JSONDecodeError, OSError):
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                outcome = parsed
    if exit_code == 0 and outcome is not None and outcome.get("status") == "completed":
        try:
            result = MissionRunResult.model_validate(outcome.get("result"))
        except ValueError:
            pass
        else:
            await client.complete_job(
                lease,
                result,
                idempotency_key=_idempotency_key(lease, "complete"),
            )
            return
    interrupted_sequence = outcome.get("final_event_sequence") if outcome is not None else None
    if (
        outcome is not None
        and outcome.get("status") == "interrupted"
        and isinstance(interrupted_sequence, int)
    ):
        sequence = interrupted_sequence
        action: Literal["pause", "cancel"] = (
            "cancel" if outcome.get("action") == "cancel" else "pause"
        )
        failure = JobFailure(
            code=f"mission_{action}",
            message=f"Mission acknowledged {action} after durable checkpoint",
            retryable=action == "pause",
            category="control",
        )
    else:
        control_action = monitor.control_action
        failure = JobFailure(
            code="mission_process_failed",
            message="Mission subprocess exited without a valid terminal result",
            retryable=control_action != "cancel",
            category="worker",
            details={"exitCode": exit_code},
        )
        sequence = await _append_failure_event(client, lease, failure)
    request = JobFailureRequest(
        worker_id=client.worker_id,
        lease_id=lease.lease_id,
        idempotency_key=_idempotency_key(lease, f"fail-{sequence}"),
        error=failure,
        final_event_sequence=sequence,
    )
    await client.fail_job(lease, request)


async def _append_failure_event(
    client: WorkerApiClient,
    lease: JobLease,
    failure: JobFailure,
) -> int:
    sequence = lease.event_sequence + 1
    for _attempt in range(4):
        event = WorkerEvent(
            event_id=uuid.uuid5(lease.job_id, str(sequence)),
            sequence=sequence,
            type="error-occurred",
            timestamp=datetime.now(UTC),
            data={
                "id": str(uuid.uuid5(lease.job_id, f"failure:{sequence}")),
                "timestamp": datetime.now(UTC).isoformat(),
                "category": "recoverable" if failure.retryable else "fatal",
                "errorType": failure.code,
                "message": failure.message,
                "context": "mission subprocess",
                "recoveryStrategy": "retry" if failure.retryable else "abort",
                "recoveryAttempts": 0,
                "resolved": False,
            },
        )
        try:
            await client.append_events(lease, (event,))
            return sequence
        except WorkerApiError as error:
            expected = (error.details or {}).get("expectedSequence")
            if error.status_code != 409 or not isinstance(expected, int):
                raise
            sequence = expected
    raise RuntimeError("could not append a terminal event after sequence reconciliation")


async def _terminate_process(process: asyncio.subprocess.Process, grace_seconds: float) -> None:
    if process.returncode is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def _capture_process_log(
    stream: asyncio.StreamReader | None,
    destination: Path,
) -> None:
    if stream is None:
        return
    written = 0
    with destination.open("xb") as target:
        os.chmod(destination, 0o600)
        while chunk := await stream.read(8192):
            if written >= _MAX_PROCESS_LOG_BYTES:
                continue
            retained = chunk[: _MAX_PROCESS_LOG_BYTES - written]
            target.write(retained)
            written += len(retained)


def _mission_environment(
    settings: WorkerSettings,
    lease: JobLease,
    worker_owner: str,
) -> dict[str, str]:
    environment = {name: os.environ[name] for name in _MISSION_ENVIRONMENT if name in os.environ}
    environment.update(
        {
            "AGENT_K_API_URL": settings.api_url,
            "AGENT_K_SANDBOX_LEASE_ID": str(lease.lease_id),
            "AGENT_K_SANDBOX_WORKER_OWNER": worker_owner,
            "AGENT_K_WORKER_ID": settings.worker_id,
            "AGENT_K_WORKER_TOKEN": settings.worker_token,
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _worker_owner(settings: WorkerSettings) -> str:
    value = f"{settings.worker_id}\0{settings.work_root}"
    return hashlib.sha256(value.encode()).hexdigest()


def _acquire_worker_lock(work_root: Path) -> int:
    lock_path = work_root / ".worker.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.chmod(lock_path, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"worker work root is already in use: {work_root}") from error
    return descriptor


def _validate_offline_resources() -> None:
    for name in (
        "AGENT_K_KUMO_CACHE",
        "AGENT_K_TRANSFORMER_CACHE",
        "AGENT_K_VISION_WEIGHTS",
    ):
        value = os.environ.get(name)
        if value and not Path(value).expanduser().resolve().exists():
            raise ValueError(f"configured offline resource does not exist: {name}")


def _write_private(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as target:
        os.chmod(path, 0o600)
        json.dump(payload, target, separators=(",", ":"), sort_keys=True)
        target.write("\n")


def _idempotency_key(lease: JobLease, operation: str) -> str:
    value = f"{lease.job_id}:{lease.attempt}:{lease.lease_id}:{operation}"
    return hashlib.sha256(value.encode()).hexdigest()


def loop_time() -> float:
    """Return the running loop's monotonic clock for control deadlines."""
    return asyncio.get_running_loop().time()


async def _interruptible_sleep(stopping: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stopping.wait(), timeout=seconds)
    except TimeoutError:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_k.worker")
    parser.add_argument("run", nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the portable worker using its validated environment contract."""
    _parser().parse_args(argv)
    settings = WorkerSettings.from_environment()
    configure_observability(service_name="agent-k-worker", environment=settings.environment)
    asyncio.run(run_worker(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
