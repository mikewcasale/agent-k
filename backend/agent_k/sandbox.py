"""Rootless Docker isolation for untrusted candidate programs.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import uuid
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = (
    "SandboxError",
    "SandboxLimits",
    "SandboxRequest",
    "SandboxResult",
    "remove_owned_sandbox_containers",
    "run_candidate_sandbox",
    "run_candidate_sandbox_sync",
    "validate_sandbox_runtime",
)

_DIGESTED_IMAGE = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
_OWNER_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ASSET_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_GPU_DEVICE = re.compile(r"^(?:[0-9]+|GPU-[A-Fa-f0-9-]+|MIG-GPU-[A-Fa-f0-9-]+/[0-9]+/[0-9]+)$")
_SAFE_ENVIRONMENT = frozenset(
    {
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
    }
)
_SECRET_NAME = re.compile(
    r"(?:aws|credential|database|kaggle|key|password|secret|token)",
    re.IGNORECASE,
)
_OUTPUT_CAPTURE_LIMIT = 64 * 1024
_LEASE_OWNER_ENV = "AGENT_K_SANDBOX_LEASE_ID"
_WORKER_OWNER_ENV = "AGENT_K_SANDBOX_WORKER_OWNER"
_DOCKER_RESOURCE_INFO_FORMAT = (
    '{"SecurityOptions":{{json .SecurityOptions}},'
    '"CgroupDriver":{{json .CgroupDriver}},'
    '"CgroupVersion":{{json .CgroupVersion}},'
    '"CpuCfsPeriod":{{json .CPUCfsPeriod}},'
    '"CpuCfsQuota":{{json .CPUCfsQuota}},'
    '"MemoryLimit":{{json .MemoryLimit}},'
    '"PidsLimit":{{json .PidsLimit}}}'
)
_REQUIRED_CGROUP_CONTROLLERS = (
    ("cpu", ("CpuCfsPeriod", "CpuCfsQuota")),
    ("memory", ("MemoryLimit",)),
    ("pids", ("PidsLimit",)),
)


def _lease_owner_from_environment() -> uuid.UUID:
    value = os.environ.get(_LEASE_OWNER_ENV)
    if value is None:
        raise ValueError("AGENT_K_SANDBOX_LEASE_ID is required for candidate execution")
    try:
        return uuid.UUID(value)
    except ValueError as error:
        raise ValueError("AGENT_K_SANDBOX_LEASE_ID must be a UUID") from error


def _worker_owner_from_environment() -> str:
    return os.environ.get(_WORKER_OWNER_ENV, "")


class SandboxError(RuntimeError):
    """Raised when candidate isolation or execution fails closed."""


class SandboxLimits(BaseModel):
    """Hard resource limits applied to one candidate container."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: float = Field(default=1.0, gt=0.0, le=64.0)
    memory_bytes: int = Field(default=2 * 1024**3, ge=256 * 1024**2)
    pids: int = Field(default=128, ge=16, le=4096)
    output_bytes: int = Field(default=512 * 1024**2, ge=1024**2)
    temporary_bytes: int = Field(default=512 * 1024**2, ge=16 * 1024**2)
    wall_seconds: float = Field(default=900.0, gt=0.0, le=86_400.0)


class SandboxRequest(BaseModel):
    """Inputs for one secretless candidate execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    image: str
    input_root: Path
    output_root: Path
    candidate_program: Path | None = None
    offline_assets: dict[str, Path] = Field(default_factory=dict)
    gpu_devices: tuple[str, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)
    limits: SandboxLimits = Field(default_factory=SandboxLimits)
    lease_owner: uuid.UUID = Field(default_factory=_lease_owner_from_environment)
    worker_owner: str = Field(default_factory=_worker_owner_from_environment)

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        """Require an immutable OCI digest rather than a mutable tag."""
        if not _DIGESTED_IMAGE.fullmatch(value):
            raise ValueError("image must be pinned with an immutable sha256 digest")
        return value

    @field_validator("offline_assets")
    @classmethod
    def validate_assets(cls, value: dict[str, Path]) -> dict[str, Path]:
        """Restrict asset names to safe single path components."""
        if any(not _SAFE_ASSET_NAME.fullmatch(name) for name in value):
            raise ValueError("offline asset names must be safe lowercase path components")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        """Permit only non-secret runtime tuning variables."""
        for name, setting in value.items():
            if name not in _SAFE_ENVIRONMENT or _SECRET_NAME.search(name):
                raise ValueError(f"environment variable is not candidate-safe: {name}")
            if "\x00" in setting or "\n" in setting:
                raise ValueError(f"environment variable contains unsafe bytes: {name}")
        return value

    @field_validator("gpu_devices")
    @classmethod
    def validate_gpu_devices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Allow only explicit NVIDIA indices, GPU UUIDs, or MIG device UUIDs."""
        if len(value) != len(set(value)) or any(not _GPU_DEVICE.fullmatch(item) for item in value):
            raise ValueError("gpu_devices must contain unique explicit NVIDIA device identifiers")
        return value

    @field_validator("worker_owner")
    @classmethod
    def validate_worker_owner(cls, value: str) -> str:
        """Require the trusted worker's stable, non-secret ownership token."""
        if not _OWNER_TOKEN.fullmatch(value):
            raise ValueError("worker_owner must be a 64-character lowercase hexadecimal token")
        return value

    @model_validator(mode="after")
    def validate_paths(self) -> SandboxRequest:
        """Require distinct, existing host paths for all mounts."""
        input_root = self.input_root.resolve(strict=True)
        output_root = self.output_root.resolve(strict=True)
        if not input_root.is_dir() or not output_root.is_dir():
            raise ValueError("input_root and output_root must be directories")
        if input_root == output_root or input_root in output_root.parents:
            raise ValueError("output_root must not be inside input_root")
        if output_root in input_root.parents:
            raise ValueError("input_root must not be inside output_root")
        if self.candidate_program is not None:
            program = self.candidate_program.resolve(strict=True)
            if not program.is_file() or program.suffix != ".py":
                raise ValueError("candidate_program must be an existing Python file")
        for asset in self.offline_assets.values():
            if not asset.resolve(strict=True).exists():
                raise ValueError(f"offline asset does not exist: {asset}")
        return self


class SandboxResult(BaseModel):
    """Auditable process outcome from an isolated candidate container."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    container_name: str
    exit_code: int
    timed_out: bool = False
    output_limit_exceeded: bool = False
    oom_killed: bool = False
    stdout: str = ""
    stderr: str = ""


async def run_candidate_sandbox(request: SandboxRequest) -> SandboxResult:
    """Run one baseline or evolved candidate in a fresh rootless Docker container."""
    docker = shutil.which("docker")
    if docker is None:
        raise SandboxError("docker CLI is required for candidate execution")
    await _require_rootless_docker(docker)
    _clear_output_root(request.output_root)
    container_name = f"agent-k-candidate-{uuid.uuid4().hex}"
    command = _docker_create_command(docker, container_name, request)
    created = await _run_control_command(command)
    if created[0] != 0:
        raise SandboxError(_control_error("docker create", created[2]))

    process: asyncio.subprocess.Process | None = None
    timed_out = False
    output_limit_exceeded = False
    try:
        process = await asyncio.create_subprocess_exec(
            docker,
            "start",
            "--attach",
            container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_task = asyncio.create_task(_read_bounded(process.stdout))
        stderr_task = asyncio.create_task(_read_bounded(process.stderr))
        monitor_task = asyncio.create_task(_monitor_output(request.output_root, request.limits))
        wait_task = asyncio.create_task(process.wait())
        done, _pending = await asyncio.wait(
            {wait_task, monitor_task},
            timeout=request.limits.wall_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            timed_out = True
            await _kill_container(docker, container_name)
            await process.wait()
        elif monitor_task in done and monitor_task.result():
            output_limit_exceeded = True
            await _kill_container(docker, container_name)
            if process.returncode is None:
                await process.wait()
        if not wait_task.done():
            wait_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wait_task
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task
        if (
            await asyncio.to_thread(_directory_size, request.output_root)
            > request.limits.output_bytes
        ):
            output_limit_exceeded = True
        stdout = await stdout_task
        stderr = await stderr_task
        inspection = await _inspect_container(docker, container_name)
        return SandboxResult(
            container_name=container_name,
            exit_code=process.returncode if process.returncode is not None else 125,
            timed_out=timed_out,
            output_limit_exceeded=output_limit_exceeded,
            oom_killed=inspection,
            stdout=stdout,
            stderr=stderr,
        )
    except asyncio.CancelledError:
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
        await _kill_container(docker, container_name)
        raise
    finally:
        await _remove_container(docker, container_name)


def run_candidate_sandbox_sync(request: SandboxRequest) -> SandboxResult:
    """Run an isolated candidate from a synchronous evaluator process."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_candidate_sandbox(request))
    raise SandboxError("sync sandbox helper cannot run inside an active event loop")


async def validate_sandbox_runtime(
    image: str,
    *,
    gpu_devices: tuple[str, ...] = (),
) -> None:
    """Fail worker startup unless its rootless daemon and pinned image are ready."""
    if not _DIGESTED_IMAGE.fullmatch(image):
        raise SandboxError("sandbox image must be pinned with an immutable sha256 digest")
    docker = shutil.which("docker")
    if docker is None:
        raise SandboxError("docker CLI is required for candidate execution")
    await _require_rootless_docker(docker)
    code, _stdout, stderr = await _run_control_command([docker, "image", "inspect", image])
    if code != 0:
        raise SandboxError(_control_error("docker image inspect", stderr))
    if gpu_devices:
        if len(gpu_devices) != len(set(gpu_devices)) or any(
            not _GPU_DEVICE.fullmatch(item) for item in gpu_devices
        ):
            raise SandboxError("GPU devices must be unique explicit NVIDIA identifiers")
        code, runtimes, stderr = await _run_control_command(
            [docker, "info", "--format", "{{json .Runtimes}}"]
        )
        if code != 0:
            raise SandboxError(_control_error("docker runtime inspection", stderr))
        if "nvidia" not in runtimes.lower():
            raise SandboxError("configured GPU devices require the NVIDIA container runtime")


def _docker_create_command(
    docker: str,
    container_name: str,
    request: SandboxRequest,
) -> list[str]:
    limits = request.limits
    command = [
        docker,
        "create",
        "--name",
        container_name,
        "--label",
        "agent-k.managed=true",
        "--label",
        f"agent-k.worker-owner={request.worker_owner}",
        "--label",
        f"agent-k.lease-owner={request.lease_owner}",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "0:0",
        "--pids-limit",
        str(limits.pids),
        "--memory",
        str(limits.memory_bytes),
        "--memory-swap",
        str(limits.memory_bytes),
        "--cpus",
        str(limits.cpus),
        "--ulimit",
        f"fsize={limits.output_bytes}:{limits.output_bytes}",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={limits.temporary_bytes}",
        "--mount",
        f"type=bind,src={request.input_root.resolve()},dst=/run/input,readonly",
        "--mount",
        f"type=bind,src={request.output_root.resolve()},dst=/run/output",
        "--workdir",
        "/run/output",
    ]
    if request.candidate_program is not None:
        program = request.candidate_program.resolve()
        command.extend(["--mount", f"type=bind,src={program},dst=/run/input/candidate.py,readonly"])
    for name, path in sorted(request.offline_assets.items()):
        destination = PurePosixPath("/run/assets") / name
        command.extend(["--mount", f"type=bind,src={path.resolve()},dst={destination},readonly"])
    environment = {"PYTHONHASHSEED": "0", **request.environment}
    for name, value in sorted(environment.items()):
        command.extend(["--env", f"{name}={value}"])
    if request.gpu_devices:
        command.extend(["--gpus", f'"device={",".join(request.gpu_devices)}"'])
    command.append(request.image)
    if request.candidate_program is None:
        command.extend(
            [
                "python",
                "-I",
                "-m",
                "agent_k.runner",
                "candidate",
                "--spec",
                "/run/input/spec.json",
                "--output",
                "/run/output/result.json",
            ]
        )
    else:
        command.extend(
            [
                "python",
                "-I",
                "/run/input/candidate.py",
                "--spec",
                "/run/input/spec.json",
                "--output",
                "/run/output/result.json",
            ]
        )
    return command


def _validate_rootless_docker_info(output: str) -> None:
    """Require the Docker Engine capabilities used by every sandbox limit."""
    try:
        info: dict[str, object] = json.loads(output)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SandboxError("docker info returned invalid capability metadata") from exc
    if not isinstance(info, dict):
        raise SandboxError("docker info returned invalid capability metadata")
    security_options = info.get("SecurityOptions")
    if not isinstance(security_options, list) or "name=rootless" not in security_options:
        raise SandboxError("candidate execution requires a rootless Docker daemon")
    if info.get("CgroupVersion") != "2":
        raise SandboxError("rootless Docker requires cgroup v2 for enforceable resource limits")
    if info.get("CgroupDriver") != "systemd":
        raise SandboxError(
            "rootless Docker requires the systemd cgroup driver for delegated resource limits"
        )
    missing = [
        controller
        for controller, fields in _REQUIRED_CGROUP_CONTROLLERS
        if any(info.get(field) is not True for field in fields)
    ]
    if missing:
        names = ", ".join(missing)
        raise SandboxError(
            f"rootless Docker lacks delegated cgroup controllers required by sandbox limits: {names}"
        )


async def _require_rootless_docker(docker: str) -> None:
    code, stdout, stderr = await _run_control_command(
        [docker, "info", "--format", _DOCKER_RESOURCE_INFO_FORMAT]
    )
    if code != 0:
        raise SandboxError(_control_error("docker info", stderr))
    _validate_rootless_docker_info(stdout)


async def remove_owned_sandbox_containers(
    worker_owner: str,
    *,
    lease_owner: uuid.UUID | None = None,
) -> None:
    """Force-remove only AGENT-K containers owned by one locked worker workspace."""
    if not _OWNER_TOKEN.fullmatch(worker_owner):
        raise SandboxError("worker sandbox ownership token is invalid")
    docker = shutil.which("docker")
    if docker is None:
        raise SandboxError("docker CLI is required for candidate cleanup")
    command = _owned_container_query(docker, worker_owner, lease_owner)
    code, stdout, stderr = await _run_control_command(command)
    if code != 0:
        raise SandboxError(_control_error("docker owned-container listing", stderr))
    for container_id in stdout.splitlines():
        code, _stdout, stderr = await _run_control_command([docker, "rm", "--force", container_id])
        if code != 0:
            raise SandboxError(_control_error("docker owned-container removal", stderr))


def _owned_container_query(
    docker: str,
    worker_owner: str,
    lease_owner: uuid.UUID | None,
) -> list[str]:
    command = [
        docker,
        "ps",
        "--all",
        "--quiet",
        "--filter",
        "label=agent-k.managed=true",
        "--filter",
        f"label=agent-k.worker-owner={worker_owner}",
    ]
    if lease_owner is not None:
        command.extend(["--filter", f"label=agent-k.lease-owner={lease_owner}"])
    return command


async def _run_control_command(command: list[str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return (
        process.returncode or 0,
        stdout.decode(errors="replace")[-_OUTPUT_CAPTURE_LIMIT:],
        stderr.decode(errors="replace")[-_OUTPUT_CAPTURE_LIMIT:],
    )


async def _read_bounded(stream: asyncio.StreamReader | None) -> str:
    if stream is None:
        return ""
    captured = bytearray()
    while chunk := await stream.read(8192):
        if len(captured) < _OUTPUT_CAPTURE_LIMIT:
            remaining = _OUTPUT_CAPTURE_LIMIT - len(captured)
            captured.extend(chunk[:remaining])
    return captured.decode(errors="replace")


async def _monitor_output(output_root: Path, limits: SandboxLimits) -> bool:
    while True:
        if await asyncio.to_thread(_directory_size, output_root) > limits.output_bytes:
            return True
        await asyncio.sleep(0.25)


def _directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        with contextlib.suppress(OSError):
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
    return total


def _clear_output_root(path: Path) -> None:
    for entry in path.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


async def _inspect_container(docker: str, name: str) -> bool:
    code, stdout, _stderr = await _run_control_command(
        [docker, "inspect", "--format", "{{.State.OOMKilled}}", name]
    )
    return code == 0 and stdout.strip().lower() == "true"


async def _kill_container(docker: str, name: str) -> None:
    await _run_control_command([docker, "kill", name])


async def _remove_container(docker: str, name: str) -> None:
    await _run_control_command([docker, "rm", "--force", name])


def _control_error(operation: str, stderr: str) -> str:
    detail = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
    return f"{operation} failed: {detail[:500]}"
