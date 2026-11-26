"""Explicit live gates and static contracts for rootless Docker and legacy sources.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import fcntl
import json
import os
import shutil
import uuid
from pathlib import Path

import pandas as pd
import pytest
from pydantic import JsonValue, ValidationError
from sklearn.datasets import load_breast_cancer

from agent_k.datasets import discover_dataset
from agent_k.evaluator import (
    CandidateEngine,
    CandidateResult,
    CandidateSpec,
    CandidateStatus,
    EvaluationRequest,
    EvaluationResult,
    PreparedSplit,
    build_candidate_spec,
    evaluate_candidate,
    materialize_trusted_splits,
)
from agent_k.metrics import MetricId
from agent_k.migration import inspect_legacy_root
from agent_k.sandbox import (
    _DOCKER_RESOURCE_INFO_FORMAT,
    SandboxError,
    SandboxLimits,
    SandboxRequest,
    SandboxResult,
    _docker_create_command,
    _owned_container_query,
    _run_control_command,
    _validate_rootless_docker_info,
    remove_owned_sandbox_containers,
    run_candidate_sandbox,
    validate_sandbox_runtime,
)
from agent_k.worker import _acquire_worker_lock

__all__ = ()

_DOCUMENTED_DOCKER_INFO: dict[str, object] = {
    "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"],
    "CgroupDriver": "systemd",
    "CgroupVersion": "2",
    "CpuCfsPeriod": True,
    "CpuCfsQuota": True,
    "MemoryLimit": True,
    "PidsLimit": True,
}
_LEASE_OWNER = uuid.UUID("00000000-0000-4000-8000-000000000001")
_WORKER_OWNER = "0" * 64
_LIVE_WORKER_OWNER = "1" * 64


def _docker_info(**overrides: object) -> str:
    values = {**_DOCUMENTED_DOCKER_INFO, **overrides}
    return json.dumps(values)


def _configured_candidate_image() -> str:
    image = os.environ.get("AGENT_K_TEST_CANDIDATE_IMAGE")
    if image is None:
        pytest.skip("requires an exact current-code image in AGENT_K_TEST_CANDIDATE_IMAGE")
    return image


def _configured_gpu_devices() -> tuple[str, ...]:
    devices = tuple(filter(None, os.environ.get("AGENT_K_TEST_GPU_DEVICES", "").split(",")))
    if not devices:
        pytest.skip("requires explicit NVIDIA devices in AGENT_K_TEST_GPU_DEVICES")
    return devices


def _configured_kumo_assets() -> Path:
    configured = os.environ.get("AGENT_K_KUMO_ASSETS")
    if configured is None:
        pytest.skip("requires approved fixed Kumo weights in AGENT_K_KUMO_ASSETS")
    source = Path(configured).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("AGENT_K_KUMO_ASSETS must be a provisioned asset directory")
    return source


def _prepare_real_tabular_split(tmp_path: Path) -> PreparedSplit:
    workspace = tmp_path / "live-acceptance"
    workspace.mkdir(mode=0o700)
    source_root = workspace / "source"
    source_root.mkdir(mode=0o700)
    frame = load_breast_cancer(as_frame=True).frame
    frame.insert(0, "row_id", [f"breast-cancer-{index}" for index in range(len(frame))])
    frame.to_csv(source_root / "train.csv", index=False)
    frame.iloc[:64].drop(columns="target").to_csv(source_root / "test.csv", index=False)
    manifest = discover_dataset(source_root, metric_id=MetricId.AUC.value)
    splits = materialize_trusted_splits(
        manifest,
        source_root,
        workspace / "candidate",
        workspace / "trusted",
    )
    split = splits.acceptance
    input_root = Path(split.candidate_input_root)
    labels_path = Path(split.trusted_labels_path)
    assert not labels_path.is_relative_to(input_root)
    assert "target" not in pd.read_csv(input_root / split.manifest.predict.path).columns
    return split


def _write_candidate_spec(
    split: PreparedSplit,
    *,
    candidate_id: str,
    engine: CandidateEngine,
    hyperparameters: dict[str, JsonValue],
    offline_assets: dict[str, str] | None = None,
) -> CandidateSpec:
    spec = build_candidate_spec(
        split,
        candidate_id=candidate_id,
        engine=engine,
        metric_id=MetricId.AUC,
        hyperparameters=hyperparameters,
        offline_assets=offline_assets,
    )
    spec_path = Path(split.candidate_input_root) / "spec.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    os.chmod(spec_path, 0o600)
    return spec


async def _execute_and_score(
    split: PreparedSplit,
    spec: CandidateSpec,
    output_root: Path,
    *,
    image: str,
    lease_owner: uuid.UUID,
    gpu_devices: tuple[str, ...] = (),
    limits: SandboxLimits | None = None,
) -> tuple[CandidateResult, EvaluationResult]:
    output_root.mkdir(mode=0o700)
    outcome = await run_candidate_sandbox(
        SandboxRequest(
            image=image,
            input_root=Path(split.candidate_input_root),
            output_root=output_root,
            gpu_devices=gpu_devices,
            environment={"OMP_NUM_THREADS": "1"},
            limits=limits or SandboxLimits(wall_seconds=300.0),
            lease_owner=lease_owner,
            worker_owner=_LIVE_WORKER_OWNER,
        )
    )
    assert outcome.exit_code == 0, outcome.stderr[-2_000:]
    assert not outcome.timed_out
    assert not outcome.output_limit_exceeded
    assert not outcome.oom_killed
    result = CandidateResult.model_validate_json(
        (output_root / "result.json").read_text(encoding="utf-8")
    )
    assert result.status is CandidateStatus.SUCCEEDED
    assert result.candidate_id == spec.candidate_id
    evaluation = evaluate_candidate(
        EvaluationRequest(
            candidate_result=result,
            metric_id=spec.metric_id,
            prediction_root=str(output_root),
            trusted_labels_path=split.trusted_labels_path,
            id_columns=spec.id_columns,
            target_columns=spec.target_columns,
            class_labels=spec.class_labels,
        )
    )
    return result, evaluation


async def _assert_no_owned_containers(lease_owner: uuid.UUID) -> None:
    docker = shutil.which("docker")
    assert docker is not None
    code, stdout, stderr = await _run_control_command(
        _owned_container_query(docker, _LIVE_WORKER_OWNER, lease_owner)
    )
    assert code == 0, stderr[-1_000:]
    assert not stdout.strip()


async def _wait_for_owned_container(
    task: asyncio.Task[SandboxResult],
    lease_owner: uuid.UUID,
) -> tuple[str, str]:
    docker = shutil.which("docker")
    assert docker is not None
    deadline = asyncio.get_running_loop().time() + 15.0
    query = _owned_container_query(docker, _LIVE_WORKER_OWNER, lease_owner)
    while asyncio.get_running_loop().time() < deadline:
        code, stdout, stderr = await _run_control_command(query)
        assert code == 0, stderr[-1_000:]
        container_ids = stdout.splitlines()
        if container_ids:
            assert len(container_ids) == 1
            code, running, stderr = await _run_control_command(
                [docker, "inspect", "--format", "{{.State.Running}}", container_ids[0]]
            )
            assert code == 0, stderr[-1_000:]
            if running.strip() == "true":
                return docker, container_ids[0]
        if task.done():
            pytest.fail("candidate exited before fenced-cleanup acceptance could observe it")
        await asyncio.sleep(0.05)
    pytest.fail("candidate container did not become observable by its ownership labels")


def test_rootless_resource_info_accepts_documented_engine_capabilities() -> None:
    """Exercise the exact capability fields returned by Docker Engine GET /info."""
    assert "{{json .CPUCfsPeriod}}" in _DOCKER_RESOURCE_INFO_FORMAT
    assert "{{json .CPUCfsQuota}}" in _DOCKER_RESOURCE_INFO_FORMAT
    _validate_rootless_docker_info(_docker_info())


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"SecurityOptions": []}, "rootless Docker daemon"),
        ({"CgroupVersion": "1"}, "cgroup v2"),
        ({"CgroupDriver": "none"}, "systemd cgroup driver"),
        ({"CpuCfsQuota": False}, "cpu"),
        ({"MemoryLimit": False}, "memory"),
        ({"PidsLimit": False}, "pids"),
    ),
)
def test_rootless_resource_info_rejects_unenforceable_limits(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SandboxError, match=message):
        _validate_rootless_docker_info(_docker_info(**overrides))


def test_sandbox_contract_rejects_mutable_images_and_secrets(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    with pytest.raises(ValidationError, match="immutable sha256"):
        SandboxRequest(
            image="agent-k:latest",
            input_root=input_root,
            output_root=output_root,
            lease_owner=_LEASE_OWNER,
            worker_owner=_WORKER_OWNER,
        )
    with pytest.raises(ValidationError, match="not candidate-safe"):
        SandboxRequest(
            image=f"agent-k@sha256:{'0' * 64}",
            input_root=input_root,
            output_root=output_root,
            environment={"OPENAI_API_KEY": "not-forwarded"},
            lease_owner=_LEASE_OWNER,
            worker_owner=_WORKER_OWNER,
        )


def test_sandbox_contract_accepts_only_explicit_gpu_devices(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    request = SandboxRequest(
        image=f"agent-k@sha256:{'0' * 64}",
        input_root=input_root,
        output_root=output_root,
        gpu_devices=("0", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
        lease_owner=_LEASE_OWNER,
        worker_owner=_WORKER_OWNER,
    )
    assert request.gpu_devices[0] == "0"
    with pytest.raises(ValidationError, match="explicit NVIDIA"):
        SandboxRequest(
            image=request.image,
            input_root=input_root,
            output_root=output_root,
            gpu_devices=("all",),
            lease_owner=_LEASE_OWNER,
            worker_owner=_WORKER_OWNER,
        )


def test_sandbox_commands_are_scoped_to_worker_and_lease(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    request = SandboxRequest(
        image=f"agent-k@sha256:{'0' * 64}",
        input_root=input_root,
        output_root=output_root,
        lease_owner=_LEASE_OWNER,
        worker_owner=_WORKER_OWNER,
    )

    create = _docker_create_command("/usr/bin/docker", "candidate", request)
    assert "agent-k.managed=true" in create
    assert f"agent-k.worker-owner={_WORKER_OWNER}" in create
    assert f"agent-k.lease-owner={_LEASE_OWNER}" in create

    query = _owned_container_query("/usr/bin/docker", _WORKER_OWNER, _LEASE_OWNER)
    assert query[:4] == ["/usr/bin/docker", "ps", "--all", "--quiet"]
    assert "label=agent-k.managed=true" in query
    assert f"label=agent-k.worker-owner={_WORKER_OWNER}" in query
    assert f"label=agent-k.lease-owner={_LEASE_OWNER}" in query
    assert "prune" not in query


def test_worker_work_root_lock_is_exclusive(tmp_path: Path) -> None:
    descriptor = _acquire_worker_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            _acquire_worker_lock(tmp_path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


async def test_configured_image_executes_real_lightgbm_with_trusted_scoring(
    tmp_path: Path,
) -> None:
    image = _configured_candidate_image()
    split = _prepare_real_tabular_split(tmp_path)
    spec = _write_candidate_spec(
        split,
        candidate_id="live-lightgbm-breast-cancer",
        engine=CandidateEngine.LIGHTGBM,
        hyperparameters={
            "n_estimators": 30,
            "loss": {"loss_id": "binary_log_loss"},
        },
    )
    lease_owner = uuid.uuid4()

    result, evaluation = await _execute_and_score(
        split,
        spec,
        tmp_path / "lightgbm-output",
        image=image,
        lease_owner=lease_owner,
    )

    assert result.diagnostics["engine"] == CandidateEngine.LIGHTGBM.value
    assert result.diagnostics["custom_loss"] == "binary_log_loss"
    assert evaluation.row_count == split.manifest.predict.row_count
    assert 0.0 <= evaluation.score <= 1.0
    await _assert_no_owned_containers(lease_owner)


async def test_configured_image_configures_bounds_and_fenced_cleanup(tmp_path: Path) -> None:
    image = _configured_candidate_image()
    split = _prepare_real_tabular_split(tmp_path)
    _write_candidate_spec(
        split,
        candidate_id="live-fenced-lightgbm",
        engine=CandidateEngine.LIGHTGBM,
        hyperparameters={
            "learning_rate": 0.01,
            "n_estimators": 50_000,
            "loss": {"loss_id": "binary_log_loss"},
        },
    )
    output_root = tmp_path / "fenced-output"
    output_root.mkdir(mode=0o700)
    lease_owner = uuid.uuid4()
    limits = SandboxLimits(
        cpus=0.5,
        memory_bytes=512 * 1024**2,
        pids=64,
        output_bytes=64 * 1024**2,
        temporary_bytes=64 * 1024**2,
        wall_seconds=120.0,
    )
    task = asyncio.create_task(
        run_candidate_sandbox(
            SandboxRequest(
                image=image,
                input_root=Path(split.candidate_input_root),
                output_root=output_root,
                environment={"OMP_NUM_THREADS": "1"},
                limits=limits,
                lease_owner=lease_owner,
                worker_owner=_LIVE_WORKER_OWNER,
            )
        )
    )
    outcome: SandboxResult | None = None
    try:
        docker, container_id = await _wait_for_owned_container(task, lease_owner)
        code, stdout, stderr = await _run_control_command(
            [
                docker,
                "inspect",
                "--format",
                (
                    '{"nanoCpus":{{.HostConfig.NanoCpus}},'
                    '"memory":{{.HostConfig.Memory}},'
                    '"pids":{{.HostConfig.PidsLimit}},'
                    '"network":{{json .HostConfig.NetworkMode}},'
                    '"readOnly":{{.HostConfig.ReadonlyRootfs}}}'
                ),
                container_id,
            ]
        )
        assert code == 0, stderr[-1_000:]
        controls = json.loads(stdout)
        assert controls == {
            "nanoCpus": 500_000_000,
            "memory": limits.memory_bytes,
            "pids": limits.pids,
            "network": "none",
            "readOnly": True,
        }
    finally:
        await remove_owned_sandbox_containers(
            _LIVE_WORKER_OWNER,
            lease_owner=lease_owner,
        )
        outcome = await asyncio.wait_for(task, timeout=30.0)

    assert outcome is not None
    assert outcome.exit_code != 0
    assert not outcome.timed_out
    await _assert_no_owned_containers(lease_owner)


async def test_configured_gpu_executes_real_kumo_with_trusted_scoring(
    tmp_path: Path,
) -> None:
    image = _configured_candidate_image()
    gpu_devices = _configured_gpu_devices()
    asset_source = _configured_kumo_assets()
    split = _prepare_real_tabular_split(tmp_path)
    input_root = Path(split.candidate_input_root)
    shutil.copytree(asset_source, input_root / "assets" / "kumo", symlinks=False)
    spec = _write_candidate_spec(
        split,
        candidate_id="live-kumo-breast-cancer",
        engine=CandidateEngine.KUMO,
        hyperparameters={
            "context_fraction": 0.5,
            "device_policy": "cuda_required",
            "max_context_rows": 256,
            "num_estimators": 1,
            "size": "small",
        },
        offline_assets={
            "hf_cache": "assets/kumo/hf-cache",
            "kumo_manifest": "assets/kumo/manifest.json",
        },
    )
    lease_owner = uuid.uuid4()

    result, evaluation = await _execute_and_score(
        split,
        spec,
        tmp_path / "kumo-output",
        image=image,
        lease_owner=lease_owner,
        gpu_devices=gpu_devices,
        limits=SandboxLimits(
            cpus=4.0,
            memory_bytes=8 * 1024**3,
            pids=512,
            output_bytes=128 * 1024**2,
            temporary_bytes=2 * 1024**3,
            wall_seconds=1_800.0,
        ),
    )

    assert result.diagnostics["engine"] == CandidateEngine.KUMO.value
    assert result.diagnostics["device"] == "cuda"
    assert evaluation.row_count == split.manifest.predict.row_count
    assert 0.0 <= evaluation.score <= 1.0
    await _assert_no_owned_containers(lease_owner)


async def test_configured_candidate_image_is_ready() -> None:
    image = os.environ.get("AGENT_K_TEST_CANDIDATE_IMAGE")
    if image is None:
        pytest.skip("requires a real immutable image in AGENT_K_TEST_CANDIDATE_IMAGE")
    devices = tuple(filter(None, os.environ.get("AGENT_K_TEST_GPU_DEVICES", "").split(",")))
    await validate_sandbox_runtime(image, gpu_devices=devices)


def test_configured_legacy_source_is_structurally_valid() -> None:
    legacy_root = os.environ.get("AGENT_K_TEST_LEGACY_ROOT")
    if legacy_root is None:
        pytest.skip("requires a real read-only legacy root in AGENT_K_TEST_LEGACY_ROOT")
    report = inspect_legacy_root(Path(legacy_root))
    assert report.mission_state_files + report.experiment_rows > 0
