"""OpenEvolve orchestration and independent sandbox scoring.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from openevolve.evaluation_result import EvaluationResult as NativeEvaluationResult
from pydantic import BaseModel, ConfigDict, Field

from .evaluator import (
    CandidateResult,
    CandidateSpec,
    EvaluationRequest,
    EvaluationResult,
    PreparedSplit,
    evaluate_candidate,
)
from .metrics import MetricDirection
from .sandbox import SandboxLimits, SandboxRequest, run_candidate_sandbox_sync

__all__ = (
    "EvolutionContext",
    "EvolutionOutcome",
    "EvolutionSettings",
    "evaluate",
    "evaluate_program",
    "evaluate_stage1",
    "evaluate_stage2",
    "run_evolution",
    "write_seed_program",
)

_CONTEXT_ENVIRONMENT = "AGENT_K_EVOLUTION_CONTEXT"


class EvolutionSettings(BaseModel):
    """A single source of truth for the native OpenEvolve search."""

    model_config = ConfigDict(frozen=True)
    iterations: int = Field(default=20, ge=1, le=200)
    minimum_iterations: int = Field(default=1, ge=1, le=200)
    minimum_improvements: int = Field(default=0, ge=0)
    model_specs: tuple[str, ...]
    parallel_evaluations: int = Field(default=1, ge=1, le=16)
    evaluation_timeout_seconds: int = Field(default=1200, ge=1)


class EvolutionContext(BaseModel):
    """Host-only evaluator configuration; this file is never mounted in a candidate."""

    model_config = ConfigDict(frozen=True)
    split: PreparedSplit
    spec: CandidateSpec
    image: str
    workdir: str
    limits: SandboxLimits = Field(default_factory=SandboxLimits)
    gpu_devices: tuple[str, ...] = ()


class EvolutionOutcome(BaseModel):
    """A fully evaluated candidate with native-search provenance."""

    model_config = ConfigDict(frozen=True)
    program_path: str
    validation: EvaluationResult
    iterations: int = Field(ge=0)
    improvements: int = Field(ge=0)
    checkpoint_path: str | None = None
    selected_baseline: bool


def write_seed_program(path: Path) -> None:
    """Write the executable baseline that OpenEvolve may mutate inside isolation."""
    path.write_text(
        '''"""Train a candidate and write prediction evidence in the isolated container."""
import json
from pathlib import Path
from agent_k.evaluator import CandidateSpec

# EVOLVE-BLOCK-START
# Tune preprocessing, model parameters, and true LightGBM gradient/Hessian losses.
# Never use validation labels or emit a claimed metric as a substitute for predictions.
def main():
    spec = CandidateSpec.model_validate_json(Path("/run/input/spec.json").read_text())
    parameters = dict(spec.hyperparameters)
    spec = CandidateSpec.model_validate_json(
        json.dumps({**spec.model_dump(mode="json"), "hyperparameters": parameters})
    )
    if spec.problem_type.modality.value == "tabular":
        from agent_k.tabular import run_candidate
    elif spec.problem_type.modality.value == "vision":
        from agent_k.vision import run_candidate
    else:
        from agent_k.text import run_candidate
    result = run_candidate(spec, Path("/run/input"), Path("/run/output"))
    Path("/run/output/result.json").write_text(result.model_dump_json())
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )


def evaluate_program(program_path: Path, context: EvolutionContext) -> EvaluationResult:
    """Run untrusted code in Docker, then score its predictions with hidden labels.

    Args:
        program_path: Candidate Python file mounted read-only into the container.
        context: Trusted split, prediction contract, immutable image, and runtime limits.

    Returns:
        A host-computed score after identity, checksum, schema, and row validation.

    Raises:
        ValueError: Candidate execution fails or its output violates the contract.
        OSError: Required candidate inputs or result artifacts cannot be read.

    Hidden labels remain on the host. Candidate stdout and claimed scores never
    determine the returned fitness.
    """
    code_hash = hashlib.sha256(
        program_path.read_bytes() + context.spec.model_dump_json().encode(),
    ).hexdigest()
    run_root = Path(context.workdir) / "evaluations" / f"{code_hash[:16]}-{uuid.uuid4().hex}"
    input_root = run_root / "input"
    output_root = run_root / "output"
    run_root.mkdir(parents=True, mode=0o700)
    shutil.copytree(context.split.candidate_input_root, input_root, copy_function=os.link)
    output_root.mkdir()
    spec = CandidateSpec.model_validate_json(
        json.dumps(
            {
                **context.spec.model_dump(mode="json"),
                "candidate_id": code_hash,
            }
        )
    )
    (input_root / "spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    outcome = run_candidate_sandbox_sync(
        SandboxRequest(
            image=context.image,
            input_root=input_root,
            output_root=output_root,
            candidate_program=program_path,
            limits=context.limits,
            gpu_devices=context.gpu_devices,
            environment={"OMP_NUM_THREADS": str(max(1, int(context.limits.cpus)))},
        ),
    )
    if (
        outcome.exit_code
        or outcome.timed_out
        or outcome.oom_killed
        or outcome.output_limit_exceeded
    ):
        raise ValueError(
            f"Candidate failed: exit={outcome.exit_code}, timeout={outcome.timed_out}, "
            f"oom={outcome.oom_killed}, output_limit={outcome.output_limit_exceeded}; "
            f"stderr={outcome.stderr[-4000:]}",
        )
    result_path = output_root / "result.json"
    if result_path.is_symlink() or result_path.stat().st_size > 1_000_000:
        raise ValueError("Invalid candidate result metadata")
    result = CandidateResult.model_validate_json(result_path.read_bytes())
    if result.candidate_id != spec.candidate_id:
        raise ValueError("Candidate result identity does not match the evaluated program")
    return evaluate_candidate(
        EvaluationRequest(
            candidate_result=result,
            metric_id=spec.metric_id,
            prediction_root=str(output_root),
            trusted_labels_path=context.split.trusted_labels_path,
            id_columns=spec.id_columns,
            target_columns=spec.target_columns,
            class_labels=spec.class_labels,
        ),
    )


def evaluate_stage1(program_path: str) -> dict[str, float]:
    """Give syntactically valid code partial fidelity, always below complete results."""
    try:
        code = Path(program_path).read_text(encoding="utf-8")
        if len(code) > 100_000:
            raise ValueError("Candidate exceeds code size limit")
        ast.parse(code)
    except (OSError, SyntaxError, ValueError):
        return {"combined_score": -1.0, "valid": 0.0, "full_evaluation": 0.0}
    return {"combined_score": 0.05, "valid": 0.0, "full_evaluation": 0.0}


def evaluate_stage2(program_path: str) -> NativeEvaluationResult:
    """Perform full validation scoring after the cheap syntax gate."""
    return evaluate(program_path)


def evaluate(program_path: str) -> NativeEvaluationResult:
    """OpenEvolve evaluator entry point loaded in the isolated mission's process pool."""
    try:
        context = EvolutionContext.model_validate_json(
            Path(os.environ[_CONTEXT_ENVIRONMENT]).read_bytes(),
        )
        result = evaluate_program(Path(program_path), context)
        quality = (
            result.score if result.direction is MetricDirection.MAXIMIZE else 1 / (1 + result.score)
        )
        fitness = 0.3 + 0.7 * max(0.0, min(1.0, quality))
        return NativeEvaluationResult(
            metrics={
                "combined_score": fitness,
                "metric_value": result.score,
                "valid": 1.0,
                "full_evaluation": 1.0,
                "rows": float(result.row_count),
            },
            artifacts={"validation": result.model_dump_json()},
        )
    except Exception as exc:
        # Invalid, incomplete, timed-out, and failed runs cannot outrank a full result.
        return NativeEvaluationResult(
            metrics={"combined_score": -1.0, "valid": 0.0, "full_evaluation": 0.0},
            artifacts={"error": f"{type(exc).__name__}: {str(exc)[:4000]}"},
        )


async def run_evolution(
    *,
    initial_program: Path,
    baseline: EvaluationResult,
    context: EvolutionContext,
    settings: EvolutionSettings,
    checkpoint_callback: Callable[[Path, int], Awaitable[None]],
    checkpoint_path: Path | None = None,
) -> EvolutionOutcome:
    """Run the sole native evolution loop and independently revalidate its winner.

    Args:
        initial_program: Existing seed program executed only inside the sandbox.
        baseline: Trusted validation score for the seed program.
        context: Host-only evaluation inputs, hidden-label paths, and sandbox limits.
        settings: Search budget, model specifications, and evaluation concurrency.
        checkpoint_callback: Persist each native checkpoint before reporting progress.
        checkpoint_path: Optional verified native checkpoint from a previous attempt.

    Returns:
        The independently validated winner, or the measured baseline when search fails
        to produce an eligible improvement, with iteration and checkpoint provenance.

    Raises:
        ValueError: No evolution model is configured or a contract is invalid.
        Exception: Native search, sandbox execution, or durable checkpointing fails.
    """
    from openevolve import OpenEvolve
    from openevolve.config import Config

    if not settings.model_specs:
        raise ValueError("At least one evolution model must be configured")
    output = Path(context.workdir) / "native"
    output.mkdir(parents=True, exist_ok=True)
    context_path = Path(context.workdir) / "context.json"
    context_path.write_text(context.model_dump_json(), encoding="utf-8")
    context_path.chmod(0o600)
    evaluator_file = Path(context.workdir) / "evaluate.py"
    evaluator_file.write_text(
        "from agent_k.evolution import evaluate, evaluate_stage1, evaluate_stage2\n",
        encoding="utf-8",
    )
    os.environ[_CONTEXT_ENVIRONMENT] = str(context_path)
    config = Config()
    config.max_iterations = settings.iterations
    config.checkpoint_interval = 1
    config.language = "python"
    config.max_code_length = 100_000
    config.evaluator.cascade_evaluation = True
    config.evaluator.cascade_thresholds = [0.01]
    config.evaluator.timeout = settings.evaluation_timeout_seconds + 60
    config.evaluator.max_retries = 0
    config.evaluator.parallel_evaluations = settings.parallel_evaluations
    config.llm.models = [_model_config(spec) for spec in settings.model_specs]
    config.prompt.system_message = (
        "Improve the supplied ML training program using validation feedback. "
        f"Keep the configured {context.spec.engine.value} engine and problem type fixed. "
        "Use LightGBM for tree models; evolve actual custom loss gradients and Hessians. "
        "Preserve the prediction/result artifact contract. No network or hidden labels exist "
        "in the sandbox. Only fully scored predictions are valid. For enabled Kumo candidates "
        "keep pretrained weights fixed; vary context, estimators, preprocessing, and blending."
    )
    evolver = OpenEvolve(
        initial_program_path=str(initial_program),
        evaluation_file=str(evaluator_file),
        config=config,
        output_dir=str(output),
    )
    observed: set[Path] = set()

    async def publish_checkpoints() -> None:
        while True:
            await _publish_new_checkpoints(output, observed, checkpoint_callback)
            await asyncio.sleep(0.5)

    publisher = asyncio.create_task(publish_checkpoints())
    native_task: asyncio.Task[Any] | None = None
    try:
        resumed_iteration = 0
        if checkpoint_path:
            evolver.database.load(str(checkpoint_path))
            resumed_iteration = int(evolver.database.last_iteration)
        remaining = max(0, settings.iterations - resumed_iteration)
        if remaining:
            native_task = asyncio.create_task(
                evolver.run(
                    iterations=remaining,
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                )
            )
            done, _ = await asyncio.wait(
                {native_task, publisher},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if publisher in done:
                await publisher
            await native_task
    finally:
        if native_task is not None and not native_task.done():
            native_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await native_task
        publisher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await publisher
        await _publish_new_checkpoints(output, observed, checkpoint_callback)
    programs = list(evolver.database.programs.values())
    full = [
        program
        for program in programs
        if program.metrics.get("valid") == 1.0 and program.metrics.get("full_evaluation") == 1.0
    ]
    completed = max((_checkpoint_iteration(path) for path in observed), default=0)
    baseline_quality = _quality(baseline.score, baseline.direction)
    improvements = 0
    best_quality = baseline_quality
    for program in sorted(full, key=lambda item: item.iteration_found):
        quality = _quality(float(program.metrics["metric_value"]), baseline.direction)
        if quality > best_quality:
            improvements += 1
            best_quality = quality
    selected = initial_program
    selected_result = baseline
    if full:
        winner = max(
            full,
            key=lambda program: _quality(
                float(program.metrics["metric_value"]), baseline.direction
            ),
        )
        candidate = Path(context.workdir) / "selected.py"
        candidate.write_text(winner.code, encoding="utf-8")
        try:
            verified = await asyncio.to_thread(evaluate_program, candidate, context)
        except ValueError:
            verified = baseline
        if _quality(verified.score, baseline.direction) > baseline_quality:
            selected, selected_result = candidate, verified
    checkpoints = sorted(observed, key=_checkpoint_iteration)
    return EvolutionOutcome(
        program_path=str(selected),
        validation=selected_result,
        iterations=completed,
        improvements=improvements,
        checkpoint_path=str(checkpoints[-1]) if checkpoints else None,
        selected_baseline=selected == initial_program,
    )


async def _publish_new_checkpoints(
    output: Path,
    observed: set[Path],
    callback: Callable[[Path, int], Awaitable[None]],
) -> None:
    markers = output.glob("checkpoints/checkpoint_*/best_program_info.json")
    for info in sorted(markers, key=lambda path: _checkpoint_iteration(path.parent)):
        try:
            json.loads(info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = info.parent
        if checkpoint in observed:
            continue
        await callback(checkpoint, _checkpoint_iteration(checkpoint))
        observed.add(checkpoint)


def _checkpoint_iteration(path: Path) -> int:
    return int(path.name.rsplit("_", maxsplit=1)[1])


def _quality(score: float, direction: MetricDirection) -> float:
    return score if direction is MetricDirection.MAXIMIZE else -score


def _model_config(spec: str) -> Any:
    from openevolve.config import LLMModelConfig

    from .providers import DEVSTRAL_BASE_URL, DEVSTRAL_MODEL_ID

    if spec.startswith("openrouter:"):
        name, base, key = (
            spec[11:],
            "https://openrouter.ai/api/v1",
            os.environ["OPENROUTER_API_KEY"],
        )
    elif spec.startswith("openai:"):
        name, base, key = spec[7:], "https://api.openai.com/v1", os.environ["OPENAI_API_KEY"]
    elif spec.startswith("devstral:"):
        suffix = spec[9:]
        name = DEVSTRAL_MODEL_ID if suffix == "local" or suffix.startswith("http") else suffix
        base = (
            suffix
            if suffix.startswith("http")
            else os.getenv("DEVSTRAL_BASE_URL", DEVSTRAL_BASE_URL)
        )
        configured = os.getenv("DEVSTRAL_BASE_URL", DEVSTRAL_BASE_URL)
        key = (
            os.getenv("DEVSTRAL_API_KEY", "not-required") if base == configured else "not-required"
        )
    else:
        raise ValueError("Evolution requires an OpenAI-compatible openai/openrouter/devstral model")
    return LLMModelConfig(name=name, api_base=base, api_key=key, weight=1.0)
