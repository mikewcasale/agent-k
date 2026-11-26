"""Durable five-phase mission graph executed only in isolated worker processes.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import hashlib
import json
import shutil
import stat
import tarfile
import time
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, End, GraphRunContext

from .config import Settings, load_settings
from .data import read_table
from .datasets import DataModality, DatasetManifest, TaskKind, discover_dataset
from .evaluator import (
    CandidateEngine,
    CandidateResult,
    CandidateSpec,
    CandidateStatus,
    EvaluationResult,
    PreparedSplitSet,
    build_candidate_spec,
    build_submission_spec,
    materialize_trusted_splits,
)
from .events import EventEmitter
from .evolution import (
    EvolutionContext,
    EvolutionOutcome,
    EvolutionSettings,
    evaluate_program,
    run_evolution,
    write_seed_program,
)
from .hints import profile_training_data
from .kaggle import KaggleAdapter, KaggleSettings
from .lobbyist import LobbyistDeps, lobbyist_agent
from .metrics import MetricDirection, MetricId
from .models import Competition, MissionCriteria, Submission
from .providers import get_model
from .sandbox import SandboxLimits, SandboxRequest, run_candidate_sandbox
from .scientist import ResearchReport, ScientistDeps, scientist_agent
from .stores import (
    ArtifactReference,
    ControlAction,
    JobLease,
    MissionInterrupted,
    MissionRunResult,
    MissionSnapshot,
)
from .tracking import MissionTracker

if TYPE_CHECKING:
    from .runner import WorkerApiClient

__all__ = (
    "DiscoveryNode",
    "EvolutionNode",
    "MissionContext",
    "MissionState",
    "PrototypeNode",
    "ResearchNode",
    "SubmissionNode",
    "run_mission",
)


class MissionState(BaseModel):
    """Versioned replay state, with submission uncertainty preserved across restarts."""

    schema_version: str = "2.0"
    mission_id: uuid.UUID
    competition: Competition | None = None
    criteria: MissionCriteria = Field(default_factory=MissionCriteria)
    phases_completed: list[str] = Field(default_factory=list)
    manifest: DatasetManifest | None = None
    splits: PreparedSplitSet | None = None
    research: ResearchReport | None = None
    baseline_spec: CandidateSpec | None = None
    baseline: EvaluationResult | None = None
    reference_baseline: EvaluationResult | None = None
    reference_spec: CandidateSpec | None = None
    baselines: list[EvaluationResult] = Field(default_factory=list)
    evolution: EvolutionOutcome | None = None
    acceptance: EvaluationResult | None = None
    native_checkpoint: str | None = None
    evolution_iterations: int = 0
    selected_program: str | None = None
    seed_program: str | None = None
    submission_state: Literal[
        "none", "prepared", "uncertain", "pending", "complete", "disabled"
    ] = "none"
    submission: Submission | None = None
    submission_sha256: str | None = None
    historical_submission_ids: tuple[str, ...] = ()
    submission_reconciliation_required: bool = False
    artifact_references: list[ArtifactReference] = Field(default_factory=list)
    legacy_program_code: str | None = None


@dataclass
class MissionContext:
    """Services and paths owned by one mission subprocess."""

    job: JobLease
    client: WorkerApiClient
    workdir: Path
    settings: Settings
    adapter: KaggleAdapter
    http_client: httpx.AsyncClient
    tracker: MissionTracker
    emitter: EventEmitter
    started: float = field(default_factory=time.monotonic)
    data_artifact: dict[str, Any] | None = None

    async def phase(
        self,
        name: str,
        state: MissionState,
        operation: Callable[[MissionState], Awaitable[None]],
    ) -> None:
        """Execute one phase with a hard deadline and durable completion boundary."""
        await self.check_control(state)
        if name in state.phases_completed:
            return
        started = time.monotonic()
        await self.emitter.emit_phase_start(name, [f"Complete {name} using validated evidence"])
        try:
            async with asyncio.timeout(self.settings.phase_timeout_seconds):
                await operation(state)
        except MissionInterrupted:
            raise
        except Exception as exc:
            await self.emitter.emit_phase_error(name, str(exc), recoverable=False)
            raise
        state.phases_completed.append(name)
        await self.emitter.emit_phase_complete(name, True, int((time.monotonic() - started) * 1000))
        await self.checkpoint(state)

    async def discovery(self, state: MissionState) -> None:
        """Select an observed competition or resolve the explicitly supplied slug."""
        await self.adapter.authenticate()
        if self.job.command.competition_id:
            state.competition = await self.adapter.get_competition(self.job.command.competition_id)
            return
        deps = LobbyistDeps(self.adapter, state.criteria)
        result = await lobbyist_agent.run(
            state.criteria.model_dump_json(),
            deps=deps,
            model=get_model(self.settings.model),
        )
        state.competition = deps.observed[result.output.competition_id]

    async def research(self, state: MissionState) -> None:
        """Download, discover, split, and research without reading acceptance labels."""
        competition = _competition(state)
        dataset_root = self.workdir / "dataset"
        await self.adapter.download_data(competition.id, str(dataset_root))
        _extract_archives(dataset_root)
        manifest = discover_dataset(dataset_root, metric_id=_metric(competition).value)
        splits = await asyncio.to_thread(
            materialize_trusted_splits,
            manifest,
            dataset_root,
            self.workdir / "inputs",
            self.workdir / "trusted",
        )
        state.manifest, state.splits = manifest, splits
        training = read_table(
            Path(splits.validation.candidate_input_root) / splits.validation.manifest.train.path
        )
        profile = profile_training_data(training)
        deps = ScientistDeps(self.http_client, self.adapter, competition, profile)
        if state.research is not None:
            return
        result = await scientist_agent.run(
            json.dumps(
                {
                    "competition": competition.model_dump(mode="json"),
                    "profile": profile.model_dump(),
                }
            ),
            deps=deps,
            model=get_model(self.settings.model),
        )
        state.research = result.output

    async def prototype(self, state: MissionState) -> None:
        """Evaluate a real baseline and, when configured, a fixed Kumo challenger."""
        manifest, splits = _dataset(state)
        engine = {
            DataModality.TABULAR: CandidateEngine.LIGHTGBM,
            DataModality.VISION: CandidateEngine.RESNET18,
            DataModality.TEXT: CandidateEngine.TFIDF_LINEAR,
        }[manifest.problem_type.modality]
        parameters: dict[str, Any] = {}
        if engine is CandidateEngine.LIGHTGBM:
            loss = "squared_error"
            if manifest.problem_type.task is TaskKind.CLASSIFICATION:
                loss = (
                    "binary_log_loss" if len(manifest.class_labels) == 2 else "multiclass_log_loss"
                )
            parameters["loss"] = {"loss_id": loss}
        assets = self._offline_assets(engine, Path(splits.validation.candidate_input_root))
        spec = build_candidate_spec(
            splits.validation,
            candidate_id="baseline",
            engine=engine,
            metric_id=_metric(_competition(state)),
            hyperparameters=parameters,
            offline_assets=assets,
        )
        program = self.workdir / "baseline.py"
        write_seed_program(program)
        context = self._evaluation_context(state, spec)
        score = await asyncio.to_thread(evaluate_program, program, context)
        state.baseline, state.baseline_spec = score, spec
        state.reference_baseline, state.reference_spec = score, spec
        state.baselines.append(score)
        state.selected_program = str(program)
        if state.criteria.enable_kumo and manifest.problem_type.modality is DataModality.TABULAR:
            await self._kumo_challenger(state, program)
        if state.legacy_program_code:
            await self._legacy_challenger(state)
        state.seed_program = state.selected_program

    async def evolution(self, state: MissionState) -> None:
        """Run OpenEvolve's native search with per-iteration durable checkpoints."""
        if state.baseline is None or state.baseline_spec is None or state.selected_program is None:
            raise ValueError("A validated baseline is required before evolution")

        async def publish(checkpoint: Path, iteration: int) -> None:
            state.native_checkpoint = str(checkpoint)
            state.evolution_iterations = iteration
            await self.checkpoint(state)
            metrics = [
                json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
                for path in checkpoint.glob("programs/*.json")
            ]
            scores = [
                float(item["combined_score"])
                for item in metrics
                if item.get("valid") == 1 and item.get("full_evaluation") == 1
            ]
            if scores:
                await self.emitter.emit_generation_complete(
                    iteration,
                    best_fitness=max(scores),
                    mean_fitness=sum(scores) / len(scores),
                    worst_fitness=min(scores),
                    population_size=len(scores),
                    mutations={},
                    iteration=iteration,
                    total_iterations=state.criteria.max_evolution_rounds,
                )

        settings = EvolutionSettings(
            iterations=state.criteria.max_evolution_rounds,
            minimum_iterations=state.criteria.min_evolution_iterations,
            minimum_improvements=state.criteria.min_improvements_required,
            model_specs=state.criteria.evolution_models or (self.settings.model,),
            parallel_evaluations=self.settings.evaluation_parallelism,
            evaluation_timeout_seconds=self.settings.evaluation_timeout_seconds,
        )
        state.evolution = await run_evolution(
            initial_program=Path(state.selected_program),
            baseline=state.baseline,
            context=self._evaluation_context(state, state.baseline_spec),
            settings=settings,
            checkpoint_callback=publish,
            checkpoint_path=Path(state.native_checkpoint) if state.native_checkpoint else None,
        )
        state.selected_program = state.evolution.program_path
        state.evolution_iterations = state.evolution.iterations
        await self.check_control(state)

    async def submission(self, state: MissionState) -> None:
        """Accept the selected candidate independently and optionally submit exactly once."""
        if state.submission_state == "uncertain":
            raise ValueError(
                "Previous submission receipt is uncertain; reconcile it before resuming"
            )
        if state.submission is not None:
            await self._await_submission(state)
            return
        artifact = self.workdir / "submission.csv"
        if state.submission_state == "prepared":
            if not artifact.is_file() or _digest(artifact) != state.submission_sha256:
                raise ValueError(
                    "Prepared submission artifact failed checkpoint integrity validation"
                )
        else:
            await self._accept_selected(state)
            artifact = await self._prepare_submission(state)
            state.artifact_references.append(await self.tracker.upload(artifact, role="submission"))
            state.submission_sha256 = _digest(artifact)
            state.submission_state = "prepared"
            await self.checkpoint(state)
        if not (self.settings.allow_submissions and state.criteria.allow_submission):
            state.submission_state = "disabled"
            return
        if state.evolution is None:
            raise ValueError("Evolution evidence is missing")
        if state.evolution.iterations < state.criteria.min_evolution_iterations:
            raise ValueError("Minimum evolution iteration requirement was not met")
        if state.evolution.improvements < state.criteria.min_improvements_required:
            raise ValueError("Minimum validated improvement requirement was not met")
        state.submission_state = "uncertain"
        await self.checkpoint(state)
        state.submission = await self.adapter.submit(
            _competition(state).id,
            str(artifact),
            message=f"AGENT-K mission {state.mission_id}",
        )
        state.submission_state = "pending"
        await self.checkpoint(state)
        await self._await_submission(state)

    async def checkpoint(self, state: MissionState) -> None:
        """Publish replayable data once and a compact runtime archive at each boundary."""
        archive_root = self.workdir / "archives"
        archive_root.mkdir(exist_ok=True)
        if self.data_artifact is None and state.splits is not None:
            data_path = archive_root / "dataset.tar.gz"
            _archive_paths(data_path, self.workdir, ("dataset", "inputs", "trusted"))
            reference = await self.tracker.upload(data_path, role="dataset-checkpoint")
            self.data_artifact = {
                "artifactId": str(reference.artifact_id),
                "sha256": _digest(data_path),
                "sizeBytes": data_path.stat().st_size,
            }
        state_file = self.workdir / "mission.json"
        state_file.write_text(state.model_dump_json(), encoding="utf-8")
        state_file.chmod(0o600)
        runtime = archive_root / f"runtime-{uuid.uuid4().hex}.tar.gz"
        names = ["mission.json", "baseline.py", "submission.csv"]
        if state.selected_program:
            names.append(str(Path(state.selected_program).relative_to(self.workdir)))
        if state.seed_program:
            names.append(str(Path(state.seed_program).relative_to(self.workdir)))
        if state.native_checkpoint:
            names.append(str(Path(state.native_checkpoint).relative_to(self.workdir)))
        _archive_paths(runtime, self.workdir, tuple(dict.fromkeys(names)))
        reference = await self.tracker.upload(runtime, role="runtime-checkpoint")
        payload = {
            "workspaceRoot": str(self.workdir),
            "state": state.model_dump(mode="json"),
            "dataArtifact": self.data_artifact,
            "runtimeArtifact": {
                "artifactId": str(reference.artifact_id),
                "sha256": _digest(runtime),
                "sizeBytes": runtime.stat().st_size,
            },
        }
        await self.emitter.emit(
            "checkpoint-created",
            {
                "name": runtime.name,
                "phase": state.phases_completed[-1] if state.phases_completed else "discovery",
            },
        )
        await self.client.save_snapshot(
            self.job,
            MissionSnapshot(
                sequence=self.tracker.sequence, created_at=datetime.now(UTC), payload=payload
            ),
        )
        runtime.unlink()

    async def check_control(self, state: MissionState) -> None:
        """Acknowledge pause or cancel only after a durable snapshot exists."""
        directive = await self.client.get_control(self.job)
        if directive.action is ControlAction.CONTINUE:
            return
        await self.checkpoint(state)
        action: Literal["pause", "cancel"] = (
            "pause" if directive.action is ControlAction.PAUSE else "cancel"
        )
        if action == "pause":
            await self.emitter.emit(
                "state-delta",
                [
                    {"op": "replace", "path": "/status", "value": "paused"},
                ],
            )
        else:
            await self.emitter.emit_mission_complete(
                success=False,
                final_rank=None,
                final_score=None,
                error_message=directive.reason or action,
                total_submissions=int(state.submission is not None),
                evolution_generations=state.evolution_iterations,
                evolution_iterations=state.evolution_iterations,
                duration_ms=int((time.monotonic() - self.started) * 1000),
                phases_completed=state.phases_completed,
            )
        raise MissionInterrupted(
            action, directive.reason, final_event_sequence=self.tracker.sequence
        )

    def _evaluation_context(
        self, state: MissionState, spec: CandidateSpec, *, acceptance: bool = False
    ) -> EvolutionContext:
        _, splits = _dataset(state)
        if self.settings.sandbox_image is None:
            raise ValueError("AGENT_K_SANDBOX_IMAGE must pin an immutable candidate image digest")
        return EvolutionContext(
            split=splits.acceptance if acceptance else splits.validation,
            spec=spec,
            image=self.settings.sandbox_image,
            workdir=str(self.workdir / "evolution"),
            limits=SandboxLimits(wall_seconds=self.settings.evaluation_timeout_seconds),
            gpu_devices=self.settings.gpu_devices
            if spec.engine
            in {
                CandidateEngine.KUMO,
                CandidateEngine.RESNET18,
                CandidateEngine.TRANSFORMER,
            }
            else (),
        )

    async def _kumo_challenger(self, state: MissionState, program: Path) -> None:
        _, splits = _dataset(state)
        if not self.settings.gpu_devices:
            await self.emitter.emit(
                "task-progress",
                {
                    "taskId": "kumo",
                    "progress": 100,
                    "message": "Kumo skipped: a validated GPU worker is not configured",
                },
            )
            return
        if self.settings.kumo_cache is None:
            await self.emitter.emit(
                "task-progress",
                {
                    "taskId": "kumo",
                    "progress": 100,
                    "message": "Kumo skipped: verified offline weights are not configured",
                },
            )
            return
        assets = self._offline_assets(
            CandidateEngine.KUMO, Path(splits.validation.candidate_input_root)
        )
        spec = build_candidate_spec(
            splits.validation,
            candidate_id="kumo",
            engine=CandidateEngine.KUMO,
            metric_id=_metric(_competition(state)),
            hyperparameters={"size": "small", "num_estimators": 8},
            offline_assets=assets,
        )
        try:
            result = await asyncio.to_thread(
                evaluate_program, program, self._evaluation_context(state, spec)
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            await self.emitter.emit(
                "task-error",
                {
                    "taskId": "kumo",
                    "error": str(exc),
                    "message": "Kumo evaluation failed; retaining the validated LightGBM baseline",
                },
            )
            return
        state.baselines.append(result)
        if state.baseline is not None and _better(result, state.baseline):
            state.baseline, state.baseline_spec = result, spec

    def _offline_assets(self, engine: CandidateEngine, input_root: Path) -> dict[str, str]:
        sources = {
            CandidateEngine.KUMO: ("kumo", self.settings.kumo_cache),
            CandidateEngine.RESNET18: ("vision", self.settings.vision_weights),
            CandidateEngine.TRANSFORMER: ("transformer", self.settings.transformer_cache),
        }
        if engine not in sources:
            return {}
        name, source = sources[engine]
        if source is None:
            if engine is CandidateEngine.RESNET18:
                return {}
            raise ValueError(f"Verified offline assets are required for {engine.value}")
        if engine is CandidateEngine.RESNET18 and not source.is_file():
            raise ValueError("Vision pretrained weights must be a checkpoint file")
        destination = input_root / "assets" / name
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        relative = str(destination.relative_to(input_root))
        if engine is CandidateEngine.KUMO:
            return {
                "hf_cache": relative + "/hf-cache",
                "kumo_manifest": relative + "/manifest.json",
            }
        if engine is CandidateEngine.RESNET18:
            return {"resnet18_weights": relative}
        return {"transformer_model": relative}

    async def _accept_selected(self, state: MissionState) -> None:
        _, splits = _dataset(state)
        if state.baseline_spec is None or state.selected_program is None:
            raise ValueError("Missing selected candidate")
        assets = self._offline_assets(
            state.baseline_spec.engine, Path(splits.acceptance.candidate_input_root)
        )
        spec = build_candidate_spec(
            splits.acceptance,
            candidate_id="acceptance",
            engine=state.baseline_spec.engine,
            metric_id=state.baseline_spec.metric_id,
            hyperparameters=state.baseline_spec.hyperparameters,
            offline_assets=assets,
        )
        context = self._evaluation_context(state, spec, acceptance=True)
        baseline_program = self.workdir / "baseline.py"
        reference_spec = state.reference_spec or state.baseline_spec
        reference_assets = self._offline_assets(
            reference_spec.engine, Path(splits.acceptance.candidate_input_root)
        )
        reference = build_candidate_spec(
            splits.acceptance,
            candidate_id="acceptance-baseline",
            engine=reference_spec.engine,
            metric_id=reference_spec.metric_id,
            hyperparameters=reference_spec.hyperparameters,
            offline_assets=reference_assets,
        )
        baseline = await asyncio.to_thread(
            evaluate_program,
            baseline_program,
            self._evaluation_context(state, reference, acceptance=True),
        )
        winning_spec = reference_spec
        winning_validation = state.reference_baseline or state.baseline
        seed_program = Path(state.seed_program) if state.seed_program else baseline_program
        if seed_program != baseline_program or state.baseline_spec != reference_spec:
            try:
                seed = await asyncio.to_thread(evaluate_program, seed_program, context)
            except ValueError as exc:
                await self.emitter.emit(
                    "task-error",
                    {
                        "taskId": "acceptance-seed",
                        "error": str(exc),
                    },
                )
            else:
                if _better(seed, baseline):
                    baseline = seed
                    baseline_program = seed_program
                    winning_spec = state.baseline_spec
                    winning_validation = state.baseline
        try:
            selected = await asyncio.to_thread(
                evaluate_program, Path(state.selected_program), context
            )
        except ValueError as exc:
            await self.emitter.emit(
                "task-error",
                {
                    "taskId": "acceptance-selected",
                    "error": str(exc),
                },
            )
            selected = baseline
        state.acceptance = selected
        if not _better(selected, baseline):
            state.selected_program = str(baseline_program)
            state.acceptance = baseline
            state.baseline_spec = winning_spec
            state.baseline = winning_validation
            if state.evolution is not None:
                state.evolution = state.evolution.model_copy(
                    update={
                        "selected_baseline": True,
                        "program_path": str(baseline_program),
                        "improvements": 0,
                        "validation": state.baseline,
                    }
                )

    async def _await_submission(self, state: MissionState) -> None:
        if state.submission is None:
            raise ValueError("Cannot await a submission without a receipt")
        deadline = time.monotonic() + min(300, self.settings.phase_timeout_seconds - 1)
        while time.monotonic() < deadline:
            await self.check_control(state)
            state.submission = await self.adapter.get_submission_status(
                _competition(state).id,
                state.submission.id,
            )
            if state.submission.status == "complete":
                state.submission_state = "complete"
                return
            if state.submission.status == "error":
                raise ValueError(state.submission.error_message or "Kaggle rejected the submission")
            await asyncio.sleep(10)
        await self.checkpoint(state)
        raise TimeoutError("Kaggle evaluation is pending; resume will poll the existing receipt")

    async def _legacy_challenger(self, state: MissionState) -> None:
        if not state.legacy_program_code or state.baseline_spec is None or state.baseline is None:
            return
        program = self.workdir / "legacy.py"
        program.write_text(state.legacy_program_code, encoding="utf-8")
        try:
            result = await asyncio.to_thread(
                evaluate_program,
                program,
                self._evaluation_context(state, state.baseline_spec),
            )
        except (ValueError, OSError, RuntimeError) as exc:
            await self.emitter.emit(
                "task-error",
                {
                    "taskId": "legacy-validation",
                    "error": str(exc),
                    "message": "Legacy program did not pass the current prediction contract; using the validated baseline",
                },
            )
        else:
            state.baselines.append(result)
            if _better(result, state.baseline):
                state.baseline = result
                state.selected_program = str(program)
        state.legacy_program_code = None

    async def _prepare_submission(self, state: MissionState) -> Path:
        from .evaluator import write_submission

        manifest, _ = _dataset(state)
        if state.baseline_spec is None or state.selected_program is None:
            raise ValueError("Missing selected candidate")
        input_root = self.workdir / "dataset"
        assets = self._offline_assets(state.baseline_spec.engine, input_root)
        spec = build_submission_spec(
            manifest,
            candidate_id="submission",
            engine=state.baseline_spec.engine,
            metric_id=state.baseline_spec.metric_id,
            hyperparameters=state.baseline_spec.hyperparameters,
            offline_assets=assets,
        )
        (input_root / "spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
        output_root = self.workdir / "submission"
        output_root.mkdir(exist_ok=True)
        if self.settings.sandbox_image is None:
            raise ValueError("Sandbox image is required")
        execution = await run_candidate_sandbox(
            SandboxRequest(
                image=self.settings.sandbox_image,
                input_root=input_root,
                output_root=output_root,
                candidate_program=Path(state.selected_program),
                limits=SandboxLimits(wall_seconds=self.settings.evaluation_timeout_seconds),
                gpu_devices=self.settings.gpu_devices
                if spec.engine
                in {
                    CandidateEngine.KUMO,
                    CandidateEngine.RESNET18,
                    CandidateEngine.TRANSFORMER,
                }
                else (),
            ),
        )
        if (
            execution.exit_code
            or execution.timed_out
            or execution.oom_killed
            or execution.output_limit_exceeded
        ):
            raise ValueError("Submission inference failed")
        result = CandidateResult.model_validate_json((output_root / "result.json").read_bytes())
        if (
            result.status is not CandidateStatus.SUCCEEDED
            or result.candidate_id != spec.candidate_id
        ):
            raise ValueError("Submission result identity or terminal status is invalid")
        if result.predictions_path is None:
            raise ValueError("Submission inference produced no predictions")
        prediction = (output_root / result.predictions_path).resolve()
        if not prediction.is_relative_to(output_root.resolve()) or prediction.is_symlink():
            raise ValueError("Submission predictions escaped their output directory")
        if _digest(prediction) != result.predictions_sha256:
            raise ValueError("Submission prediction digest does not match the result contract")
        destination = self.workdir / "submission.csv"
        write_submission(prediction, manifest, input_root, destination, metric_id=spec.metric_id)
        return destination


@dataclass
class DiscoveryNode(BaseNode[MissionState, MissionContext, MissionRunResult]):
    """Select a competition and transition to research."""

    async def run(self, ctx: GraphRunContext[MissionState, MissionContext]) -> ResearchNode:
        """Complete discovery through the shared phase lifecycle."""
        await ctx.deps.phase("discovery", ctx.state, ctx.deps.discovery)
        return ResearchNode()


@dataclass
class ResearchNode(BaseNode[MissionState, MissionContext, MissionRunResult]):
    """Discover dataset structure and collect evidence."""

    async def run(self, ctx: GraphRunContext[MissionState, MissionContext]) -> PrototypeNode:
        """Complete research through the shared phase lifecycle."""
        await ctx.deps.phase("research", ctx.state, ctx.deps.research)
        return PrototypeNode()


@dataclass
class PrototypeNode(BaseNode[MissionState, MissionContext, MissionRunResult]):
    """Establish independently scored baselines."""

    async def run(self, ctx: GraphRunContext[MissionState, MissionContext]) -> EvolutionNode:
        """Complete prototyping through the shared phase lifecycle."""
        await ctx.deps.phase("prototype", ctx.state, ctx.deps.prototype)
        return EvolutionNode()


@dataclass
class EvolutionNode(BaseNode[MissionState, MissionContext, MissionRunResult]):
    """Optimize with OpenEvolve only."""

    async def run(self, ctx: GraphRunContext[MissionState, MissionContext]) -> SubmissionNode:
        """Complete native evolution through the shared phase lifecycle."""
        await ctx.deps.phase("evolution", ctx.state, ctx.deps.evolution)
        return SubmissionNode()


@dataclass
class SubmissionNode(BaseNode[MissionState, MissionContext, MissionRunResult]):
    """Perform independent acceptance and prepare the final artifact."""

    async def run(
        self, ctx: GraphRunContext[MissionState, MissionContext]
    ) -> End[MissionRunResult]:
        """Finish the mission with truthful local and external result separation."""
        await ctx.deps.phase("submission", ctx.state, ctx.deps.submission)
        state = ctx.state
        duration = int((time.monotonic() - ctx.deps.started) * 1000)
        score = state.submission.public_score if state.submission else None
        await ctx.deps.emitter.emit_mission_complete(
            success=True,
            error_message=None,
            final_rank=None,
            final_score=score,
            total_submissions=int(state.submission is not None),
            evolution_generations=state.evolution_iterations,
            evolution_iterations=state.evolution_iterations,
            duration_ms=duration,
            phases_completed=state.phases_completed,
        )
        return End(
            MissionRunResult(
                mission_id=state.mission_id,
                competition_id=_competition(state).id,
                final_score=score,
                total_submissions=int(state.submission is not None),
                evolution_generations=state.evolution_iterations,
                evolution_iterations=state.evolution_iterations,
                duration_ms=duration,
                phases_completed=tuple(state.phases_completed),
                final_event_sequence=ctx.deps.tracker.sequence,
                artifacts=tuple(state.artifact_references),
            )
        )


async def run_mission(job: JobLease, client: WorkerApiClient, workdir: Path) -> MissionRunResult:
    """Execute one fenced mission, restoring verified durable state when present.

    Args:
        job: Current server-issued job lease, command, and event sequence.
        client: Authenticated worker client used for fenced state and artifact writes.
        workdir: Private directory owned exclusively by this worker attempt.

    Returns:
        A terminal result referencing verified artifacts and the final event sequence.

    Raises:
        MissionInterrupted: A pause or cancellation has been durably acknowledged.
        ValueError: Configuration, restored state, or a mission contract is invalid.
        TimeoutError: A phase or remote submission remains incomplete past its deadline.

    Provider and Kaggle failures propagate to the worker's retry policy. A submission
    with an uncertain remote outcome is reconciled before any new submission attempt.
    """
    from .lycurgus import mission_graph

    settings = load_settings()
    if not settings.sandbox_image:
        raise ValueError("Worker requires AGENT_K_SANDBOX_IMAGE before accepting ML jobs")
    workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
    tracker = MissionTracker(job, client, sequence=job.event_sequence)
    state = MissionState(
        mission_id=job.mission_id, criteria=MissionCriteria.model_validate(job.command.criteria)
    )
    async with httpx.AsyncClient(follow_redirects=False) as http:
        context = MissionContext(
            job=job,
            client=client,
            workdir=workdir,
            settings=settings,
            adapter=KaggleAdapter(KaggleSettings(), http),
            http_client=http,
            tracker=tracker,
            emitter=EventEmitter(str(job.mission_id), tracker),
        )
        if job.command.checkpoint_id:
            state = await _restore(context, job.command.checkpoint_id)
        if (
            state.submission_state == "uncertain"
            and state.historical_submission_ids
            and not state.submission_reconciliation_required
        ):
            receipts = [
                await context.adapter.get_submission_status(_competition(state).id, receipt_id)
                for receipt_id in state.historical_submission_ids
            ]
            if all(receipt.status == "complete" for receipt in receipts):
                state.submission = receipts[-1]
                state.submission_state = "complete"
                await context.checkpoint(state)
        if state.submission_state == "uncertain":
            raise ValueError(
                "A previous submission may have succeeded; reconcile its receipt before resuming"
            )
        first: DiscoveryNode | SubmissionNode = (
            SubmissionNode() if state.submission else DiscoveryNode()
        )
        result = await mission_graph.run(first, state=state, deps=context)
        return result.output


async def _restore(context: MissionContext, snapshot_id: uuid.UUID) -> MissionState:
    snapshot = await context.client.get_snapshot(context.job, snapshot_id)
    payload = snapshot.payload
    if isinstance(payload.get("migration"), dict):
        return await _restore_legacy(context, payload["migration"])
    for key in ("dataArtifact", "runtimeArtifact"):
        reference = payload.get(key)
        if not isinstance(reference, dict):
            continue
        signed = await context.client.request_download(
            context.job, uuid.UUID(reference["artifactId"])
        )
        archive = context.workdir / f"{key}.tar.gz"
        await context.client.download_file(
            signed,
            archive,
            expected_sha256=reference["sha256"],
            expected_size=reference["sizeBytes"],
        )
        with tarfile.open(archive) as bundle:
            bundle.extractall(context.workdir, filter="data")
        archive.unlink()
    previous = str(payload.get("workspaceRoot", ""))
    state_data = payload.get("state")
    if not isinstance(state_data, dict):
        raise ValueError("Legacy snapshot requires validated migration before execution")
    restored = _rebase(state_data, previous, str(context.workdir))
    state = MissionState.model_validate(restored)
    if state.mission_id != context.job.mission_id:
        raise ValueError("Checkpoint belongs to another mission")
    context.data_artifact = payload.get("dataArtifact")
    if state.baseline_spec is not None and state.splits is not None:
        for split in (state.splits.validation, state.splits.acceptance):
            context._offline_assets(state.baseline_spec.engine, Path(split.candidate_input_root))
    return state


async def _restore_legacy(context: MissionContext, migration: dict[str, Any]) -> MissionState:
    """Revalidate legacy evidence while preserving irreversible submission history."""
    if migration.get("kind") != "legacy-v1":
        raise ValueError("Unsupported legacy migration kind")
    legacy = migration.get("legacyState")
    if not isinstance(legacy, dict):
        raise ValueError("Legacy migration has no mission state")
    if str(legacy.get("mission_id")) != str(context.job.mission_id):
        raise ValueError("Legacy migration belongs to another mission")
    criteria = MissionCriteria.model_validate(legacy.get("criteria", {}))
    competition_id = legacy.get("competition_id")
    if not isinstance(competition_id, str):
        raise ValueError("Legacy mission has no selected competition")
    competition = await context.adapter.get_competition(competition_id)
    state = MissionState(
        mission_id=context.job.mission_id,
        competition=competition,
        criteria=criteria,
        phases_completed=["discovery"],
    )
    findings = legacy.get("research_findings")
    if isinstance(findings, dict) and findings.get("strategy_recommendations"):
        state.research = ResearchReport(
            summary="Imported research recommendations; measurements are revalidated independently.",
            recommendations=tuple(findings["strategy_recommendations"]),
        )
    code = legacy.get("prototype_code")
    if isinstance(code, str) and code:
        if hashlib.sha256(code.encode()).hexdigest() != migration.get("prototypeCodeSha256"):
            raise ValueError("Legacy candidate digest does not match the imported source")
        state.legacy_program_code = code
    guard = migration.get("submissionGuard", {})
    receipt = legacy.get("final_submission_id")
    receipt_ids = guard.get("submissionIds", []) if isinstance(guard, dict) else []
    if isinstance(receipt, str) and receipt and receipt not in receipt_ids:
        receipt_ids.append(receipt)
    state.historical_submission_ids = tuple(str(receipt_id) for receipt_id in receipt_ids)
    if isinstance(guard, dict) and (guard.get("ambiguous") or guard.get("status") == "ambiguous"):
        state.submission_state = "uncertain"
        state.submission_reconciliation_required = True
    elif receipt_ids:
        for receipt_id in receipt_ids:
            observed = await context.adapter.get_submission_status(competition_id, str(receipt_id))
            if observed.status != "complete":
                state.submission_state = "uncertain"
                break
            state.submission = observed
            state.submission_state = "complete"
    elif legacy.get("current_phase") == "submission" or "submission" in legacy.get(
        "phases_completed", []
    ):
        state.submission_state = "uncertain"
    await context.checkpoint(state)
    return state


def _rebase(value: Any, previous: str, current: str) -> Any:
    if isinstance(value, str) and previous and value.startswith(previous + "/"):
        return current + value[len(previous) :]
    if isinstance(value, dict):
        return {key: _rebase(item, previous, current) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebase(item, previous, current) for item in value]
    return value


def _competition(state: MissionState) -> Competition:
    if state.competition is None:
        raise ValueError("Mission has no validated competition")
    return state.competition


def _dataset(state: MissionState) -> tuple[DatasetManifest, PreparedSplitSet]:
    if state.manifest is None or state.splits is None:
        raise ValueError("Mission has no discovered dataset and trusted splits")
    return state.manifest, state.splits


def _metric(competition: Competition) -> MetricId:
    return MetricId(competition.metric.value)


def _better(candidate: EvaluationResult, baseline: EvaluationResult) -> bool:
    return (
        candidate.score > baseline.score
        if candidate.direction is MetricDirection.MAXIMIZE
        else candidate.score < baseline.score
    )


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _archive_paths(destination: Path, root: Path, names: tuple[str, ...]) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        for name in names:
            path = root / name
            if path.exists():
                archive.add(path, arcname=name, recursive=True)
    destination.chmod(0o600)


def _extract_archives(root: Path) -> None:
    total = 0
    for path in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                destination = (root / item.filename).resolve()
                if not destination.is_relative_to(root.resolve()) or stat.S_ISLNK(
                    item.external_attr >> 16
                ):
                    raise ValueError("Dataset archive contains an unsafe path")
                total += item.file_size
                if total > 50_000_000_000:
                    raise ValueError("Expanded dataset exceeds the configured extraction limit")
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
