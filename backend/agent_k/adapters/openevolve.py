"""OpenEvolve integration adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import csv
import hashlib
import inspect
import json
import os
import tempfile
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypeAliasType

# Third-party (alphabetical)
import logfire
from openevolve import OpenEvolve
from openevolve.config import Config, LLMModelConfig, load_config
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.agents.prompts import OPENEVOLVE_MUTATION_PROMPT
from agent_k.core.exceptions import (
    AdapterError,
    CompetitionNotFoundError,
    DeadlinePassedError,
    EvolutionError,
    SubmissionError,
)
from agent_k.core.hints import PreprocessingHint
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry, Submission
from agent_k.core.protocols import PlatformAdapter
from agent_k.infra.providers import DEVSTRAL_BASE_URL, DEVSTRAL_MODEL_ID, OPENROUTER_FREE_MODELS

__all__ = (
    "OpenEvolveAdapter",
    "OpenEvolveEvolutionConfig",
    "OpenEvolveJobState",
    "OpenEvolveJobStatus",
    "OpenEvolveRunner",
    "OpenEvolveSettings",
    "OpenEvolveSolution",
    "SCHEMA_VERSION",
)

SCHEMA_VERSION: Final[str] = "1.0.0"
_DEFAULT_PRIZE_POOL: Final[int] = 50_000
_DEFAULT_SUBMISSION_ROWS: Final[int] = 50
_LEADERBOARD_SIZE: Final[int] = 25


class OpenEvolveSettings(BaseSettings):
    """Settings for OpenEvolve adapter.

    Environment variables are prefixed with OPENEVOLVE_.
    """

    model_config = SettingsConfigDict(env_prefix="OPENEVOLVE_", env_file=".env", extra="ignore", validate_default=True)
    api_url: str = Field(default="http://localhost:8080", description="Base URL for OpenEvolve API")
    timeout: int = Field(default=30, ge=1, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts for failed requests")
    poll_interval: float = Field(default=5.0, ge=0.1, description="Seconds between status polls")
    simulated_latency: float = Field(
        default=0.0, ge=0.0, description="Delay before marking jobs complete in in-memory mode"
    )
    job_ttl_seconds: int = Field(default=86_400, ge=0, description="Retention for completed jobs in memory")


class OpenEvolveJobState(StrEnum):
    """Lifecycle state for an OpenEvolve job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class OpenEvolveEvolutionConfig(BaseModel):
    """Evolution configuration for OpenEvolve jobs."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION)
    population_size: int = Field(default=50, ge=1)
    max_generations: int = Field(default=50, ge=1)
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    target_score: float | None = Field(default=None)
    seed: int | None = Field(default=None)


class OpenEvolveJobStatus(BaseModel):
    """Status snapshot for an OpenEvolve evolution job."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION)
    job_id: str = Field(..., min_length=1)
    state: OpenEvolveJobState = Field(..., description="Current lifecycle state")
    complete: bool = Field(..., description="Whether the job has reached a terminal state")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last status update time")
    best_score: float | None = Field(default=None, description="Best fitness achieved")
    error_message: str | None = Field(default=None, description="Failure reason, if any")


class OpenEvolveSolution(BaseModel):
    """Best solution produced by an OpenEvolve job."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION)
    job_id: str = Field(..., min_length=1)
    solution_code: str = Field(..., min_length=1)
    fitness: float = Field(..., description="Fitness score for the solution")
    created_at: datetime = Field(..., description="When the solution was finalized")


_OPENEVOLVE_CONTEXT_ENV: Final[str] = "AGENT_K_OPENEVOLVE_CONTEXT"
_OPENROUTER_API_BASE: Final[str] = "https://openrouter.ai/api/v1"
_OPENAI_API_BASE: Final[str] = "https://api.openai.com/v1"
_DEFAULT_OPENEVOLVE_MODEL: Final[str] = OPENROUTER_FREE_MODELS["devstral"]
_DEFAULT_VALIDATION_SPLIT: Final[float] = 0.2


@dataclass
class OpenEvolveRunner:
    """Runner for OpenEvolve evolutionary optimization."""

    config_path: Path | None = None
    work_dir: Path | None = None
    hints: list[PreprocessingHint] = field(default_factory=list)
    model_specs: list[str] = field(default_factory=list)
    metric_direction: str = "minimize"
    validation_split: float = _DEFAULT_VALIDATION_SPLIT
    timeout_seconds: int = 180
    base_prompt: str | None = None

    async def run_evolution(
        self, initial_program: str, max_iterations: int = 50, target_score: float | None = None
    ) -> dict[str, Any]:
        """Run OpenEvolve evolution on a solution."""
        if not initial_program.strip():
            raise EvolutionError("OpenEvolve requires a non-empty initial program.")

        work_dir = self._resolve_work_dir()
        evaluation_file = self._resolve_evaluator_path()

        initial_path = work_dir / "initial_program.py"
        initial_path.write_text(initial_program, encoding="utf-8")

        config = self._load_config()
        config = self._configure_llm(config)
        config = self._configure_prompt(config)
        config = self._configure_evaluator(config)
        config.max_iterations = max_iterations
        config.diff_based_evolution = False

        self._export_context(work_dir)

        evolver = OpenEvolve(
            initial_program_path=str(initial_path),
            evaluation_file=str(evaluation_file),
            config=config,
            output_dir=str(work_dir / "openevolve_output"),
        )

        logfire.info("openevolve_starting", iterations=max_iterations, target_score=target_score)
        best_program = await evolver.run(iterations=max_iterations, target_score=target_score)

        if best_program is None:
            raise EvolutionError("OpenEvolve did not return a best program.")

        artifacts = evolver.database.get_artifacts(best_program.id)
        applied_hints = _parse_applied_hints(artifacts.get("applied_hints"))
        program_summaries = _summarize_programs(evolver)
        iterations_run = max((summary["iteration"] for summary in program_summaries), default=0)

        return {
            "best_solution": best_program.code,
            "best_fitness": _extract_fitness(best_program.metrics),
            "cv_score": best_program.metrics.get("cv_score"),
            "iterations_run": iterations_run,
            "artifacts": artifacts,
            "applied_hints": applied_hints,
            "programs": program_summaries,
        }

    def _resolve_work_dir(self) -> Path:
        if self.work_dir is not None:
            path = self.work_dir.expanduser()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(tempfile.mkdtemp())

    def _resolve_evaluator_path(self) -> Path:
        path = Path(__file__).resolve().parents[1] / "evolution" / "evaluator.py"
        if not path.exists():
            raise EvolutionError(f"OpenEvolve evaluator not found at {path}")
        return path

    def _load_config(self) -> Config:
        config_path = self.config_path or (Path(__file__).resolve().parents[1] / "evolution" / "openevolve_config.yaml")
        if config_path.exists():
            return load_config(config_path)
        return Config()

    def _configure_llm(self, config: Config) -> Config:
        model_specs = _resolve_model_specs(self.model_specs)
        model_configs = [_build_model_config(spec) for spec in model_specs]
        config.llm.models = model_configs
        config.llm.evaluator_models = list(model_configs)
        config.llm.update_model_params({"system_message": config.prompt.system_message}, overwrite=True)
        return config

    def _configure_prompt(self, config: Config) -> Config:
        """Configure prompt settings including template stochasticity.

        Enables template variations that OpenEvolve will randomly sample
        to break repetitive mutation patterns and encourage exploration.
        """
        hint_block = _format_hint_block(self.hints)
        base_prompt = self.base_prompt.strip() if self.base_prompt else ""
        system_message = OPENEVOLVE_MUTATION_PROMPT.strip()
        if base_prompt:
            system_message = f"{system_message}\n\nCONTEXT:\n{base_prompt}"
        if hint_block:
            system_message = f"{system_message}\n\nPREPROCESSING HINTS (REQUIRED):\n{hint_block}"
        config.prompt.system_message = system_message
        config.llm.update_model_params({"system_message": system_message}, overwrite=True)

        # Enable template stochasticity to break repetitive mutation patterns
        config.prompt.use_template_stochasticity = True

        # Define template variations for different mutation aspects
        # OpenEvolve randomly samples from these to add variety to prompts
        config.prompt.template_variations = {
            "optimization_focus": [
                "Focus on improving model accuracy through hyperparameter tuning",
                "Focus on feature engineering to capture more signal",
                "Focus on reducing overfitting through regularization",
                "Focus on ensemble diversity and blending weights",
                "Focus on optimizing the loss function for the metric",
            ],
            "mutation_style": [
                "Make small, incremental improvements to the current approach",
                "Try a different algorithmic approach entirely",
                "Optimize the preprocessing pipeline",
                "Adjust the cross-validation strategy for better stability",
                "Explore different feature combinations",
            ],
        }

        logfire.info(
            "openevolve_template_stochasticity_configured",
            use_template_stochasticity=True,
            variation_categories=list(config.prompt.template_variations.keys()),
            total_variations=sum(len(v) for v in config.prompt.template_variations.values()),
        )

        return config

    def _configure_evaluator(self, config: Config) -> Config:
        config.evaluator.timeout = self.timeout_seconds
        config.evaluator.parallel_evaluations = 1
        config.evaluator.cascade_evaluation = False
        config.prompt.include_artifacts = True
        return config

    def _export_context(self, work_dir: Path) -> None:
        payload = {
            "work_dir": str(work_dir),
            "timeout": self.timeout_seconds,
            "validation_split": self.validation_split,
            "metric_direction": self.metric_direction,
            "hints": [hint.to_dict() for hint in self.hints],
        }
        os.environ[_OPENEVOLVE_CONTEXT_ENV] = json.dumps(payload)
        os.environ.setdefault("ENABLE_ARTIFACTS", "true")


FitnessFunction = TypeAliasType("FitnessFunction", Callable[[str], float | Awaitable[float]])
EvolutionConfigInput = TypeAliasType("EvolutionConfigInput", OpenEvolveEvolutionConfig | dict[str, Any] | None)


@dataclass
class _OpenEvolveJob:
    """Internal tracking for OpenEvolve evolution jobs."""

    job_id: str
    prototype: str
    config: OpenEvolveEvolutionConfig
    created_at: datetime
    updated_at: datetime
    ready_at: datetime
    state: OpenEvolveJobState
    best_solution: str | None = None
    best_score: float | None = None
    error_message: str | None = None


@dataclass
class OpenEvolveAdapter(PlatformAdapter):
    """In-memory adapter for OpenEvolve integration."""

    config: OpenEvolveSettings = field(default_factory=OpenEvolveSettings)
    api_url: str | None = None
    _platform_name: str = "openevolve"
    catalog: list[Competition] = field(default_factory=list)
    _submissions: dict[str, dict[str, Submission]] = field(
        default_factory=lambda: defaultdict(dict), init=False, repr=False
    )
    _submission_payloads: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _evolution_jobs: dict[str, _OpenEvolveJob] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api_url:
            self.config = self.config.model_copy(update={"api_url": self.api_url})
        if not self.catalog:
            self.catalog = _build_default_catalog()

    @property
    def platform_name(self) -> str:
        """Return the platform identifier."""
        return self._platform_name

    async def authenticate(self) -> bool:
        """Authenticate the adapter (no-op for in-memory implementation)."""
        return True

    async def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        """Search the in-memory competition catalog."""
        keyword_terms = [term.lower() for term in (keywords or [])]
        category_terms = {_normalize_category(value) for value in (categories or [])}
        for competition in self.catalog:
            if active_only and not competition.is_active:
                continue
            if category_terms and competition.competition_type.value not in category_terms:
                continue
            if min_prize and (competition.prize_pool or 0) < min_prize:
                continue
            if keyword_terms and not _matches_keywords(competition, keyword_terms):
                continue
            yield competition

    async def get_competition(self, competition_id: str) -> Competition:
        """Get a competition by ID."""
        for competition in self.catalog:
            if competition.id == competition_id:
                return competition
        raise CompetitionNotFoundError(competition_id)

    async def get_leaderboard(self, competition_id: str, *, limit: int = 100) -> list[LeaderboardEntry]:
        """Return leaderboard entries for a competition."""
        competition = await self.get_competition(competition_id)
        submissions = list(self._submissions.get(competition_id, {}).values())
        leaderboard_entries = _build_leaderboard(submissions, competition, limit=limit)
        if leaderboard_entries:
            return leaderboard_entries
        return _build_baseline_leaderboard(competition, limit=limit)

    async def submit(self, competition_id: str, file_path: str, message: str = "") -> Submission:
        """Submit a file payload to the in-memory competition."""
        competition = await self.get_competition(competition_id)
        if not competition.is_active:
            raise DeadlinePassedError(competition_id, competition.deadline.isoformat())

        payload = _load_submission_payload(file_path)
        submission_id = _submission_id(competition_id, payload, message)
        submission = Submission(
            id=submission_id,
            competition_id=competition_id,
            file_name=_submission_filename(file_path, submission_id),
            status="pending",
        )
        self._submissions[competition_id][submission_id] = submission
        self._submission_payloads[submission_id] = payload
        logfire.info("openevolve_submission_received", competition_id=competition_id, submission_id=submission_id)
        return submission

    async def get_submission_status(self, competition_id: str, submission_id: str) -> Submission:
        """Get the latest status for a submission."""
        submissions = self._submissions.get(competition_id, {})
        submission = submissions.get(submission_id)
        if not submission:
            raise SubmissionError(competition_id, "Submission not found", submission_id=submission_id)

        if submission.status == "complete":
            return submission

        payload = self._submission_payloads.get(submission_id, "")
        score = _score_payload(payload, competition_id)
        updated = Submission(
            id=submission.id,
            competition_id=submission.competition_id,
            file_name=submission.file_name,
            submitted_at=submission.submitted_at,
            public_score=score,
            status="complete",
        )
        submissions[submission_id] = updated
        return updated

    async def download_data(self, competition_id: str, destination: str) -> list[str]:
        """Generate local data files for the competition."""
        competition = await self.get_competition(competition_id)
        dest_path = Path(destination)
        dest_path.mkdir(parents=True, exist_ok=True)
        train_path = dest_path / "train.csv"
        test_path = dest_path / "test.csv"
        submission_path = dest_path / "sample_submission.csv"

        _write_tabular_dataset(train_path, include_target=True)
        _write_tabular_dataset(test_path, include_target=False)
        _write_submission_template(submission_path)

        logfire.info("openevolve_data_prepared", competition_id=competition.id, destination=str(dest_path))
        return [str(train_path), str(test_path), str(submission_path)]

    async def submit_evolution(
        self, prototype: str, fitness_function: FitnessFunction | None = None, config: EvolutionConfigInput = None
    ) -> str:
        """Submit an evolution job for a prototype solution."""
        if not prototype:
            raise EvolutionError("Prototype solution cannot be empty")

        evolution_config = _coerce_evolution_config(config)
        created_at = datetime.now(UTC)
        ready_at = created_at + timedelta(seconds=self.config.simulated_latency)
        job_id = _evolution_job_id()
        job = _OpenEvolveJob(
            job_id=job_id,
            prototype=prototype,
            config=evolution_config,
            created_at=created_at,
            updated_at=created_at,
            ready_at=ready_at,
            state=OpenEvolveJobState.RUNNING,
        )

        try:
            job.best_score = await _evaluate_fitness(prototype, fitness_function, evolution_config, job_id)
            job.best_solution = prototype
        except Exception as exc:
            job.state = OpenEvolveJobState.FAILED
            job.error_message = str(exc)
            job.ready_at = created_at
            job.updated_at = created_at

        if job.state != OpenEvolveJobState.FAILED and ready_at <= created_at:
            job.state = OpenEvolveJobState.COMPLETE
            job.updated_at = created_at

        self._prune_jobs()
        self._evolution_jobs[job_id] = job

        logfire.info("openevolve_evolution_submitted", job_id=job_id, state=job.state.value)

        return job_id

    async def get_status(self, job_id: str) -> OpenEvolveJobStatus:
        """Get the status of an evolution job."""
        job = self._evolution_jobs.get(job_id)
        if not job:
            raise AdapterError(f"OpenEvolve job not found: {job_id}")

        now = datetime.now(UTC)
        _refresh_job(job, now)
        self._prune_jobs()

        return OpenEvolveJobStatus(
            job_id=job.job_id,
            state=job.state,
            complete=job.state in {OpenEvolveJobState.COMPLETE, OpenEvolveJobState.FAILED},
            progress=_job_progress(job, now),
            created_at=job.created_at,
            updated_at=job.updated_at,
            best_score=job.best_score,
            error_message=job.error_message,
        )

    async def get_best_solution(self, job_id: str) -> OpenEvolveSolution:
        """Return the best solution for a completed evolution job."""
        job = self._evolution_jobs.get(job_id)
        if not job:
            raise AdapterError(f"OpenEvolve job not found: {job_id}")

        now = datetime.now(UTC)
        _refresh_job(job, now)

        if job.state != OpenEvolveJobState.COMPLETE or not job.best_solution or job.best_score is None:
            raise EvolutionError(f"OpenEvolve job {job_id} is not complete")

        return OpenEvolveSolution(
            job_id=job.job_id, solution_code=job.best_solution, fitness=job.best_score, created_at=job.updated_at
        )

    def _prune_jobs(self) -> None:
        ttl_seconds = self.config.job_ttl_seconds
        if ttl_seconds <= 0:
            return

        cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
        expired = [
            job_id
            for job_id, job in self._evolution_jobs.items()
            if job.updated_at < cutoff and job.state in {OpenEvolveJobState.COMPLETE, OpenEvolveJobState.FAILED}
        ]
        for job_id in expired:
            self._evolution_jobs.pop(job_id, None)


def _resolve_model_specs(model_specs: list[str]) -> list[str]:
    cleaned: list[str] = []
    for spec in model_specs:
        if not isinstance(spec, str):
            continue
        trimmed = spec.strip()
        if not trimmed:
            continue
        if trimmed.startswith("anthropic:"):
            logfire.warning("openevolve_model_unsupported", model=trimmed)
            continue
        cleaned.append(trimmed)
    if cleaned:
        return cleaned
    logfire.info("openevolve_default_model", model=_DEFAULT_OPENEVOLVE_MODEL)
    return [_DEFAULT_OPENEVOLVE_MODEL]


def _build_model_config(model_spec: str) -> LLMModelConfig:
    api_base = _OPENAI_API_BASE
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    name = model_spec

    if model_spec.startswith("openrouter:"):
        name = model_spec[len("openrouter:") :]
        api_base = _OPENROUTER_API_BASE
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logfire.warning("openevolve_missing_api_key", provider="openrouter", model=name)
    elif model_spec.startswith("openai:"):
        name = model_spec[len("openai:") :]
        api_base = _OPENAI_API_BASE
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logfire.warning("openevolve_missing_api_key", provider="openai", model=name)
    elif model_spec.startswith("devstral:"):
        suffix = model_spec[len("devstral:") :]
        if suffix == "local":
            name = DEVSTRAL_MODEL_ID
            api_base = DEVSTRAL_BASE_URL
        elif suffix.startswith("http"):
            name = DEVSTRAL_MODEL_ID
            api_base = suffix
        else:
            name = suffix
            api_base = DEVSTRAL_BASE_URL
        api_key = "not-required"

    return LLMModelConfig(name=name, api_base=api_base, api_key=api_key, weight=1.0)


def _format_hint_block(hints: list[PreprocessingHint], limit: int = 5) -> str:
    if not hints:
        return ""
    ordered = sorted(hints, key=lambda hint: hint.priority, reverse=True)
    lines: list[str] = []
    for idx, hint in enumerate(ordered[:limit], start=1):
        snippet = hint.code_snippet.strip()
        lines.extend(
            [
                f"### Hint {idx}: {hint.id} (priority={hint.priority:.2f})",
                f"Add comment: # Applied hint: {hint.id}",
                f"Description: {hint.description}",
                "```python",
                snippet,
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _parse_applied_hints(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(value).strip() for value in raw if str(value).strip()]
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []
        if cleaned.startswith("["):
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                return [str(value).strip() for value in payload if str(value).strip()]
        return [value.strip() for value in cleaned.split(",") if value.strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _extract_fitness(metrics: dict[str, Any]) -> float:
    for key in ("fitness", "combined_score"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _summarize_programs(evolver: OpenEvolve) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for program in evolver.database.programs.values():
        metrics = program.metrics or {}
        artifacts = evolver.database.get_artifacts(program.id)
        summaries.append(
            {
                "program_id": program.id,
                "iteration": int(program.iteration_found or 0),
                "generation": int(program.generation or 0),
                "cv_score": metrics.get("cv_score"),
                "fitness": _extract_fitness(metrics),
                "applied_hints": _parse_applied_hints(artifacts.get("applied_hints")),
            }
        )
    summaries.sort(key=lambda item: (item["iteration"], item["program_id"]))
    return summaries


def _build_default_catalog() -> list[Competition]:
    """Create a default set of OpenEvolve competitions."""
    now = datetime.now(UTC)
    return [
        Competition(
            id="oe-tabular-classification",
            title="OpenEvolve Tabular Classification",
            description="Binary classification on synthetic tabular data.",
            competition_type=CompetitionType.PLAYGROUND,
            metric=EvaluationMetric.AUC,
            metric_direction="maximize",
            deadline=now + timedelta(days=45),
            prize_pool=_DEFAULT_PRIZE_POOL,
            tags=frozenset({"tabular", "classification"}),
            url="https://openevolve.example.com/competitions/oe-tabular-classification",
        ),
        Competition(
            id="oe-tabular-regression",
            title="OpenEvolve Tabular Regression",
            description="Regression on synthetic tabular data with noisy targets.",
            competition_type=CompetitionType.RESEARCH,
            metric=EvaluationMetric.RMSE,
            metric_direction="minimize",
            deadline=now + timedelta(days=60),
            prize_pool=_DEFAULT_PRIZE_POOL * 2,
            tags=frozenset({"tabular", "regression"}),
            url="https://openevolve.example.com/competitions/oe-tabular-regression",
        ),
        Competition(
            id="oe-timeseries-forecast",
            title="OpenEvolve Time Series Forecasting",
            description="Forecast weekly demand using synthetic time series.",
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.MAE,
            metric_direction="minimize",
            deadline=now + timedelta(days=30),
            prize_pool=_DEFAULT_PRIZE_POOL // 2,
            tags=frozenset({"time_series", "forecasting"}),
            url="https://openevolve.example.com/competitions/oe-timeseries-forecast",
        ),
    ]


def _matches_keywords(competition: Competition, keywords: list[str]) -> bool:
    haystack = " ".join(filter(None, [competition.title, competition.description or ""])).lower()
    return all(keyword in haystack for keyword in keywords)


def _normalize_category(value: str | CompetitionType) -> str:
    if isinstance(value, CompetitionType):
        return value.value
    return str(value).lower()


def _evolution_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _coerce_evolution_config(config: EvolutionConfigInput) -> OpenEvolveEvolutionConfig:
    if config is None:
        return OpenEvolveEvolutionConfig()
    if isinstance(config, OpenEvolveEvolutionConfig):
        return config
    if isinstance(config, dict):
        return OpenEvolveEvolutionConfig.model_validate(config)
    raise TypeError("Unsupported OpenEvolve config type")


async def _evaluate_fitness(
    prototype: str, fitness_function: FitnessFunction | None, config: OpenEvolveEvolutionConfig, seed: str
) -> float:
    if fitness_function is None:
        salt = str(config.seed) if config.seed is not None else seed
        return _score_evolution(prototype, salt)

    result = fitness_function(prototype)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, (int, float)):
        raise ValueError("Fitness function must return a numeric score")
    return float(result)


def _refresh_job(job: _OpenEvolveJob, now: datetime) -> None:
    if job.state in {OpenEvolveJobState.COMPLETE, OpenEvolveJobState.FAILED}:
        return
    if now >= job.ready_at:
        job.state = OpenEvolveJobState.COMPLETE
    job.updated_at = now


def _job_progress(job: _OpenEvolveJob, now: datetime) -> float:
    if job.state in {OpenEvolveJobState.COMPLETE, OpenEvolveJobState.FAILED}:
        return 1.0
    total = (job.ready_at - job.created_at).total_seconds()
    if total <= 0:
        return 1.0
    elapsed = (now - job.created_at).total_seconds()
    return max(0.0, min(elapsed / total, 1.0))


def _submission_id(competition_id: str, payload: str, message: str) -> str:
    seed = f"{competition_id}:{message}:{payload}".encode()
    return hashlib.sha256(seed).hexdigest()[:12]


def _submission_filename(file_path: str, submission_id: str) -> str:
    path = Path(file_path)
    return path.name if path.exists() else f"inline_{submission_id}.csv"


def _load_submission_payload(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else file_path


def _score_payload(payload: str, competition_id: str) -> float:
    return _score_evolution(payload, competition_id)


def _score_evolution(payload: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{payload}".encode()).hexdigest()
    raw_score = int(digest[:8], 16) / 0xFFFFFFFF
    return round(0.5 + raw_score * 0.5, 6)


def _build_leaderboard(
    submissions: list[Submission], competition: Competition, *, limit: int
) -> list[LeaderboardEntry]:
    scored = [s for s in submissions if s.public_score is not None]
    if not scored:
        return []

    reverse = competition.metric_direction == "maximize"
    scored_sorted = sorted(scored, key=lambda s: s.public_score or 0.0, reverse=reverse)
    return [
        LeaderboardEntry(
            rank=idx,
            team_name=f"open-evolve-team-{idx}",
            score=submission.public_score or 0.0,
            entries=1,
            last_submission=submission.submitted_at,
        )
        for idx, submission in enumerate(scored_sorted[:limit], start=1)
    ]


def _build_baseline_leaderboard(competition: Competition, *, limit: int) -> list[LeaderboardEntry]:
    entries: list[LeaderboardEntry] = []
    max_rank = min(limit, _LEADERBOARD_SIZE)
    step = 0.4 / _LEADERBOARD_SIZE
    for idx in range(1, max_rank + 1):
        score = 0.05 + (idx - 1) * step if competition.metric_direction == "minimize" else 0.95 - (idx - 1) * step
        entries.append(
            LeaderboardEntry(
                rank=idx,
                team_name=f"baseline-team-{idx}",
                score=round(score, 6),
                entries=1,
                last_submission=datetime.now(UTC),
            )
        )
    return entries


def _write_tabular_dataset(path: Path, *, include_target: bool) -> None:
    headers = ["id", "feature_1", "feature_2", "feature_3"]
    if include_target:
        headers.append("target")

    rows = []
    for idx in range(_DEFAULT_SUBMISSION_ROWS):
        row = {
            "id": idx,
            "feature_1": round((idx % 10) * 0.1, 4),
            "feature_2": round((idx % 7) * 0.15, 4),
            "feature_3": round((idx % 5) * 0.2, 4),
        }
        if include_target:
            row["target"] = 1 if idx % 2 == 0 else 0
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_submission_template(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "target"])
        writer.writeheader()
        for idx in range(_DEFAULT_SUBMISSION_ROWS):
            writer.writerow({"id": idx, "target": 0.5})
