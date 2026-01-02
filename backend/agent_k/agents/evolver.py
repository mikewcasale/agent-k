"""Evolver agent - evolutionary optimization for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import csv
import hashlib
import json
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, cast

# Third-party (alphabetical)
import logfire
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, ModelSettings, RunContext, ToolOutput, ToolReturn
from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.agents import register_agent
from agent_k.agents.base import MemoryMixin, universal_tool_preparation
from agent_k.agents.prompts import EVOLVER_SYSTEM_PROMPT
from agent_k.core.constants import (
    DEFAULT_KAGGLE_MCP_URL,
    DEFAULT_MODEL,
    EVOLUTION_POPULATION_SIZE,
    MAX_EVOLUTION_GENERATIONS,
    SOLUTION_EXECUTION_TIMEOUT_SECONDS,
)
from agent_k.core.data import stage_competition_data
from agent_k.core.solution import execute_solution, parse_baseline_score
from agent_k.infra.providers import get_model
from agent_k.toolsets import code_toolset, create_production_toolset, prepare_code_execution_tool, prepare_memory_tool

if TYPE_CHECKING:
    from agent_k.core.models import Competition
    from agent_k.core.protocols import PlatformAdapter
    from agent_k.ui.ag_ui import EventEmitter

__all__ = (
    'EVOLUTION_OUTPUT_TYPE',
    'EvolutionFailure',
    'EvolutionResult',
    'EvolverAgent',
    'EvolverDeps',
    'EvolverSettings',
    'EVOLVER_SYSTEM_PROMPT',
    'SCHEMA_VERSION',
    'evolver_agent',
    'settings',
)

SCHEMA_VERSION: Final[str] = "1.0.0"
_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])(-?\d+\.?\d*)(?![\w.])")
_FILLNA_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>\s*)(?P<var>\w+)\s*=\s*pd\.read_csv\(.*\)$", re.MULTILINE
)
_DEF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(def (?!__)\w+).*?(?=^(?:def |class )|\Z)", re.MULTILINE | re.DOTALL
)
_NUMERIC_PIPELINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(numeric_transformer\s*=\s*Pipeline\(steps=\[\s*)(?P<steps>.*?)(\s*\]\))",
    re.DOTALL,
)
_HYPERPARAM_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "n_estimators": re.compile(r"(n_estimators\s*=\s*)(\d+)", re.IGNORECASE),
    "learning_rate": re.compile(r"(learning_rate\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_depth": re.compile(r"(max_depth\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_leaf": re.compile(r"(min_samples_leaf\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_split": re.compile(r"(min_samples_split\s*=\s*)(\d+)", re.IGNORECASE),
    "max_features": re.compile(r"(max_features\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "subsample": re.compile(r"(subsample\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "n_neighbors": re.compile(r"(n_neighbors\s*=\s*)(\d+)", re.IGNORECASE),
    "C": re.compile(r"(\bC\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "alpha": re.compile(r"(alpha\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "l1_ratio": re.compile(r"(l1_ratio\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_iter": re.compile(r"(max_iter\s*=\s*)(\d+)", re.IGNORECASE),
}
_HYPERPARAM_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "learning_rate": (0.001, 1.0),
    "subsample": (0.1, 1.0),
    "max_features": (0.1, 1.0),
    "C": (0.01, 100.0),
    "alpha": (0.0001, 10.0),
    "l1_ratio": (0.0, 1.0),
}
_HYPERPARAM_INTEGER_KEYS: Final[set[str]] = {
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "n_neighbors",
    "max_iter",
}
_MODEL_SWAPS: Final[dict[str, str]] = {
    "RandomForestClassifier": "GradientBoostingClassifier",
    "RandomForestRegressor": "GradientBoostingRegressor",
    "GradientBoostingClassifier": "RandomForestClassifier",
    "GradientBoostingRegressor": "RandomForestRegressor",
    "ExtraTreesClassifier": "RandomForestClassifier",
    "ExtraTreesRegressor": "RandomForestRegressor",
    "HistGradientBoostingClassifier": "GradientBoostingClassifier",
    "HistGradientBoostingRegressor": "GradientBoostingRegressor",
    "LogisticRegression": "LinearSVC",
    "LinearRegression": "Ridge",
}
_MODEL_IMPORTS: Final[dict[str, str]] = {
    "RandomForestClassifier": "sklearn.ensemble",
    "RandomForestRegressor": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "GradientBoostingRegressor": "sklearn.ensemble",
    "ExtraTreesClassifier": "sklearn.ensemble",
    "ExtraTreesRegressor": "sklearn.ensemble",
    "HistGradientBoostingClassifier": "sklearn.ensemble",
    "HistGradientBoostingRegressor": "sklearn.ensemble",
    "LogisticRegression": "sklearn.linear_model",
    "LinearRegression": "sklearn.linear_model",
    "LinearSVC": "sklearn.svm",
    "Ridge": "sklearn.linear_model",
}
_MODEL_FAMILY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("random_forest", re.compile(r"\b(RandomForestClassifier|RandomForestRegressor)\b")),
    ("extra_trees", re.compile(r"\b(ExtraTreesClassifier|ExtraTreesRegressor)\b")),
    ("gradient_boosting", re.compile(r"\b(GradientBoostingClassifier|GradientBoostingRegressor)\b")),
    (
        "hist_gradient_boosting",
        re.compile(r"\b(HistGradientBoostingClassifier|HistGradientBoostingRegressor)\b"),
    ),
    (
        "linear",
        re.compile(
            r"\b(LogisticRegression|LinearRegression|Ridge|Lasso|ElasticNet|SGDClassifier|SGDRegressor)\b"
        ),
    ),
    ("svm", re.compile(r"\b(LinearSVC|LinearSVR|SVC|SVR)\b")),
    ("knn", re.compile(r"\b(KNeighborsClassifier|KNeighborsRegressor)\b")),
    ("naive_bayes", re.compile(r"\b(GaussianNB|MultinomialNB|BernoulliNB)\b")),
    ("tree", re.compile(r"\b(DecisionTreeClassifier|DecisionTreeRegressor)\b")),
    (
        "ensemble",
        re.compile(r"\b(BaggingClassifier|BaggingRegressor|AdaBoostClassifier|AdaBoostRegressor)\b"),
    ),
    ("stacking", re.compile(r"\b(StackingClassifier|StackingRegressor)\b")),
)
_COMPLEXITY_BINS: Final[tuple[int, ...]] = (120, 200, 320, 480, 640)


@dataclass(frozen=True, slots=True)
class EvolutionArchiveEntry:
    """Tracked candidate for MAP-Elites-style sampling."""

    code: str
    fitness: float
    cv_score: float
    complexity: int
    complexity_bin: int
    model_family: str
    signature: str

    def to_payload(self, *, max_chars: int) -> dict[str, Any]:
        """Serialize entry for tool outputs."""
        truncated = False
        code = self.code
        if max_chars > 0 and len(code) > max_chars:
            code = code[:max_chars].rstrip() + "\n# ... truncated"
            truncated = True
        return {
            "fitness": self.fitness,
            "cv_score": self.cv_score,
            "complexity": self.complexity,
            "complexity_bin": self.complexity_bin,
            "model_family": self.model_family,
            "signature": self.signature,
            "code": code,
            "truncated": truncated,
        }


class EvolverSettings(BaseSettings):
    """Configuration for the Evolver agent."""

    model_config = SettingsConfigDict(env_prefix='EVOLVER_', env_file='.env', extra='ignore', validate_default=True)
    model: str = Field(default=DEFAULT_MODEL, description='Model identifier for evolution tasks')
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description='Sampling temperature for evolution prompts')
    max_tokens: int = Field(default=4096, ge=1, description='Maximum tokens for responses')
    solution_timeout: int = Field(
        default=SOLUTION_EXECUTION_TIMEOUT_SECONDS,
        ge=1,
        description='Timeout for executing a candidate solution (seconds)',
    )
    tool_retries: int = Field(default=3, ge=0, description='Tool retry attempts')
    output_retries: int = Field(default=2, ge=0, description='Output validation retry attempts')
    population_size: int = Field(default=EVOLUTION_POPULATION_SIZE, ge=1, description='Population size for evolution')
    max_generations: int = Field(default=MAX_EVOLUTION_GENERATIONS, ge=1, description='Maximum evolution generations')
    min_generations: int = Field(default=0, ge=0, description='Minimum generations before convergence checks')
    convergence_threshold: int = Field(default=5, ge=1, description='Generations without improvement before stopping')
    enable_thinking: bool = Field(default=True, description='Enable extended reasoning mode for supported models')
    thinking_budget_tokens: int = Field(default=4096, ge=0, description='Token budget for model thinking mode')
    enable_kaggle_mcp: bool = Field(default=False, description='Enable Kaggle MCP tool access')
    kaggle_mcp_url: str = Field(default=DEFAULT_KAGGLE_MCP_URL, description='Kaggle MCP endpoint')
    enable_submission_tool: bool = Field(default=False, description='Allow submissions during evolution')
    cascade_evaluation: bool = Field(
        default=True,
        description="Enable cascade evaluation for candidate filtering",
    )
    cascade_stage1_rows: int = Field(
        default=300,
        ge=0,
        description="Max rows for quick evaluation stage",
    )
    cascade_stage1_timeout: int = Field(
        default=45,
        ge=1,
        description="Timeout for quick evaluation stage",
    )
    cascade_relative_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Fraction of best fitness required to run full evaluation",
    )
    cascade_floor_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum quick fitness required to run full evaluation",
    )
    elite_sample_top: int = Field(default=3, ge=0, description="Default top elites to sample")
    elite_sample_diverse: int = Field(default=2, ge=0, description="Default diverse elites to sample")
    elite_code_max_chars: int = Field(
        default=8000,
        ge=256,
        description="Max chars per elite code snippet",
    )

    @model_validator(mode='after')
    def validate_evolution_params(self) -> Self:
        """Validate cross-field evolution configuration."""
        if self.min_generations > self.max_generations:
            raise ValueError('min_generations cannot exceed max_generations')
        if self.convergence_threshold > self.max_generations:
            raise ValueError('convergence_threshold cannot exceed max_generations')
        if self.population_size < 2 and self.max_generations > 1:
            raise ValueError('population_size must be >= 2 when running multiple generations')
        return self

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings from configuration."""
        settings: ModelSettings = {'temperature': self.temperature, 'max_tokens': self.max_tokens}

        if self.enable_thinking and 'anthropic' in self.model:
            logfire.info('evolver_thinking_disabled', model=self.model, reason='anthropic_output_tools_incompatible')
            return settings

        return settings


@dataclass
class EvolverDeps:
    """Dependencies for the Evolver agent."""

    competition: Competition
    event_emitter: EventEmitter
    platform_adapter: PlatformAdapter
    data_dir: Path
    train_path: Path
    test_path: Path
    sample_path: Path
    target_columns: list[str]
    train_target_columns: list[str]
    initial_solution: str = ''
    population_size: int = EVOLUTION_POPULATION_SIZE
    max_generations: int = MAX_EVOLUTION_GENERATIONS
    min_generations: int = 0
    solution_timeout: int = SOLUTION_EXECUTION_TIMEOUT_SECONDS
    target_score: float = 0.0
    generation_offset: int = 0
    best_solution: str | None = None
    best_fitness: float | None = None
    generation_history: list[dict[str, Any]] = field(default_factory=list)
    elite_archive: dict[tuple[int, str], EvolutionArchiveEntry] = field(default_factory=dict)


class EvolutionResult(BaseModel):
    """Result of evolution process."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    best_solution: str = Field(description='Best solution code')
    best_fitness: float = Field(description='Fitness score of best solution')
    generations_completed: int = Field(default=0, ge=0, description='Number of generations completed')
    convergence_achieved: bool = Field(default=False, description='Whether convergence criteria were met')
    convergence_reason: str | None = Field(default=None, description='Reason for convergence if achieved')
    submission_ready: bool = Field(default=False, description='Whether output is ready for submission')


class EvolutionFailure(BaseModel):
    """Failure result for evolution process."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    error_type: str = Field(description='Classification of failure')
    error_message: str = Field(description='Human-readable error')
    partial_solution: str | None = Field(default=None, description='Best available solution snippet, if any')
    recoverable: bool = Field(default=True, description='Whether the failure is likely recoverable')


EVOLUTION_OUTPUT_TYPE: Final[list[Any]] = [
    ToolOutput[EvolutionResult](EvolutionResult, name='return_success'),
    ToolOutput[EvolutionFailure](EvolutionFailure, name='return_failure'),
    DeferredToolRequests,
]


class EvolverAgent(MemoryMixin):
    """Evolver agent encapsulating evolutionary optimization functionality.

    This class wraps the pydantic-ai Agent and provides all evolution tools
    as instance methods for cleaner organization and testing.
    """

    def __init__(self, settings: EvolverSettings | None = None, *, register: bool = True) -> None:
        """Initialize the Evolver agent.

        Args:
            settings: Configuration for the agent. Uses defaults if not provided.
            register: Whether to register this agent in the global registry.
        """
        self._settings = settings or EvolverSettings()
        self._toolset: FunctionToolset[EvolverDeps] = FunctionToolset(id='evolver')
        self._memory_backend = self._init_memory_backend()
        self._register_tools()
        self._agent = self._create_agent()
        if register:
            register_agent('evolver', self._agent)
        self._setup_memory()

    @property
    def agent(self) -> Agent[EvolverDeps, EvolutionResult | EvolutionFailure]:
        """Return the underlying pydantic-ai Agent."""
        return self._agent

    @property
    def settings(self) -> EvolverSettings:
        """Return current settings."""
        return self._settings

    async def mutate_solution(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        mutation_type: str,
        mutation_params: dict[str, Any] | None = None,
    ) -> str:
        """Apply mutation to a solution.

        Args:
            ctx: Run context with dependencies.
            solution_code: The solution code to mutate.
            mutation_type: Type of mutation (point, structural, hyperparameter, crossover).
            mutation_params: Optional parameters for the mutation.

        Returns:
            Mutated solution code.
        """
        with logfire.span('evolver.mutate', mutation_type=mutation_type):
            await ctx.deps.event_emitter.emit(
                'tool-start',
                {
                    'taskId': 'evolution_mutate',
                    'toolCallId': f'mutate_{mutation_type}',
                    'toolType': 'code_executor',
                    'operation': f'mutate_{mutation_type}',
                },
            )

            params = mutation_params or {}
            mutations = {
                'crossover': lambda: self._apply_crossover(solution_code, params.get('other_solution', ''), params),
                'hyperparameter': lambda: self._apply_hyperparameter_mutation(solution_code, params),
                'point': lambda: self._apply_point_mutation(solution_code, params),
                'structural': lambda: self._apply_structural_mutation(solution_code, params),
            }
            return mutations.get(mutation_type, lambda: solution_code)()

    async def evaluate_fitness(
        self, ctx: RunContext[EvolverDeps], solution_code: str, validation_split: float = 0.2
    ) -> ToolReturn:
        """Evaluate solution fitness.

        Args:
            ctx: Run context with dependencies.
            solution_code: Solution code to evaluate.
            validation_split: Fraction of data for validation.

        Returns:
            ToolReturn with fitness results.
        """
        with logfire.span('evolver.evaluate_fitness'):
            tool_call_id = f'fitness_{id(solution_code):x}'
            await ctx.deps.event_emitter.emit_tool_start(
                task_id='evolution_evaluate',
                tool_call_id=tool_call_id,
                tool_type='code_executor',
                operation='evaluate_fitness',
            )

            result = await self._run_evaluation(ctx, solution_code, validation_split=validation_split)
            eligible_for_archive = result["valid"] and result.get("stage") != "stage1"

            if eligible_for_archive:
                if ctx.deps.best_fitness is None or result["fitness"] > ctx.deps.best_fitness:
                    ctx.deps.best_fitness = result["fitness"]
                    ctx.deps.best_solution = solution_code

            if result["valid"]:
                archive_entry = self._build_archive_entry(
                    solution_code,
                    result["fitness"],
                    result["cv_score"],
                )
                if eligible_for_archive:
                    self._update_elite_archive(ctx.deps, archive_entry)
                result.update(
                    {
                        "complexity": archive_entry.complexity,
                        "complexity_bin": archive_entry.complexity_bin,
                        "model_family": archive_entry.model_family,
                        "archive_size": len(ctx.deps.elite_archive),
                    }
                )

                await ctx.deps.event_emitter.emit(
                    "fitness-update",
                    {
                        "fitness": result["fitness"],
                        "cv_score": result["cv_score"],
                        "validation_split": validation_split,
                        "stage": result.get("stage", "full"),
                    },
                )
            else:
                await ctx.deps.event_emitter.emit_tool_error(
                    task_id="evolution_evaluate",
                    tool_call_id=tool_call_id,
                    error=result.get("error") or "Invalid solution",
                )

            await ctx.deps.event_emitter.emit_tool_result(
                task_id="evolution_evaluate",
                tool_call_id=tool_call_id,
                result=result,
                duration_ms=result["runtime_ms"],
            )

            summary = f'Fitness {result["fitness"]:.4f}, CV {result["cv_score"]:.4f}, valid={result["valid"]}'
            return ToolReturn(
                return_value=result,
                content=summary,
                metadata={'tool_call_id': tool_call_id, 'runtime_ms': result['runtime_ms']},
            )

    async def record_generation(
        self,
        ctx: RunContext[EvolverDeps],
        generation: int,
        best_fitness: float,
        mean_fitness: float,
        worst_fitness: float,
        mutations: dict[str, int],
    ) -> None:
        """Record generation metrics.

        Args:
            ctx: Run context with dependencies.
            generation: Generation number.
            best_fitness: Best fitness in generation.
            mean_fitness: Mean fitness in generation.
            worst_fitness: Worst fitness in generation.
            mutations: Count of each mutation type applied.
        """
        global_generation = generation + ctx.deps.generation_offset
        metrics = {
            'generation': global_generation,
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'worst_fitness': worst_fitness,
            'mutations': mutations,
        }

        ctx.deps.generation_history.append(metrics)
        await ctx.deps.event_emitter.emit_generation_complete(
            generation=global_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            worst_fitness=worst_fitness,
            population_size=ctx.deps.population_size,
            mutations=mutations,
        )

        logfire.info(
            'evolution_generation', generation=global_generation, best_fitness=best_fitness, mean_fitness=mean_fitness
        )

    async def check_convergence(
        self, ctx: RunContext[EvolverDeps], threshold_generations: int = 5, improvement_threshold: float = 0.001
    ) -> ToolReturn:
        """Check if evolution has converged.

        Args:
            ctx: Run context with dependencies.
            threshold_generations: Generations to check for improvement.
            improvement_threshold: Minimum improvement required.

        Returns:
            Convergence status dictionary.
        """
        history = ctx.deps.generation_history
        if ctx.deps.min_generations and len(history) < ctx.deps.min_generations:
            result = {
                'converged': False,
                'reason': f'Minimum generations not reached ({len(history)}/{ctx.deps.min_generations})',
            }
            return ToolReturn(return_value=result, content=json.dumps(result))

        if len(history) < threshold_generations:
            result = {'converged': False, 'reason': 'Not enough generations'}
            return ToolReturn(return_value=result, content=json.dumps(result))

        recent_fitness = [g['best_fitness'] for g in history[-threshold_generations:]]
        best = max(recent_fitness)
        improvement = best - min(recent_fitness)
        if improvement < improvement_threshold:
            result = {
                'converged': True,
                'reason': f'No improvement for {threshold_generations} generations',
                'best_fitness': best,
            }
            return ToolReturn(return_value=result, content=json.dumps(result))

        if ctx.deps.target_score > 0:
            target_fitness = self._fitness_from_score(ctx.deps.target_score, ctx.deps.competition.metric_direction)
            if best >= target_fitness:
                result = {'converged': True, 'reason': 'Target score achieved', 'best_fitness': best}
                return ToolReturn(return_value=result, content=json.dumps(result))

        result = {'converged': False, 'reason': 'Evolution in progress', 'recent_improvement': improvement}
        return ToolReturn(return_value=result, content=json.dumps(result))

    async def sample_elites(
        self,
        ctx: RunContext[EvolverDeps],
        num_top: int | None = None,
        num_diverse: int | None = None,
    ) -> ToolReturn:
        """Sample elite solutions for prompt construction."""
        top = self._settings.elite_sample_top if num_top is None else max(0, num_top)
        diverse = self._settings.elite_sample_diverse if num_diverse is None else max(0, num_diverse)
        entries = self._select_elite_samples(ctx.deps, top=top, diverse=diverse)
        payload = [entry.to_payload(max_chars=self._settings.elite_code_max_chars) for entry in entries]
        summary = f"Sampled {len(payload)} elites from {len(ctx.deps.elite_archive)} archive cells."
        return ToolReturn(return_value=payload, content=summary)

    async def submit_to_kaggle(
        self, ctx: RunContext[EvolverDeps], solution_code: str, message: str = 'AGENT-K submission'
    ) -> ToolReturn:
        """Submit solution to Kaggle via the platform adapter.

        Args:
            ctx: Run context with dependencies.
            solution_code: Solution code to submit.
            message: Submission message.

        Returns:
            Submission result dictionary.
        """
        with logfire.span('evolver.submit', competition_id=ctx.deps.competition.id):
            tool_call_id = f'submit_{len(ctx.deps.generation_history)}'
            await ctx.deps.event_emitter.emit(
                'tool-start',
                {
                    'taskId': 'evolution_submit',
                    'toolCallId': tool_call_id,
                    'toolType': 'kaggle_mcp',
                    'operation': 'competitions.submit',
                },
            )

            result = await self._submit_solution(ctx, solution_code, message=message)
            if result.get('status') == 'failed':
                await ctx.deps.event_emitter.emit_tool_error(
                    task_id='evolution_submit',
                    tool_call_id=tool_call_id,
                    error=result.get('error', 'Submission failed'),
                )
                summary = f'Submission failed: {result.get("error", "Unknown error")}'
                return ToolReturn(return_value=result, content=summary)

            await ctx.deps.event_emitter.emit_tool_result(
                task_id='evolution_submit',
                tool_call_id=tool_call_id,
                result=result,
                duration_ms=result.get('runtime_ms', 0),
            )

            summary = f'Submission status: {result.get("status", "unknown")}'
            return ToolReturn(return_value=result, content=summary)

    def _create_agent(self) -> Agent[EvolverDeps, EvolutionResult | EvolutionFailure]:
        """Create the underlying pydantic-ai agent."""
        builtin_tools: list[Any] = [prepare_code_execution_tool]
        if self._settings.enable_kaggle_mcp:
            builtin_tools.insert(0, MCPServerTool(id='kaggle', url=self._settings.kaggle_mcp_url))
        if self._memory_backend is not None:
            builtin_tools.append(prepare_memory_tool)

        require_approval = ['submit_to_kaggle'] if self._settings.enable_submission_tool else None
        agent: Agent[EvolverDeps, EvolutionResult | EvolutionFailure] = Agent(
            model=get_model(self._settings.model),
            deps_type=EvolverDeps,
            output_type=EVOLUTION_OUTPUT_TYPE,
            instructions=EVOLVER_SYSTEM_PROMPT,
            name='evolver',
            model_settings=self._settings.model_settings,
            retries=self._settings.tool_retries,
            output_retries=self._settings.output_retries,
            builtin_tools=builtin_tools,
            toolsets=[
                create_production_toolset(
                    [self._toolset, cast('FunctionToolset[EvolverDeps]', code_toolset)],
                    require_approval_for=require_approval,
                )
            ],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

        agent.output_validator(self._validate_evolution_result)
        agent.instructions(self._add_evolution_context)
        return agent

    def _register_tools(self) -> None:
        """Register all evolution tools with the toolset."""
        self._toolset.tool(self.mutate_solution)
        self._toolset.tool(self.evaluate_fitness)
        self._toolset.tool(self.record_generation)
        self._toolset.tool(self.check_convergence)
        self._toolset.tool(self.sample_elites)
        if self._settings.enable_submission_tool:
            self._toolset.tool(requires_approval=True)(self.submit_to_kaggle)

    async def _validate_evolution_result(
        self, ctx: RunContext[EvolverDeps], result: EvolutionResult | EvolutionFailure
    ) -> EvolutionResult | EvolutionFailure:
        """Validate evolution results."""
        if ctx.partial_output:
            return result
        match result:
            case EvolutionFailure(error_type=error_type, error_message=error_message) if (
                not error_type or not error_message
            ):
                raise ModelRetry('EvolutionFailure must include error_type and error_message.')
            case EvolutionResult(best_fitness=best_fitness) if best_fitness < 0:
                raise ModelRetry('best_fitness must be >= 0. Provide a valid fitness score.')
            case EvolutionResult(best_solution=best_solution) if not best_solution:
                raise ModelRetry('best_solution is required. Provide the best solution code.')
        return result

    async def _add_evolution_context(self, ctx: RunContext[EvolverDeps]) -> str:
        """Add evolution-specific context to instructions."""
        comp = ctx.deps.competition
        deps = ctx.deps
        sections = [
            (
                'COMPETITION CONTEXT:\n'
                f'- ID: {comp.id}\n'
                f'- Title: {comp.title}\n'
                f'- Metric: {comp.metric.value} ({comp.metric_direction})\n'
                f'- Target Score: {deps.target_score}'
            ),
            (
                'EVOLUTION STATE:\n'
                f'- Generations Completed: {len(deps.generation_history)}\n'
                f'- Population Size: {deps.population_size}\n'
                f'- Max Generations: {deps.max_generations}\n'
                f'- Minimum Generations: {deps.min_generations}\n'
                f'- Generation Offset: {deps.generation_offset}'
            ),
        ]
        if deps.generation_history:
            last_gen = deps.generation_history[-1]
            sections.append(
                'LAST GENERATION:\n'
                f'- Best Fitness: {last_gen.get("best_fitness", "N/A")}\n'
                f'- Mean Fitness: {last_gen.get("mean_fitness", "N/A")}'
            )
        if deps.best_solution:
            sections.append(f'BEST SOLUTION AVAILABLE: {len(deps.best_solution)} chars')
        if deps.elite_archive:
            families = sorted({entry.model_family for entry in deps.elite_archive.values()})
            family_preview = ", ".join(families[:6])
            suffix = "..." if len(families) > 6 else ""
            sections.append(
                "ELITE ARCHIVE:\n"
                f"- Cells: {len(deps.elite_archive)}\n"
                f"- Families: {family_preview}{suffix}\n"
                "- Use sample_elites to retrieve top + diverse candidates."
            )

        return '\n\n'.join(sections)

    def _build_execution_env(self, validation_split: float) -> dict[str, str]:
        return {'AGENT_K_VALIDATION_SPLIT': f'{validation_split:.6f}'}

    def _fitness_from_score(self, score: float, direction: str) -> float:
        return 1.0 / (1.0 + max(score, 0.0)) if direction == 'minimize' else max(score, 0.0)

    def _solution_complexity(self, code: str) -> int:
        return sum(1 for line in code.splitlines() if line.strip())

    def _complexity_bin(self, complexity: int) -> int:
        for idx, threshold in enumerate(_COMPLEXITY_BINS):
            if complexity <= threshold:
                return idx
        return len(_COMPLEXITY_BINS)

    def _model_family(self, code: str) -> str:
        for family, pattern in _MODEL_FAMILY_PATTERNS:
            if pattern.search(code):
                return family
        return "unknown"

    def _build_archive_entry(self, code: str, fitness: float, cv_score: float) -> EvolutionArchiveEntry:
        complexity = self._solution_complexity(code)
        return EvolutionArchiveEntry(
            code=code,
            fitness=fitness,
            cv_score=cv_score,
            complexity=complexity,
            complexity_bin=self._complexity_bin(complexity),
            model_family=self._model_family(code),
            signature=hashlib.sha256(code.encode()).hexdigest()[:12],
        )

    def _update_elite_archive(self, deps: EvolverDeps, entry: EvolutionArchiveEntry) -> None:
        key = (entry.complexity_bin, entry.model_family)
        existing = deps.elite_archive.get(key)
        if existing is None or entry.fitness > existing.fitness:
            deps.elite_archive[key] = entry

    def _select_elite_samples(
        self, deps: EvolverDeps, *, top: int, diverse: int
    ) -> list[EvolutionArchiveEntry]:
        entries = list(deps.elite_archive.values())
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda entry: entry.fitness, reverse=True)
        selected: list[EvolutionArchiveEntry] = []
        for entry in sorted_entries[:top]:
            selected.append(entry)

        used_signatures = {entry.signature for entry in selected}
        used_families = {entry.model_family for entry in selected}
        used_bins = {entry.complexity_bin for entry in selected}

        target_size = top + diverse
        if diverse > 0:
            for entry in sorted_entries:
                if entry.signature in used_signatures:
                    continue
                if entry.model_family not in used_families or entry.complexity_bin not in used_bins:
                    selected.append(entry)
                    used_signatures.add(entry.signature)
                    used_families.add(entry.model_family)
                    used_bins.add(entry.complexity_bin)
                if len(selected) >= target_size:
                    break

        if len(selected) < target_size:
            for entry in sorted_entries:
                if entry.signature in used_signatures:
                    continue
                selected.append(entry)
                if len(selected) >= target_size:
                    break

        return selected

    async def _run_evaluation(
        self, ctx: RunContext[EvolverDeps], solution_code: str, *, validation_split: float
    ) -> dict[str, Any]:
        if not self._settings.cascade_evaluation or self._settings.cascade_stage1_rows <= 0:
            result = await self._evaluate_solution(ctx, solution_code, validation_split=validation_split)
            result["stage"] = "full"
            return result

        quick_timeout = min(self._settings.cascade_stage1_timeout, ctx.deps.solution_timeout)
        quick_result = await self._evaluate_solution(
            ctx,
            solution_code,
            validation_split=validation_split,
            max_rows=self._settings.cascade_stage1_rows,
            timeout_seconds=quick_timeout,
        )
        quick_result["stage"] = "stage1"
        if not quick_result["valid"]:
            return quick_result

        threshold = None
        if ctx.deps.best_fitness is not None:
            threshold = max(
                ctx.deps.best_fitness * self._settings.cascade_relative_threshold,
                self._settings.cascade_floor_threshold,
            )
        if threshold is not None and quick_result["fitness"] < threshold:
            quick_result["stage_threshold"] = threshold
            return quick_result

        full_result = await self._evaluate_solution(ctx, solution_code, validation_split=validation_split)
        full_result["stage"] = "full"
        full_result["stage1_fitness"] = quick_result["fitness"]
        full_result["stage1_cv_score"] = quick_result["cv_score"]
        full_result["stage1_runtime_ms"] = quick_result["runtime_ms"]
        full_result["runtime_ms"] += quick_result["runtime_ms"]
        return full_result

    async def _evaluate_solution(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        *,
        validation_split: float,
        max_rows: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(dir=str(ctx.deps.data_dir)) as run_dir:
            run_path = Path(run_dir)
            stage_competition_data(
                ctx.deps.train_path,
                ctx.deps.test_path,
                ctx.deps.sample_path,
                run_path,
                competition_id=ctx.deps.competition.id,
            )
            if max_rows is not None and max_rows > 0:
                _truncate_csv(run_path / "train.csv", max_rows=max_rows)
                _truncate_csv(run_path / "test.csv", max_rows=max_rows)
                _truncate_csv(run_path / "sample_submission.csv", max_rows=max_rows)
            elif ctx.deps.max_generations <= 5:
                _truncate_csv(run_path / "train.csv", max_rows=800)
                _truncate_csv(run_path / "test.csv", max_rows=800)
                _truncate_csv(run_path / "sample_submission.csv", max_rows=800)
            execution = await execute_solution(
                solution_code,
                run_path,
                timeout_seconds=timeout_seconds or ctx.deps.solution_timeout,
                env=self._build_execution_env(validation_split),
                use_builtin_code_execution=True,
                model_spec=self._settings.model,
            )

        score = parse_baseline_score(execution.stdout)
        error = (
            'Execution timed out'
            if execution.timed_out
            else f'Execution failed (exit {execution.returncode})'
            if execution.returncode != 0
            else 'Baseline score not found in output'
            if score is None
            else None
        )
        cv_score = score if score is not None else 0.0
        fitness = self._fitness_from_score(cv_score, ctx.deps.competition.metric_direction) if error is None else 0.0
        return {
            "fitness": round(fitness, 6),
            "cv_score": round(cv_score, 6),
            "valid": error is None,
            "runtime_ms": execution.runtime_ms,
            "timed_out": execution.timed_out,
            "returncode": execution.returncode,
            "error": error,
            "stderr": execution.stderr.strip() if execution.stderr else None,
        }

    async def _submit_solution(
        self, ctx: RunContext[EvolverDeps], solution_code: str, *, message: str
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(dir=str(ctx.deps.data_dir)) as run_dir:
            run_path = Path(run_dir)
            stage_competition_data(
                ctx.deps.train_path,
                ctx.deps.test_path,
                ctx.deps.sample_path,
                run_path,
                competition_id=ctx.deps.competition.id,
            )
            execution = await execute_solution(
                solution_code,
                run_path,
                timeout_seconds=ctx.deps.solution_timeout,
                env=self._build_execution_env(0.2),
                use_builtin_code_execution=True,
                model_spec=self._settings.model,
            )
            submission_path = run_path / 'submission.csv'
            error = (
                'Execution timed out'
                if execution.timed_out
                else f'Execution failed (exit {execution.returncode})'
                if execution.returncode != 0
                else 'submission.csv not found after execution'
                if not submission_path.exists()
                else None
            )

            if error:
                return {
                    'submission_id': None,
                    'status': 'failed',
                    'error': error,
                    'generation': len(ctx.deps.generation_history),
                    'runtime_ms': execution.runtime_ms,
                }

            submission = await ctx.deps.platform_adapter.submit(
                ctx.deps.competition.id, str(submission_path), message=message
            )

        return {
            'submission_id': submission.id,
            'status': submission.status,
            'generation': len(ctx.deps.generation_history),
            'runtime_ms': execution.runtime_ms,
        }

    def _seeded_rng(self, solution_code: str, params: dict[str, Any], salt: str) -> random.Random:
        seed_input = f'{salt}:{solution_code}:{sorted(params.items())}'.encode()
        seed = int(hashlib.sha256(seed_input).hexdigest(), 16)
        return random.Random(seed)

    def _mutate_numbers(self, code: str, rng: random.Random, *, max_changes: int, magnitude: float) -> str:
        changes = 0

        def replacer(match: re.Match[str]) -> str:
            nonlocal changes
            if changes >= max_changes:
                return match.group(0)
            raw = match.group(1)
            try:
                value = float(raw)
            except ValueError:
                return raw
            if value == 0:
                value = 0.1
            direction = rng.choice([-1, 1])
            mutated = value * (1 + direction * magnitude)
            changes += 1
            return str(max(1, int(round(mutated)))) if raw.isdigit() else f'{mutated:.6g}'

        return _NUMBER_PATTERN.sub(replacer, code)

    def _ensure_import(self, code: str, module: str, symbol: str) -> str:
        imports, body = self._split_imports(code)
        import_prefix = f"from {module} import"
        for line in imports:
            stripped = line.strip()
            if stripped.startswith(import_prefix) and symbol in stripped:
                return code

        imports.append(f"{import_prefix} {symbol}")
        merged_imports = self._merge_imports(imports, [])
        if merged_imports:
            return "\n".join([*merged_imports, "", *body])
        return "\n".join(body)

    def _swap_model_family(self, code: str) -> str:
        ordered_swaps = sorted(_MODEL_SWAPS.items(), key=lambda item: len(item[0]), reverse=True)
        for source, target in ordered_swaps:
            pattern = re.compile(rf"\b{re.escape(source)}\b")
            if not pattern.search(code):
                continue
            updated = pattern.sub(target, code)
            module = _MODEL_IMPORTS.get(target)
            if module:
                updated = self._ensure_import(updated, module, target)
            return updated
        return code

    def _inject_scaler(self, code: str) -> str:
        if "StandardScaler" in code:
            return code
        match = _NUMERIC_PIPELINE_PATTERN.search(code)
        if not match:
            return code
        steps_block = match.group("steps")
        if "SimpleImputer" not in steps_block or "StandardScaler" in steps_block:
            return code

        lines = steps_block.splitlines()
        inserted = False
        for idx, line in enumerate(lines):
            if "SimpleImputer" in line:
                indent = re.match(r"\s*", line).group(0)
                lines.insert(idx + 1, f'{indent}("scaler", StandardScaler()),')
                inserted = True
                break
        if not inserted:
            return code

        updated_steps = "\n".join(lines)
        updated = f"{code[:match.start('steps')]}{updated_steps}{code[match.end('steps'):]}"
        return self._ensure_import(updated, "sklearn.preprocessing", "StandardScaler")

    def _inject_fillna(self, code: str) -> str:
        if 'fillna(' in code or not (match := _FILLNA_PATTERN.search(code)):
            return code
        insert = f'\n{match["indent"]}{match["var"]} = {match["var"]}.fillna(0)'
        return f'{code[: match.end()]}{insert}{code[match.end() :]}'

    def _merge_imports(self, primary: list[str], secondary: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for line in primary + secondary:
            normalized = line.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(line)
        return result

    def _split_imports(self, code: str) -> tuple[list[str], list[str]]:
        def is_import(line: str) -> bool:
            stripped = line.lstrip()
            return stripped.startswith(('import ', 'from ')) and not stripped.startswith('from __future__')

        lines = code.splitlines()
        return [line for line in lines if is_import(line)], [line for line in lines if not is_import(line)]

    def _extract_top_level_defs(self, code: str) -> dict[str, str]:
        return {
            match.group(1).split()[1].split('(')[0]: match.group(0).rstrip() for match in _DEF_PATTERN.finditer(code)
        }

    def _apply_point_mutation(self, code: str, params: dict[str, Any]) -> str:
        rng = self._seeded_rng(code, params, 'point')
        magnitude = float(params.get('delta', 0.1))
        max_changes = int(params.get('max_changes', 2))
        return self._mutate_numbers(code, rng, max_changes=max_changes, magnitude=magnitude)

    def _apply_structural_mutation(self, code: str, params: dict[str, Any]) -> str:
        for mutate in (self._swap_model_family, self._inject_scaler, self._inject_fillna):
            if (result := mutate(code)) != code:
                return result
        return self._apply_point_mutation(code, params)

    def _apply_hyperparameter_mutation(self, code: str, params: dict[str, Any]) -> str:
        rng = self._seeded_rng(code, params, "hyperparameter")
        magnitude = float(params.get("magnitude", 0.2))
        requested = str(params.get("param", "")).strip()

        candidates: list[tuple[str, re.Pattern[str], re.Match[str]]] = []
        for name, pattern in _HYPERPARAM_PATTERNS.items():
            if requested and name != requested:
                continue
            if match := pattern.search(code):
                candidates.append((name, pattern, match))

        if not candidates:
            return self._apply_point_mutation(code, params)

        name, pattern, match = rng.choice(candidates)
        try:
            value = float(match.group(2))
        except ValueError:
            return self._apply_point_mutation(code, params)

        mutated = value * (1 + magnitude * rng.choice([-1, 1]))
        bounds = _HYPERPARAM_BOUNDS.get(name)
        if bounds is not None:
            mutated = min(max(mutated, bounds[0]), bounds[1])
        if name in _HYPERPARAM_INTEGER_KEYS:
            mutated_text = str(max(1, int(round(mutated))))
        else:
            mutated_text = f"{max(0.0001, mutated):.6g}"
        return pattern.sub(f"{match.group(1)}{mutated_text}", code, count=1)

    def _apply_crossover(self, code: str, other: str, params: dict[str, Any]) -> str:
        if not other.strip():
            return code
        primary_imports, primary_body = self._split_imports(code)
        other_imports, _ = self._split_imports(other)
        primary_defs = self._extract_top_level_defs(code)
        extra_defs = [block for name, block in self._extract_top_level_defs(other).items() if name not in primary_defs]
        body_parts = ['\n'.join(primary_body).strip(), *extra_defs]
        imports = '\n'.join(self._merge_imports(primary_imports, other_imports))
        body = '\n\n'.join(part for part in body_parts if part)
        return f'{imports}\n\n{body}' if imports else body


def _truncate_csv(path: Path, *, max_rows: int) -> None:
    if max_rows <= 0:
        return
    temp_path = path.with_suffix(f'{path.suffix}.tmp')
    with path.open('r', encoding='utf-8', errors='ignore', newline='') as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None:
            return
        with temp_path.open('w', encoding='utf-8', errors='ignore', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(header)
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                writer.writerow(row)
    temp_path.replace(path)


# Module-level singleton for backward compatibility
evolver_agent_instance = EvolverAgent()
evolver_agent = evolver_agent_instance.agent
settings = evolver_agent_instance.settings
