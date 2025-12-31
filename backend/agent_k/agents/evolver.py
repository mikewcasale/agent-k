"""Evolver agent - evolutionary optimization for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Standard library (alphabetical)
import hashlib
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, cast

# Third-party (alphabetical)
import logfire
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, ModelRetry, ModelSettings, RunContext, ToolOutput, ToolReturn
from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local imports (core first, then alphabetical)
from agent_k.agents import register_agent
from agent_k.agents.base import universal_tool_preparation
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
from agent_k.toolsets import (
    AgentKMemoryTool,
    code_toolset,
    create_memory_backend,
    create_production_toolset,
    prepare_code_execution_tool,
    prepare_memory_tool,
    register_memory_tool,
)

if TYPE_CHECKING:
    from agent_k.core.models import Competition
    from agent_k.core.protocols import PlatformAdapter
    from agent_k.ui.ag_ui import EventEmitter

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "EVOLUTION_OUTPUT_TYPE",
    "EvolutionFailure",
    "EvolutionResult",
    "EvolverDeps",
    "EvolverSettings",
    "EVOLVER_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "evolver_agent",
)

# =============================================================================
# Section 3: Constants
# =============================================================================
SCHEMA_VERSION: Final[str] = "1.0.0"
_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])(-?\d+\.?\d*)(?![\w.])")
_HYPERPARAM_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "n_estimators": re.compile(r"(n_estimators\s*=\s*)(\d+)", re.IGNORECASE),
    "learning_rate": re.compile(r"(learning_rate\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_depth": re.compile(r"(max_depth\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_leaf": re.compile(r"(min_samples_leaf\s*=\s*)(\d+)", re.IGNORECASE),
    "subsample": re.compile(r"(subsample\s*=\s*)([\d\.]+)", re.IGNORECASE),
}
_MODEL_SWAPS: Final[dict[str, str]] = {
    "RandomForestClassifier": "GradientBoostingClassifier",
    "RandomForestRegressor": "GradientBoostingRegressor",
    "GradientBoostingClassifier": "RandomForestClassifier",
    "GradientBoostingRegressor": "RandomForestRegressor",
    "LogisticRegression": "LinearSVC",
    "LinearRegression": "Ridge",
}


# =============================================================================
# Section 4: Settings
# =============================================================================
class EvolverSettings(BaseSettings):
    """Configuration for the Evolver agent."""

    model_config = SettingsConfigDict(
        env_prefix="EVOLVER_",
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model identifier for evolution tasks",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for evolution prompts",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens for responses",
    )
    solution_timeout: int = Field(
        default=SOLUTION_EXECUTION_TIMEOUT_SECONDS,
        ge=1,
        description="Timeout for executing a candidate solution (seconds)",
    )

    tool_retries: int = Field(
        default=3,
        ge=0,
        description="Tool retry attempts",
    )
    output_retries: int = Field(
        default=2,
        ge=0,
        description="Output validation retry attempts",
    )

    population_size: int = Field(
        default=EVOLUTION_POPULATION_SIZE,
        ge=1,
        description="Population size for evolution",
    )
    max_generations: int = Field(
        default=MAX_EVOLUTION_GENERATIONS,
        ge=1,
        description="Maximum evolution generations",
    )
    convergence_threshold: int = Field(
        default=5,
        ge=1,
        description="Generations without improvement before stopping",
    )

    enable_thinking: bool = Field(
        default=True,
        description="Enable extended reasoning mode for supported models",
    )
    thinking_budget_tokens: int = Field(
        default=4096,
        ge=0,
        description="Token budget for model thinking mode",
    )

    @model_validator(mode="after")
    def validate_evolution_params(self) -> Self:
        """Validate cross-field evolution configuration."""
        if self.convergence_threshold > self.max_generations:
            raise ValueError("convergence_threshold cannot exceed max_generations")
        if self.population_size < 2 and self.max_generations > 1:
            raise ValueError("population_size must be >= 2 when running multiple generations")
        return self

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings from configuration."""
        settings: ModelSettings = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.enable_thinking and "anthropic" in self.model:
            return cast(
                ModelSettings,
                {
                    **settings,
                    "anthropic_thinking": {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget_tokens,
                    },
                },
            )

        return settings


# =============================================================================
# Section 5: Dependencies
# =============================================================================
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
    initial_solution: str = ""
    population_size: int = EVOLUTION_POPULATION_SIZE
    max_generations: int = MAX_EVOLUTION_GENERATIONS
    solution_timeout: int = SOLUTION_EXECUTION_TIMEOUT_SECONDS
    target_score: float = 0.0
    best_solution: str | None = None
    best_fitness: float | None = None
    generation_history: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Section 6: Output Types
# =============================================================================
class EvolutionResult(BaseModel):
    """Result of evolution process."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    best_solution: str = Field(description="Best solution code")
    best_fitness: float = Field(description="Fitness score of best solution")
    generations_completed: int = Field(
        default=0,
        ge=0,
        description="Number of generations completed",
    )
    convergence_achieved: bool = Field(
        default=False,
        description="Whether convergence criteria were met",
    )
    convergence_reason: str | None = Field(
        default=None,
        description="Reason for convergence if achieved",
    )
    submission_ready: bool = Field(
        default=False,
        description="Whether output is ready for submission",
    )


class EvolutionFailure(BaseModel):
    """Failure result for evolution process."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    error_type: str = Field(description="Classification of failure")
    error_message: str = Field(description="Human-readable error")
    partial_solution: str | None = Field(
        default=None,
        description="Best available solution snippet, if any",
    )
    recoverable: bool = Field(
        default=True,
        description="Whether the failure is likely recoverable",
    )


EVOLUTION_OUTPUT_TYPE: Final[list[ToolOutput[EvolutionResult] | ToolOutput[EvolutionFailure]]] = [
    ToolOutput[EvolutionResult](EvolutionResult, name="return_success"),
    ToolOutput[EvolutionFailure](EvolutionFailure, name="return_failure"),
]


# =============================================================================
# Section 7: System Prompt
# =============================================================================
EVOLVER_SYSTEM_PROMPT: Final[str] = """You are the EVOLVER agent in the AGENT-K multi-agent system.

Your mission is to optimize competition solutions using evolutionary code search.

AVAILABLE BUILTIN TOOLS:
- Kaggle MCP: Use for all Kaggle platform operations (submit, download data, check leaderboard)
- Memory: Use to persist and retrieve context across long evolution runs
- Code Executor: Use to safely execute and evaluate solution candidates

CUSTOM TOOLS:
- mutate_solution: Apply mutations to solutions
- evaluate_fitness: Compute fitness scores
- record_generation: Log generation metrics
- check_convergence: Detect when to stop evolution
- submit_to_kaggle: Submit best solution

EVOLUTION WORKFLOW:
1. Initialize population from the provided prototype solution
2. For each generation:
   a. Evaluate fitness of all candidates using evaluate_fitness
   b. Select top performers based on fitness
   c. Apply mutations using mutate_solution (vary mutation types)
   d. Record metrics using record_generation
   e. Check convergence using check_convergence
   f. Save best solution to Memory for recovery
3. When converged or max generations reached:
   a. Submit best solution using submit_to_kaggle
   b. Return EvolutionResult with final metrics (or EvolutionFailure on errors)

MUTATION STRATEGY:
- Use point mutations for fine-tuning (small parameter changes)
- Use structural mutations for exploring new architectures
- Use hyperparameter mutations for learning rate, regularization
- Use crossover to combine successful solutions

IMPORTANT:
- Always save promising solutions to Memory before applying risky mutations
- Use submit_to_kaggle periodically for leaderboard validation
- Respect rate limits when submitting to Kaggle
- Record all generation metrics for convergence analysis
- Keep the baseline print line in candidate code: "Baseline <metric> score: <value>"
- Preserve TARGET_COLUMNS and TRAIN_TARGET_COLUMNS to support multi-target submissions
"""


# =============================================================================
# Section 8: Agent Singleton
# =============================================================================
settings = EvolverSettings()

evolver_toolset: FunctionToolset[EvolverDeps] = FunctionToolset(id="evolver")

_kaggle_mcp = MCPServerTool(
    id="kaggle",
    url=DEFAULT_KAGGLE_MCP_URL,
)
try:
    _memory_backend: AgentKMemoryTool | None = create_memory_backend()
except RuntimeError:  # pragma: no cover - optional dependency
    _memory_backend = None

_builtin_tools: list[Any] = [_kaggle_mcp, prepare_code_execution_tool]
if _memory_backend is not None:
    _builtin_tools.append(prepare_memory_tool)

evolver_agent: Agent[EvolverDeps, EvolutionResult | EvolutionFailure] = Agent(
    model=get_model(settings.model),
    deps_type=EvolverDeps,
    output_type=EVOLUTION_OUTPUT_TYPE,
    instructions=EVOLVER_SYSTEM_PROMPT,
    name="evolver",
    model_settings=settings.model_settings,
    retries=settings.tool_retries,
    output_retries=settings.output_retries,
    builtin_tools=_builtin_tools,
    toolsets=[
        create_production_toolset(
            [evolver_toolset, cast(FunctionToolset[EvolverDeps], code_toolset)],
            require_approval_for=["submit_to_kaggle"],
        ),
    ],
    prepare_tools=universal_tool_preparation,
    instrument=True,
)

register_agent("evolver", evolver_agent)

if _memory_backend is not None:
    register_memory_tool(evolver_agent, _memory_backend)


# =============================================================================
# Section 9: Tools
# =============================================================================
@evolver_toolset.tool
async def mutate_solution(
    ctx: RunContext[EvolverDeps],
    solution_code: str,
    mutation_type: str,
    mutation_params: dict[str, Any] | None = None,
) -> str:
    """Apply mutation to a solution."""
    with logfire.span("evolver.mutate", mutation_type=mutation_type):
        await ctx.deps.event_emitter.emit(
            "tool-start",
            {
                "taskId": "evolution_mutate",
                "toolCallId": f"mutate_{mutation_type}",
                "toolType": "code_executor",
                "operation": f"mutate_{mutation_type}",
            },
        )

        params = mutation_params or {}

        if mutation_type == "point":
            mutated = _apply_point_mutation(solution_code, params)
        elif mutation_type == "structural":
            mutated = _apply_structural_mutation(solution_code, params)
        elif mutation_type == "hyperparameter":
            mutated = _apply_hyperparameter_mutation(solution_code, params)
        elif mutation_type == "crossover":
            other_solution = params.get("other_solution", "")
            mutated = _apply_crossover(solution_code, other_solution, params)
        else:
            mutated = solution_code

        return mutated


@evolver_toolset.tool
async def evaluate_fitness(
    ctx: RunContext[EvolverDeps],
    solution_code: str,
    validation_split: float = 0.2,
) -> ToolReturn:
    """Evaluate solution fitness."""
    with logfire.span("evolver.evaluate_fitness"):
        tool_call_id = f"fitness_{id(solution_code):x}"

        await ctx.deps.event_emitter.emit_tool_start(
            task_id="evolution_evaluate",
            tool_call_id=tool_call_id,
            tool_type="code_executor",
            operation="evaluate_fitness",
        )

        result = await _evaluate_solution(
            ctx,
            solution_code,
            validation_split=validation_split,
        )

        if result["valid"]:
            if ctx.deps.best_fitness is None or result["fitness"] > ctx.deps.best_fitness:
                ctx.deps.best_fitness = result["fitness"]
                ctx.deps.best_solution = solution_code

            await ctx.deps.event_emitter.emit(
                "fitness-update",
                {
                    "fitness": result["fitness"],
                    "cv_score": result["cv_score"],
                    "validation_split": validation_split,
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

        summary = (
            f"Fitness {result['fitness']:.4f}, CV {result['cv_score']:.4f}, valid={result['valid']}"
        )

        return ToolReturn(
            return_value=result,
            content=summary,
            metadata={
                "tool_call_id": tool_call_id,
                "runtime_ms": result["runtime_ms"],
            },
        )


@evolver_toolset.tool
async def record_generation(
    ctx: RunContext[EvolverDeps],
    generation: int,
    best_fitness: float,
    mean_fitness: float,
    worst_fitness: float,
    mutations: dict[str, int],
) -> None:
    """Record generation metrics."""
    metrics = {
        "generation": generation,
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "worst_fitness": worst_fitness,
        "mutations": mutations,
    }

    ctx.deps.generation_history.append(metrics)

    await ctx.deps.event_emitter.emit_generation_complete(
        generation=generation,
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        worst_fitness=worst_fitness,
        population_size=ctx.deps.population_size,
        mutations=mutations,
    )

    logfire.info(
        "evolution_generation",
        generation=generation,
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
    )


@evolver_toolset.tool
async def check_convergence(
    ctx: RunContext[EvolverDeps],
    threshold_generations: int = 5,
    improvement_threshold: float = 0.001,
) -> dict[str, Any]:
    """Check if evolution has converged."""
    history = ctx.deps.generation_history

    if len(history) < threshold_generations:
        return {"converged": False, "reason": "Not enough generations"}

    recent = history[-threshold_generations:]
    fitness_values = [g["best_fitness"] for g in recent]
    max_improvement = max(fitness_values) - min(fitness_values)

    if max_improvement < improvement_threshold:
        return {
            "converged": True,
            "reason": f"No improvement for {threshold_generations} generations",
            "best_fitness": max(fitness_values),
        }

    if ctx.deps.target_score > 0 and max(fitness_values) >= ctx.deps.target_score:
        return {
            "converged": True,
            "reason": "Target score achieved",
            "best_fitness": max(fitness_values),
        }

    return {
        "converged": False,
        "reason": "Evolution in progress",
        "recent_improvement": max_improvement,
    }


@evolver_toolset.tool(requires_approval=True)
async def submit_to_kaggle(
    ctx: RunContext[EvolverDeps],
    solution_code: str,
    message: str = "AGENT-K submission",
) -> dict[str, Any]:
    """Submit solution to Kaggle via the platform adapter."""
    with logfire.span(
        "evolver.submit",
        competition_id=ctx.deps.competition.id,
    ):
        tool_call_id = f"submit_{len(ctx.deps.generation_history)}"
        await ctx.deps.event_emitter.emit(
            "tool-start",
            {
                "taskId": "evolution_submit",
                "toolCallId": tool_call_id,
                "toolType": "kaggle_mcp",
                "operation": "competitions.submit",
            },
        )

        result = await _submit_solution(ctx, solution_code, message=message)

        if result.get("status") == "failed":
            await ctx.deps.event_emitter.emit_tool_error(
                task_id="evolution_submit",
                tool_call_id=tool_call_id,
                error=result.get("error", "Submission failed"),
            )
            return result

        await ctx.deps.event_emitter.emit_tool_result(
            task_id="evolution_submit",
            tool_call_id=tool_call_id,
            result=result,
            duration_ms=result.get("runtime_ms", 0),
        )

        return result


# =============================================================================
# Section 10: Validators
# =============================================================================
@evolver_agent.output_validator
async def validate_evolution_result(
    ctx: RunContext[EvolverDeps],
    result: EvolutionResult | EvolutionFailure,
) -> EvolutionResult | EvolutionFailure:
    """Validate evolution results."""
    if ctx.partial_output:
        return result
    if isinstance(result, EvolutionFailure):
        if not result.error_type or not result.error_message:
            raise ModelRetry("EvolutionFailure must include error_type and error_message.")
        return result
    if result.best_fitness < 0:
        raise ModelRetry("best_fitness must be >= 0. Provide a valid fitness score.")
    if not result.best_solution:
        raise ModelRetry("best_solution is required. Provide the best solution code.")
    return result


# =============================================================================
# Section 11: Dynamic Instructions
# =============================================================================
@evolver_agent.instructions
async def add_evolution_context(ctx: RunContext[EvolverDeps]) -> str:
    """Add evolution-specific context to instructions."""
    comp = ctx.deps.competition
    history = ctx.deps.generation_history

    context = (
        "COMPETITION CONTEXT:\n"
        f"- ID: {comp.id}\n"
        f"- Title: {comp.title}\n"
        f"- Metric: {comp.metric.value} ({comp.metric_direction})\n"
        f"- Target Score: {ctx.deps.target_score}\n\n"
        "EVOLUTION STATE:\n"
        f"- Generations Completed: {len(history)}\n"
        f"- Population Size: {ctx.deps.population_size}\n"
        f"- Max Generations: {ctx.deps.max_generations}"
    )

    if history:
        last_gen = history[-1]
        context += (
            "\n\nLAST GENERATION:\n"
            f"- Best Fitness: {last_gen.get('best_fitness', 'N/A')}\n"
            f"- Mean Fitness: {last_gen.get('mean_fitness', 'N/A')}"
        )

    if ctx.deps.best_solution:
        context += f"\n\nBEST SOLUTION AVAILABLE: {len(ctx.deps.best_solution)} chars"

    return context


def _build_execution_env(validation_split: float) -> dict[str, str]:
    return {"AGENT_K_VALIDATION_SPLIT": f"{validation_split:.6f}"}


def _fitness_from_score(score: float, direction: str) -> float:
    if direction == "minimize":
        return 1.0 / (1.0 + max(score, 0.0))
    return max(score, 0.0)


async def _evaluate_solution(
    ctx: RunContext[EvolverDeps],
    solution_code: str,
    *,
    validation_split: float,
) -> dict[str, Any]:
    error: str | None = None
    score: float | None = None
    stderr: str | None = None
    returncode = 0
    timed_out = False
    runtime_ms = 0

    with tempfile.TemporaryDirectory(dir=str(ctx.deps.data_dir)) as run_dir:
        run_path = Path(run_dir)
        stage_competition_data(
            ctx.deps.train_path,
            ctx.deps.test_path,
            ctx.deps.sample_path,
            run_path,
        )
        execution = await execute_solution(
            solution_code,
            run_path,
            timeout_seconds=ctx.deps.solution_timeout,
            env=_build_execution_env(validation_split),
            use_builtin_code_execution=True,
            model_spec=settings.model,
        )
        runtime_ms = execution.runtime_ms
        timed_out = execution.timed_out
        returncode = execution.returncode
        stderr = execution.stderr.strip() if execution.stderr else None
        score = parse_baseline_score(execution.stdout)

        if timed_out:
            error = "Execution timed out"
        elif returncode != 0:
            error = f"Execution failed (exit {returncode})"
        elif score is None:
            error = "Baseline score not found in output"

    cv_score = score if score is not None else 0.0
    fitness = (
        _fitness_from_score(cv_score, ctx.deps.competition.metric_direction)
        if error is None
        else 0.0
    )

    return {
        "fitness": round(fitness, 6),
        "cv_score": round(cv_score, 6),
        "valid": error is None,
        "runtime_ms": runtime_ms,
        "timed_out": timed_out,
        "returncode": returncode,
        "error": error,
        "stderr": stderr,
    }


async def _submit_solution(
    ctx: RunContext[EvolverDeps],
    solution_code: str,
    *,
    message: str,
) -> dict[str, Any]:
    error: str | None = None
    runtime_ms = 0
    submission_id: str | None = None
    status: str = "failed"

    with tempfile.TemporaryDirectory(dir=str(ctx.deps.data_dir)) as run_dir:
        run_path = Path(run_dir)
        stage_competition_data(
            ctx.deps.train_path,
            ctx.deps.test_path,
            ctx.deps.sample_path,
            run_path,
        )
        execution = await execute_solution(
            solution_code,
            run_path,
            timeout_seconds=ctx.deps.solution_timeout,
            env=_build_execution_env(0.2),
            use_builtin_code_execution=True,
            model_spec=settings.model,
        )
        runtime_ms = execution.runtime_ms
        if execution.timed_out:
            error = "Execution timed out"
        elif execution.returncode != 0:
            error = f"Execution failed (exit {execution.returncode})"

        submission_path = run_path / "submission.csv"
        if error is None and not submission_path.exists():
            error = "submission.csv not found after execution"

        if error is None:
            submission = await ctx.deps.platform_adapter.submit(
                ctx.deps.competition.id,
                str(submission_path),
                message=message,
            )
            submission_id = submission.id
            status = submission.status

    payload: dict[str, Any] = {
        "submission_id": submission_id,
        "status": status,
        "generation": len(ctx.deps.generation_history),
        "runtime_ms": runtime_ms,
    }
    if error:
        payload["error"] = error
    return payload


def _seeded_rng(solution_code: str, params: dict[str, Any], salt: str) -> random.Random:
    seed_input = f"{salt}:{solution_code}:{sorted(params.items())}".encode()
    seed = int(hashlib.sha256(seed_input).hexdigest(), 16)
    return random.Random(seed)


def _mutate_numbers(
    code: str,
    rng: random.Random,
    *,
    max_changes: int,
    magnitude: float,
) -> str:
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
        direction = -1 if rng.random() < 0.5 else 1
        mutated = value * (1 + direction * magnitude)
        changes += 1
        if raw.isdigit():
            return str(max(1, int(round(mutated))))
        return f"{mutated:.6g}"

    return _NUMBER_PATTERN.sub(replacer, code)


def _swap_model_family(code: str) -> str:
    for source, target in _MODEL_SWAPS.items():
        if source in code:
            return code.replace(source, target)
    return code


def _inject_fillna(code: str) -> str:
    if "fillna(" in code:
        return code
    pattern = re.compile(
        r"^(?P<indent>\s*)(?P<var>\w+)\s*=\s*pd\.read_csv\(.*\)$",
        re.MULTILINE,
    )
    match = pattern.search(code)
    if not match:
        return code
    indent = match.group("indent")
    var = match.group("var")
    insert = f"\n{indent}{var} = {var}.fillna(0)"
    return code[: match.end()] + insert + code[match.end() :]


def _merge_imports(primary: list[str], secondary: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for line in primary + secondary:
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(line)
    return merged


def _split_imports(code: str) -> tuple[list[str], list[str]]:
    import_lines: list[str] = []
    body_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ")) and not stripped.startswith("from __future__"):
            import_lines.append(line)
        else:
            body_lines.append(line)
    return import_lines, body_lines


def _extract_top_level_defs(code: str) -> dict[str, str]:
    lines = code.splitlines()
    blocks: dict[str, list[str]] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("def ") and not line.startswith("def __"):
            if current_name and current_lines:
                blocks[current_name] = current_lines
            current_name = line.split("def ")[1].split("(")[0].strip()
            current_lines = [line]
            continue
        if current_name:
            if line.startswith(("def ", "class ")):
                blocks[current_name] = current_lines
                current_name = None
                current_lines = []
            else:
                current_lines.append(line)

    if current_name and current_lines:
        blocks[current_name] = current_lines

    return {name: "\n".join(lines) for name, lines in blocks.items()}


def _apply_point_mutation(code: str, params: dict[str, Any]) -> str:
    rng = _seeded_rng(code, params, "point")
    magnitude = float(params.get("delta", 0.1))
    max_changes = int(params.get("max_changes", 2))
    return _mutate_numbers(code, rng, max_changes=max_changes, magnitude=magnitude)


def _apply_structural_mutation(code: str, params: dict[str, Any]) -> str:
    swapped = _swap_model_family(code)
    if swapped != code:
        return swapped
    injected = _inject_fillna(code)
    if injected != code:
        return injected
    return _apply_point_mutation(code, params)


def _apply_hyperparameter_mutation(code: str, params: dict[str, Any]) -> str:
    rng = _seeded_rng(code, params, "hyperparameter")
    magnitude = float(params.get("magnitude", 0.2))
    for name, pattern in _HYPERPARAM_PATTERNS.items():
        match = pattern.search(code)
        if not match:
            continue
        prefix, value_text = match.groups()
        try:
            value = float(value_text)
        except ValueError:
            continue
        direction = -1 if rng.random() < 0.5 else 1
        mutated = value * (1 + direction * magnitude)
        if name in {"n_estimators", "max_depth", "min_samples_leaf"}:
            mutated_text = str(max(1, int(round(mutated))))
        else:
            mutated_text = f"{max(0.0001, mutated):.6g}"
        return pattern.sub(f"{prefix}{mutated_text}", code, count=1)
    return _apply_point_mutation(code, params)


def _apply_crossover(code: str, other: str, params: dict[str, Any]) -> str:
    if not other.strip():
        return code
    primary_imports, primary_body = _split_imports(code)
    other_imports, _ = _split_imports(other)
    merged_imports = _merge_imports(primary_imports, other_imports)

    primary_defs = _extract_top_level_defs(code)
    other_defs = _extract_top_level_defs(other)
    extra_defs = [block for name, block in other_defs.items() if name not in primary_defs]

    body = "\n".join(primary_body).strip()
    if extra_defs:
        body = f"{body}\n\n" + "\n\n".join(extra_defs)
    if merged_imports:
        return "\n".join(merged_imports) + "\n\n" + body
    return body
