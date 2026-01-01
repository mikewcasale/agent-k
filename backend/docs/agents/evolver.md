# EVOLVER - Optimization Agent

The EVOLVER agent optimizes solutions through evolutionary code search. It evolves candidate solutions, evaluates fitness, and tracks convergence.

## Role

- Initialize a population from the prototype
- Evaluate candidate fitness
- Apply mutations and crossover
- Track best solutions and convergence
- Submit best candidates when appropriate

## Built-in Tools

| Tool | Purpose |
|------|---------|
| `CodeExecutionTool` | Execute and evaluate solutions (supported providers) |
| `MemoryTool` | Store and retrieve evolution context (Anthropic only) |
| `MCPServerTool` (Kaggle) | Submit solutions and query Kaggle MCP |

## Custom Tools

| Tool | Purpose |
|------|---------|
| `mutate_solution` | Apply mutations to candidate code |
| `evaluate_fitness` | Execute code and compute fitness |
| `record_generation` | Log generation metrics |
| `check_convergence` | Determine if evolution should stop |
| `sample_elites` | Fetch top + diverse elite candidates |
| `submit_to_kaggle` | Submit best solution (approval required) |

## Basic Usage

Direct usage requires a staged dataset and a configured platform adapter:

```python
from pathlib import Path

from agent_k.agents.evolver import EvolverDeps, evolver_agent
from agent_k.core.data import infer_competition_schema, locate_data_files, stage_competition_data
from agent_k.ui.ag_ui import EventEmitter

# Download data through the adapter
files = await adapter.download_data(competition.id, work_dir)
train_path, test_path, sample_path = locate_data_files(files)

staged = stage_competition_data(train_path, test_path, sample_path, Path(work_dir))
schema = infer_competition_schema(staged["train"], staged["test"], staged["sample"])

deps = EvolverDeps(
    competition=competition,
    event_emitter=EventEmitter(),
    platform_adapter=adapter,
    data_dir=Path(work_dir),
    train_path=staged["train"],
    test_path=staged["test"],
    sample_path=staged["sample"],
    target_columns=schema.target_columns,
    train_target_columns=schema.train_target_columns,
    initial_solution=baseline_code,
)

run_result = await evolver_agent.run(
    "Improve the baseline solution",
    deps=deps,
)

output = run_result.output
print(output)
```

## Evolver Settings

Key configuration fields from `EvolverSettings`:

- `model` - model spec for evolution tasks
- `solution_timeout` - execution timeout for candidate solutions
- `population_size` - number of candidates per generation
- `max_generations` - maximum number of generations
- `convergence_threshold` - generations without improvement before stopping
- `enable_thinking` - enable Anthropic thinking mode
- `cascade_evaluation` - run quick evaluation stages before full evaluation
- `elite_sample_top` / `elite_sample_diverse` - default elite sampler sizes

## Output Models

The agent returns either `EvolutionResult` or `EvolutionFailure`:

```python
class EvolutionResult(BaseModel):
    best_solution: str
    best_fitness: float
    generations_completed: int
    convergence_achieved: bool
    convergence_reason: str | None = None
    submission_ready: bool

class EvolutionFailure(BaseModel):
    error_type: str
    error_message: str
    partial_solution: str | None = None
    recoverable: bool
```

## Notes

- Code execution uses a provider tool when available; otherwise local execution is used.
- `submit_to_kaggle` requires tool approval when used with Pydantic-AI.
- The evolver maintains an elite archive keyed by model family + complexity; use `sample_elites` to pull top/diverse seeds.
