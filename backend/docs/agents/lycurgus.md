# LYCURGUS - Orchestrator Agent

LYCURGUS is the central orchestrator that coordinates the multi-agent mission lifecycle. It runs a state machine graph and manages transitions across discovery, research, prototype, evolution, and submission.

## Role

- Initialize and configure specialized agents
- Execute the state machine graph
- Manage phase transitions
- Handle error recovery and retries
- Emit events for the UI dashboard

## Basic Usage

```python
from agent_k.agents.lycurgus import LycurgusOrchestrator
from agent_k.core.models import MissionCriteria

async def run_mission():
    async with LycurgusOrchestrator() as orchestrator:
        result = await orchestrator.execute_mission(
            competition_id="titanic",
            criteria=MissionCriteria(target_leaderboard_percentile=0.10),
        )
        print(result)
```

## Configuration

### LycurgusSettings

```python
from agent_k.agents.lycurgus import LycurgusSettings

config = LycurgusSettings(
    default_model="anthropic:claude-3-haiku-20240307",
    max_evolution_rounds=100,
)
```

`LycurgusSettings.from_file()` expects a JSON file:

```python
from pathlib import Path
from agent_k.agents.lycurgus import LycurgusSettings

settings = LycurgusSettings.from_file(Path("config/mission.json"))
```

### Devstral Helper

```python
config = LycurgusSettings.with_devstral(
    base_url="http://localhost:1234/v1"  # Optional
)
```

### Model Override

```python
orchestrator = LycurgusOrchestrator(model="openrouter:mistralai/devstral-small-2-2512")
```

## Platform Adapter Selection

- If `KAGGLE_USERNAME` and `KAGGLE_KEY` are present, LYCURGUS uses `KaggleAdapter`.
- Otherwise it falls back to `OpenEvolveAdapter` for an in-memory workflow.

## Mission Execution Flow

1. Build the graph:
   ```python
   Graph(nodes=(DiscoveryNode, ResearchNode, PrototypeNode, EvolutionNode, SubmissionNode))
   ```
2. Initialize `MissionState` with criteria.
3. Execute the graph starting at `DiscoveryNode`.
4. Return `MissionResult` with final score/rank metadata.

## Event Emission

LYCURGUS emits events through `EventEmitter` for the AG-UI dashboard, including:

- `phase-start` / `phase-complete`
- `task-start` / `task-complete`
- `tool-start` / `tool-result`
- `generation-start` / `generation-complete`
- `submission-result`
- `error-occurred`
