# Multi-Agent Architecture

AGENT-K uses a multi-agent architecture where specialized agents collaborate to complete Kaggle competition missions. Each agent has a specific role and uses tools to accomplish its objectives.

## Overview

```mermaid
graph TB
    L[LYCURGUS<br/>Orchestrator] --> LB[LOBBYIST<br/>Discovery]
    L --> SC[SCIENTIST<br/>Research]
    L --> EV[EVOLVER<br/>Optimization]

    LB --> KT[Kaggle Toolset]
    LB --> WS[WebSearchTool]
    LB --> MT[MemoryTool (Anthropic)]

    SC --> KT
    SC --> WS
    SC --> MT

    EV --> CE[CodeExecutionTool]
    EV --> MCP[Kaggle MCP]
    EV --> MT
```

## The Agents

### LYCURGUS (Orchestrator)

The central coordinator that manages the mission lifecycle using a [pydantic-graph](https://ai.pydantic.dev/graph/) state machine.

Responsibilities:

- Initialize and configure specialized agents
- Execute the state machine graph
- Handle phase transitions
- Manage error recovery and retries

```python
from agent_k.agents.lycurgus import LycurgusOrchestrator

async with LycurgusOrchestrator() as orchestrator:
    result = await orchestrator.execute_mission(
        competition_id="titanic",
        criteria=criteria,
    )
```

### LOBBYIST (Discovery)

Discovers and evaluates competitions that match user criteria.

Tools used:

- `search_kaggle_competitions`, `get_competition_details`, `score_competition_fit`
- `kaggle_*` toolset helpers
- Built-in `web_search`
- Built-in `memory` (Anthropic only)

### SCIENTIST (Research)

Analyzes the selected competition, leaderboards, and relevant literature.

Tools used:

- `analyze_leaderboard`, `get_kaggle_notebooks`, `analyze_data_characteristics`
- `kaggle_*` toolset helpers
- Built-in `web_search`
- Built-in `memory` (Anthropic only)

### EVOLVER (Optimization)

Optimizes the prototype solution using evolutionary search.

Tools used:

- `mutate_solution`, `evaluate_fitness`, `record_generation`, `check_convergence`, `submit_to_kaggle`
- Built-in `CodeExecutionTool` when supported by the provider
- `MCPServerTool` for Kaggle submissions
- Built-in `memory` (Anthropic only)

## Agent Singleton Pattern

Each agent module defines a class wrapper around a Pydantic-AI `Agent` and exposes a module-level singleton.

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

from agent_k.agents import register_agent
from agent_k.agents.base import MemoryMixin, universal_tool_preparation
from agent_k.infra.providers import get_model
from agent_k.toolsets import create_production_toolset, kaggle_toolset

class LobbyistAgent(MemoryMixin):
    def __init__(self, settings: LobbyistSettings | None = None) -> None:
        self._settings = settings or LobbyistSettings()
        self._toolset: FunctionToolset[LobbyistDeps] = FunctionToolset(id="lobbyist")
        self._register_tools()
        self._agent = self._create_agent()
        register_agent("lobbyist", self._agent)

    def _create_agent(self) -> Agent[LobbyistDeps, DiscoveryResult]:
        return Agent(
            model=get_model(self._settings.model),
            deps_type=LobbyistDeps,
            output_type=DiscoveryResult,
            toolsets=[
                create_production_toolset([self._toolset, kaggle_toolset])
            ],
            prepare_tools=universal_tool_preparation,
        )

lobbyist_agent_instance = LobbyistAgent()
lobbyist_agent = lobbyist_agent_instance.agent
```

## Dependency Injection

Dependencies are passed to agents using dataclasses:

```python
from dataclasses import dataclass, field
import httpx

@dataclass
class LobbyistDeps:
    """Dependencies for the LOBBYIST agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    search_cache: dict[str, Any] = field(default_factory=dict)
```

This allows:

- Testability via dependency substitution
- Flexibility for swapping adapters
- Type safety for agent dependencies

## Agent Communication

Agents communicate through:

1. Mission state passed through the graph
2. Optional MemoryTool for shared notes (Anthropic only)
3. Event emission for UI updates

```python
await memory(command="create", path="shared/target_competition.md", file_text="Titanic details")
notes = await memory(command="view", path="shared/target_competition.md")
```

## Next Steps

- [State Machine Graph](graph.md) - How phases are orchestrated
- [Toolsets](toolsets.md) - FunctionToolset implementations
- [Model Configuration](models.md) - Supported model providers
