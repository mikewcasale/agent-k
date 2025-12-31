# Quick Start

This guide walks you through running your first AGENT-K multi-agent mission.

## Prerequisites

Make sure you've completed the [installation](install.md) steps:

- ✅ Python 3.11+ with uv installed
- ✅ `uv sync` completed in `backend/`
- ✅ Environment variables configured (Kaggle + model API key)

## Running the Multi-Agent Demo

The easiest way to see AGENT-K in action is the multi-agent playbook demo.

### 1. Navigate to Backend

```bash
cd backend
```

### 2. Choose Your Model

AGENT-K supports multiple model providers:

=== "Local Devstral (LM Studio)"

    ```bash
    uv run python examples/multi_agent_playbook.py --model devstral:local
    ```

    !!! note
        Requires [LM Studio](https://lmstudio.ai/) running locally with Devstral loaded.

=== "Claude Haiku (Anthropic)"

    ```bash
    uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307
    ```

    !!! note
        Requires `ANTHROPIC_API_KEY` environment variable.

=== "Devstral via OpenRouter"

    ```bash
    uv run python examples/multi_agent_playbook.py --model openrouter:mistralai/devstral-small
    ```

    !!! note
        Requires `OPENROUTER_API_KEY` environment variable.

### 3. Watch the Agents Work

The demo runs through two phases:

1. **LOBBYIST (Discovery)** — Searches for active Kaggle competitions
2. **SCIENTIST (Research)** — Analyzes the selected competition

You'll see output like:

```
╭──────────────────────────────────────────────────────────────╮
│ 🎯 Discovery Phase                                            │
│                                                               │
│ Phase 1: LOBBYIST - Competition Discovery                     │
│                                                               │
│ Tools being used:                                             │
│ • kaggle_search_competitions (Kaggle API)                     │
│ • web_search (Built-in WebSearchTool)                         │
│ • memory (Built-in MemoryTool, Anthropic only)                │
╰──────────────────────────────────────────────────────────────╯
```

## Understanding the Output

### Tool Calls

Each agent uses toolsets and built-in tools to interact with external services:

| Toolset / Tool | Tools | Purpose |
|---------|-------|---------|
| **KaggleToolset** | `kaggle_search_competitions`, `kaggle_get_leaderboard` | Kaggle API operations |
| **WebSearchTool** | `web_search` | Web search (built-in) |
| **MemoryTool** | `memory` | Cross-agent persistence (Anthropic only) |

### Memory Persistence

Information is shared between agents via the MemoryTool:

```python
# LOBBYIST stores findings
memory(command="create", path="shared/target_competition.md", file_text="Titanic summary")

# SCIENTIST retrieves them
notes = memory(command="view", path="shared/target_competition.md")
```

Memory is persisted to `examples/mission_memory/`.

## Programmatic Usage

You can also use AGENT-K programmatically:

```python
import asyncio
from agent_k.agents.lycurgus import LycurgusOrchestrator, LycurgusSettings
from agent_k.core.models import MissionCriteria, CompetitionType

async def run_mission():
    # Configure the orchestrator
    config = LycurgusSettings(
        default_model='anthropic:claude-3-haiku-20240307',
        max_evolution_rounds=50,
    )
    
    # Define mission criteria
    criteria = MissionCriteria(
        target_competition_types=frozenset({
            CompetitionType.FEATURED,
            CompetitionType.RESEARCH,
        }),
        min_prize_pool=10000,
        min_days_remaining=14,
        target_leaderboard_percentile=0.10,
    )
    
    # Execute the mission
    async with LycurgusOrchestrator(config=config) as orchestrator:
        result = await orchestrator.execute_mission(
            competition_id="titanic",
            criteria=criteria,
        )
        
        print(f"Success: {result.success}")
        print(f"Final Rank: {result.final_rank}")
        print(f"Final Score: {result.final_score}")

asyncio.run(run_mission())
```

## Using the Dashboard

For a visual interface, start both servers:

```bash
./run.sh
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

The dashboard shows:

- **Plan View** — Mission phases and tasks
- **Evolution View** — Fitness charts during optimization
- **Research View** — Findings from the SCIENTIST
- **Memory View** — Persistent data across agents
- **Logs View** — Real-time agent activity

## Next Steps

- [Concepts: Agents](concepts/agents.md) — Understand the multi-agent architecture
- [Concepts: Toolsets](concepts/toolsets.md) — Learn about FunctionToolsets
- [Examples: Custom Agent](examples/custom-agent.md) — Create your own agent
