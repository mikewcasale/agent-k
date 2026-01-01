# Quick Start

This guide walks you through running your first AGENT-K mission using the API or programmatically.

## Prerequisites

Make sure you have completed the [installation](install.md) steps:

- Python 3.11+ with uv installed
- `uv sync` completed in `backend/`
- Environment variables configured (Kaggle + model API key)

## Start the API Server

```bash
cd backend
python -m agent_k.ui.ag_ui
```

The API server runs on `http://localhost:9000`.

## Start a Mission (Chat Endpoint)

The `/agentic_generative_ui/` endpoint accepts Vercel AI chat messages. When a mission intent is detected, it runs the mission and streams events.

```bash
curl -N -X POST http://localhost:9000/agentic_generative_ui/ \
  -H "Content-Type: application/json" \
  -d '{"id":"demo","messages":[{"role":"user","parts":[{"type":"text","text":"Find a Kaggle competition with a $10k prize"}]}]}'
```

## Programmatic Usage

```python
import asyncio
from agent_k import LycurgusOrchestrator
from agent_k.core.models import MissionCriteria

async def main():
    async with LycurgusOrchestrator() as orchestrator:
        result = await orchestrator.execute_mission(
            competition_id="titanic",
            criteria=MissionCriteria(
                target_leaderboard_percentile=0.10,
                max_evolution_rounds=50,
            ),
        )
        print(f"Final rank: {result.final_rank}")

asyncio.run(main())
```

## Tooling Notes

- `web_search` and `web_fetch` are built-in tools and only available for supported providers.
- `memory` is only available for Anthropic models and stores files under `.agent_k_memory` by default.
- Kaggle operations use the Kaggle adapter when credentials are available; otherwise OpenEvolve is used.

## Using the Dashboard

For a visual interface, start both servers:

```bash
./run.sh
```

Then open `http://localhost:3000` in your browser.

## Next Steps

- [Concepts: Agents](concepts/agents.md) - Understand the multi-agent architecture
- [Concepts: Toolsets](concepts/toolsets.md) - Learn about FunctionToolsets
- [Examples: Multi-Agent Demo](examples/multi-agent-demo.md) - Walkthrough using the core agents
