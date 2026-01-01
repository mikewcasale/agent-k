# Multi-Agent Demo

This demo runs the LOBBYIST (discovery) and SCIENTIST (research) agents in sequence.

## Running the Demo

```bash
cd backend
uv run python - <<'PY'
import asyncio
import os
from contextlib import AsyncExitStack

import httpx

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.adapters.openevolve import OpenEvolveAdapter
from agent_k.agents.lobbyist import LobbyistDeps, lobbyist_agent
from agent_k.agents.scientist import ScientistDeps, scientist_agent
from agent_k.ui.ag_ui import EventEmitter


async def run_agents(adapter, http_client):
    emitter = EventEmitter()
    lobbyist_deps = LobbyistDeps(
        http_client=http_client,
        platform_adapter=adapter,
        event_emitter=emitter,
    )

    discovery = await lobbyist_agent.run(
        "Find featured competitions with 14+ days remaining",
        deps=lobbyist_deps,
    )

    competition = discovery.output.competitions[0]
    scientist_deps = ScientistDeps(
        http_client=http_client,
        platform_adapter=adapter,
        competition=competition,
    )

    research = await scientist_agent.run(
        f"Research competition: {competition.title}",
        deps=scientist_deps,
    )

    print(research.output.recommended_approaches)


async def main():
    async with httpx.AsyncClient() as http:
        async with AsyncExitStack() as stack:
            if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
                adapter = await stack.enter_async_context(
                    KaggleAdapter(
                        KaggleSettings(
                            username=os.environ["KAGGLE_USERNAME"],
                            api_key=os.environ["KAGGLE_KEY"],
                        )
                    )
                )
            else:
                adapter = OpenEvolveAdapter()
                await adapter.authenticate()

            await run_agents(adapter, http)


asyncio.run(main())
PY
```

## Notes

- If Kaggle credentials are missing, the demo uses OpenEvolve's in-memory catalog.
- The orchestrator runs the full five-phase lifecycle; use it for end-to-end missions.
