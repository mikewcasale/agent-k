# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AGENT-K is an autonomous multi-agent system for participating in Kaggle competitions. Built with Pydantic-AI and FastAPI, it orchestrates specialized agents through five mission phases: Discovery, Research, Prototype, Evolution, and Submission.

## Backend Python Style (Required)

All backend code under `backend/` must follow `docs/python-ai-style-guide.md`. If this file conflicts with that guide, the style guide wins.

Required backend conventions (summary):
- Module layout: module docstring with MIT license notice, `from __future__ import annotations as _annotations`, ordered imports, `TYPE_CHECKING` block, module TypeVars, `__all__` tuple, then constants/enums/ABCs/models/classes/functions.
- Typing: lowercase generics (`list[str]`), unions with `|`, `TypeAliasType` for complex aliases, `collections.abc` for Callables/Iterators.
- Formatting: 4-space indent, 88-100 char lines, trailing commas for multiline, double quotes, early-return control flow.
- Observability: use `logfire` (avoid `logging.getLogger()` in backend).
- Prefer built-in `pydantic-ai`, `pydantic-graph`, `pydantic-evals`, and `logfire` tools before custom implementations.

## Backend Module Layout (Example)

```python
"""Agent module.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""
from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from pydantic_ai import Agent

if TYPE_CHECKING:
    import httpx

__all__ = (
    "AgentDeps",
    "AgentResult",
    "agent",
)

SCHEMA_VERSION: str = "1.0.0"
AGENT_SYSTEM_PROMPT = """System prompt for the agent."""


@dataclass
class AgentDeps:
    """Dependencies for the agent."""

    http_client: httpx.AsyncClient


class AgentResult(BaseModel):
    """Result payload for the agent."""

    schema_version: str = Field(default=SCHEMA_VERSION)


agent: Agent[AgentDeps, AgentResult] = ...
```

## Development Commands

### Backend (`backend/`)

```bash
cd backend

# Install dependencies
uv sync

# Run tests
uv run pytest -v

# Run specific test
uv run pytest tests/test_file.py::test_name -v

# Linting and formatting
uv run ruff check .
uv run ruff format .
uv run mypy .

# Run multi-agent demo
uv run python examples/multi_agent_playbook.py --model devstral:local
uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

# Start FastAPI server (port 9000)
uvicorn cli:app --host 0.0.0.0 --port 9000 --reload
# Or:
python -m agent_k.ui.ag_ui
```

### Frontend (`frontend/`)

```bash
cd frontend

# Install dependencies
pnpm install

# Development server (port 3000)
pnpm dev

# Production build
pnpm build

# Linting (uses Ultracite)
pnpm lint
pnpm format

# E2E tests
pnpm test:e2e
```

### Start Both Servers

```bash
./run.sh  # Starts backend (9000) and frontend (3000)
```

## Pre-Commit Checks (Required)

Before each commit (or pre-commit run), always execute and fix failures:

### Backend (`backend/`)

```bash
uv run ruff format .
uv run ruff check .
uv run pytest -v --tb=short
```

### Frontend (`frontend/`)

```bash
pnpm lint
pnpm build
```

If any formatter/linter changes files, re-run the checks before committing.

## Project Architecture

```
agent-k/
├── backend/
│   ├── agent_k/
│   │   ├── agents/           # Multi-agent system
│   │   │   ├── base.py
│   │   │   ├── evolver.py
│   │   │   ├── lobbyist.py
│   │   │   ├── lycurgus.py
│   │   │   └── scientist.py
│   │   ├── adapters/         # External service adapters
│   │   │   ├── kaggle.py
│   │   │   └── openevolve.py
│   │   ├── mission/          # Pydantic-Graph state machine
│   │   │   ├── nodes.py      # Phase nodes (Discovery, Research, etc.)
│   │   │   ├── state.py      # MissionState model
│   │   │   └── edges.py      # Transition logic
│   │   ├── toolsets/         # FunctionToolset implementations
│   │   │   ├── browser.py
│   │   │   ├── code.py
│   │   │   ├── kaggle.py
│   │   │   ├── memory.py
│   │   │   ├── scholarly.py
│   │   │   └── search.py
│   │   ├── core/             # Domain models and types
│   │   ├── embeddings/       # RAG utilities
│   │   ├── evals/            # Evaluation framework
│   │   ├── infra/            # Config, logging, model providers
│   │   └── ui/               # AG-UI protocol (FastAPI)
│   └── examples/             # Demo scripts
├── frontend/
│   ├── components/agent-k/   # Mission dashboard components
│   ├── hooks/                # useAgentKState, etc.
│   └── lib/ai/               # Model configuration
└── docs/                     # Project docs and style guide
```

## Key Patterns

Snippets omit the standard module header; include the Backend Module Layout above for all backend files.

### Agent Singleton Pattern

Each agent is defined once at module level and registered:

```python
settings = LobbyistSettings()

lobbyist_agent: Agent[LobbyistDeps, DiscoveryResult] = Agent(
    model=get_model(settings.model),
    deps_type=LobbyistDeps,
    output_type=DiscoveryResult,
    instructions=LOBBYIST_SYSTEM_PROMPT,
    name="lobbyist",
    model_settings=settings.model_settings,
    retries=settings.tool_retries,
    output_retries=settings.output_retries,
    instrument=True,
)

register_agent("lobbyist", lobbyist_agent)
```

### Dependency Injection

Dependencies are passed via dataclasses:

```python
from dataclasses import dataclass


@dataclass
class LobbyistDeps:
    """Dependencies for the Lobbyist agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
```

### FunctionToolset Pattern

Toolsets are model-agnostic (work with any provider):

```python
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.deps import KaggleDeps

kaggle_toolset: FunctionToolset[KaggleDeps] = FunctionToolset(id="kaggle")


@kaggle_toolset.tool
async def kaggle_search_competitions(
    ctx: RunContext[KaggleDeps],
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search Kaggle for active competitions."""
    competitions: list[dict[str, Any]] = []
    async for comp in ctx.deps.kaggle_adapter.search_competitions(
        categories=categories,
    ):
        competitions.append(comp.model_dump())
    return competitions
```

### Graph State Machine

Mission phases are nodes in a pydantic-graph:

```python
from dataclasses import dataclass


@dataclass
class DiscoveryNode(BaseNode[MissionState, MissionResult]):
    """Discovery phase for the mission graph."""

    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> ResearchNode | End[MissionResult]:
        return ResearchNode(scientist_agent=...)
```

## Model Configuration

Supported model specifications via `get_model()` in `backend/agent_k/infra/providers.py`:

| Spec | Description |
|------|-------------|
| `devstral:local` | Local LM Studio (default: `http://192.168.105.1:1234/v1`) |
| `devstral:http://host:port/v1` | Custom Devstral endpoint |
| `anthropic:claude-3-haiku-20240307` | Claude Haiku |
| `anthropic:claude-sonnet-4-20250514` | Claude Sonnet |
| `openrouter:mistralai/devstral-small` | Devstral via OpenRouter |
| `openai:gpt-4o` | GPT-4o |

Devstral is treated like any other model provider. Use the standard examples with a `--model` flag
(`devstral:local`, `openrouter:mistralai/devstral-small`, etc.) and avoid creating dedicated
Devstral-only scripts.

## Environment Variables

Required in `backend/.env`:

```bash
# Kaggle API (required)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Model providers (at least one)
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...

# Optional
DEVSTRAL_BASE_URL=http://192.168.105.1:1234/v1
LOGFIRE_TOKEN=...
```

## Key Files

| File | Purpose |
|------|---------|
| `docs/python-ai-style-guide.md` | Backend coding conventions (required) |
| `backend/agent_k/infra/providers.py` | Model factory (`get_model()`) |
| `backend/agent_k/toolsets/__init__.py` | Toolset exports |
| `backend/agent_k/agents/lycurgus.py` | Orchestrator |
| `backend/agent_k/mission/nodes.py` | State machine phases |
| `backend/examples/multi_agent_playbook.py` | Full demo |
| `frontend/components/agent-k/mission-dashboard.tsx` | Main UI |
| `render.yaml` | Render deployment config |

## Testing

```bash
# Backend
cd backend
uv run pytest -v                          # All tests
uv run pytest tests/test_file.py -v       # Single file
uv run pytest -k "discovery" -v           # Pattern match

# Frontend E2E
cd frontend
pnpm test:e2e
```

## Mission Lifecycle

```
Discovery → Research → Prototype → Evolution → Submission
    │           │          │           │           │
    ▼           ▼          ▼           ▼           ▼
  Find      Analyze     Build      Optimize    Submit
  comps    leaderboard  baseline   solution    final
```

Each phase is a `BaseNode` that returns either the next node or `End[MissionResult]`.

## Adding a New Agent

1. Create `backend/agent_k/agents/<agentname>.py` (single lowercase word, no underscores).
2. Keep settings, deps, output models, prompts, toolsets, and the agent singleton in the same file unless it exceeds ~500-800 lines or splits along distinct domains.
3. Follow `docs/python-ai-style-guide.md` for module layout, docstrings, import order, `__all__`, typing, and formatting.
4. Export from `backend/agent_k/agents/__init__.py`.

## Adding a New Toolset

1. Create `backend/agent_k/toolsets/<toolsetname>.py` (single lowercase word, no underscores).
2. Use the Backend Module Layout above, then define a module-level `<toolsetname>_toolset` with `FunctionToolset`.
3. Export from `backend/agent_k/toolsets/__init__.py` and register in `TOOLSET_REGISTRY` when needed.

```python
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

analysis_toolset: FunctionToolset[Any] = FunctionToolset(id="analysis")


@analysis_toolset.tool
async def analyze(query: str) -> str:
    """Tool description for the LLM."""
    return f"Result for {query}"
```

## Deployment

Deploys to Render via `render.yaml`:
- Backend: FastAPI on port 9000
- Frontend: Next.js on port 3000

Environment variables are set in Render's `agent-k-secrets` group.
