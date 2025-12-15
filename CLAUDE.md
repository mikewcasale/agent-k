# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AGENT-K is an autonomous multi-agent system for participating in Kaggle competitions. Built with Pydantic-AI and FastAPI, it orchestrates specialized agents through five mission phases: Discovery, Research, Prototype, Evolution, and Submission.

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

## Project Architecture

```
agent-k/
├── backend/
│   └── agent_k/
│       ├── agents/           # Multi-agent system
│       │   ├── lycurgus/     # Orchestrator (state machine coordinator)
│       │   ├── lobbyist/     # Competition discovery
│       │   ├── scientist/    # Research and analysis
│       │   └── evolver/      # Evolutionary optimization
│       ├── adapters/         # External service adapters
│       │   ├── kaggle/       # Kaggle API
│       │   └── openevolve/   # OpenEvolve integration
│       ├── graph/            # Pydantic-Graph state machine
│       │   ├── nodes.py      # Phase nodes (Discovery, Research, etc.)
│       │   ├── state.py      # MissionState model
│       │   └── edges.py      # Transition logic
│       ├── toolsets/         # FunctionToolset implementations
│       │   ├── kaggle.py     # kaggle_search_competitions, kaggle_get_leaderboard
│       │   ├── search.py     # web_search, search_papers, search_kaggle
│       │   └── memory.py     # memory_store, memory_retrieve
│       ├── core/             # Domain models and types
│       ├── services/         # Business logic
│       ├── infra/            # Config, logging, model factory
│       └── ui/ag_ui/         # AG-UI protocol (FastAPI)
├── frontend/
│   ├── components/agent-k/   # Mission dashboard components
│   ├── hooks/                # useAgentKState, etc.
│   └── lib/ai/               # Model configuration
└── examples/                 # Demo scripts
```

## Key Patterns

### Agent Factory Pattern

Each agent has a factory function that creates a configured Pydantic-AI Agent:

```python
def create_lobbyist_agent(model: str) -> Agent[LobbyistDeps, DiscoveryResult]:
    model = get_model(model)
    agent = Agent(
        model,
        deps_type=LobbyistDeps,
        output_type=DiscoveryResult,
        instructions="...",
        toolsets=[kaggle_toolset, search_toolset],
    )
    
    @agent.tool
    async def custom_tool(ctx: RunContext[LobbyistDeps]) -> str:
        return await ctx.deps.platform_adapter.do_something()
    
    return agent
```

### Dependency Injection

Dependencies are passed via dataclasses:

```python
@dataclass
class LobbyistDeps:
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
```

### FunctionToolset Pattern

Toolsets are model-agnostic (work with any provider):

```python
def create_kaggle_toolset(adapter: KaggleAdapter) -> FunctionToolset[Any]:
    toolset = FunctionToolset(id='kaggle')
    
    @toolset.tool
    async def kaggle_search_competitions(
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search Kaggle for active competitions."""
        async for comp in adapter.search_competitions(categories=categories):
            yield comp.model_dump()
    
    return toolset
```

### Graph State Machine

Mission phases are nodes in a pydantic-graph:

```python
@dataclass
class DiscoveryNode(BaseNode[MissionState, MissionResult]):
    async def run(
        self,
        ctx: GraphRunContext[MissionState, GraphContext],
    ) -> ResearchNode | End[MissionResult]:
        # Execute discovery logic
        return ResearchNode(scientist_agent=...)
```

## Model Configuration

Supported model specifications via `get_model()` in `backend/agent_k/infra/models.py`:

| Spec | Description |
|------|-------------|
| `devstral:local` | Local LM Studio (default: `http://192.168.105.1:1234/v1`) |
| `devstral:http://host:port/v1` | Custom Devstral endpoint |
| `anthropic:claude-3-haiku-20240307` | Claude Haiku |
| `anthropic:claude-sonnet-4-20250514` | Claude Sonnet |
| `openrouter:mistralai/devstral-small` | Devstral via OpenRouter |
| `openai:gpt-4o` | GPT-4o |

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
| `backend/agent_k/infra/models.py` | Model factory (`get_model()`) |
| `backend/agent_k/toolsets/__init__.py` | Toolset exports |
| `backend/agent_k/agents/lycurgus/agent.py` | Orchestrator |
| `backend/agent_k/graph/nodes.py` | State machine phases |
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

1. Create directory `backend/agent_k/agents/my_agent/`
2. Add files:
   - `agent.py` - Agent factory and class
   - `prompts.py` - System instructions
   - `tools.py` - Agent-specific tools (optional)
   - `__init__.py` - Exports
3. Export from `backend/agent_k/agents/__init__.py`

## Adding a New Toolset

1. Create `backend/agent_k/toolsets/my_toolset.py`:

```python
from pydantic_ai.toolsets import FunctionToolset

def create_my_toolset() -> FunctionToolset:
    toolset = FunctionToolset(id='my_toolset')
    
    @toolset.tool
    async def my_tool(query: str) -> str:
        """Tool description for the LLM."""
        return f"Result for {query}"
    
    return toolset
```

2. Export from `backend/agent_k/toolsets/__init__.py`

## Deployment

Deploys to Render via `render.yaml`:
- Backend: FastAPI on port 9000
- Frontend: Next.js on port 3000

Environment variables are set in Render's `agent-k-secrets` group.
