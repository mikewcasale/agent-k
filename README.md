<div align="center">
  <img src="backend/docs/logo.png" alt="AGENT-K" width="400">
</div>

<div align="center">
  <h3>Multi-Agent Kaggle GrandMaster (🧐)</h3>
  <p><em>Autonomous multi-agent framework for discovering, entering, and winning Kaggle competitions</em></p>
</div>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/mikewcasale/agent-k/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://ai.pydantic.dev/"><img src="https://img.shields.io/badge/built%20with-Pydantic--AI-orange.svg" alt="Built with Pydantic-AI"></a>
  <a href="https://pydantic.dev/logfire"><img src="https://img.shields.io/badge/observability-Logfire-purple.svg" alt="Observability with Logfire"></a>
</div>

<div align="center">
  <br>
  <a href="https://agents-k.com"><img src="https://img.shields.io/badge/🚀_Live_Demo-agents--k.com-ff6b6b?style=for-the-badge&labelColor=1a1a2e" alt="Live Demo"></a>
  <br><br>
</div>

---

## Overview

AGENT-K is an autonomous multi-agent system that discovers, researches, prototypes, evolves, and submits solutions to Kaggle competitions. The system leverages:

- **Pydantic-AI** agents with FunctionToolsets (Kaggle, Search, Memory)
- **Pydantic-Graph** state machine for orchestration
- **OpenEvolve** framework for evolutionary code search
- **Pydantic Logfire** for comprehensive observability
- **Next.js** frontend for real-time mission monitoring
- **SAGE Spec** structured docstrings and `.sage/` metadata for agent + human guidance

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LYCURGUS ORCHESTRATOR                              │
│                    (Pydantic-Graph State Machine)                            │
│                                                                              │
│  Discovery -> Research -> Prototype -> Evolution -> Submission               │
│     |           |             |            |           |                     │
│  LOBBYIST    SCIENTIST      baseline     EVOLVER    adapter submit           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           TOOLING & ADAPTERS                          │   │
│  │  • Kaggle Toolset (FunctionToolset)                                    │   │
│  │  • Built-in WebSearch/WebFetch                                         │   │
│  │  • MemoryTool + AgentKMemoryTool (Anthropic only)                      │   │
│  │  • CodeExecutionTool (provider)                                        │   │
│  │  • Kaggle MCP (evolver submissions)                                    │   │
│  │  • Platform Adapters: Kaggle API or OpenEvolve (in-memory)             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agents

### LYCURGUS (Orchestrator | Law Giver)

The central orchestrator coordinating the multi-agent competition lifecycle. Implements a state machine using `pydantic-graph` to manage phase transitions, resource allocation, error recovery, and mission persistence.

### LOBBYIST (Discovery)

Discovers and evaluates Kaggle competitions matching user-specified criteria. Uses web search and Kaggle API to find competitions based on prize pool, deadline, domain alignment, and team constraints.

### SCIENTIST (Research)

Conducts comprehensive research including literature review, leaderboard analysis, exploratory data analysis, and strategy synthesis. Identifies winning approaches from similar past competitions.

### EVOLVER (Optimization)

Evolves solutions using evolutionary code search to maximize competition score. Manages population-based optimization with mutations, crossover, and fitness evaluation.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Orchestration** | Pydantic-Graph state machine coordinates specialized agents through competition lifecycle |
| **Evolutionary Code Search** | OpenEvolve integration for population-based solution optimization |
| **Kaggle Integration** | FunctionToolset-based platform operations with OpenEvolve fallback for offline runs |
| **Real-Time Observability** | Pydantic Logfire instrumentation for tracing, metrics, and debugging |
| **SAGE Spec Documentation** | Structured docstrings and `.sage/` artifacts for agent navigation and review workflows |
| **Interactive Dashboard** | Next.js frontend with mission monitoring, evolution visualization, and tool call inspection |
| **Memory Persistence** | Cross-session context and checkpoint management for long-running missions |
| **Error Recovery** | Automatic retry, fallback, and replanning strategies for robust execution |

---

## SAGE Spec (Structured Agent Guidance Embeddings)

AGENT-K uses the SAGE spec to encode agent guidance and machine-readable metadata directly in
docstrings and type annotations. The upstream spec lives at
[github.com/mikewcasale/sage-spec](https://github.com/mikewcasale/sage-spec), and a snapshot is
vendored at `backend/docs/sage-spec.md`.

Key pieces in this repo:

- Docstrings include SAGE tags like `@notice`, `@dev`, `@graph`, `@agent-guidance`, and
  `@human-review` to capture the Contextual Relationship Graph (CRG).
- Parameter metadata is co-located with signatures using `typing.Annotated` plus
  `Doc`/`Range`/`Constraint`/`Default` from `agent_k.core.sage` (`backend/agent_k/core/sage.py`).
- Machine-readable exports live in `backend/.sage/` (for example `components.json`,
  `canonical-homes.json`, and `errors.json`) to support tooling and fast lookups without scanning
  the full tree.

Developer workflow:
- Use the SAGE Docstrings VS Code extension (https://marketplace.visualstudio.com/items?itemName=gcode-ai.sage-vscode-extension) for tag templates and validation.
- Refresh `.sage` artifacts before committing:
  `backend/.venv/bin/python backend/tools/sageextract.py --emit` (pre-commit runs validation).

---

## Mission Lifecycle

AGENT-K executes missions through a 5-phase lifecycle:

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────────┐
│ DISCOVERY │───▶│ RESEARCH  │───▶│ PROTOTYPE │───▶│ EVOLUTION │───▶│ SUBMISSION │
└───────────┘    └───────────┘    └───────────┘    └───────────┘    └────────────┘
     │                │                │                │                 │
     ▼                ▼                ▼                ▼                 ▼
  Find and         Analyze          Build           Optimize          Submit
  validate       leaderboard,      baseline         solution          final
 competition      research,       working          using ECS         solution
                    EDA           solution
```

### Phase Details

1. **Discovery** — Search for competitions matching criteria (prize, deadline, domain), validate accessibility, rank candidates
2. **Research** — Analyze leaderboard distribution, review academic papers and winning solutions, perform EDA, synthesize strategy
3. **Prototype** — Generate baseline solution from research findings, validate execution, establish baseline score
4. **Evolution** — Initialize population, evaluate fitness, apply mutations/crossover, detect convergence, submit checkpoints
5. **Submission** — Generate final predictions, submit to Kaggle, wait for scoring, record final rank

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Agent Framework | [Pydantic-AI](https://ai.pydantic.dev/) | Agent definitions, tool registration, structured outputs |
| Orchestration | [Pydantic-Graph](https://ai.pydantic.dev/graph/) | State machine, phase transitions |
| Evolution | OpenEvolve | Evolutionary code search |
| Kaggle API | KaggleToolset | Platform operations |
| Observability | [Pydantic Logfire](https://pydantic.dev/logfire) | Tracing, metrics, logging |
| HTTP Client | HTTPX | Async HTTP requests |

### Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Next.js 16 | React server components, routing |
| UI Library | React 19 | Component rendering |
| Protocol | AG-UI | Agent-to-UI event streaming |
| Styling | Tailwind CSS | Utility-first styling |
| Charts | Recharts | Evolution visualization |
| State | SWR | Data fetching and caching |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Node.js 20+
- pnpm
- Kaggle API credentials

### Backend Setup

```bash
cd backend

# Install dependencies with uv
uv sync

# Activate virtual environment (uv creates .venv automatically)
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Set environment variables (or create backend/.env)
# At least one model provider API key is required
export ANTHROPIC_API_KEY="your-api-key"
# or: OPENROUTER_API_KEY / OPENAI_API_KEY

# Kaggle credentials are required for live competitions
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="your-kaggle-key"
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Set environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Run development server
pnpm dev
```

### Run Both Servers

```bash
# From project root - starts backend (9000) and frontend (3000)
./run.sh
```

### Run Backend API (AG-UI)

```bash
cd backend
python -m agent_k.ui.agui
```

Run a mission through the chat endpoint (streams Vercel AI data events):

```bash
curl -N -X POST http://localhost:9000/agentic_generative_ui/ \
  -H "Content-Type: application/json" \
  -d '{"id":"demo","messages":[{"role":"user","parts":[{"type":"text","text":"Find a Kaggle competition with a $10k+ prize"}]}]}'
```

### Run a Mission (Programmatic)

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
        print(f"Final score: {result.final_score}")

asyncio.run(main())
```

---

## Model Configuration

AGENT-K supports multiple model providers via `get_model()`. Standard Pydantic-AI model
strings are passed through; `devstral:` and `openrouter:` specs resolve to
OpenAI-compatible models.

| Model Spec | Description |
|------------|-------------|
| `devstral:local` | Local LM Studio server (default: `http://192.168.105.1:1234/v1`) |
| `devstral:http://host:port/v1` | Custom Devstral endpoint |
| `devstral:mistralai/devstral-small-2-2512` | Local LM Studio with explicit model id |
| `anthropic:claude-3-haiku-20240307` | Claude Haiku via Anthropic |
| `anthropic:claude-sonnet-4-5` | Claude Sonnet (backend default) |
| `openrouter:mistralai/devstral-small-2-2512` | Devstral via OpenRouter |
| `openai:gpt-4o` | GPT-4o via OpenAI |

---

## Project Structure

```
agent-k/
├── backend/
│   ├── .sage/                     # SAGE metadata artifacts (CRG exports)
│   ├── agent_k/
│   │   ├── agents/                 # Pydantic-AI agent definitions
│   │   │   ├── base.py
│   │   │   ├── evolver.py
│   │   │   ├── lobbyist.py
│   │   │   ├── lycurgus.py
│   │   │   ├── scientist.py
│   │   │   └── prompts.py
│   │   ├── adapters/               # Platform integrations
│   │   │   ├── kaggle.py
│   │   │   └── openevolve.py
│   │   ├── core/                   # Domain models and helpers
│   │   │   ├── constants.py
│   │   │   ├── data.py
│   │   │   ├── deps.py
│   │   │   ├── exceptions.py
│   │   │   ├── models.py
│   │   │   ├── protocols.py
│   │   │   ├── settings.py
│   │   │   ├── solution.py
│   │   │   ├── sage.py
│   │   │   └── types.py
│   │   ├── mission/                # State machine
│   │   │   ├── nodes.py
│   │   │   ├── persistence.py
│   │   │   └── state.py
│   │   ├── toolsets/               # FunctionToolset helpers
│   │   │   ├── code.py
│   │   │   ├── kaggle.py
│   │   │   ├── memory.py
│   │   │   ├── search.py
│   │   │   ├── browser.py          # Placeholder
│   │   │   └── scholarly.py        # Placeholder
│   │   ├── embeddings/             # RAG support
│   │   │   ├── embedder.py
│   │   │   ├── retriever.py
│   │   │   └── store.py
│   │   ├── evals/                  # Evaluation framework
│   │   │   ├── datasets.py
│   │   │   ├── evaluators.py
│   │   │   ├── discovery.yaml
│   │   │   ├── evolution.yaml
│   │   │   └── submission.yaml
│   │   ├── infra/                  # Infrastructure
│   │   │   ├── config.py
│   │   │   ├── instrumentation.py
│   │   │   ├── logging.py
│   │   │   └── providers.py
│   │   └── ui/                     # AG-UI protocol (FastAPI)
│   │       └── agui.py
│   ├── cli.py                      # FastAPI app entrypoint
│   ├── docs/                       # Backend docs (mkdocs + logo.png)
│   └── tests/
│
├── frontend/
│   ├── app/                        # Next.js app router
│   │   ├── (auth)/                 # Authentication
│   │   └── (chat)/                 # Chat interface
│   ├── components/
│   │   ├── agent-k/                # Mission dashboard
│   │   │   ├── mission-dashboard.tsx
│   │   │   ├── phase-card.tsx
│   │   │   ├── evolution-view.tsx
│   │   │   ├── fitness-chart.tsx
│   │   │   └── tool-call-card.tsx
│   │   └── ui/                     # Shared UI components
│   ├── hooks/                      # React hooks
│   │   └── use-agent-k-state.tsx   # Mission state hook
│   └── lib/
│       ├── ai/                     # Model configuration
│       │   └── models.ts           # Available chat models
│       └── types/
│           └── agent-k.ts          # TypeScript types
│
├── run.sh                          # Start both servers
└── render.yaml                     # Render deployment config
```

---

## Configuration

### Environment Variables

#### Backend (`backend/.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `KAGGLE_USERNAME` | Kaggle account username | Yes* |
| `KAGGLE_KEY` | Kaggle API key | Yes* |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | Yes** |
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes** |
| `OPENAI_API_KEY` | OpenAI API key | Yes** |
| `DEVSTRAL_BASE_URL` | Local LM Studio endpoint (default: `http://192.168.105.1:1234/v1`) | No |
| `LOGFIRE_TOKEN` | Pydantic Logfire token | No |
| `AGENT_K_MEMORY_DIR` | Memory tool storage path | No |

#### Frontend (`frontend/.env.local`)

| Variable | Description | Required |
|----------|-------------|----------|
| `AUTH_SECRET` | Auth.js signing secret | Yes |
| `AUTH_TRUST_HOST` | Required behind reverse proxies (Render, etc.) | Conditional |
| `AUTH_URL` | Base URL for Auth.js callbacks | Yes |
| `POSTGRES_URL` | PostgreSQL connection string | Yes |
| `PYTHON_BACKEND_URL` | Agent K backend SSE endpoint | Yes (Agent K) |
| `ANTHROPIC_API_KEY` | Claude models for in-app chat | Optional |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob storage for uploads | Optional |
| `REDIS_URL` | Redis cache | Optional |

*Required for Kaggle platform access. If absent, the orchestrator falls back to OpenEvolve.

**At least one model provider API key is required.

### Mission Criteria

```python
from agent_k.core.models import MissionCriteria, CompetitionType

criteria = MissionCriteria(
    target_competition_types=frozenset({
        CompetitionType.FEATURED,
        CompetitionType.RESEARCH,
    }),
    min_prize_pool=10000,
    min_days_remaining=14,
    target_domains=frozenset({"computer_vision", "nlp"}),
    max_evolution_rounds=100,
    target_leaderboard_percentile=0.10,
)
```

---

## Observability

AGENT-K uses Pydantic Logfire for comprehensive observability:

```python
from agent_k.infra.instrumentation import configure_instrumentation

configure_instrumentation(
    service_name="agent-k",
    environment="production",
)
```

### Metrics Tracked

- Phase completion times and success rates
- Evolution generation fitness progression
- Tool call latency and error rates
- Kaggle submission scores and rankings
- API token usage and costs

---

## Development

### Running Tests

```bash
# Backend tests
cd backend
uv run pytest -v

# Run specific test
uv run pytest tests/test_file.py::test_name -v

# Frontend E2E tests
cd frontend
pnpm test:e2e
```

### Code Quality

```bash
# Backend linting
cd backend
uv run ruff check .
uv run ruff format .
uv run mypy .

# Frontend linting (uses Ultracite)
cd frontend
pnpm lint
pnpm format
```

### Pre-Commit (Backend)

```bash
cd backend
uv run ruff format .
uv run ruff check .
backend/.venv/bin/python backend/tools/sageextract.py --emit
backend/.venv/bin/python backend/tools/sageextract.py --validate
uv run pytest -v --tb=short
```

---

## Deployment

Deploys to Render via `render.yaml`:

- **Backend**: FastAPI on port 9000 (`agent-k-api`)
- **Frontend**: Next.js on port 3000 (`agent-k-frontend`)
- **Database**: PostgreSQL (`agent-k-postgres`)

Environment variables are set in Render's `agent-k-secrets` group.

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the `main` branch.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Pydantic](https://pydantic.dev/) for the excellent AI framework and observability tools
- [Kaggle](https://www.kaggle.com/) for the competition platform
- [OpenEvolve](https://github.com/codelion/openevolve) for evolutionary code search inspiration
- [SAGE Spec](https://github.com/mikewcasale/sage-spec) for structured agent guidance metadata

---

<!-- <div align="center">
  <sub>Built with ❤️</sub>
</div> -->
