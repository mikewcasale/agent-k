<div align="center">
  <img src="docs/logo.png" alt="AGENT-K" width="400">
</div>

<div align="center">
  <h3>Multi-Agent Kaggle GrandMaster (🧐)</h3>
  <p><em>Autonomous multi-agent framework for discovering, entering, and winning Kaggle competitions</em></p>
</div>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/casalexyz/agent-k/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
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

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LYCURGUS ORCHESTRATOR                              │
│                    (Pydantic-Graph State Machine)                            │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   LOBBYIST   │───▶│  SCIENTIST   │───▶│   EVOLVER    │───▶│ SUBMITTER │  │
│  │  Discovery   │    │   Research   │    │  Evolution   │    │   Final   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        SHARED TOOLSETS                                │   │
│  │  • KaggleToolset (API)              • CodeExecutorToolset             │   │
│  │  • SearchToolset (Web/Papers)       • MemoryToolset (Persistence)     │   │
│  │  • BrowserToolset                   • ScholarlyToolset                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                          ┌─────────────────────┐
                          │    KAGGLE PLATFORM  │
                          │   (via Kaggle API)  │
                          └─────────────────────┘
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
| **Kaggle Integration** | FunctionToolset-based platform operations for seamless competition interaction |
| **Real-Time Observability** | Pydantic Logfire instrumentation for tracing, metrics, and debugging |
| **Interactive Dashboard** | Next.js frontend with mission monitoring, evolution visualization, and tool call inspection |
| **Memory Persistence** | Cross-session context and checkpoint management for long-running missions |
| **Error Recovery** | Automatic retry, fallback, and replanning strategies for robust execution |

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
export ANTHROPIC_API_KEY="your-api-key"
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

### Run Multi-Agent Demo

```bash
cd backend

# Local Devstral (LM Studio)
uv run python examples/multi_agent_playbook.py --model devstral:local

# Claude Haiku (Anthropic)
uv run python examples/multi_agent_playbook.py --model anthropic:claude-3-haiku-20240307

# Devstral via OpenRouter
uv run python examples/multi_agent_playbook.py --model openrouter:mistralai/devstral-small
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

AGENT-K supports multiple model providers via `get_model()`:

| Model Spec | Description |
|------------|-------------|
| `devstral:local` | Local LM Studio server (default: `http://192.168.105.1:1234/v1`) |
| `devstral:http://custom:port/v1` | Custom Devstral endpoint |
| `anthropic:claude-3-haiku-20240307` | Claude Haiku via Anthropic |
| `anthropic:claude-sonnet-4-20250514` | Claude Sonnet via Anthropic |
| `openrouter:mistralai/devstral-small` | Devstral via OpenRouter |
| `openai:gpt-4o` | GPT-4o via OpenAI |

---

## Project Structure

```
agent-k/
├── backend/
│   └── agent_k/
│       ├── agents/                 # Pydantic-AI agent definitions
│       │   ├── lobbyist/           # Competition discovery
│       │   ├── scientist/          # Research and analysis
│       │   ├── evolver/            # Solution optimization
│       │   └── lycurgus/           # Orchestration
│       ├── adapters/               # Platform integrations
│       │   ├── kaggle/             # Kaggle API adapter
│       │   └── openevolve/         # OpenEvolve integration
│       ├── core/                   # Domain models and protocols
│       │   ├── models.py           # Pydantic models
│       │   ├── protocols.py        # Interface definitions
│       │   ├── exceptions.py       # Exception hierarchy
│       │   └── types.py            # Type aliases
│       ├── graph/                  # State machine
│       │   ├── nodes.py            # Phase nodes
│       │   ├── state.py            # Mission state
│       │   └── persistence.py      # Checkpoint management
│       ├── toolsets/               # FunctionToolset implementations
│       │   ├── kaggle.py           # Kaggle API operations
│       │   ├── search.py           # Web/paper search
│       │   ├── memory.py           # Persistent memory
│       │   ├── browser.py          # Browser automation
│       │   ├── code_executor.py    # Code execution
│       │   └── scholarly.py        # Academic search
│       ├── services/               # Application services
│       │   ├── competition.py      # Competition management
│       │   ├── evolution.py        # Evolution orchestration
│       │   └── submission.py       # Submission handling
│       ├── ui/                     # UI adapters
│       │   ├── ag_ui/              # AG-UI protocol (FastAPI)
│       │   └── console/            # Terminal console
│       └── infra/                  # Infrastructure
│           ├── config.py           # Configuration
│           ├── models.py           # Model factory (get_model)
│           ├── logging.py          # Logging setup
│           └── instrumentation.py  # Observability
│   └── examples/                   # Demo scripts
│       └── multi_agent_playbook.py # Full multi-agent demo
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
├── docs/                           # Documentation
│   └── logo.png                    # Project logo
│
├── run.sh                          # Start both servers
├── render.yaml                     # Render deployment config
│
└── refs/                           # Reference documentation
    ├── python_spec_v2.md           # Architecture specification
    └── agent_k_playbook.md         # Operational playbook
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | Yes* |
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes* |
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `KAGGLE_USERNAME` | Kaggle account username | Yes |
| `KAGGLE_KEY` | Kaggle API key | Yes |
| `DEVSTRAL_BASE_URL` | Local LM Studio endpoint (default: `http://192.168.105.1:1234/v1`) | No |
| `LOGFIRE_TOKEN` | Pydantic Logfire token | No |
| `DATABASE_URL` | PostgreSQL connection string | Frontend |

*At least one model provider API key is required.

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
import logfire

# Configure at startup
logfire.configure(
    service_name='agent-k',
    environment='production',
)

# Automatic instrumentation
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()
logfire.instrument_asyncio()
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

---

<div align="center">
  <sub>Built with Pydantic-AI • Powered by Claude</sub>
</div>
