<div align="center">
  <img src="../docs/logo.png" alt="AGENT-K" width="300">
  <h1>AGENT-K Backend</h1>
  <h3>Multi-Agent Kaggle GrandMaster (🧐)</h3>
  <p><em>Python multi-agent system for autonomous Kaggle competition participation</em></p>
</div>

<div align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://ai.pydantic.dev/"><img src="https://img.shields.io/badge/built%20with-Pydantic--AI-orange.svg" alt="Built with Pydantic-AI"></a>
  <a href="https://pydantic.dev/logfire"><img src="https://img.shields.io/badge/observability-Logfire-purple.svg" alt="Observability with Logfire"></a>
</div>

---

## Overview

The AGENT-K backend is a Python package implementing a multi-agent system for autonomous Kaggle competition participation. Built on Pydantic-AI, it coordinates specialized agents through a state machine to discover competitions, conduct research, prototype solutions, evolve optimizations, and submit entries.

---

## Installation

### Using pip

```bash
cd backend
pip install -e .
```

### Using uv (recommended)

```bash
cd backend
uv pip install -e .
```

### Dependencies

Core dependencies from `pyproject.toml`:

- `pydantic-ai[ag-ui]>=0.2.0` - Agent framework with AG-UI support
- `pydantic-graph>=0.2.0` - State machine orchestration
- `logfire>=0.44.0` - Observability and tracing
- `httpx>=0.27.0` - Async HTTP client
- `opentelemetry-api>=1.27.0` - Telemetry instrumentation

---

## Package Structure

```
agent_k/
├── __init__.py                 # Package exports
├── __main__.py                 # CLI entry point
├── _version.py                 # Version singleton
│
├── agents/                     # Pydantic-AI agent definitions
│   ├── _base.py                # Base agent abstractions
│   ├── lobbyist/               # Competition discovery
│   │   ├── agent.py            # LobbyistAgent
│   │   ├── tools.py            # Discovery tools
│   │   └── prompts.py          # System prompts
│   ├── scientist/              # Research and analysis
│   │   ├── agent.py            # ScientistAgent
│   │   ├── tools.py            # Research tools
│   │   └── prompts.py          # System prompts
│   ├── evolver/                # Solution optimization
│   │   ├── agent.py            # EvolverAgent
│   │   ├── tools.py            # Evolution tools
│   │   └── prompts.py          # System prompts
│   └── lycurgus/               # Orchestration
│       ├── agent.py            # LycurgusOrchestrator
│       ├── tools.py            # Orchestration tools
│       └── prompts.py          # System prompts
│
├── adapters/                   # Platform integrations
│   ├── _base.py                # Adapter protocol
│   ├── kaggle/                 # Kaggle platform
│   │   ├── adapter.py          # KaggleAdapter
│   │   ├── api.py              # API client
│   │   └── models.py           # Kaggle-specific models
│   └── openevolve/             # OpenEvolve integration
│       ├── adapter.py          # Evolution engine adapter
│       └── models.py           # Evolution models
│
├── core/                       # Domain primitives
│   ├── constants.py            # Domain constants
│   ├── deps.py                 # Shared dependency containers
│   ├── exceptions.py           # Exception hierarchy
│   ├── models.py               # Core Pydantic models
│   ├── protocols.py            # Interface definitions
│   ├── settings.py             # Shared settings
│   └── types.py                # Type aliases
│
├── mission/                    # State machine
│   ├── nodes.py                # Phase nodes (Discovery, Research, etc.)
│   ├── edges.py                # Transition definitions
│   ├── state.py                # Mission state models
│   └── persistence.py          # Checkpoint management
│
├── toolsets/                   # Agent tools
│   ├── kaggle.py               # Kaggle toolset
│   ├── search.py               # Web + paper search
│   ├── memory.py               # Persistent memory
│   ├── browser.py              # Browser automation
│   ├── code.py                 # Sandboxed code execution
│   └── scholarly.py            # Academic search
│
├── embeddings/                 # RAG support
│   ├── embedder.py             # Embedding utilities
│   ├── retriever.py            # Retrieval logic
│   └── store.py                # Vector store helpers
│
├── evals/                      # Evaluation framework
│   ├── datasets.py             # Dataset definitions
│   ├── evaluators.py           # Evaluation logic
│   └── discovery.yaml          # Sample eval cases
│
├── ui/                         # UI adapters
│   └── ag_ui.py                # AG-UI protocol + EventEmitter
│
└── infra/                      # Infrastructure
    ├── config.py               # Configuration management
    ├── providers.py            # Model provider configuration
    ├── logging.py              # Centralized logging
    └── instrumentation.py      # Logfire setup
```

---

## Agents

### LYCURGUS (Orchestrator | Law Giver)

The central orchestrator coordinating the multi-agent competition lifecycle. Implements a state machine using `pydantic-graph` to manage phase transitions.

```python
from agent_k import LycurgusOrchestrator

orchestrator = LycurgusOrchestrator(model='anthropic:claude-sonnet-4-5')
```

### LOBBYIST (Discovery)

Discovers and evaluates Kaggle competitions matching user-specified criteria.

```python
from agent_k import lobbyist_agent

result = await lobbyist_agent.run(
    'Find featured competitions with $10k+ prize',
    deps=deps,
)
```

### SCIENTIST (Research)

Conducts comprehensive research including literature review, leaderboard analysis, and EDA.

```python
from agent_k import scientist_agent

report = await scientist_agent.run(
    'Analyze this tabular competition',
    deps=deps,
)
```

### EVOLVER (Optimization)

Evolves solutions using evolutionary code search to maximize competition score.

```python
from agent_k import evolver_agent

result = await evolver_agent.run(
    'Optimize this XGBoost solution',
    deps=deps,
)
```

---

## Usage

### Basic Mission Execution

```python
import asyncio
from agent_k import LycurgusOrchestrator
from agent_k.core.models import MissionCriteria, CompetitionType

async def main():
    # Configure mission criteria
    criteria = MissionCriteria(
        target_competition_types=frozenset({
            CompetitionType.FEATURED,
            CompetitionType.RESEARCH,
        }),
        min_prize_pool=10000,
        min_days_remaining=14,
        target_domains=frozenset({'computer_vision', 'nlp'}),
        max_evolution_rounds=100,
        target_leaderboard_percentile=0.10,
    )
    
    # Execute mission
    async with LycurgusOrchestrator() as orchestrator:
        result = await orchestrator.execute_mission(
            competition_id='titanic',
            criteria=criteria,
        )
        
        print(f'Success: {result.success}')
        print(f'Final Rank: {result.final_rank}')
        print(f'Final Score: {result.final_score}')
        print(f'Generations: {result.evolution_generations}')

asyncio.run(main())
```

### Configuration from File

```python
from pathlib import Path
from agent_k import LycurgusOrchestrator

orchestrator = LycurgusOrchestrator.from_config_file(
    Path('config/mission.json')
)
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | Yes |
| `KAGGLE_USERNAME` | Kaggle account username | Yes |
| `KAGGLE_KEY` | Kaggle API key | Yes |
| `LOGFIRE_TOKEN` | Pydantic Logfire token | No |

### Setting Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key"
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="your-kaggle-key"
export LOGFIRE_TOKEN="your-logfire-token"  # Optional
```

---

## Development

### Running Tests

```bash
pytest
pytest --cov=agent_k  # With coverage
```

### Linting and Formatting

```bash
# Check code style
ruff check .

# Format code
ruff format .

# Type checking
mypy .
```

### Code Quality Tools

The project uses:
- **Ruff** - Fast Python linter and formatter
- **MyPy** - Static type checking with Pydantic plugin
- **Pytest** - Testing with async support

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
```

---

## Observability

AGENT-K uses Pydantic Logfire for comprehensive observability:

```python
import logfire
from agent_k.infra.instrumentation import configure_instrumentation

# Configure at startup
configure_instrumentation(
    service_name='agent-k',
    environment='production',
)

# Automatic instrumentation
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()
logfire.instrument_asyncio()
```

### Custom Spans

```python
import logfire

with logfire.span('custom_operation', param=value):
    # Your code here
    pass
```

---

## Architecture Reference

For detailed architecture specifications, see:

- [Python Enterprise Architecture Specification](../refs/python_spec_v2.md) - Code organization and patterns
- [AGENT-K Operational Playbook](../refs/agent_k_playbook.md) - Mission lifecycle and agent coordination

---

## License

MIT License - see [LICENSE](../LICENSE) for details.
