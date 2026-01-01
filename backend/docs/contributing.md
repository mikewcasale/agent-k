# Contributing to AGENT-K

Thank you for your interest in contributing to AGENT-K. This guide covers the backend development workflow and conventions.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 20+ and pnpm (frontend only)

### Clone and Install

```bash
# Clone repository
git clone https://github.com/mikewcasale/agent-k.git
cd agent-k

# Backend
cd backend
uv sync --all-extras
source .venv/bin/activate
```

### Configure Environment

Create `backend/.env`:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
ANTHROPIC_API_KEY=sk-ant-...
```

## Development Workflow

### Running Tests

```bash
cd backend

uv run pytest -v
uv run pytest tests/test_file.py -v
```

### Linting and Formatting

```bash
cd backend

uv run ruff check .
uv run ruff format .
uv run mypy .
```

### Pre-Commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Code Style

### Python (Backend)

All backend code under `backend/` must follow `docs/python-ai-style-guide.md`.

Key points:

- Module header with MIT license notice and `from __future__ import annotations as _annotations`
- Import ordering and `TYPE_CHECKING` blocks
- Double quotes and 88-100 character line length
- Early-return control flow
- Use `logfire` for observability

### TypeScript (Frontend)

- Formatter: Ultracite (Biome)
- Framework: Next.js with App Router

## Project Structure

```
agent-k/
├── backend/
│   └── agent_k/
│       ├── agents/       # Agent implementations
│       ├── adapters/     # External service adapters
│       ├── mission/      # State machine
│       ├── toolsets/     # FunctionToolset helpers
│       ├── core/         # Domain models and helpers
│       ├── embeddings/   # RAG support
│       ├── evals/        # Evaluation framework
│       ├── infra/        # Config, logging, providers
│       └── ui/           # AG-UI protocol
└── frontend/
```

## Adding Features

### New Agent

1. Create `backend/agent_k/agents/<agentname>.py` (single lowercase word, no underscores)
2. Keep settings, deps, output models, prompts, tool registrations, and the singleton in the same file
3. Follow the module layout in `docs/python-ai-style-guide.md`
4. Export the agent from `backend/agent_k/agents/__init__.py`
5. Document the agent in `backend/docs/agents/`

### New Toolset

1. Create `backend/agent_k/toolsets/<toolsetname>.py` (single lowercase word, no underscores)
2. Define a module-level `<toolsetname>_toolset` using `FunctionToolset`
3. Export from `backend/agent_k/toolsets/__init__.py`
4. Document the toolset in `backend/docs/toolsets/`

### New API Endpoint

1. Add a route in `backend/agent_k/ui/ag_ui.py`
2. Update API docs if needed
3. Add tests under `backend/tests/`

## Pull Request Process

1. Create a branch
2. Make changes with tests and docs
3. Run checks:

```bash
cd backend
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest -v
```

4. Submit PR against `main`
