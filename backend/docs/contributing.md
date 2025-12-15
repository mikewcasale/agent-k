# Contributing to AGENT-K

Thank you for your interest in contributing to AGENT-K! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 20+ (for frontend)
- pnpm (for frontend)

### Clone and Install

```bash
# Clone repository
git clone https://github.com/mikewcasale/agent-k.git
cd agent-k

# Backend
cd backend
uv sync --all-extras
source .venv/bin/activate

# Frontend (optional)
cd ../frontend
pnpm install
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

# All tests
uv run pytest -v

# Specific test file
uv run pytest tests/test_file.py -v

# With coverage
uv run pytest --cov=agent_k --cov-report=html
```

### Linting and Formatting

```bash
cd backend

# Check linting
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run mypy .
```

### Pre-Commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Code Style

### Python

- **Formatter**: Ruff
- **Line Length**: 120 characters
- **Quote Style**: Single quotes
- **Type Hints**: Required for all public functions

Example:

```python
from dataclasses import dataclass
from typing import Any

import logfire
from pydantic import BaseModel


@dataclass
class MyConfig:
    """Configuration for something."""
    
    name: str
    value: int = 10


async def my_function(config: MyConfig) -> dict[str, Any]:
    """Do something with config.
    
    Args:
        config: The configuration to use.
    
    Returns:
        Result dictionary.
    """
    with logfire.span('my_function'):
        return {'name': config.name, 'result': config.value * 2}
```

### TypeScript (Frontend)

- **Formatter**: Ultracite (Biome)
- **Framework**: Next.js 16 with App Router

## Project Structure

```
agent-k/
├── backend/
│   └── agent_k/
│       ├── agents/       # Agent implementations
│       ├── adapters/     # External service adapters
│       ├── graph/        # State machine
│       ├── toolsets/     # FunctionToolset implementations
│       ├── core/         # Domain models
│       ├── infra/        # Config, logging, models
│       └── ui/           # AG-UI protocol
└── frontend/
    ├── components/       # React components
    ├── hooks/            # Custom hooks
    └── lib/              # Utilities
```

## Adding Features

### New Agent

1. Create `backend/agent_k/agents/my_agent/`
2. Add `agent.py`, `prompts.py`, `tools.py`, `__init__.py`
3. Export from `backend/agent_k/agents/__init__.py`
4. Add tests in `backend/tests/agents/`
5. Document in `backend/docs/agents/`

See [Creating a Custom Agent](examples/custom-agent.md) for details.

### New Toolset

1. Create `backend/agent_k/toolsets/my_toolset.py`
2. Export from `backend/agent_k/toolsets/__init__.py`
3. Add tests in `backend/tests/toolsets/`
4. Document in `backend/docs/toolsets/`

### New API Endpoint

1. Add route in `backend/agent_k/ui/ag_ui/routes/`
2. Update OpenAPI schema if needed
3. Add tests in `backend/tests/ui/`

## Pull Request Process

### 1. Create Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Run Checks

```bash
cd backend
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pytest -v
```

### 4. Commit

Use conventional commits:

```bash
git commit -m "feat: add new toolset for X"
git commit -m "fix: handle edge case in discovery"
git commit -m "docs: update installation guide"
```

### 5. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub.

### PR Checklist

- [ ] Tests pass
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)

## Documentation

### Building Docs

```bash
cd backend
uv run mkdocs build
```

### Serving Docs Locally

```bash
uv run mkdocs serve
```

Open [http://localhost:8000](http://localhost:8000).

### Documentation Style

- Use clear, concise language
- Include code examples
- Add diagrams where helpful (Mermaid supported)

## Testing Guidelines

### Unit Tests

Test individual functions and classes:

```python
async def test_my_function():
    result = await my_function(MyConfig(name='test'))
    assert result['name'] == 'test'
    assert result['result'] == 20
```

### Integration Tests

Test components working together:

```python
async def test_agent_with_toolset():
    agent = create_my_agent()
    result = await agent.run('do something', deps=mock_deps)
    assert result.data.success
```

### Mocking

Mock external services:

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mock_http():
    client = AsyncMock()
    client.get.return_value.json.return_value = {'data': 'test'}
    return client
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/mikewcasale/agent-k/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mikewcasale/agent-k/discussions)

## License

AGENT-K is MIT licensed. By contributing, you agree that your contributions will be licensed under the MIT License.

