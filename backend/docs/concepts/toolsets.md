# Toolsets

Agent-K uses Pydantic-AI's `FunctionToolset` pattern for **custom** tools that are not covered by built-in tools. Built-in tools (e.g., `WebSearchTool`, `MemoryTool`) are preferred whenever available.

## Why FunctionToolset?

Built-in tools are executed by the model provider, but custom tools are executed **client-side** by pydantic-ai. Function toolsets are ideal for:

- ✅ Domain-specific APIs (e.g., Kaggle adapters)
- ✅ Local resources (files, databases)
- ✅ Custom business logic
- ✅ Easy testing and mocking

## Built-in Tools

Use built-in tools for web search, URL fetching, memory, and code execution. Agent-K exposes helpers for configuring them:

```python
from pydantic_ai import Agent

from agent_k.toolsets import prepare_web_search

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    builtin_tools=[prepare_web_search],
)
```

Memory (Anthropic only) requires registering a backend:

```python
from pydantic_ai import Agent

from agent_k.toolsets import create_memory_backend, prepare_memory_tool, register_memory_tool

memory_backend = create_memory_backend()
agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    builtin_tools=[prepare_memory_tool],
)
register_memory_tool(agent, memory_backend)
```

## Available Toolsets

### KaggleToolset

Provides Kaggle API operations:

| Tool | Description |
|------|-------------|
| `kaggle_search_competitions` | Search for active competitions |
| `kaggle_get_competition` | Get competition details |
| `kaggle_get_leaderboard` | Get leaderboard entries |
| `kaggle_list_datasets` | List competition data files |

**Usage:**

```python
from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.deps import KaggleDeps
from agent_k.toolsets import kaggle_toolset
from agent_k.ui.ag_ui import EventEmitter

# Create adapter and deps
config = KaggleSettings(username="you", api_key="your-key")
adapter = KaggleAdapter(config)
deps = KaggleDeps(kaggle_adapter=adapter, event_emitter=EventEmitter())

# Use with agent
agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    deps_type=KaggleDeps,
    toolsets=[kaggle_toolset],
)

result = await agent.run("Find featured competitions", deps=deps)
```

## Creating a FunctionToolset

The pattern for creating toolsets:

```python
from typing import Any
from pydantic_ai.toolsets import FunctionToolset
import logfire

def create_my_toolset(some_dependency: SomeDep) -> FunctionToolset[Any]:
    """Create a custom toolset."""

    toolset: FunctionToolset[Any] = FunctionToolset(id='my_toolset')

    @toolset.tool
    async def my_tool(
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for something.

        Args:
            query: The search query.
            limit: Maximum results to return.

        Returns:
            List of matching results.
        """
        with logfire.span('my_tool', query=query):
            results = await some_dependency.search(query, limit=limit)
            return [r.model_dump() for r in results]

    @toolset.tool
    async def another_tool(item_id: str) -> dict[str, Any]:
        """Get details about an item.

        Args:
            item_id: The item identifier.

        Returns:
            Item details.
        """
        item = await some_dependency.get(item_id)
        return item.model_dump()

    return toolset
```

### Key Points

1. **Docstrings matter** — The LLM uses them to understand when to call the tool
2. **Type hints matter** — Pydantic-AI generates JSON schema from them
3. **Return serializable data** — Dicts, lists, primitives (not Pydantic models)
4. **Use `logfire.span`** — For observability and debugging

## Combining Toolsets

Pass multiple toolsets to an agent alongside built-in tools:

```python
from agent_k.toolsets import kaggle_toolset, prepare_web_search

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[kaggle_toolset],
    builtin_tools=[prepare_web_search],
    instructions="You are the LOBBYIST agent...",
)
```

## Error Handling

Handle errors gracefully and return informative responses:

```python
@toolset.tool
async def get_competition(competition_id: str) -> dict[str, Any]:
    """Get competition details."""
    try:
        comp = await adapter.get_competition(competition_id)
        return comp.model_dump()
    except CompetitionNotFoundError:
        return {"error": f"Competition '{competition_id}' not found"}
    except Exception as e:
        return {"error": str(e)}
```

## Testing Toolsets

Toolsets are easy to test:

```python
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent_k.core.deps import KaggleDeps
from agent_k.toolsets.kaggle import kaggle_search_competitions
from agent_k.ui.ag_ui import EventEmitter

@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.search_competitions.return_value = [
        Competition(id="titanic", title="Titanic", ...)
    ]
    return adapter

async def test_kaggle_search(mock_adapter):
    deps = KaggleDeps(kaggle_adapter=mock_adapter, event_emitter=EventEmitter())
    ctx = SimpleNamespace(deps=deps)
    results = await kaggle_search_competitions(ctx, categories=["Featured"])
    assert results
```
