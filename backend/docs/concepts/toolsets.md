# Toolsets

AGENT-K uses Pydantic-AI's `FunctionToolset` pattern to provide tools that work with **any model provider**, including OpenAI-compatible endpoints like local LM Studio servers.

## Why FunctionToolset?

Unlike builtin tools (`WebSearchTool`, `MCPServerTool`) which are executed server-side by the model provider, FunctionToolsets are executed **client-side** by pydantic-ai. This means:

- ✅ Works with any model (Anthropic, OpenRouter, local Devstral)
- ✅ Full control over tool implementation
- ✅ Easy to test and mock
- ✅ Can access local resources and databases

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
from agent_k.adapters.kaggle import KaggleAdapter, KaggleConfig
from agent_k.toolsets import create_kaggle_toolset

# Create adapter
config = KaggleConfig(username="you", api_key="your-key")
adapter = KaggleAdapter(config)

# Create toolset
kaggle_toolset = create_kaggle_toolset(adapter)

# Use with agent
agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[kaggle_toolset],
)
```

### SearchToolset

Provides web and academic search:

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via DuckDuckGo |
| `search_papers` | Search academic papers |
| `search_kaggle` | Search kaggle.com discussions |

**Usage:**

```python
from agent_k.toolsets import create_search_toolset

search_toolset = create_search_toolset()

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[search_toolset],
)
```

### MemoryToolset

Provides persistent key-value storage for cross-agent communication:

| Tool | Description |
|------|-------------|
| `memory_store` | Store a value by key |
| `memory_retrieve` | Retrieve a value by key |
| `memory_search` | Search memory by pattern |

**Usage:**

```python
from pathlib import Path
from agent_k.toolsets import create_memory_toolset

memory_toolset = create_memory_toolset(
    storage_path=Path("mission_memory.json")
)

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[memory_toolset],
)
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

Pass multiple toolsets to an agent:

```python
from agent_k.toolsets import (
    create_kaggle_toolset,
    create_search_toolset,
    create_memory_toolset,
)

kaggle_toolset = create_kaggle_toolset(adapter)
search_toolset = create_search_toolset()
memory_toolset = create_memory_toolset(storage_path)

agent = Agent(
    'anthropic:claude-3-haiku-20240307',
    toolsets=[kaggle_toolset, search_toolset, memory_toolset],
    instructions="You are the LOBBYIST agent...",
)
```

## Tool Parameters

Tools can have various parameter types:

```python
@toolset.tool
async def search_competitions(
    categories: list[str] | None = None,      # Optional list
    keywords: list[str] | None = None,        # Optional list
    min_prize: int | None = None,             # Optional int
    active_only: bool = True,                 # Default value
) -> list[dict[str, Any]]:
    """Search Kaggle competitions.
    
    Args:
        categories: Filter by type (Featured, Research, etc.)
        keywords: Keywords to search for
        min_prize: Minimum prize pool in USD
        active_only: Only return active competitions
    """
    ...
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
from unittest.mock import AsyncMock

@pytest.fixture
def mock_adapter():
    adapter = AsyncMock()
    adapter.search_competitions.return_value = [
        Competition(id="titanic", title="Titanic", ...)
    ]
    return adapter

async def test_kaggle_search(mock_adapter):
    toolset = create_kaggle_toolset(mock_adapter)
    
    # Get the tool function
    search_fn = toolset.get_tool("kaggle_search_competitions")
    
    # Call it
    results = await search_fn(categories=["Featured"])
    
    assert len(results) == 1
    assert results[0]["id"] == "titanic"
```

## Next Steps

- [Model Configuration](models.md) — Supported model providers
- [Agents](agents.md) — How agents use toolsets
- [API Reference: Toolsets](../api/toolsets/kaggle.md) — Full API documentation

