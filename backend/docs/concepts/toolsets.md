# Toolsets

AGENT-K uses Pydantic-AI's `FunctionToolset` for custom tools and relies on built-in tools when a provider supports them. Built-in tools are preferred for web search, memory, and code execution.

## Built-in Tools

Use built-in tools for web search, URL fetching, memory, and code execution. AGENT-K exposes helpers for configuring them:

```python
from pydantic_ai import Agent
from agent_k.toolsets import prepare_web_search, prepare_web_fetch

agent = Agent(
    "anthropic:claude-3-haiku-20240307",
    builtin_tools=[prepare_web_search, prepare_web_fetch],
)
```

Memory (Anthropic only) requires a backend:

```python
from pydantic_ai import Agent
from agent_k.toolsets import create_memory_backend, prepare_memory_tool, register_memory_tool

memory_backend = create_memory_backend()
agent = Agent(
    "anthropic:claude-3-haiku-20240307",
    builtin_tools=[prepare_memory_tool],
)
register_memory_tool(agent, memory_backend)
```

Code execution is enabled only for supported providers:

```python
from agent_k.toolsets import prepare_code_execution_tool

agent = Agent(
    "anthropic:claude-3-haiku-20240307",
    builtin_tools=[prepare_code_execution_tool],
)
```

The Evolver also uses `MCPServerTool` to call Kaggle MCP endpoints for submissions.

## Available Toolsets

### Kaggle Toolset

Provides Kaggle API operations:

| Tool | Description |
|------|-------------|
| `kaggle_search_competitions` | Search for active competitions |
| `kaggle_get_competition` | Get competition details |
| `kaggle_get_leaderboard` | Get leaderboard entries |
| `kaggle_list_datasets` | List competition datasets |

Usage:

```python
from pydantic_ai import Agent
from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.deps import KaggleDeps
from agent_k.toolsets import kaggle_toolset
from agent_k.ui.ag_ui import EventEmitter

config = KaggleSettings(username="you", api_key="your-key")
adapter = KaggleAdapter(config)

deps = KaggleDeps(kaggle_adapter=adapter, event_emitter=EventEmitter())

agent = Agent(
    "anthropic:claude-3-haiku-20240307",
    deps_type=KaggleDeps,
    toolsets=[kaggle_toolset],
)

result = await agent.run("Find featured competitions", deps=deps)
```

### Code Toolset

`code_toolset` provides helper tools used by the Evolver for code mutation and evaluation workflows. It is composed with the Evolver's own toolset via `create_production_toolset`.

## Combining Toolsets

Use `create_production_toolset` to apply strict settings and approvals:

```python
from agent_k.toolsets import create_production_toolset, kaggle_toolset

toolset = create_production_toolset(
    [kaggle_toolset],
    prefix="kaggle",
)
```

## Error Handling

Toolsets return serializable data. Errors should return a structured dict or list so the model can respond gracefully.
