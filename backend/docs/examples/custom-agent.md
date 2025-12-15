# Creating a Custom Agent

This guide shows how to create a new agent for AGENT-K, following the established patterns.

## Agent Structure

Each agent consists of:

1. **agent.py** — Factory function and agent class
2. **prompts.py** — System instructions
3. **tools.py** — Custom tools (optional)
4. **__init__.py** — Module exports

## Step 1: Create Directory

```bash
mkdir -p backend/agent_k/agents/my_agent
touch backend/agent_k/agents/my_agent/{__init__,agent,prompts,tools}.py
```

## Step 2: Define Dependencies

In `agent.py`, define what your agent needs:

```python
"""My custom agent."""
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from agent_k.core.protocols import EventEmitter, PlatformAdapter
from agent_k.infra.models import get_model, DEFAULT_MODEL


@dataclass
class MyAgentDeps:
    """Dependencies for my agent."""
    
    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    cache: dict[str, Any] = field(default_factory=dict)
```

## Step 3: Define Output Model

Define the structured output your agent produces:

```python
class MyAgentResult(BaseModel):
    """Output from my agent."""
    
    findings: list[str]
    confidence: float
    recommendations: list[str]
    metadata: dict[str, Any] = {}
```

## Step 4: Write Instructions

In `prompts.py`:

```python
"""System prompts for my agent."""


def get_my_agent_instructions() -> str:
    """Get the system instructions for my agent."""
    return """You are MY_AGENT in the AGENT-K system.

Your role is to [describe purpose].

AVAILABLE TOOLS:
- tool_1: Description of tool 1
- tool_2: Description of tool 2
- memory_store: Save findings for other agents
- memory_retrieve: Get data from other agents

WORKFLOW:
1. [First step]
2. [Second step]
3. [Store findings for next agent]

OUTPUT:
Return a MyAgentResult with:
- findings: List of key findings
- confidence: Confidence score 0-1
- recommendations: Actionable recommendations
"""
```

## Step 5: Create Factory Function

In `agent.py`:

```python
from agent_k.toolsets import create_memory_toolset, create_search_toolset
from .prompts import get_my_agent_instructions


def create_my_agent(
    model: str = DEFAULT_MODEL,
    toolsets: list | None = None,
) -> Agent[MyAgentDeps, MyAgentResult]:
    """Create and configure my agent.
    
    Args:
        model: Model specification (e.g., 'anthropic:claude-3-haiku-20240307')
        toolsets: Optional list of toolsets to use
    
    Returns:
        Configured pydantic-ai Agent
    """
    resolved_model = get_model(model)
    
    # Default toolsets
    if toolsets is None:
        toolsets = [
            create_search_toolset(),
            create_memory_toolset(),
        ]
    
    agent = Agent(
        resolved_model,
        deps_type=MyAgentDeps,
        output_type=MyAgentResult,
        instructions=get_my_agent_instructions(),
        toolsets=toolsets,
        retries=2,
        name='my_agent',
    )
    
    # Add custom tools
    @agent.tool
    async def custom_tool(
        ctx: RunContext[MyAgentDeps],
        param: str,
    ) -> dict[str, Any]:
        """Custom tool description for the LLM.
        
        Args:
            param: Description of parameter.
        
        Returns:
            Result of the operation.
        """
        # Use dependencies
        result = await ctx.deps.platform_adapter.do_something(param)
        
        # Cache if needed
        ctx.deps.cache[param] = result
        
        return {"result": result}
    
    return agent
```

## Step 6: Create Agent Class (Optional)

For more complex agents, wrap in a class:

```python
class MyAgent:
    """High-level interface for my agent."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        deps: MyAgentDeps | None = None,
    ):
        self._agent = create_my_agent(model)
        self._deps = deps
    
    async def run(self, prompt: str) -> MyAgentResult:
        """Run the agent with the given prompt.
        
        Args:
            prompt: Task description for the agent
        
        Returns:
            Agent's structured result
        """
        if self._deps is None:
            raise ValueError("Dependencies not configured")
        
        result = await self._agent.run(prompt, deps=self._deps)
        return result.data
    
    @classmethod
    async def create(cls, model: str = DEFAULT_MODEL) -> "MyAgent":
        """Factory method with default dependencies."""
        async with httpx.AsyncClient() as http:
            deps = MyAgentDeps(
                http_client=http,
                platform_adapter=DefaultAdapter(),
                event_emitter=NoOpEmitter(),
            )
            return cls(model=model, deps=deps)
```

## Step 7: Export from __init__.py

```python
"""My custom agent."""
from .agent import (
    MyAgent,
    MyAgentDeps,
    MyAgentResult,
    create_my_agent,
)
from .prompts import get_my_agent_instructions

__all__ = [
    "MyAgent",
    "MyAgentDeps",
    "MyAgentResult",
    "create_my_agent",
    "get_my_agent_instructions",
]
```

## Step 8: Register in Parent __init__.py

Add to `backend/agent_k/agents/__init__.py`:

```python
from .my_agent import (
    MyAgent,
    MyAgentDeps,
    MyAgentResult,
    create_my_agent,
)

__all__ = [
    # ... existing exports ...
    "MyAgent",
    "MyAgentDeps",
    "MyAgentResult",
    "create_my_agent",
]
```

## Step 9: Test Your Agent

Create a test file:

```python
# tests/agents/test_my_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_k.agents.my_agent import create_my_agent, MyAgentDeps


@pytest.fixture
def mock_deps():
    return MyAgentDeps(
        http_client=AsyncMock(),
        platform_adapter=AsyncMock(),
        event_emitter=MagicMock(),
    )


async def test_my_agent_creation():
    """Test that agent can be created."""
    agent = create_my_agent('anthropic:claude-3-haiku-20240307')
    assert agent is not None
    assert agent.name == 'my_agent'


async def test_my_agent_tools(mock_deps):
    """Test that custom tools are available."""
    agent = create_my_agent()
    tools = [t.name for t in agent.tools]
    assert 'custom_tool' in tools
```

Run tests:

```bash
uv run pytest tests/agents/test_my_agent.py -v
```

## Step 10: Use Your Agent

```python
import asyncio
from agent_k.agents.my_agent import MyAgent

async def main():
    agent = await MyAgent.create(model='anthropic:claude-3-haiku-20240307')
    
    result = await agent.run(
        prompt="Perform your task..."
    )
    
    print(f"Findings: {result.findings}")
    print(f"Confidence: {result.confidence}")
    print(f"Recommendations: {result.recommendations}")

asyncio.run(main())
```

## Best Practices

### 1. Use Type Hints

```python
def create_my_agent(
    model: str = DEFAULT_MODEL,
) -> Agent[MyAgentDeps, MyAgentResult]:
    ...
```

### 2. Document Tools

The LLM uses docstrings to understand when to call tools:

```python
@agent.tool
async def my_tool(ctx: RunContext[Deps], param: str) -> dict:
    """Clear description of what this tool does.
    
    Use this when you need to [specific use case].
    
    Args:
        param: What this parameter controls.
    
    Returns:
        Description of the return value.
    """
```

### 3. Handle Errors Gracefully

```python
@agent.tool
async def risky_tool(ctx: RunContext[Deps]) -> dict:
    try:
        result = await ctx.deps.something_risky()
        return {"success": True, "data": result}
    except SomeError as e:
        return {"success": False, "error": str(e)}
```

### 4. Use Logfire for Observability

```python
import logfire

@agent.tool
async def my_tool(ctx: RunContext[Deps], query: str) -> dict:
    with logfire.span('my_tool', query=query):
        result = await do_something(query)
        logfire.info('Tool completed', result_size=len(result))
        return result
```

### 5. Cache Expensive Operations

```python
@agent.tool
async def expensive_search(ctx: RunContext[Deps], query: str) -> list:
    if query in ctx.deps.cache:
        return ctx.deps.cache[query]
    
    result = await ctx.deps.http_client.get(f"/search?q={query}")
    ctx.deps.cache[query] = result.json()
    
    return ctx.deps.cache[query]
```

## Next Steps

- [Toolsets](../concepts/toolsets.md) — Create custom toolsets
- [State Machine](../concepts/graph.md) — Integrate with LYCURGUS
- [API Reference](../api/agents/lycurgus.md) — See existing implementations

