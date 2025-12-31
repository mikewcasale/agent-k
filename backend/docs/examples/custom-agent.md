# Creating a Custom Agent

This guide shows how to create a new agent for AGENT-K, following the singleton pattern used throughout the codebase.

## Agent Structure

Each agent consists of:

1. **agent.py** — Agent singleton and registration
2. **config.py** — Pydantic Settings configuration
3. **deps.py** — Dependency container dataclass
4. **output.py** — Output models
5. **prompts.py** — System instructions
6. **tools.py** — Custom tools (optional)
7. **validators.py** — Output validators (optional)
8. **__init__.py** — Module exports

## Step 1: Create Directory

```bash
mkdir -p backend/agent_k/agents/my_agent
touch backend/agent_k/agents/my_agent/{__init__,agent,config,deps,output,prompts,tools,validators}.py
```

## Step 2: Define Settings

In `config.py`, define configuration for your agent:

```python
from typing import Final

from pydantic import Field
from pydantic_ai import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_k.core.constants import DEFAULT_MODEL

SCHEMA_VERSION: Final[str] = '1.0.0'


class MyAgentSettings(BaseSettings):
    """Configuration for my agent."""

    model_config = SettingsConfigDict(
        env_prefix='MY_AGENT_',
        env_file='.env',
        extra='ignore',
        validate_default=True,
    )

    model: str = Field(default=DEFAULT_MODEL, description='Model identifier')
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description='Sampling temperature')
    max_tokens: int = Field(default=2048, ge=1, description='Maximum tokens')
    tool_retries: int = Field(default=2, ge=0, description='Tool retry attempts')
    output_retries: int = Field(default=1, ge=0, description='Output validation retries')

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings for the configured model."""
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
```

## Step 3: Define Dependencies

In `deps.py`, define what your agent needs:

```python
"""Dependencies for my agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_k.core.protocols import PlatformAdapter
from agent_k.ui.ag_ui import EventEmitter


@dataclass
class MyAgentDeps:
    """Dependencies for my agent."""

    http_client: httpx.AsyncClient
    platform_adapter: PlatformAdapter
    event_emitter: EventEmitter
    cache: dict[str, Any] = field(default_factory=dict)
```

## Step 4: Define Output Model

In `output.py`, define the structured output your agent produces:

```python
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Final[str] = '1.0.0'


class MyAgentResult(BaseModel):
    """Output from my agent."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, description='Schema version')
    findings: list[str]
    confidence: float
    recommendations: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
```

## Step 5: Write Instructions

In `prompts.py`:

```python
"""System prompts for my agent."""
from __future__ import annotations

from typing import Final

MY_AGENT_SYSTEM_PROMPT: Final[str] = """You are MY_AGENT in the AGENT-K system.

Your role is to [describe purpose].

AVAILABLE TOOLS:
- tool_1: Description of tool 1
- tool_2: Description of tool 2
- memory: Built-in MemoryTool for shared notes (create/view)

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

## Step 6: Register Tools (Optional)

In `tools.py`:

```python
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from pydantic_ai import RunContext

from .deps import MyAgentDeps

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from .output import MyAgentResult


def register_tools(agent: 'Agent[MyAgentDeps, MyAgentResult]') -> None:
    """Register tools for my agent."""

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
        result = await ctx.deps.platform_adapter.do_something(param)
        ctx.deps.cache[param] = result
        return {"result": result}
```

## Step 7: Add Output Validators (Optional)

In `validators.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import RunContext

from .deps import MyAgentDeps
from .output import MyAgentResult

if TYPE_CHECKING:
    from pydantic_ai import Agent


def register_validators(agent: 'Agent[MyAgentDeps, MyAgentResult]') -> None:
    """Register output validators for my agent."""

    @agent.output_validator
    async def validate_confidence(
        output: MyAgentResult,
        ctx: RunContext[MyAgentDeps],
    ) -> MyAgentResult:
        if not 0.0 <= output.confidence <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        return output
```

## Step 8: Create Agent Singleton

In `agent.py`:

```python
from pydantic_ai import Agent

from agent_k.agents import register_agent
from agent_k.infra.providers import get_model
from agent_k.toolsets import (
    create_memory_backend,
    prepare_memory_tool,
    prepare_web_search,
    register_memory_tool,
)
from .config import MyAgentSettings
from .deps import MyAgentDeps
from .output import MyAgentResult
from .prompts import MY_AGENT_SYSTEM_PROMPT
from .tools import register_tools
from .validators import register_validators

settings = MyAgentSettings()

memory_backend = create_memory_backend()

my_agent: Agent[MyAgentDeps, MyAgentResult] = Agent(
    model=get_model(settings.model),
    deps_type=MyAgentDeps,
    output_type=MyAgentResult,
    instructions=MY_AGENT_SYSTEM_PROMPT,
    builtin_tools=[prepare_web_search, prepare_memory_tool],
    model_settings=settings.model_settings,
    retries=settings.tool_retries,
    output_retries=settings.output_retries,
    name='my_agent',
)

register_agent('my_agent', my_agent)
register_tools(my_agent)
register_validators(my_agent)
register_memory_tool(my_agent, memory_backend)
```

## Step 9: Export from __init__.py

```python
"""My custom agent."""
from .agent import my_agent
from .config import MyAgentSettings
from .deps import MyAgentDeps
from .output import MyAgentResult
from .prompts import MY_AGENT_SYSTEM_PROMPT

__all__ = [
    'my_agent',
    'MyAgentSettings',
    'MyAgentDeps',
    'MyAgentResult',
    'MY_AGENT_SYSTEM_PROMPT',
]
```

## Step 10: Register in Parent __init__.py

Add to `backend/agent_k/agents/__init__.py`:

```python
from .my_agent import MyAgentDeps, MyAgentResult, MyAgentSettings, my_agent

__all__ = [
    # ... existing exports ...
    'my_agent',
    'MyAgentDeps',
    'MyAgentResult',
    'MyAgentSettings',
]
```

## Step 11: Test Your Agent

Create a test file:

```python
# tests/agents/test_my_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_k.agents.my_agent import MyAgentDeps, my_agent


@pytest.fixture
def mock_deps():
    return MyAgentDeps(
        http_client=AsyncMock(),
        platform_adapter=AsyncMock(),
        event_emitter=MagicMock(),
    )


async def test_my_agent_creation():
    """Test that agent is available."""
    assert my_agent is not None
    assert my_agent.name == 'my_agent'


async def test_my_agent_tools(mock_deps):
    """Test that custom tools are available."""
    tools = [t.name for t in my_agent.tools]
    assert 'custom_tool' in tools
    assert mock_deps.cache == {}
```

Run tests:

```bash
uv run pytest tests/agents/test_my_agent.py -v
```

## Step 12: Use Your Agent

```python
import asyncio
import httpx

from agent_k.agents.my_agent import MyAgentDeps, my_agent
from agent_k.ui.ag_ui import EventEmitter


class DefaultAdapter:
    async def do_something(self, param: str) -> str:
        return f"processed:{param}"


async def main():
    async with httpx.AsyncClient() as http:
        deps = MyAgentDeps(
            http_client=http,
            platform_adapter=DefaultAdapter(),
            event_emitter=EventEmitter(),
        )

        result = await my_agent.run(
            prompt="Perform your task...",
            deps=deps,
        )

        output = result.output
        print(f"Findings: {output.findings}")
        print(f"Confidence: {output.confidence}")
        print(f"Recommendations: {output.recommendations}")

asyncio.run(main())
```

## Best Practices

### 1. Use Type Hints

```python
from pydantic_ai import Agent

my_agent: Agent[MyAgentDeps, MyAgentResult] = Agent(...)
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
