# Creating a Custom Agent

This guide shows how to create a new agent following the current single-file pattern.

## Step 1: Create the Module

Create `backend/agent_k/agents/strategist.py` (single lowercase word, no underscores).

## Step 2: Implement the Agent

```python
"""Strategist agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""
from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_k.agents import register_agent
from agent_k.agents.base import MemoryMixin, universal_tool_preparation
from agent_k.infra.providers import get_model

if TYPE_CHECKING:
    import httpx

__all__ = (
    "StrategistDeps",
    "StrategistResult",
    "StrategistSettings",
    "strategist_agent",
)

SCHEMA_VERSION: Final[str] = "1.0.0"


class StrategistSettings(BaseSettings):
    """Configuration for the Strategist agent."""

    model_config = SettingsConfigDict(env_prefix="STRATEGIST_", env_file=".env", extra="ignore")
    model: str = Field(default="anthropic:claude-sonnet-4-5")


@dataclass
class StrategistDeps:
    """Dependencies for the Strategist agent."""

    http_client: httpx.AsyncClient


class StrategistResult(BaseModel):
    """Structured output for Strategist."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = Field(default=SCHEMA_VERSION)
    summary: str


class StrategistAgent(MemoryMixin):
    """Strategist agent wrapper."""

    def __init__(self, settings: StrategistSettings | None = None) -> None:
        self._settings = settings or StrategistSettings()
        self._toolset: FunctionToolset[StrategistDeps] = FunctionToolset(id="strategist")
        self._register_tools()
        self._agent = self._create_agent()
        register_agent("strategist", self._agent)
        self._setup_memory()

    def _create_agent(self) -> Agent[StrategistDeps, StrategistResult]:
        return Agent(
            model=get_model(self._settings.model),
            deps_type=StrategistDeps,
            output_type=StrategistResult,
            instructions="You are the Strategist agent.",
            toolsets=[self._toolset],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

    def _register_tools(self) -> None:
        self._toolset.tool(self.plan_strategy)

    async def plan_strategy(self, *_: object) -> str:
        return "Strategy drafted."


strategist_agent_instance = StrategistAgent()
strategist_agent = strategist_agent_instance.agent
```

## Step 3: Export the Agent

Add the export to `backend/agent_k/agents/__init__.py`.

## Step 4: Document the Agent

Add a new doc in `backend/docs/agents/strategist.md` describing its role and outputs.
