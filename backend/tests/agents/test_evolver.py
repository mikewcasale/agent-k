"""Tests for the EVOLVER optimization agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent

__all__ = ()

try:
    from agent_k.agents.evolver import evolver_agent
except TypeError as exc:
    if 'MCPServerTool' in str(exc):
        pytest.skip(f'MCPServerTool API issue: {exc}', allow_module_level=True)
    raise

pytestmark = pytest.mark.anyio


class TestEvolverAgentSingleton:
    """Tests for the Evolver agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent('evolver') is evolver_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(evolver_agent, Agent)
        assert evolver_agent.name == 'evolver'
