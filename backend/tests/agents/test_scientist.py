"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import scientist_agent

__all__ = ()

pytestmark = pytest.mark.anyio


class TestScientistAgentSingleton:
    """Tests for the Scientist agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent('scientist') is scientist_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(scientist_agent, Agent)
        assert scientist_agent.name == 'scientist'
