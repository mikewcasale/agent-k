"""Tests for the SCIENTIST research agent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_k.agents.scientist import ScientistAgent

pytestmark = pytest.mark.anyio


class TestScientistAgent:
    """Tests for the ScientistAgent class."""
    
    def test_agent_initialization(self) -> None:
        """Agent should initialize with devstral model."""
        agent = ScientistAgent(model='devstral:local')
        assert agent is not None
    
    def test_agent_initialization_with_model(self) -> None:
        """Agent should accept custom model."""
        agent = ScientistAgent(model='devstral:local')
        assert agent is not None

