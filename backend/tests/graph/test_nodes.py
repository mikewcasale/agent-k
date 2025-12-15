"""Tests for the graph nodes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_k.graph.nodes import (
    DiscoveryNode,
    ResearchNode,
    PrototypeNode,
    EvolutionNode,
    SubmissionNode,
)

pytestmark = pytest.mark.anyio


class TestDiscoveryNode:
    """Tests for the DiscoveryNode."""
    
    def test_node_creation(self) -> None:
        """Node should be creatable with mock agent."""
        mock_agent = MagicMock()
        node = DiscoveryNode(lobbyist_agent=mock_agent)
        assert node is not None
        assert node.lobbyist_agent is mock_agent


class TestResearchNode:
    """Tests for the ResearchNode."""
    
    def test_node_creation(self) -> None:
        """Node should be creatable with mock agent."""
        mock_agent = MagicMock()
        node = ResearchNode(scientist_agent=mock_agent)
        assert node is not None
        assert node.scientist_agent is mock_agent


class TestPrototypeNode:
    """Tests for the PrototypeNode."""
    
    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = PrototypeNode()
        assert node is not None


class TestEvolutionNode:
    """Tests for the EvolutionNode."""
    
    def test_node_creation(self) -> None:
        """Node should be creatable with mock agent."""
        mock_agent = MagicMock()
        node = EvolutionNode(evolver_agent=mock_agent)
        assert node is not None
        assert node.evolver_agent is mock_agent


class TestSubmissionNode:
    """Tests for the SubmissionNode."""
    
    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = SubmissionNode()
        assert node is not None

