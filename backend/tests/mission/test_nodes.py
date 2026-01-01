"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from unittest.mock import MagicMock

import pytest

from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _evaluate_metric,
)

__all__ = ()

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


class TestEvaluateMetric:
    """Tests for metric evaluation helpers."""

    def test_rmsle_ignores_negative_values(self) -> None:
        """RMSLE should ignore negative targets in the denominator."""
        score = _evaluate_metric(EvaluationMetric.RMSLE, [1.0, -1.0], prediction=0.0)
        assert score == pytest.approx(math.log1p(1.0))
