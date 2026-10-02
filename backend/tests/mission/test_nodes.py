"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math

import pytest
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError

from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _evaluate_metric,
    _is_rate_limit_error,
)

__all__ = ()

pytestmark = pytest.mark.anyio


class TestDiscoveryNode:
    """Tests for the DiscoveryNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = DiscoveryNode()
        assert node is not None


class TestResearchNode:
    """Tests for the ResearchNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = ResearchNode()
        assert node is not None


class TestPrototypeNode:
    """Tests for the PrototypeNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = PrototypeNode()
        assert node is not None


class TestEvolutionNode:
    """Tests for the EvolutionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = EvolutionNode()
        assert node is not None


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


class TestIsRateLimitError:
    """Tests for rate-limit classification of model failures."""

    def test_detects_rate_limited_status(self) -> None:
        """A 429 from a provider should be classified as rate limited."""
        error = ModelHTTPError(status_code=429, model_name="m", body=None)

        assert _is_rate_limit_error(error) is True

    def test_ignores_unrelated_error(self) -> None:
        """An unrelated failure should not be classified as rate limited."""
        assert _is_rate_limit_error(ValueError("bad column name")) is False

    def test_unwraps_fallback_exception_group(self) -> None:
        """A chain that exhausted every model on 429s should still read as rate limited."""
        group = FallbackExceptionGroup(
            "all models failed",
            [
                ModelHTTPError(status_code=429, model_name="primary", body=None),
                ModelHTTPError(status_code=429, model_name="backup", body=None),
            ],
        )

        assert _is_rate_limit_error(group) is True

    def test_fallback_group_without_rate_limits(self) -> None:
        """A chain that failed for non-rate-limit reasons should not be misread."""
        group = FallbackExceptionGroup(
            "all models failed", [ModelHTTPError(status_code=404, model_name="primary", body=None)]
        )

        assert _is_rate_limit_error(group) is False
