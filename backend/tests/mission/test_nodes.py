"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
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

    def test_generated_prototype_clips_rmsle_predictions(self) -> None:
        """Generated RMSLE prototypes must clip predictions to non-negative.

        sklearn.metrics.mean_squared_log_error raises when values are less than
        -1, and Kaggle rejects RMSLE submissions with negative predictions, so
        the generated prototype must clip both the validation score inputs and
        the test-set predictions.
        """
        competition = Competition(
            id="rmsle-test",
            title="RMSLE Test",
            competition_type=CompetitionType.PLAYGROUND,
            metric=EvaluationMetric.RMSLE,
            metric_direction="minimize",
            deadline=datetime(2099, 12, 31, tzinfo=UTC),
        )
        research = SimpleNamespace(strategy_recommendations=["lightgbm"], recommended_approaches=[])

        code = PrototypeNode()._generate_prototype(
            competition, research, target_columns=["target"], train_target_columns=["target"], id_column="id"
        )

        assert 'if METRIC_KEY == "rmsle":\n                preds = np.maximum' in code
        assert 'if METRIC_KEY == "rmsle":\n                test_preds = np.maximum' in code


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
