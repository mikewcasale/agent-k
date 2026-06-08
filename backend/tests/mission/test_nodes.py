"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math

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

    @pytest.mark.parametrize(
        ("metric", "expected_scorer"),
        [(EvaluationMetric.MAP, "average_precision_score"), (EvaluationMetric.NDCG, "ndcg_score")],
    )
    def test_generate_prototype_ranking_metric_uses_classifier_and_proba(
        self, metric: EvaluationMetric, expected_scorer: str
    ) -> None:
        """Ranking-metric prototypes should pick a classifier head and produce probabilities."""
        from datetime import UTC, datetime

        from agent_k.core.models import Competition, CompetitionType

        competition = Competition(
            id="ranking-comp",
            title="Ranking Competition",
            description=None,
            competition_type=CompetitionType.FEATURED,
            metric=metric,
            metric_direction="maximize",
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            prize_pool=None,
            max_team_size=1,
            max_daily_submissions=5,
            tags=frozenset({"tabular"}),
            url=None,
        )

        node = PrototypeNode()
        prototype = node._generate_prototype(
            competition, object(), target_columns=["target"], train_target_columns=["target"], id_column="id"
        )

        assert "IS_CLASSIFICATION = True" in prototype
        assert "USES_PROBA = True" in prototype
        assert expected_scorer in prototype
        assert "RandomForestRegressor" not in prototype.split("base_model")[0] or "RandomForestClassifier" in prototype


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

    @pytest.mark.parametrize("metric", [EvaluationMetric.MAP, EvaluationMetric.NDCG])
    def test_ranking_metric_constant_predictor_returns_positive_rate(self, metric: EvaluationMetric) -> None:
        """Ranking metrics with a constant predictor should baseline at the positive rate."""
        values = [0.0, 1.0, 1.0, 0.0, 1.0]
        score = _evaluate_metric(metric, values, prediction=0.5)
        assert score == pytest.approx(3 / 5)

    @pytest.mark.parametrize("metric", [EvaluationMetric.MAP, EvaluationMetric.NDCG])
    def test_ranking_metric_empty_values(self, metric: EvaluationMetric) -> None:
        """Ranking metrics should return 0.0 for empty inputs."""
        assert _evaluate_metric(metric, [], prediction=0.5) == 0.0
