"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from datetime import UTC, datetime

import pytest

from agent_k.core.models import EvaluationMetric, LeaderboardEntry
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _build_leaderboard_analysis,
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


def _make_entry(rank: int, score: float) -> LeaderboardEntry:
    return LeaderboardEntry(
        rank=rank, team_name=f"team-{rank}", score=score, last_submission=datetime(2026, 1, 1, tzinfo=UTC)
    )


class TestBuildLeaderboardAnalysis:
    """Tests for ``_build_leaderboard_analysis`` direction-aware summary."""

    def test_returns_none_when_no_scores(self) -> None:
        assert _build_leaderboard_analysis([], target_percentile=0.1, metric_direction="maximize") is None

    def test_maximize_top_score_is_largest(self) -> None:
        entries = [_make_entry(1, 0.95), _make_entry(2, 0.90), _make_entry(3, 0.80)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="maximize")

        assert analysis is not None
        assert analysis.top_score == pytest.approx(0.95)
        assert analysis.median_score == pytest.approx(0.90)

    def test_minimize_top_score_is_smallest(self) -> None:
        """For RMSE-style metrics, the best score is the minimum."""
        entries = [_make_entry(1, 0.10), _make_entry(2, 0.25), _make_entry(3, 0.40)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="minimize")

        assert analysis is not None
        assert analysis.top_score == pytest.approx(0.10)
        assert analysis.median_score == pytest.approx(0.25)

    def test_median_uses_statistics_for_even_length(self) -> None:
        """Even-length lists should average the two central values, not pick the upper one."""
        entries = [_make_entry(i, score) for i, score in enumerate([0.1, 0.2, 0.3, 0.4], start=1)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="maximize")

        assert analysis is not None
        assert analysis.median_score == pytest.approx(0.25)
