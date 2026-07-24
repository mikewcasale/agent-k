"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from dataclasses import dataclass

import pytest

from agent_k.core.models import EvaluationMetric
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


@dataclass(frozen=True)
class _StubEntry:
    """Minimal leaderboard entry stub for analysis unit tests."""

    score: float


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


class TestBuildLeaderboardDistribution:
    """Tests for ``_build_leaderboard_analysis`` score_distribution direction handling."""

    def test_distribution_maximize_places_best_score_at_top_percentile(self) -> None:
        """For a maximize metric, percentile 1.0 must be the highest (best) score."""
        entries = [_StubEntry(score=v) for v in (0.10, 0.30, 0.50, 0.70, 0.90)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="maximize")

        assert analysis is not None
        distribution = {entry["percentile"]: entry["score"] for entry in analysis.score_distribution}
        assert distribution[1.0] == pytest.approx(0.90)
        assert distribution[0.0] == pytest.approx(0.10)
        assert distribution[0.5] == pytest.approx(0.50)

    def test_distribution_minimize_places_best_score_at_top_percentile(self) -> None:
        """For a minimize metric, percentile 1.0 must be the lowest (best) score.

        This is the regression: sampling the ascending sort ignored
        ``metric_direction``, so RMSE/MAE/RMSLE/log-loss competitions
        surfaced the worst score at the top-percentile slot and the best
        score at the bottom.
        """
        entries = [_StubEntry(score=v) for v in (0.10, 0.30, 0.50, 0.70, 0.90)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="minimize")

        assert analysis is not None
        distribution = {entry["percentile"]: entry["score"] for entry in analysis.score_distribution}
        assert distribution[1.0] == pytest.approx(0.10)
        assert distribution[0.0] == pytest.approx(0.90)
        assert distribution[0.5] == pytest.approx(0.50)

    def test_distribution_percentile_1_matches_top_ranked_score(self) -> None:
        """Percentile 1.0 must equal the top-ranked score under both directions."""
        entries = [_StubEntry(score=v) for v in (1.5, 2.5, 3.5, 4.5)]

        maximize = _build_leaderboard_analysis(entries, target_percentile=0.0, metric_direction="maximize")
        minimize = _build_leaderboard_analysis(entries, target_percentile=0.0, metric_direction="minimize")

        assert maximize is not None
        assert minimize is not None
        max_top = next(entry["score"] for entry in maximize.score_distribution if entry["percentile"] == 1.0)
        min_top = next(entry["score"] for entry in minimize.score_distribution if entry["percentile"] == 1.0)
        assert max_top == pytest.approx(4.5)
        assert min_top == pytest.approx(1.5)

    def test_distribution_is_monotonic_in_leaderboard_rank(self) -> None:
        """Distribution should be monotonic when read in leaderboard-rank order.

        Under either metric direction, walking percentile from 0.0 to 1.0
        traces the leaderboard from worst to best. For maximize that means
        scores should be non-decreasing; for minimize they should be
        non-increasing.
        """
        entries = [_StubEntry(score=v) for v in (5.0, 4.0, 3.0, 2.0, 1.0)]

        maximize = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="maximize")
        minimize = _build_leaderboard_analysis(entries, target_percentile=0.5, metric_direction="minimize")

        assert maximize is not None
        assert minimize is not None

        max_scores = [entry["score"] for entry in maximize.score_distribution]
        min_scores = [entry["score"] for entry in minimize.score_distribution]

        for prev, curr in zip(max_scores, max_scores[1:], strict=False):
            assert prev <= curr, "maximize distribution must be non-decreasing across percentiles"
        for prev, curr in zip(min_scores, min_scores[1:], strict=False):
            assert prev >= curr, "minimize distribution must be non-increasing across percentiles"

    def test_distribution_handles_single_entry(self) -> None:
        """Single-entry leaderboards should surface the same score at every percentile."""
        entries = [_StubEntry(score=0.42)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=1.0, metric_direction="minimize")

        assert analysis is not None
        for row in analysis.score_distribution:
            assert row["score"] == pytest.approx(0.42)
