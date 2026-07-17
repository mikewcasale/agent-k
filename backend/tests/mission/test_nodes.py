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

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _StubEntry:
    """Minimal leaderboard entry stub for analysis unit tests."""

    score: float


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


class TestBuildLeaderboardAnalysis:
    """Tests for ``_build_leaderboard_analysis`` direction handling."""

    def test_returns_none_when_no_scores(self) -> None:
        """Analysis is unavailable when every entry lacks a score."""
        assert _build_leaderboard_analysis([], 0.1, "maximize") is None

    def test_top_score_for_maximize_is_highest(self) -> None:
        """For a maximize metric the top of the leaderboard is the largest score."""
        entries = [_StubEntry(score=v) for v in (0.90, 0.75, 0.60, 0.55, 0.40)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.4, metric_direction="maximize")

        assert analysis is not None
        assert analysis.top_score == pytest.approx(0.90)
        assert analysis.target_score == pytest.approx(0.75)

    def test_top_score_for_minimize_is_lowest(self) -> None:
        """For a minimize metric the top of the leaderboard is the smallest score."""
        entries = [_StubEntry(score=v) for v in (0.40, 0.55, 0.60, 0.75, 0.90)]

        analysis = _build_leaderboard_analysis(entries, target_percentile=0.4, metric_direction="minimize")

        assert analysis is not None
        assert analysis.top_score == pytest.approx(0.40)
        assert analysis.target_score == pytest.approx(0.55)

    def test_top_score_matches_target_for_top_percentile(self) -> None:
        """A target percentile of 0.0 (top rank) matches the best score."""
        entries = [_StubEntry(score=v) for v in (0.1, 0.2, 0.3, 0.4)]

        maximize = _build_leaderboard_analysis(entries, target_percentile=0.0, metric_direction="maximize")
        minimize = _build_leaderboard_analysis(entries, target_percentile=0.0, metric_direction="minimize")

        assert maximize is not None
        assert minimize is not None
        assert maximize.top_score == pytest.approx(maximize.target_score)
        assert minimize.top_score == pytest.approx(minimize.target_score)
