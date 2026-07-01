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
    _best_score,
    _build_leaderboard_analysis,
    _evaluate_metric,
)


@dataclass
class _LBEntry:
    """Lightweight leaderboard entry stand-in for tests."""

    rank: int
    score: float


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


class TestBestScore:
    """Tests for the ``_best_score`` metric-direction helper."""

    def test_maximize_returns_max(self) -> None:
        """Maximize competitions rank highest score as best."""
        assert _best_score([0.1, 0.9, 0.5], "maximize") == 0.9

    def test_minimize_returns_min(self) -> None:
        """Minimize competitions (RMSE, log_loss, ...) rank lowest score as best."""
        assert _best_score([1.4, 0.2, 0.8], "minimize") == 0.2

    def test_unknown_direction_defaults_to_min(self) -> None:
        """Only the explicit ``maximize`` string flips to max; everything else is min."""
        assert _best_score([1.4, 0.2, 0.8], "") == 0.2


class TestBuildLeaderboardAnalysis:
    """Tests for ``_build_leaderboard_analysis``."""

    def test_empty_entries_returns_none(self) -> None:
        """No parsable scores means no analysis."""
        assert _build_leaderboard_analysis([], 0.1, "maximize") is None

    def test_maximize_top_score_is_max(self) -> None:
        """For maximize metrics, top_score matches the highest observed score."""
        entries = [_LBEntry(rank=i, score=s) for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.5], start=1)]
        analysis = _build_leaderboard_analysis(entries, 0.4, "maximize")
        assert analysis is not None
        assert analysis.top_score == 0.9
        assert analysis.target_score == 0.8

    def test_minimize_top_score_is_min(self) -> None:
        """Regression on the RMSE-style bug: top_score must be min for minimize metrics."""
        entries = [_LBEntry(rank=i, score=s) for i, s in enumerate([0.10, 0.25, 0.40, 0.55, 0.70], start=1)]
        analysis = _build_leaderboard_analysis(entries, 0.4, "minimize")
        assert analysis is not None
        assert analysis.top_score == 0.10
        assert analysis.target_score == 0.25
