"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import _apply_difficulty_multiplier, scientist_agent

__all__ = ()

pytestmark = pytest.mark.anyio


class TestScientistAgentSingleton:
    """Tests for the Scientist agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent("scientist") is scientist_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(scientist_agent, Agent)
        assert scientist_agent.name == "scientist"


class TestApplyDifficultyMultiplier:
    """Tests for the difficulty-aware baseline estimator."""

    _MAXIMIZE_SCORES: list[float] = [0.60, 0.70, 0.80, 0.90, 0.95]  # median = 0.80
    _MINIMIZE_SCORES: list[float] = [0.05, 0.10, 0.20, 0.30, 0.50]  # median = 0.20

    def test_empty_scores_returns_zero(self) -> None:
        """No scores → 0.0 baseline for either direction."""
        assert _apply_difficulty_multiplier([], competition_difficulty="medium", metric_direction="maximize") == 0.0
        assert _apply_difficulty_multiplier([], competition_difficulty="medium", metric_direction="minimize") == 0.0

    def test_maximize_shrinks_median(self) -> None:
        """Maximize baselines sit below the leaderboard median."""
        value = _apply_difficulty_multiplier(
            self._MAXIMIZE_SCORES, competition_difficulty="medium", metric_direction="maximize"
        )
        assert value == pytest.approx(0.80 * 0.85)
        assert value < 0.80

    def test_minimize_expands_median(self) -> None:
        """Minimize baselines sit *above* the leaderboard median (worse = higher error)."""
        value = _apply_difficulty_multiplier(
            self._MINIMIZE_SCORES, competition_difficulty="medium", metric_direction="minimize"
        )
        assert value == pytest.approx(0.20 / 0.85)
        assert value > 0.20

    @pytest.mark.parametrize(
        ("difficulty", "multiplier"), [("easy", 0.95), ("medium", 0.85), ("hard", 0.70), ("unknown", 0.80)]
    )
    def test_difficulty_labels_and_default(self, difficulty: str, multiplier: float) -> None:
        """Known labels use their multiplier; unknown labels fall back to 0.80."""
        maximize = _apply_difficulty_multiplier(
            self._MAXIMIZE_SCORES, competition_difficulty=difficulty, metric_direction="maximize"
        )
        minimize = _apply_difficulty_multiplier(
            self._MINIMIZE_SCORES, competition_difficulty=difficulty, metric_direction="minimize"
        )
        assert maximize == pytest.approx(0.80 * multiplier)
        assert minimize == pytest.approx(0.20 / multiplier)

    def test_minimize_non_positive_median_falls_back(self) -> None:
        """Non-positive medians (e.g. maximized R^2 leaderboards misrouted as minimize) fall back to multiplication."""
        scores = [-0.5, -0.2, 0.0, 0.1, 0.2]  # median = 0.0
        value = _apply_difficulty_multiplier(scores, competition_difficulty="medium", metric_direction="minimize")
        assert value == pytest.approx(0.0 * 0.85)
