"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import scientist_agent, scientist_agent_instance
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry

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


def _make_competition(*, metric: EvaluationMetric, metric_direction: str) -> Competition:
    return Competition(
        id="demo",
        title="Demo competition",
        competition_type=CompetitionType.PLAYGROUND,
        metric=metric,
        metric_direction=metric_direction,
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _make_entries(scores: list[float]) -> list[LeaderboardEntry]:
    return [LeaderboardEntry(rank=i, team_name=f"team-{i}", score=score) for i, score in enumerate(scores, start=1)]


class TestAnalyzeLeaderboard:
    """``analyze_leaderboard`` should respect the competition metric direction."""

    @staticmethod
    def _ctx(deps: Any) -> Any:
        return SimpleNamespace(deps=deps)

    async def test_minimize_metric_returns_smallest_top_score(self) -> None:
        """For RMSE/log_loss-style metrics the best score is the minimum."""
        agent = scientist_agent_instance
        competition = _make_competition(metric=EvaluationMetric.RMSE, metric_direction="minimize")
        deps = SimpleNamespace(competition=competition, leaderboard=_make_entries([0.10, 0.25, 0.40]))

        result = await agent.analyze_leaderboard(cast("Any", self._ctx(deps)), refresh=False)

        assert result["metric_direction"] == "minimize"
        assert result["top_score"] == pytest.approx(0.10)
        assert result["median_score"] == pytest.approx(0.25)
        assert [team["rank"] for team in result["top_teams"]] == [1, 2, 3]

    async def test_maximize_metric_returns_largest_top_score(self) -> None:
        agent = scientist_agent_instance
        competition = _make_competition(metric=EvaluationMetric.AUC, metric_direction="maximize")
        deps = SimpleNamespace(competition=competition, leaderboard=_make_entries([0.95, 0.90, 0.80]))

        result = await agent.analyze_leaderboard(cast("Any", self._ctx(deps)), refresh=False)

        assert result["metric_direction"] == "maximize"
        assert result["top_score"] == pytest.approx(0.95)
        assert result["median_score"] == pytest.approx(0.90)

    async def test_median_handles_even_length_lists(self) -> None:
        """Even-length lists must average the two central values, not pick the upper one."""
        agent = scientist_agent_instance
        competition = _make_competition(metric=EvaluationMetric.AUC, metric_direction="maximize")
        deps = SimpleNamespace(competition=competition, leaderboard=_make_entries([0.1, 0.2, 0.3, 0.4]))

        result = await agent.analyze_leaderboard(cast("Any", self._ctx(deps)), refresh=False)

        assert result["median_score"] == pytest.approx(0.25)

    async def test_empty_leaderboard_returns_error(self) -> None:
        agent = scientist_agent_instance
        competition = _make_competition(metric=EvaluationMetric.RMSE, metric_direction="minimize")
        deps = SimpleNamespace(competition=competition, leaderboard=[])

        result = await agent.analyze_leaderboard(cast("Any", self._ctx(deps)), refresh=False)

        assert result == {"error": "No leaderboard data available"}
