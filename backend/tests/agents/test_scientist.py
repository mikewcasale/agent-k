"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import LeaderboardAnalysis, ResearchReport, scientist_agent

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


class TestLeaderboardAnalysisFinite:
    """LeaderboardAnalysis must reject non-finite scores from free models."""

    def _payload(self, **overrides: float) -> dict[str, object]:
        payload: dict[str, object] = {
            "top_score": 0.95,
            "median_score": 0.80,
            "score_distribution": "narrow",
            "common_approaches": ["gbdt"],
            "improvement_opportunities": ["feature_eng"],
        }
        payload.update(overrides)
        return payload

    def test_accepts_finite_scores(self) -> None:
        analysis = LeaderboardAnalysis(**self._payload())
        assert analysis.top_score == 0.95
        assert analysis.median_score == 0.80

    def test_rejects_nan_top_score(self) -> None:
        with pytest.raises(ValidationError):
            LeaderboardAnalysis(**self._payload(top_score=math.nan))

    def test_rejects_inf_top_score(self) -> None:
        with pytest.raises(ValidationError):
            LeaderboardAnalysis(**self._payload(top_score=math.inf))

    def test_rejects_nan_median_score(self) -> None:
        with pytest.raises(ValidationError):
            LeaderboardAnalysis(**self._payload(median_score=math.nan))

    def test_rejects_inf_median_score(self) -> None:
        with pytest.raises(ValidationError):
            LeaderboardAnalysis(**self._payload(median_score=-math.inf))


class TestResearchReportFinite:
    """ResearchReport must reject non-finite estimated baseline scores."""

    def test_accepts_finite_estimate(self) -> None:
        report = ResearchReport(competition_id="comp", estimated_baseline_score=0.5)
        assert report.estimated_baseline_score == 0.5

    def test_accepts_none_estimate(self) -> None:
        report = ResearchReport(competition_id="comp")
        assert report.estimated_baseline_score is None

    def test_rejects_nan_estimate(self) -> None:
        with pytest.raises(ValidationError):
            ResearchReport(competition_id="comp", estimated_baseline_score=math.nan)

    def test_rejects_inf_estimate(self) -> None:
        with pytest.raises(ValidationError):
            ResearchReport(competition_id="comp", estimated_baseline_score=math.inf)
