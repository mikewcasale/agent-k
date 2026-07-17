"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import scientist_agent, scientist_agent_instance

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


@dataclass
class _StubEntry:
    """Minimal leaderboard entry stub."""

    score: float
    rank: int = 1
    team_name: str = "stub_team"


@dataclass
class _StubCompetition:
    """Minimal competition stub carrying metric direction."""

    metric_direction: str = "maximize"


@dataclass
class _StubDeps:
    """Minimal Scientist deps stub bypassing platform/HTTP wiring."""

    competition: _StubCompetition
    leaderboard: list[_StubEntry] = field(default_factory=list)

    async def refresh_leaderboard(self) -> None:  # pragma: no cover - not invoked when refresh=False
        """No-op refresh — tests use pre-populated leaderboards."""
        return None


@dataclass
class _StubCtx:
    """Minimal ctx stub exposing only ``.deps`` used by ``analyze_leaderboard``."""

    deps: _StubDeps


class TestAnalyzeLeaderboardDirection:
    """Tests for direction-aware ``top_score`` reporting."""

    async def test_top_score_uses_max_for_maximize(self) -> None:
        """Higher-is-better metrics report the highest score as the top."""
        deps = _StubDeps(
            competition=_StubCompetition(metric_direction="maximize"),
            leaderboard=[_StubEntry(score=v) for v in (0.9, 0.8, 0.7, 0.6)],
        )
        result: dict[str, Any] = await scientist_agent_instance.analyze_leaderboard(_StubCtx(deps=deps), refresh=False)

        assert result["top_score"] == pytest.approx(0.9)

    async def test_top_score_uses_min_for_minimize(self) -> None:
        """Lower-is-better metrics report the smallest score as the top."""
        deps = _StubDeps(
            competition=_StubCompetition(metric_direction="minimize"),
            leaderboard=[_StubEntry(score=v) for v in (0.15, 0.20, 0.30, 0.45)],
        )
        result: dict[str, Any] = await scientist_agent_instance.analyze_leaderboard(_StubCtx(deps=deps), refresh=False)

        assert result["top_score"] == pytest.approx(0.15)
