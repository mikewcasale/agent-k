"""Tests for the LOBBYIST discovery agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext

from agent_k.agents import get_agent
from agent_k.agents.lobbyist import DiscoveryResult, LobbyistDeps, lobbyist_agent, lobbyist_agent_instance
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric

__all__ = ()

if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.anyio


class TestLobbyistDeps:
    """Tests for the LobbyistDeps dependency container."""

    def test_deps_creation(
        self, mock_http_client: AsyncMock, mock_platform_adapter: AsyncMock, mock_event_emitter: MagicMock
    ) -> None:
        """Dependencies should be properly structured."""
        deps = LobbyistDeps(
            http_client=mock_http_client, platform_adapter=mock_platform_adapter, event_emitter=mock_event_emitter
        )

        assert deps.http_client is mock_http_client
        assert deps.platform_adapter is mock_platform_adapter
        assert deps.event_emitter is mock_event_emitter
        assert deps.search_cache == {}


class TestDiscoveryResult:
    """Tests for the DiscoveryResult output model."""

    def test_empty_result(self) -> None:
        """Empty result should have default values."""
        result = DiscoveryResult()

        assert result.competitions == []
        assert result.total_searched == 0
        assert result.filters_applied == []

    def test_result_with_competitions(self) -> None:
        """Result should accept competition list."""
        competitions = [
            Competition(
                id="titanic",
                title="Titanic",
                competition_type=CompetitionType.GETTING_STARTED,
                metric=EvaluationMetric.ACCURACY,
                deadline=datetime(2030, 1, 1, tzinfo=UTC),
            )
        ]

        result = DiscoveryResult(competitions=competitions, total_searched=10, filters_applied=["featured", "active"])

        assert len(result.competitions) == 1
        assert result.competitions[0].id == "titanic"
        assert result.total_searched == 10
        assert result.filters_applied == ["featured", "active"]

    def test_result_is_frozen(self) -> None:
        """Result should be immutable."""
        result = DiscoveryResult()

        with pytest.raises(ValidationError):
            result.total_searched = 100  # type: ignore


class TestLobbyistAgentSingleton:
    """Tests for the Lobbyist agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent("lobbyist") is lobbyist_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(lobbyist_agent, Agent)
        assert lobbyist_agent.name == "lobbyist"


def _competition(competition_id: str, **overrides: Any) -> Competition:
    """Build a competition with permissive defaults."""
    fields: dict[str, Any] = {
        "id": competition_id,
        "title": "Example",
        "competition_type": CompetitionType.FEATURED,
        "metric": EvaluationMetric.AUC,
        "deadline": datetime(2030, 1, 1, tzinfo=UTC),
        "tags": frozenset({"tabular"}),
    }
    fields.update(overrides)
    return Competition(**fields)


def _context(search_cache: dict[str, object]) -> RunContext[LobbyistDeps]:
    """Build a minimal run context exposing only the search cache."""
    deps = SimpleNamespace(search_cache=search_cache)
    return cast("RunContext[LobbyistDeps]", SimpleNamespace(deps=deps))


class TestScoreCompetitionFit:
    """Tests for the competition fit scoring tool."""

    async def test_enterable_competition_scores(self) -> None:
        """An enterable competition should accumulate fit score."""
        competition = _competition("open-comp", prize_pool=50000)
        ctx = _context({competition.id: competition})

        result = await lobbyist_agent_instance.score_competition_fit(
            ctx, competition_id="open-comp", target_domains=["tabular"], min_days_remaining=7, target_percentile=10.0
        )

        assert result["score"] > 0.0
        assert "matches_domain" in result["reasons"]

    async def test_kernels_only_competition_scores_zero(self) -> None:
        """A code competition should be rejected outright."""
        competition = _competition("code-comp", prize_pool=50000, kernels_submissions_only=True)
        ctx = _context({competition.id: competition})

        result = await lobbyist_agent_instance.score_competition_fit(
            ctx, competition_id="code-comp", target_domains=["tabular"], min_days_remaining=7, target_percentile=10.0
        )

        assert result["score"] == 0.0
        assert result["reasons"] == ["not_enterable", "kernels_submissions_only"]

    async def test_closed_entry_competition_scores_zero(self) -> None:
        """A competition past its new-entrant deadline should be rejected."""
        competition = _competition("closed-comp", new_entrant_deadline=datetime(2020, 1, 1, tzinfo=UTC))
        ctx = _context({competition.id: competition})

        result = await lobbyist_agent_instance.score_competition_fit(
            ctx, competition_id="closed-comp", target_domains=["tabular"], min_days_remaining=7, target_percentile=10.0
        )

        assert result["score"] == 0.0
        assert "new_entrant_deadline_passed" in result["reasons"]
