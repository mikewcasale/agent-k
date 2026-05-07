"""Tests for the LOBBYIST discovery agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent

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


class TestScoreCompetitionFitDomainMatching:
    """Regression tests for ``score_competition_fit`` domain matching.

    The previous implementation joined ``competition.tags`` into a single
    string and used substring matching, so short domain labels like ``"ai"``
    matched unrelated tags like ``"audio"``. These tests pin the corrected
    token-aware behavior through the public scoring tool.
    """

    @staticmethod
    def _ctx(competition: Competition) -> AsyncMock:
        deps = LobbyistDeps(
            http_client=MagicMock(),
            platform_adapter=AsyncMock(),
            event_emitter=MagicMock(),
            search_cache={competition.id: competition},
        )
        ctx = MagicMock()
        ctx.deps = deps
        return ctx

    @staticmethod
    def _competition(*, tags: tuple[str, ...], title: str = "Sample") -> Competition:
        return Competition(
            id="comp-1",
            title=title,
            competition_type=CompetitionType.FEATURED,
            metric=EvaluationMetric.ACCURACY,
            deadline=datetime(2030, 1, 1, tzinfo=UTC),
            tags=frozenset(tags),
        )

    @pytest.mark.anyio
    async def test_short_domain_label_does_not_substring_match(self) -> None:
        """``"ai"`` must not match a competition tagged only ``"audio"``."""
        comp = self._competition(tags=("audio",))
        ctx = self._ctx(comp)

        result = await lobbyist_agent_instance.score_competition_fit(
            ctx, competition_id=comp.id, target_domains=["ai"], min_days_remaining=0, target_percentile=10.0
        )
        assert "matches_domain" not in result["reasons"]

    @pytest.mark.anyio
    async def test_exact_tag_match_scores_domain(self) -> None:
        """A direct tag match still earns the ``matches_domain`` bonus."""
        comp = self._competition(tags=("tabular",))
        ctx = self._ctx(comp)

        result = await lobbyist_agent_instance.score_competition_fit(
            ctx, competition_id=comp.id, target_domains=["tabular"], min_days_remaining=0, target_percentile=10.0
        )
        assert "matches_domain" in result["reasons"]

    @pytest.mark.anyio
    async def test_keyword_expansion_matches_canonical_domain(self) -> None:
        """``computer_vision`` expands to keywords that match a CV-tagged comp."""
        comp = self._competition(tags=("computer vision",))
        ctx = self._ctx(comp)

        result = await lobbyist_agent_instance.score_competition_fit(
            ctx,
            competition_id=comp.id,
            target_domains=["computer_vision"],
            min_days_remaining=0,
            target_percentile=10.0,
        )
        assert "matches_domain" in result["reasons"]
