"""Tests for the LOBBYIST discovery agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.lobbyist import DiscoveryResult, LobbyistDeps, lobbyist_agent
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
                id='titanic',
                title='Titanic',
                competition_type=CompetitionType.GETTING_STARTED,
                metric=EvaluationMetric.ACCURACY,
                deadline=datetime(2030, 1, 1, tzinfo=UTC),
            )
        ]

        result = DiscoveryResult(competitions=competitions, total_searched=10, filters_applied=['featured', 'active'])

        assert len(result.competitions) == 1
        assert result.competitions[0].id == 'titanic'
        assert result.total_searched == 10
        assert result.filters_applied == ['featured', 'active']

    def test_result_is_frozen(self) -> None:
        """Result should be immutable."""
        result = DiscoveryResult()

        with pytest.raises(ValidationError):
            result.total_searched = 100  # type: ignore


class TestLobbyistAgentSingleton:
    """Tests for the Lobbyist agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent('lobbyist') is lobbyist_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(lobbyist_agent, Agent)
        assert lobbyist_agent.name == 'lobbyist'
