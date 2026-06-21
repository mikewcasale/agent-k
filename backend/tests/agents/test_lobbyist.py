"""Tests for the LOBBYIST discovery agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.lobbyist import DiscoveryResult, LobbyistAgent, LobbyistDeps, LobbyistSettings, lobbyist_agent
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric

__all__ = ()

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

pytestmark = pytest.mark.anyio


def _make_competition(comp_id: str) -> Competition:
    """Build a minimal Competition fixture for tool tests."""
    return Competition(
        id=comp_id,
        title=f"Competition {comp_id}",
        competition_type=CompetitionType.FEATURED,
        metric=EvaluationMetric.ACCURACY,
        deadline=datetime(2099, 12, 31, tzinfo=UTC),
    )


def _build_adapter(competitions: list[Competition]) -> MagicMock:
    """Build a platform adapter mock whose search_competitions yields the given items."""

    async def _iter(**_: object) -> AsyncIterator[Competition]:
        for comp in competitions:
            yield comp

    adapter = MagicMock()
    adapter.search_competitions = _iter
    return adapter


def _build_emitter() -> MagicMock:
    emitter = MagicMock()
    emitter.emit_tool_start = AsyncMock()
    emitter.emit_tool_result = AsyncMock()
    return emitter


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


class TestSearchKaggleCompetitions:
    """Tests for the bounded search_kaggle_competitions tool."""

    @pytest.fixture
    def fresh_agent_slot(self) -> Iterator[None]:
        """Pop the module singleton so each test can build its own LobbyistAgent."""
        from agent_k.agents import AGENT_REGISTRY

        previous = AGENT_REGISTRY.pop("lobbyist", None)
        try:
            yield
        finally:
            AGENT_REGISTRY.pop("lobbyist", None)
            if previous is not None:
                AGENT_REGISTRY["lobbyist"] = previous

    async def test_caps_results_at_settings_default(self, fresh_agent_slot: None) -> None:
        """When max_results is omitted the limit falls back to settings.max_results."""
        agent = LobbyistAgent(settings=LobbyistSettings(max_results=5))
        adapter = _build_adapter([_make_competition(f"c{i}") for i in range(50)])
        async with httpx.AsyncClient() as client:
            deps = LobbyistDeps(http_client=client, platform_adapter=adapter, event_emitter=_build_emitter())
            ctx = SimpleNamespace(deps=deps)

            results = await agent.search_kaggle_competitions(ctx, categories=["featured"])  # type: ignore[arg-type]

        assert len(results) == 5
        assert [item["id"] for item in results] == [f"c{i}" for i in range(5)]
        assert list(deps.search_cache.keys()) == [f"c{i}" for i in range(5)]

    async def test_respects_explicit_max_results(self, fresh_agent_slot: None) -> None:
        """Explicit max_results overrides the settings default."""
        agent = LobbyistAgent(settings=LobbyistSettings(max_results=50))
        adapter = _build_adapter([_make_competition(f"c{i}") for i in range(20)])
        async with httpx.AsyncClient() as client:
            deps = LobbyistDeps(http_client=client, platform_adapter=adapter, event_emitter=_build_emitter())
            ctx = SimpleNamespace(deps=deps)

            results = await agent.search_kaggle_competitions(
                ctx,  # type: ignore[arg-type]
                categories=["featured"],
                max_results=3,
            )

        assert len(results) == 3

    async def test_returns_all_when_below_limit(self, fresh_agent_slot: None) -> None:
        """No truncation when the adapter yields fewer items than the cap."""
        agent = LobbyistAgent(settings=LobbyistSettings(max_results=50))
        adapter = _build_adapter([_make_competition(f"c{i}") for i in range(4)])
        async with httpx.AsyncClient() as client:
            deps = LobbyistDeps(http_client=client, platform_adapter=adapter, event_emitter=_build_emitter())
            ctx = SimpleNamespace(deps=deps)

            results = await agent.search_kaggle_competitions(ctx, categories=["featured"])  # type: ignore[arg-type]

        assert len(results) == 4
        assert len(deps.search_cache) == 4
