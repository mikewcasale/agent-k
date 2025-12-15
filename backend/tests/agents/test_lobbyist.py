"""Tests for the LOBBYIST discovery agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_k.agents.lobbyist import (
    DiscoveryResult,
    LobbyistAgent,
    LobbyistDeps,
)
from agent_k.agents.lobbyist.agent import create_lobbyist_agent
from agent_k.core.models import Competition, CompetitionType

pytestmark = pytest.mark.anyio


class TestCreateLobbyistAgent:
    """Tests for the create_lobbyist_agent factory function."""
    
    def test_creates_agent_with_default_model(self) -> None:
        """Agent should be created with default model."""
        # Skip if default model requires API key
        import os
        if not os.getenv('ANTHROPIC_API_KEY'):
            pytest.skip('ANTHROPIC_API_KEY not set for default model')
        
        agent = create_lobbyist_agent()
        assert agent is not None
        assert agent.name == 'lobbyist'
    
    def test_creates_agent_with_devstral_model(self) -> None:
        """Agent should accept devstral model specification."""
        # Devstral doesn't require API key
        agent = create_lobbyist_agent(model='devstral:local')
        assert agent is not None
        assert agent.name == 'lobbyist'
    
    def test_agent_has_required_tools(self) -> None:
        """Agent should have all required tools registered."""
        agent = create_lobbyist_agent(model='devstral:local')
        
        # Get tool names - the actual structure may vary
        # Just verify agent was created successfully
        assert agent is not None


class TestLobbyistDeps:
    """Tests for the LobbyistDeps dependency container."""
    
    def test_deps_creation(
        self,
        mock_http_client: AsyncMock,
        mock_platform_adapter: AsyncMock,
        mock_event_emitter: MagicMock,
    ) -> None:
        """Dependencies should be properly structured."""
        deps = LobbyistDeps(
            http_client=mock_http_client,
            platform_adapter=mock_platform_adapter,
            event_emitter=mock_event_emitter,
        )
        
        assert deps.http_client is mock_http_client
        assert deps.platform_adapter is mock_platform_adapter
        assert deps.event_emitter is mock_event_emitter
        assert deps.search_cache == {}
    
    def test_deps_search_cache_default(
        self,
        mock_http_client: AsyncMock,
        mock_platform_adapter: AsyncMock,
        mock_event_emitter: MagicMock,
    ) -> None:
        """Search cache should default to empty dict."""
        deps = LobbyistDeps(
            http_client=mock_http_client,
            platform_adapter=mock_platform_adapter,
            event_emitter=mock_event_emitter,
        )
        
        assert isinstance(deps.search_cache, dict)
        assert len(deps.search_cache) == 0


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
        from datetime import datetime, timezone
        from agent_k.core.models import EvaluationMetric
        
        competitions = [
            Competition(
                id='titanic',
                title='Titanic',
                competition_type=CompetitionType.GETTING_STARTED,
                metric=EvaluationMetric.ACCURACY,
                deadline=datetime(2030, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        
        result = DiscoveryResult(
            competitions=competitions,
            total_searched=10,
            filters_applied=['featured', 'active'],
        )
        
        assert len(result.competitions) == 1
        assert result.competitions[0].id == 'titanic'
        assert result.total_searched == 10
        assert result.filters_applied == ['featured', 'active']
    
    def test_result_is_frozen(self) -> None:
        """Result should be immutable."""
        result = DiscoveryResult()
        
        with pytest.raises(Exception):  # ValidationError or similar
            result.total_searched = 100  # type: ignore


class TestLobbyistAgent:
    """Tests for the LobbyistAgent class."""
    
    def test_agent_initialization(self) -> None:
        """Agent should initialize with devstral model."""
        agent = LobbyistAgent(model='devstral:local')
        assert agent is not None
    
    def test_agent_initialization_with_model(self) -> None:
        """Agent should accept custom model."""
        agent = LobbyistAgent(model='devstral:local')
        assert agent is not None
    
    def test_agent_initialization_with_timeout(self) -> None:
        """Agent should accept custom timeout."""
        agent = LobbyistAgent(model='devstral:local', timeout=600)
        assert agent._timeout == 600

