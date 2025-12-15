"""Tests for the OpenEvolve adapter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_k.adapters.openevolve import OpenEvolveAdapter

pytestmark = pytest.mark.anyio


class TestOpenEvolveAdapter:
    """Tests for the OpenEvolveAdapter class."""
    
    def test_adapter_creation(self) -> None:
        """Adapter should be created as a dataclass."""
        adapter = OpenEvolveAdapter()
        
        assert adapter is not None
        assert adapter.platform_name == 'openevolve'
    
    async def test_authenticate_returns_true(self) -> None:
        """Authenticate should return True (stub)."""
        adapter = OpenEvolveAdapter()
        
        result = await adapter.authenticate()
        assert result is True
    
    async def test_get_leaderboard_returns_empty(self) -> None:
        """Get leaderboard should return empty list (stub)."""
        adapter = OpenEvolveAdapter()
        
        result = await adapter.get_leaderboard('test_comp')
        assert result == []

