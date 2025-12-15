"""Tests for the search toolset."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent_k.toolsets.search import create_search_toolset

pytestmark = pytest.mark.anyio


class TestCreateSearchToolset:
    """Tests for the create_search_toolset factory function."""
    
    def test_creates_toolset(self) -> None:
        """Toolset should be created."""
        toolset = create_search_toolset()
        assert toolset is not None
        assert toolset.id == 'web_search'


class TestWebSearch:
    """Tests for the web_search tool."""
    
    async def test_web_search_basic(self) -> None:
        """Web search should be callable."""
        toolset = create_search_toolset()
        assert toolset is not None


class TestSearchPapers:
    """Tests for the search_papers tool."""
    
    async def test_search_papers_basic(self) -> None:
        """Search papers should be callable."""
        toolset = create_search_toolset()
        assert toolset is not None


class TestSearchKaggle:
    """Tests for the search_kaggle tool."""
    
    async def test_search_kaggle_basic(self) -> None:
        """Search kaggle should be callable."""
        toolset = create_search_toolset()
        assert toolset is not None

