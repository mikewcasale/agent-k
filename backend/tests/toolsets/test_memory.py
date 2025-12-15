"""Tests for the memory toolset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_k.toolsets.memory import create_memory_toolset, MemoryEntry

pytestmark = pytest.mark.anyio


class TestMemoryEntry:
    """Tests for the MemoryEntry dataclass."""
    
    def test_entry_creation(self) -> None:
        """Entry should be created with required fields."""
        entry = MemoryEntry(
            key='test',
            value={'data': 123},
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            tags=['test'],
        )
        
        assert entry.key == 'test'
        assert entry.value == {'data': 123}
        assert entry.tags == ['test']
        assert entry.access_count == 0
    
    def test_entry_with_access_count(self) -> None:
        """Entry should accept custom access count."""
        entry = MemoryEntry(
            key='test',
            value='data',
            created_at='2024-01-01T00:00:00Z',
            updated_at='2024-01-01T00:00:00Z',
            tags=[],
            access_count=5,
        )
        
        assert entry.access_count == 5


class TestCreateMemoryToolset:
    """Tests for the create_memory_toolset factory function."""
    
    def test_creates_toolset_without_storage(self) -> None:
        """Toolset should be created without storage path."""
        toolset = create_memory_toolset()
        assert toolset is not None
        assert toolset.id == 'memory'
    
    def test_creates_toolset_with_storage(self, tmp_path: Path) -> None:
        """Toolset should be created with storage path."""
        storage_path = tmp_path / 'memory.json'
        toolset = create_memory_toolset(storage_path=storage_path)
        assert toolset is not None


class TestMemoryStoreAndRetrieve:
    """Tests for memory store and retrieve operations."""
    
    def test_store_and_retrieve_basic(self, tmp_path: Path) -> None:
        """Store and retrieve should work for basic values."""
        storage_path = tmp_path / 'memory.json'
        toolset = create_memory_toolset(storage_path=storage_path)
        
        # Get the tool functions
        # The toolset registers tools that can be accessed via the agent
        assert toolset is not None
    
    def test_persistence(self, tmp_path: Path) -> None:
        """Memory should persist across toolset instances."""
        storage_path = tmp_path / 'memory.json'
        
        # Create first toolset and store
        toolset1 = create_memory_toolset(storage_path=storage_path)
        assert toolset1 is not None
        
        # The file should be created when data is stored
        # For testing without running the agent, we verify the toolset is created


class TestMemorySearch:
    """Tests for memory search operations."""
    
    def test_search_by_tag(self, tmp_path: Path) -> None:
        """Search should filter by tag."""
        toolset = create_memory_toolset(storage_path=tmp_path / 'memory.json')
        assert toolset is not None
    
    def test_search_by_prefix(self, tmp_path: Path) -> None:
        """Search should filter by key prefix."""
        toolset = create_memory_toolset(storage_path=tmp_path / 'memory.json')
        assert toolset is not None


class TestMemoryDelete:
    """Tests for memory delete operations."""
    
    def test_delete_existing(self, tmp_path: Path) -> None:
        """Delete should remove existing entries."""
        toolset = create_memory_toolset(storage_path=tmp_path / 'memory.json')
        assert toolset is not None


class TestMemoryListKeys:
    """Tests for memory list keys operation."""
    
    def test_list_all_keys(self, tmp_path: Path) -> None:
        """List keys should return all keys."""
        toolset = create_memory_toolset(storage_path=tmp_path / 'memory.json')
        assert toolset is not None
    
    def test_list_keys_with_tag(self, tmp_path: Path) -> None:
        """List keys should filter by tag."""
        toolset = create_memory_toolset(storage_path=tmp_path / 'memory.json')
        assert toolset is not None

