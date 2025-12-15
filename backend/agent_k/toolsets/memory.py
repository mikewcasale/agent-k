"""Memory toolset for AGENT-K agents.

Provides persistent memory functionality as a pydantic-ai toolset
that works with any model provider.

Uses FunctionToolset to properly integrate with pydantic-ai's tool system.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logfire
from pydantic_ai.toolsets import FunctionToolset

__all__ = ['create_memory_toolset', 'MemoryEntry']


@dataclass
class MemoryEntry:
    """A single memory entry."""
    key: str
    value: Any
    created_at: str
    updated_at: str
    tags: list[str]
    access_count: int = 0


def create_memory_toolset(
    storage_path: Path | None = None,
) -> FunctionToolset[Any]:
    """Create a memory toolset for persistent storage.
    
    This creates a FunctionToolset with tools for storing and retrieving
    data. Works with any model provider including OpenAI-compatible endpoints.
    
    Example:
        >>> toolset = create_memory_toolset(Path('./memory.json'))
        >>> agent = Agent('devstral:local', toolsets=[toolset])
    """
    toolset: FunctionToolset[Any] = FunctionToolset(id='memory')
    
    # In-memory storage
    _memory: dict[str, MemoryEntry] = {}
    
    def _load() -> None:
        """Load memory from disk."""
        if storage_path and storage_path.exists():
            try:
                data = json.loads(storage_path.read_text())
                for entry_data in data.get('entries', []):
                    _memory[entry_data['key']] = MemoryEntry(**entry_data)
                logfire.info('memory_loaded', count=len(_memory))
            except Exception as e:
                logfire.error('memory_load_failed', error=str(e))
    
    def _save() -> None:
        """Save memory to disk."""
        if storage_path:
            try:
                data = {
                    'version': '1.0',
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                    'entries': [
                        {
                            'key': e.key,
                            'value': e.value,
                            'created_at': e.created_at,
                            'updated_at': e.updated_at,
                            'tags': e.tags,
                            'access_count': e.access_count,
                        }
                        for e in _memory.values()
                    ],
                }
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                storage_path.write_text(json.dumps(data, indent=2))
            except Exception as e:
                logfire.error('memory_save_failed', error=str(e))
    
    # Load existing memory on creation
    _load()
    
    @toolset.tool
    def memory_store(
        key: str,
        value: Any,
        tags: list[str] | None = None,
    ) -> dict[str, str]:
        """Store a value in persistent memory.
        
        Args:
            key: Unique key for the value
            value: Value to store (any JSON-serializable data)
            tags: Optional tags for categorization
        
        Returns:
            Status of the store operation.
        """
        with logfire.span('memory_store', key=key):
            now = datetime.now(timezone.utc).isoformat()
            
            if key in _memory:
                entry = _memory[key]
                _memory[key] = MemoryEntry(
                    key=key,
                    value=value,
                    created_at=entry.created_at,
                    updated_at=now,
                    tags=tags if tags is not None else entry.tags,
                    access_count=entry.access_count,
                )
                action = 'updated'
            else:
                _memory[key] = MemoryEntry(
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                    tags=tags or [],
                )
                action = 'created'
            
            _save()
            return {'status': 'success', 'action': action, 'key': key}
    
    @toolset.tool
    def memory_retrieve(
        key: str,
    ) -> dict[str, Any]:
        """Retrieve a value from memory by its key.
        
        Args:
            key: Key to look up
        
        Returns:
            The stored value and metadata, or not_found status.
        """
        with logfire.span('memory_retrieve', key=key):
            entry = _memory.get(key)
            if entry is None:
                return {'found': False, 'key': key}
            
            entry.access_count += 1
            return {
                'found': True,
                'key': key,
                'value': entry.value,
                'tags': entry.tags,
                'created_at': entry.created_at,
                'access_count': entry.access_count,
            }
    
    @toolset.tool
    def memory_search(
        tag: str | None = None,
        prefix: str | None = None,
    ) -> dict[str, Any]:
        """Search memory entries by tag or key prefix.
        
        Args:
            tag: Find entries with this tag
            prefix: Find entries with keys starting with this prefix
        
        Returns:
            List of matching entries.
        """
        with logfire.span('memory_search', tag=tag, prefix=prefix):
            results = []
            
            for key, entry in _memory.items():
                if tag and tag not in entry.tags:
                    continue
                if prefix and not key.startswith(prefix):
                    continue
                results.append({
                    'key': key,
                    'tags': entry.tags,
                    'created_at': entry.created_at,
                })
            
            return {'count': len(results), 'entries': results}
    
    @toolset.tool
    def memory_delete(
        key: str,
    ) -> dict[str, Any]:
        """Delete an entry from memory.
        
        Args:
            key: Key to delete
        
        Returns:
            Status of the delete operation.
        """
        with logfire.span('memory_delete', key=key):
            if key in _memory:
                del _memory[key]
                _save()
                return {'status': 'deleted', 'key': key}
            return {'status': 'not_found', 'key': key}
    
    @toolset.tool
    def memory_list_keys(
        tag: str | None = None,
    ) -> dict[str, Any]:
        """List all keys in memory.
        
        Args:
            tag: Optional tag to filter by
        
        Returns:
            List of keys in memory.
        """
        with logfire.span('memory_list_keys', tag=tag):
            keys = []
            for key, entry in _memory.items():
                if tag and tag not in entry.tags:
                    continue
                keys.append(key)
            return {'count': len(keys), 'keys': keys}
    
    return toolset
