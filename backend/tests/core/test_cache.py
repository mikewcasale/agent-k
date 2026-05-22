"""Tests for BoundedTTLCache.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.cache import BoundedTTLCache

__all__ = ()


class _ManualClock:
    """Test-controllable monotonic clock."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


def test_set_then_get_returns_value() -> None:
    """A freshly inserted entry is retrievable until it expires."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=60)
    cache.set("k", 1)
    assert cache.get("k") == 1


def test_get_returns_none_for_missing_key() -> None:
    """Missing keys return ``None`` rather than raising."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=60)
    assert cache.get("missing") is None


def test_entry_expires_after_ttl() -> None:
    """Entries past the TTL are evicted on access."""
    clock = _ManualClock()
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=30, time_source=clock)
    cache.set("k", 1)
    clock.advance(31.0)
    assert cache.get("k") is None
    assert len(cache) == 0


def test_ttl_zero_disables_expiry() -> None:
    """TTL of zero keeps entries indefinitely."""
    clock = _ManualClock()
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=0, time_source=clock)
    cache.set("k", 1)
    clock.advance(1_000_000.0)
    assert cache.get("k") == 1


def test_lru_eviction_when_full() -> None:
    """Inserting beyond max_size evicts the least-recently-used entry."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    # Access "a" so it becomes most-recently used.
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_set_refreshes_ttl_on_overwrite() -> None:
    """Overwriting an entry refreshes its TTL."""
    clock = _ManualClock()
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=10, time_source=clock)
    cache.set("k", 1)
    clock.advance(8.0)
    cache.set("k", 2)
    clock.advance(8.0)
    assert cache.get("k") == 2


def test_get_refreshes_lru_position() -> None:
    """Reading an entry promotes it so it survives subsequent evictions."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_max_size_zero_disables_cache() -> None:
    """A max_size of zero acts as a null cache."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=0, ttl_seconds=60)
    cache.set("k", 1)
    assert cache.get("k") is None
    assert len(cache) == 0


def test_contains_matches_get_semantics() -> None:
    """``key in cache`` follows the same expiry/eviction semantics as ``get``."""
    clock = _ManualClock()
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=2, ttl_seconds=5, time_source=clock)
    cache.set("k", 1)
    assert "k" in cache
    clock.advance(6.0)
    assert "k" not in cache


def test_pop_removes_and_returns() -> None:
    """``pop`` removes the entry and returns its value when live."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=2, ttl_seconds=60)
    cache.set("k", 1)
    assert cache.pop("k") == 1
    assert cache.get("k") is None


def test_pop_returns_none_for_expired_entry() -> None:
    """``pop`` skips expired entries even though the key was present."""
    clock = _ManualClock()
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=2, ttl_seconds=5, time_source=clock)
    cache.set("k", 1)
    clock.advance(6.0)
    assert cache.pop("k") is None


def test_clear_removes_all_entries() -> None:
    """``clear`` empties the cache."""
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert len(cache) == 0


def test_len_purges_expired_entries() -> None:
    """``len`` reflects only live entries by purging expired ones first."""
    clock = _ManualClock()
    cache: BoundedTTLCache[str, int] = BoundedTTLCache(max_size=4, ttl_seconds=5, time_source=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    clock.advance(6.0)
    assert len(cache) == 0


@pytest.mark.parametrize("max_size,ttl_seconds", [(-1, 60), (0, -1)], ids=["negative-max-size", "negative-ttl"])
def test_invalid_constructor_args_raise(max_size: int, ttl_seconds: int) -> None:
    """Negative max_size or ttl_seconds is rejected at construction."""
    with pytest.raises(ValueError):
        BoundedTTLCache(max_size=max_size, ttl_seconds=ttl_seconds)
