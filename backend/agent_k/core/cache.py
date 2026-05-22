"""Bounded, TTL-aware caching primitives for AGENT-K.

@notice: |
    Bounded, TTL-aware caching primitives for AGENT-K.

@dev: |
    Plain ``dict`` caches grow without bound and never refresh, which causes
    long-running processes to leak memory and serve stale upstream data across
    missions. ``BoundedTTLCache`` puts a hard ceiling on entry count using
    insertion-order LRU eviction and refreshes entries after a configurable TTL.

@graph:
    id: agent_k.core.cache
    provides:
        - agent_k.core.cache:BoundedTTLCache
    pattern: cache

@agent-guidance:
    do:
        - "Use BoundedTTLCache for any process-wide cache that holds upstream payloads."
    do_not:
        - "Reintroduce raw module-level dicts as caches; they leak memory and serve stale data."

@human-review:
    last-verified: 2026-05-22
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, Final, Generic, TypeVar

from .sage import Doc, Range

__all__ = ("BoundedTTLCache",)

CacheKeyT = TypeVar("CacheKeyT")
"""Type variable for cache keys."""

CacheValueT = TypeVar("CacheValueT")
"""Type variable for cache values."""

DEFAULT_MAX_SIZE: Final[int] = 256
"""Default maximum entries retained in a BoundedTTLCache."""

DEFAULT_TTL_SECONDS: Final[float] = 300.0
"""Default per-entry time-to-live in seconds (5 minutes)."""


class BoundedTTLCache(Generic[CacheKeyT, CacheValueT]):
    """LRU cache with hard size cap and per-entry TTL eviction.

    @notice: |
        LRU cache with hard size cap and per-entry TTL eviction.

    @dev: |
        Entries are evicted on access if they exceed the TTL, and the
        oldest entry is evicted on insert when the size cap is exceeded.
        TTL of zero disables expiry; a max_size of zero disables the cache.
        Not thread-safe; intended for asyncio single-loop use.

        @pattern:
            name: cache
            rationale: "Bounded TTL cache for upstream payloads."
            violations: "Unbounded dict caches leak memory and serve stale data."

        @concurrency:
            model: asyncio
            safe: false
            reason: "Mutates internal OrderedDict without locks."
    """

    __slots__ = ("_entries", "_max_size", "_ttl_seconds", "_time_source")

    def __init__(
        self,
        *,
        max_size: Annotated[int, Doc("Maximum entries retained before LRU eviction."), Range(0, 1_000_000)] = (
            DEFAULT_MAX_SIZE
        ),
        ttl_seconds: Annotated[float, Doc("Per-entry TTL in seconds; 0 disables expiry."), Range(0, 86_400)] = (
            DEFAULT_TTL_SECONDS
        ),
        time_source: Annotated[Callable[[], float] | None, Doc("Optional monotonic clock override for tests.")] = None,
    ) -> None:
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._time_source = time_source or time.monotonic
        self._entries: OrderedDict[CacheKeyT, tuple[float, CacheValueT]] = OrderedDict()

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` has a live (non-expired) entry."""
        return self.get(key) is not None  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Return the current number of live entries after expiry cleanup."""
        self._purge_expired()
        return len(self._entries)

    def get(self, key: Annotated[CacheKeyT, Doc("Cache key to look up.")]) -> CacheValueT | None:
        """Return the cached value for ``key`` or ``None`` if missing/expired.

        @notice: |
            Refreshes LRU recency on hit and drops the entry on expiry.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._is_expired(expires_at):
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def set(
        self,
        key: Annotated[CacheKeyT, Doc("Cache key to insert or refresh.")],
        value: Annotated[CacheValueT, Doc("Value to store.")],
    ) -> None:
        """Insert ``value`` for ``key``, evicting the oldest entry when full.

        @notice: |
            Refreshes the TTL on overwrite and trims to ``max_size`` when needed.
        """
        if self._max_size == 0:
            return
        expires_at = self._compute_expiry()
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = (expires_at, value)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def pop(self, key: Annotated[CacheKeyT, Doc("Cache key to remove if present.")]) -> CacheValueT | None:
        """Remove and return the value for ``key`` if present, else ``None``."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        expires_at, value = entry
        if self._is_expired(expires_at):
            return None
        return value

    def clear(self) -> None:
        """Remove every entry from the cache."""
        self._entries.clear()

    def _is_expired(self, expires_at: float) -> bool:
        if self._ttl_seconds == 0:
            return False
        return self._time_source() >= expires_at

    def _compute_expiry(self) -> float:
        if self._ttl_seconds == 0:
            return float("inf")
        return self._time_source() + self._ttl_seconds

    def _purge_expired(self) -> None:
        if self._ttl_seconds == 0:
            return
        now = self._time_source()
        expired = [key for key, (expires_at, _) in self._entries.items() if now >= expires_at]
        for key in expired:
            del self._entries[key]
