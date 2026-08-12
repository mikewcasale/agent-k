"""Tests for the process-wide Kaggle competition cache.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import RunContext

if TYPE_CHECKING:
    from collections.abc import Iterator

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.toolsets import kaggle as kaggle_toolset_module
from agent_k.toolsets.kaggle import COMPETITION_CACHE_MAX_ENTRIES, clear_competition_cache, kaggle_get_competition

__all__ = ()


def _make_competition(comp_id: str) -> Competition:
    """Build a minimal Competition object for cache tests."""
    return Competition(
        id=comp_id,
        title=f"Competition {comp_id}",
        competition_type=CompetitionType.PLAYGROUND,
        metric=EvaluationMetric.RMSE,
        metric_direction="minimize",
        deadline=datetime.now(UTC) + timedelta(days=30),
    )


@dataclass
class _StubDeps:
    """Minimal deps container so the tool wrapper can find search_cache."""

    search_cache: dict[str, Competition]
    kaggle_adapter: Any


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:
    """Ensure each test starts and ends with an empty module-level cache."""
    clear_competition_cache()
    yield
    clear_competition_cache()


class TestClearCompetitionCache:
    """Tests for ``clear_competition_cache``."""

    def test_clear_removes_all_entries(self) -> None:
        """After clear, the module-level cache must be empty."""
        cache = kaggle_toolset_module._cache
        cache["a"] = _make_competition("a")
        cache["b"] = _make_competition("b")
        assert len(cache) == 2

        clear_competition_cache()

        assert len(cache) == 0

    def test_clear_is_idempotent(self) -> None:
        """Calling clear on an already-empty cache must not raise."""
        clear_competition_cache()
        clear_competition_cache()
        assert len(kaggle_toolset_module._cache) == 0


class TestStoreCompetitionBounded:
    """Tests for ``_store_competition``'s bounded-cache behavior."""

    def test_cache_respects_max_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inserting more than the cap must evict oldest entries."""
        monkeypatch.setattr(kaggle_toolset_module, "COMPETITION_CACHE_MAX_ENTRIES", 3)

        ctx = _fake_run_context()
        for i in range(5):
            kaggle_toolset_module._store_competition(ctx, _make_competition(f"c{i}"))

        cache = kaggle_toolset_module._cache
        assert len(cache) == 3
        assert list(cache.keys()) == ["c2", "c3", "c4"]

    def test_repeated_store_moves_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-storing an existing key must refresh its LRU position."""
        monkeypatch.setattr(kaggle_toolset_module, "COMPETITION_CACHE_MAX_ENTRIES", 3)

        ctx = _fake_run_context()
        for cid in ("a", "b", "c"):
            kaggle_toolset_module._store_competition(ctx, _make_competition(cid))

        # Re-store 'a' to promote it, then add 'd' and 'e' to evict cold entries.
        kaggle_toolset_module._store_competition(ctx, _make_competition("a"))
        kaggle_toolset_module._store_competition(ctx, _make_competition("d"))
        kaggle_toolset_module._store_competition(ctx, _make_competition("e"))

        cache = kaggle_toolset_module._cache
        assert set(cache.keys()) == {"a", "d", "e"}

    def test_store_updates_per_deps_search_cache(self) -> None:
        """Storing must still populate the per-deps ``search_cache`` mapping."""
        deps = _StubDeps(search_cache={}, kaggle_adapter=AsyncMock())
        ctx = _fake_run_context(deps=deps)

        competition = _make_competition("kaggle-x")
        kaggle_toolset_module._store_competition(ctx, competition)

        assert deps.search_cache["kaggle-x"] is competition

    def test_default_cache_limit_reasonable(self) -> None:
        """The default cap must be a positive int well above single-mission use."""
        assert COMPETITION_CACHE_MAX_ENTRIES >= 32


class TestGetCompetitionLruPromotion:
    """Tests for cache-hit LRU promotion in ``kaggle_get_competition``."""

    @pytest.mark.asyncio
    async def test_cache_hit_promotes_entry_to_mru(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cache hit must survive eviction when the cap is reached."""
        monkeypatch.setattr(kaggle_toolset_module, "COMPETITION_CACHE_MAX_ENTRIES", 3)

        adapter = AsyncMock()
        deps = _StubDeps(search_cache={}, kaggle_adapter=adapter)
        ctx = _fake_run_context(deps=deps)

        for cid in ("a", "b", "c"):
            kaggle_toolset_module._store_competition(ctx, _make_competition(cid))

        # Access 'a' via the tool — cache hit should not call the adapter.
        result = await kaggle_get_competition(ctx, "a")
        assert result["id"] == "a"
        adapter.get_competition.assert_not_awaited()

        # Now insert two new entries; 'a' should remain because it was touched.
        kaggle_toolset_module._store_competition(ctx, _make_competition("d"))
        kaggle_toolset_module._store_competition(ctx, _make_competition("e"))

        cache = kaggle_toolset_module._cache
        assert set(cache.keys()) == {"a", "d", "e"}


def _fake_run_context(*, deps: Any | None = None) -> RunContext[Any]:
    """Return a stub RunContext sufficient for the tool wrapper.

    The Kaggle tool functions only touch ``ctx.deps`` (for the adapter, search
    cache, and event emitter). Bypass RunContext's normal init to avoid pulling
    in a live pydantic-ai model/agent for a pure unit test.
    """
    if deps is None:
        deps = _StubDeps(search_cache={}, kaggle_adapter=AsyncMock())
    ctx = RunContext.__new__(RunContext)
    object.__setattr__(ctx, "deps", deps)
    return ctx
