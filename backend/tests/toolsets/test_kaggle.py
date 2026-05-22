"""Tests for the Kaggle toolset.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.toolsets import FunctionToolset

import agent_k.toolsets.kaggle as kaggle_toolset_module
from agent_k.core.cache import BoundedTTLCache
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.toolsets.kaggle import kaggle_get_competition, kaggle_toolset

__all__ = ()


def _competition(comp_id: str, *, prize_pool: int | None = None) -> Competition:
    return Competition(
        id=comp_id,
        title=f"Title for {comp_id}",
        description="desc",
        competition_type=CompetitionType.PLAYGROUND,
        metric=EvaluationMetric.RMSE,
        metric_direction="minimize",
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
        prize_pool=prize_pool,
        max_team_size=1,
        max_daily_submissions=5,
        tags=frozenset(),
        url=None,
    )


def _make_ctx(adapter: Any) -> Any:
    deps = MagicMock()
    deps.platform_adapter = adapter
    deps.kaggle_adapter = adapter
    deps.search_cache = None
    deps.event_emitter = None
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a fresh cache for each test to avoid cross-test contamination."""
    fresh: BoundedTTLCache[str, Competition] = BoundedTTLCache(max_size=256, ttl_seconds=300.0)
    monkeypatch.setattr(kaggle_toolset_module, "_cache", fresh)


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance."""
    assert isinstance(kaggle_toolset, FunctionToolset)


def test_toolset_id() -> None:
    """Toolset should have the expected id."""
    assert kaggle_toolset.id == "kaggle"


def test_module_cache_is_bounded_ttl_cache() -> None:
    """The module-level cache must be a bounded TTL cache, not an unbounded dict."""
    assert isinstance(kaggle_toolset_module._cache, BoundedTTLCache)


async def test_get_competition_serves_cached_value_on_repeat_call() -> None:
    """A second call with the same id reuses the cached entry instead of refetching."""
    adapter = MagicMock()
    adapter.get_competition = AsyncMock(return_value=_competition("titanic"))
    ctx = _make_ctx(adapter)

    await kaggle_get_competition(ctx, "titanic")
    await kaggle_get_competition(ctx, "titanic")

    assert adapter.get_competition.await_count == 1


async def test_get_competition_refetches_after_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache entries expire after the configured TTL so stale data is refreshed."""
    clock_value = {"now": 0.0}

    def clock() -> float:
        return clock_value["now"]

    expiring: BoundedTTLCache[str, Competition] = BoundedTTLCache(max_size=4, ttl_seconds=10.0, time_source=clock)
    monkeypatch.setattr(kaggle_toolset_module, "_cache", expiring)

    adapter = MagicMock()
    adapter.get_competition = AsyncMock(
        side_effect=[_competition("titanic", prize_pool=1_000), _competition("titanic", prize_pool=2_000)]
    )
    ctx = _make_ctx(adapter)

    first = await kaggle_get_competition(ctx, "titanic")
    clock_value["now"] = 100.0
    second = await kaggle_get_competition(ctx, "titanic")

    assert adapter.get_competition.await_count == 2
    assert first["prize_pool"] == 1_000
    assert second["prize_pool"] == 2_000


async def test_get_competition_bounds_cache_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache size is capped so long-running processes do not leak memory."""
    bounded: BoundedTTLCache[str, Competition] = BoundedTTLCache(max_size=2, ttl_seconds=300.0)
    monkeypatch.setattr(kaggle_toolset_module, "_cache", bounded)

    adapter = MagicMock()
    adapter.get_competition = AsyncMock(side_effect=lambda comp_id: _competition(comp_id))
    ctx = _make_ctx(adapter)

    for comp_id in ("a", "b", "c"):
        await kaggle_get_competition(ctx, comp_id)

    assert len(bounded) <= 2
    # "a" was evicted; refetching it must hit the adapter again.
    await kaggle_get_competition(ctx, "a")
    assert adapter.get_competition.await_count == 4
