"""Tests for the Kaggle toolset.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.toolsets.kaggle import (
    _cached_competition,
    _resolve_search_cache,
    _store_competition,
    kaggle_get_competition,
    kaggle_toolset,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ()

pytestmark = pytest.mark.anyio


def _make_competition(competition_id: str = "titanic") -> Competition:
    return Competition(
        id=competition_id,
        title=f"Competition {competition_id}",
        competition_type=CompetitionType.GETTING_STARTED,
        metric=EvaluationMetric.ACCURACY,
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _make_ctx(deps: Any) -> Any:
    """Build a RunContext-like stub: tools only read ``ctx.deps``."""
    return SimpleNamespace(deps=deps)


class _FakeAdapter:
    """Adapter stub that tracks `get_competition` invocations."""

    def __init__(self, competition: Competition) -> None:
        self._competition = competition
        self.calls: int = 0

    async def get_competition(self, competition_id: str) -> Competition:
        self.calls += 1
        if competition_id != self._competition.id:
            raise ValueError(f"Unknown competition: {competition_id}")
        return self._competition

    async def search_competitions(
        self,
        categories: list[str] | None = None,
        keywords: list[str] | None = None,
        min_prize: int | None = None,
        active_only: bool = True,
    ) -> AsyncIterator[Competition]:
        if False:  # pragma: no cover - keep async-iterator typing happy
            yield self._competition


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance."""
    assert isinstance(kaggle_toolset, FunctionToolset)


def test_toolset_id() -> None:
    """Toolset should have the expected id."""
    assert kaggle_toolset.id == "kaggle"


class TestResolveSearchCache:
    """Tests for ``_resolve_search_cache`` cache discovery."""

    def test_returns_dict_when_present(self) -> None:
        cache: dict[str, Any] = {}
        ctx = _make_ctx(SimpleNamespace(search_cache=cache))
        assert _resolve_search_cache(ctx) is cache

    def test_returns_none_when_missing(self) -> None:
        ctx = _make_ctx(SimpleNamespace())
        assert _resolve_search_cache(ctx) is None

    def test_returns_none_when_not_a_dict(self) -> None:
        ctx = _make_ctx(SimpleNamespace(search_cache="not-a-dict"))
        assert _resolve_search_cache(ctx) is None


class TestStoreCompetition:
    """Tests for ``_store_competition`` cache write behavior."""

    def test_writes_to_deps_cache(self) -> None:
        cache: dict[str, Any] = {}
        ctx = _make_ctx(SimpleNamespace(search_cache=cache))
        comp = _make_competition()

        _store_competition(ctx, comp)

        assert cache == {"titanic": comp}

    def test_no_op_when_deps_lack_cache(self) -> None:
        ctx = _make_ctx(SimpleNamespace())
        comp = _make_competition()

        _store_competition(ctx, comp)

        assert _resolve_search_cache(ctx) is None


class TestCachedCompetition:
    """Tests for ``_cached_competition`` cache read behavior."""

    def test_hits_when_cached(self) -> None:
        comp = _make_competition()
        ctx = _make_ctx(SimpleNamespace(search_cache={comp.id: comp}))

        assert _cached_competition(ctx, comp.id) is comp

    def test_miss_when_absent(self) -> None:
        ctx = _make_ctx(SimpleNamespace(search_cache={}))
        assert _cached_competition(ctx, "titanic") is None

    def test_returns_none_for_non_competition_payload(self) -> None:
        ctx = _make_ctx(SimpleNamespace(search_cache={"titanic": {"id": "titanic"}}))
        assert _cached_competition(ctx, "titanic") is None

    def test_returns_none_when_cache_missing(self) -> None:
        ctx = _make_ctx(SimpleNamespace())
        assert _cached_competition(ctx, "titanic") is None


class TestKaggleGetCompetitionCaching:
    """End-to-end caching contract for ``kaggle_get_competition``.

    The tool wrapper enforces the per-deps cache pattern: a second call with
    the same deps must reuse the cached competition; a fresh deps instance
    must trigger a new adapter fetch (no cross-mission leak).
    """

    async def test_second_call_hits_cache(self) -> None:
        comp = _make_competition()
        adapter = _FakeAdapter(comp)
        cache: dict[str, Any] = {}
        deps = SimpleNamespace(kaggle_adapter=adapter, search_cache=cache, event_emitter=None)
        ctx = _make_ctx(deps)

        first = await kaggle_get_competition(ctx, comp.id)
        second = await kaggle_get_competition(ctx, comp.id)

        assert first["id"] == comp.id
        assert second["id"] == comp.id
        assert adapter.calls == 1
        assert cache[comp.id] is comp

    async def test_different_deps_do_not_share_cache(self) -> None:
        comp = _make_competition()
        adapter_a = _FakeAdapter(comp)
        adapter_b = _FakeAdapter(comp)
        deps_a = SimpleNamespace(kaggle_adapter=adapter_a, search_cache={}, event_emitter=None)
        deps_b = SimpleNamespace(kaggle_adapter=adapter_b, search_cache={}, event_emitter=None)

        await kaggle_get_competition(_make_ctx(deps_a), comp.id)
        await kaggle_get_competition(_make_ctx(deps_b), comp.id)

        assert adapter_a.calls == 1
        assert adapter_b.calls == 1
        assert deps_a.search_cache != deps_b.search_cache or deps_a.search_cache is not deps_b.search_cache

    async def test_cache_is_optional(self) -> None:
        """Deps without a search_cache still work: every call hits the adapter."""
        comp = _make_competition()
        adapter = _FakeAdapter(comp)
        deps = SimpleNamespace(kaggle_adapter=adapter, event_emitter=None)
        ctx = _make_ctx(deps)

        await kaggle_get_competition(ctx, comp.id)
        await kaggle_get_competition(ctx, comp.id)

        assert adapter.calls == 2
