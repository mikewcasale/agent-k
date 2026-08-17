"""Tests for the tracking toolset and its use of the tracker cache.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from agent_k.core.tracking import (
    ExperimentRecord,
    ExperimentTracker,
    create_experiment_tracker,
    reset_experiment_trackers,
)
from agent_k.toolsets.tracking import (
    tracking_best_experiment,
    tracking_list_experiments,
    tracking_record_experiment,
    tracking_toolset,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ()


@pytest.fixture(autouse=True)
def _reset_cache() -> Iterator[None]:
    reset_experiment_trackers()
    yield
    reset_experiment_trackers()


def _make_ctx() -> RunContext[Any]:
    """Return a minimal RunContext-shaped stub; the tools only use it to satisfy their signature."""
    return cast("RunContext[Any]", MagicMock())


def _record_payload(competition_id: str = "titanic", *, public_score: float | None = None) -> dict[str, Any]:
    return {
        "competition_id": competition_id,
        "phase": "prototype",
        "model_name": "LGBMRegressor",
        "model_family": "lightgbm",
        "public_score": public_score,
    }


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance with the expected id."""
    assert isinstance(tracking_toolset, FunctionToolset)
    assert tracking_toolset.id == "tracking"


def test_create_experiment_tracker_caches_by_path(tmp_path: Path) -> None:
    """Trackers are memoized by resolved database path."""
    db_path = tmp_path / "cache.sqlite"
    first = create_experiment_tracker(db_path=db_path)
    second = create_experiment_tracker(db_path=db_path)
    assert first is second


def test_create_experiment_tracker_distinct_paths(tmp_path: Path) -> None:
    """Different paths yield different tracker instances."""
    first = create_experiment_tracker(db_path=tmp_path / "a.sqlite")
    second = create_experiment_tracker(db_path=tmp_path / "b.sqlite")
    assert first is not second


def test_reset_experiment_trackers_forces_new_instance(tmp_path: Path) -> None:
    """After reset, a new tracker instance is constructed."""
    db_path = tmp_path / "cache.sqlite"
    first = create_experiment_tracker(db_path=db_path)
    reset_experiment_trackers()
    second = create_experiment_tracker(db_path=db_path)
    assert first is not second


def test_create_experiment_tracker_respects_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no path is passed the env override is honored and cached."""
    override = tmp_path / "env.sqlite"
    monkeypatch.setenv("AGENT_K_EXPERIMENT_DB", str(override))
    reset_experiment_trackers()
    tracker = create_experiment_tracker()
    assert tracker.db_path == override
    again = create_experiment_tracker()
    assert tracker is again


def test_tracking_record_experiment_runs_off_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool wraps the sync SQLite call in ``asyncio.to_thread``."""
    monkeypatch.setenv("AGENT_K_EXPERIMENT_DB", str(tmp_path / "off_loop.sqlite"))
    reset_experiment_trackers()

    async def _run() -> dict[str, Any]:
        result = await tracking_record_experiment(_make_ctx(), record=_record_payload(public_score=0.42))
        return cast("dict[str, Any]", result)

    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
    original = asyncio.to_thread

    async def _spy(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        calls.append((func, args, kwargs))
        return await original(func, *args, **kwargs)

    monkeypatch.setattr("agent_k.toolsets.tracking.asyncio.to_thread", _spy)

    result = asyncio.run(_run())

    assert result["competition_id"] == "titanic"
    assert result["public_score"] == pytest.approx(0.42)
    assert len(calls) == 1
    invoked, _args, _kwargs = calls[0]
    tracker = create_experiment_tracker()
    assert invoked == tracker.record_experiment


def test_tracking_list_and_best_experiment_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Record two experiments then list/rank via the async tools."""
    monkeypatch.setenv("AGENT_K_EXPERIMENT_DB", str(tmp_path / "e2e.sqlite"))
    reset_experiment_trackers()

    tracker = create_experiment_tracker()
    tracker.record_experiment(ExperimentRecord.model_validate(_record_payload(public_score=0.5)))
    tracker.record_experiment(ExperimentRecord.model_validate(_record_payload(public_score=0.9)))

    async def _list() -> list[dict[str, Any]]:
        result = await tracking_list_experiments(_make_ctx(), competition_id="titanic", limit=5)
        return cast("list[dict[str, Any]]", result)

    async def _best() -> dict[str, Any] | None:
        result = await tracking_best_experiment(
            _make_ctx(), competition_id="titanic", metric="public_score", direction="maximize"
        )
        return cast("dict[str, Any] | None", result)

    listed = asyncio.run(_list())
    assert len(listed) == 2

    best = asyncio.run(_best())
    assert best is not None
    assert best["public_score"] == pytest.approx(0.9)


def test_tracking_best_experiment_returns_none_for_missing_competition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``tracking_best_experiment`` returns ``None`` when there are no records."""
    monkeypatch.setenv("AGENT_K_EXPERIMENT_DB", str(tmp_path / "empty.sqlite"))
    reset_experiment_trackers()

    async def _best() -> dict[str, Any] | None:
        result = await tracking_best_experiment(_make_ctx(), competition_id="none-here")
        return cast("dict[str, Any] | None", result)

    assert asyncio.run(_best()) is None


def test_tracker_cache_survives_direct_construction(tmp_path: Path) -> None:
    """Directly constructing ``ExperimentTracker`` should not poison the factory cache."""
    db_path = tmp_path / "direct.sqlite"
    direct = ExperimentTracker(db_path=db_path)
    factory = create_experiment_tracker(db_path=db_path)
    assert factory is not direct
    cached_again = create_experiment_tracker(db_path=db_path)
    assert cached_again is factory
