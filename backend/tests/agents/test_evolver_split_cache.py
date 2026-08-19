"""Tests for the Evolver staged validation-split cache.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import csv
import shutil
from typing import TYPE_CHECKING

import pytest

from agent_k.agents.evolver import (
    _copy_staged_inputs,
    _resolve_split_max_rows,
    _split_cache_key,
    _SplitStageCache,
    _stage_validation_split,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

pytestmark = pytest.mark.anyio


def _write_train_csv(path: Path, *, rows: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "feature_a", "feature_b", "target"])
        for index in range(rows):
            writer.writerow([index, index * 0.5, (index % 7) * 1.25, index % 3])


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def test_resolve_split_max_rows_prefers_explicit_cap() -> None:
    assert _resolve_split_max_rows(max_rows=300, max_generations=20) == 300


def test_resolve_split_max_rows_caps_short_runs() -> None:
    assert _resolve_split_max_rows(max_rows=None, max_generations=3) == 800
    assert _resolve_split_max_rows(max_rows=0, max_generations=3) == 800


def test_resolve_split_max_rows_uncapped_for_long_runs() -> None:
    assert _resolve_split_max_rows(max_rows=None, max_generations=20) is None


def test_stage_validation_split_writes_reusable_inputs(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=100)

    staged = _stage_validation_split(
        cache_root=tmp_path / "cache",
        key="abc123",
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=None,
    )

    assert staged.id_column == "id"
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        assert (staged.directory / name).exists()

    assert _row_count(staged.directory / "train.csv") == 80
    assert _row_count(staged.directory / "test.csv") == 20
    assert _row_count(staged.directory / "sample_submission.csv") == 20
    assert list(staged.y_val.columns) == ["id", "target"]
    assert len(staged.y_val) == 20

    with (staged.directory / "test.csv").open("r", encoding="utf-8", newline="") as handle:
        test_header = next(csv.reader(handle))
    assert "target" not in test_header


def test_stage_validation_split_respects_max_rows(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=500)

    staged = _stage_validation_split(
        cache_root=tmp_path / "cache",
        key="capped",
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=50,
    )

    assert _row_count(staged.directory / "train.csv") == 50
    assert _row_count(staged.directory / "test.csv") == 50
    assert len(staged.y_val) == 50


def test_split_cache_key_changes_with_training_data(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=40)
    first = _split_cache_key(
        train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, max_rows=None
    )

    _write_train_csv(train_path, rows=90)
    second = _split_cache_key(
        train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, max_rows=None
    )

    assert first != second


def test_split_cache_key_changes_with_max_rows(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=40)
    uncapped = _split_cache_key(
        train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, max_rows=None
    )
    capped = _split_cache_key(
        train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, max_rows=10
    )

    assert uncapped != capped


def test_copy_staged_inputs_creates_independent_files(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=60)
    staged = _stage_validation_split(
        cache_root=tmp_path / "cache",
        key="independent",
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.25,
        max_rows=None,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _copy_staged_inputs(staged.directory, run_dir)

    original = (staged.directory / "train.csv").read_text(encoding="utf-8")
    (run_dir / "train.csv").write_text("clobbered\n", encoding="utf-8")

    assert (staged.directory / "train.csv").read_text(encoding="utf-8") == original


async def test_split_cache_reuses_staged_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "session" / "data"
    data_dir.mkdir(parents=True)
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=120)

    cache = _SplitStageCache()
    run_one = tmp_path / "run_one"
    run_one.mkdir()
    first = await cache.stage_into(
        run_one,
        data_dir=data_dir,
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=None,
    )
    run_two = tmp_path / "run_two"
    run_two.mkdir()
    second = await cache.stage_into(
        run_two,
        data_dir=data_dir,
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=None,
    )

    assert second is first
    assert first.directory.parent == data_dir.parent / ".split_cache"
    for run_dir in (run_one, run_two):
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            assert (run_dir / name).exists()
    assert _row_count(run_two / "train.csv") == _row_count(first.directory / "train.csv")


async def test_split_cache_restages_when_directory_removed(tmp_path: Path) -> None:
    data_dir = tmp_path / "session" / "data"
    data_dir.mkdir(parents=True)
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=120)

    cache = _SplitStageCache()
    run_one = tmp_path / "run_one"
    run_one.mkdir()
    first = await cache.stage_into(
        run_one,
        data_dir=data_dir,
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=None,
    )
    shutil.rmtree(first.directory)

    run_two = tmp_path / "run_two"
    run_two.mkdir()
    second = await cache.stage_into(
        run_two,
        data_dir=data_dir,
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=None,
    )

    assert second is not first
    assert (second.directory / "train.csv").exists()


async def test_split_cache_serializes_concurrent_staging(tmp_path: Path) -> None:
    data_dir = tmp_path / "session" / "data"
    data_dir.mkdir(parents=True)
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=150)

    cache = _SplitStageCache()
    run_dirs = [tmp_path / f"run_{index}" for index in range(4)]
    for run_dir in run_dirs:
        run_dir.mkdir()

    staged = await asyncio.gather(
        *(
            cache.stage_into(
                run_dir,
                data_dir=data_dir,
                train_path=train_path,
                id_column="id",
                target_columns=["target"],
                validation_split=0.2,
                max_rows=None,
            )
            for run_dir in run_dirs
        )
    )

    assert all(entry is staged[0] for entry in staged)
    for run_dir in run_dirs:
        assert _row_count(run_dir / "train.csv") == 120
        assert _row_count(run_dir / "test.csv") == 30


async def test_split_cache_evicts_least_recently_used(tmp_path: Path) -> None:
    data_dir = tmp_path / "session" / "data"
    data_dir.mkdir(parents=True)
    train_path = tmp_path / "train.csv"
    _write_train_csv(train_path, rows=200)

    cache = _SplitStageCache(max_entries=1)
    run_one = tmp_path / "run_one"
    run_one.mkdir()
    first = await cache.stage_into(
        run_one,
        data_dir=data_dir,
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=40,
    )
    run_two = tmp_path / "run_two"
    run_two.mkdir()
    second = await cache.stage_into(
        run_two,
        data_dir=data_dir,
        train_path=train_path,
        id_column="id",
        target_columns=["target"],
        validation_split=0.2,
        max_rows=None,
    )

    assert second.directory != first.directory
    assert not first.directory.exists()
    assert (second.directory / "train.csv").exists()
