"""Tests for experiment tracker connection handling.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import RunContext

from agent_k.core.tracking import SQLITE_BUSY_TIMEOUT_MS, ExperimentRecord, ExperimentTracker, KaggleSubmissionRecord


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    """Return a tracker backed by an isolated database file."""
    return ExperimentTracker(db_path=tmp_path / "experiments.sqlite")


def _open_fd_count() -> int:
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def test_repeated_writes_do_not_leak_connections(tracker: ExperimentTracker) -> None:
    """Each unit of work closes its connection instead of holding the descriptor open."""
    tracker.record_experiment(ExperimentRecord(competition_id="c", phase="prototype", cv_score=0.1))
    baseline = _open_fd_count()

    for index in range(200):
        tracker.record_experiment(ExperimentRecord(competition_id="c", phase="evolution", cv_score=float(index)))

    # Without closing, sqlite3 keeps one descriptor per query; allow slack for pytest's own I/O.
    assert _open_fd_count() - baseline < 10


def test_reads_do_not_leak_connections(tracker: ExperimentTracker) -> None:
    """Query helpers release their connections too, not just the write path."""
    tracker.record_experiment(ExperimentRecord(competition_id="c", phase="evolution", cv_score=0.5))
    baseline = _open_fd_count()

    for _ in range(200):
        tracker.list_experiments("c", limit=5)
        tracker.best_experiment("c", metric="cv_score", direction="maximize")

    assert _open_fd_count() - baseline < 10


def test_database_uses_wal_and_busy_timeout(tracker: ExperimentTracker) -> None:
    """WAL is persisted in the database header and the busy timeout is applied per connection."""
    with closing(sqlite3.connect(tracker.db_path)) as probe:
        assert probe.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    # Asserting the connection is tuned, not the public API.
    with closing(tracker._connect()) as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == SQLITE_BUSY_TIMEOUT_MS
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL, safe under WAL


class _WalRefusingConnection(sqlite3.Connection):
    """Connection that rejects ``PRAGMA journal_mode``, as a filesystem without WAL support does."""

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        """Delegate to SQLite unless WAL is being requested."""
        if "journal_mode" in sql.lower():
            raise sqlite3.OperationalError("cannot change into wal mode from within a transaction")
        return super().execute(sql, parameters)


def test_wal_failure_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem that refuses WAL leaves the tracker fully usable on the rollback journal."""
    monkeypatch.setattr(
        "agent_k.core.tracking.sqlite3.connect", partial(sqlite3.connect, factory=_WalRefusingConnection)
    )
    tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
    monkeypatch.undo()

    with closing(sqlite3.connect(tracker.db_path)) as probe:
        assert probe.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"

    # Relaxed durability is only safe under WAL, so the rollback journal keeps FULL syncs.
    with closing(tracker._connect()) as conn:
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL

    stored = tracker.record_experiment(ExperimentRecord(competition_id="c", phase="prototype", cv_score=0.2))
    assert tracker.list_experiments("c")[0].record_id == stored.record_id


def test_concurrent_writers_all_commit(tmp_path: Path) -> None:
    """Parallel writers against one database file every record, with no lock errors."""
    db_path = tmp_path / "experiments.sqlite"
    ExperimentTracker(db_path=db_path)
    writers, per_writer = 8, 25

    def write(worker: int) -> None:
        tracker = ExperimentTracker(db_path=db_path)
        for index in range(per_writer):
            tracker.record_experiment(
                ExperimentRecord(competition_id="c", phase="evolution", cv_score=float(worker * 100 + index))
            )

    with ThreadPoolExecutor(max_workers=writers) as pool:
        for future in [pool.submit(write, worker) for worker in range(writers)]:
            future.result()

    reader = ExperimentTracker(db_path=db_path)
    assert len(reader.list_experiments("c", limit=10_000)) == writers * per_writer


def test_reads_observe_writes_from_another_connection(tmp_path: Path) -> None:
    """A tracker opened separately sees committed rows, so WAL does not strand readers."""
    db_path = tmp_path / "experiments.sqlite"
    writer = ExperimentTracker(db_path=db_path)
    writer.record_submission(
        KaggleSubmissionRecord(competition_id="c", submission_id="s1", public_score=0.9, model_config_hash="hash-1")
    )

    reader = ExperimentTracker(db_path=db_path)
    duplicate = reader.find_duplicate_config("c", model_config_hash="hash-1")
    assert duplicate is not None
    assert duplicate.submission_id == "s1"


async def test_tracking_tools_do_not_block_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool wrappers run SQLite off the loop, so other tasks keep making progress."""
    monkeypatch.setenv("AGENT_K_EXPERIMENT_DB", str(tmp_path / "experiments.sqlite"))
    from agent_k.toolsets import tracking as tracking_tools

    tracking_tools._tracker.cache_clear()  # Reset the per-process cache for this test.
    try:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0)
                ticks += 1

        ctx = cast("RunContext[Any]", None)
        task = asyncio.create_task(ticker())
        for index in range(25):
            await tracking_tools.tracking_record_experiment(
                ctx, {"competition_id": "c", "phase": "evolution", "cv_score": float(index)}
            )
        task.cancel()

        # A blocking sqlite call inside the coroutine would starve the ticker entirely.
        assert ticks > 0
        listed = await tracking_tools.tracking_list_experiments(ctx, "c", limit=100)
        assert len(listed) == 25
    finally:
        tracking_tools._tracker.cache_clear()  # Avoid leaking the tmp_path tracker.
