"""Tests for the experiment tracker persistence layer.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from agent_k.core.tracking import _SQLITE_BUSY_TIMEOUT_MS, ExperimentRecord, ExperimentTracker, KaggleSubmissionRecord

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _tracker(tmp_path: Path) -> ExperimentTracker:
    return ExperimentTracker(db_path=tmp_path / "experiments.sqlite")


def test_connect_enables_wal_journal(tmp_path: Path) -> None:
    """WAL journal mode is persistent across connections once enabled."""
    tracker = _tracker(tmp_path)
    with tracker._connect() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode.lower() == "wal"

    with sqlite3.connect(tracker.db_path) as fresh:
        journal_mode = fresh.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode.lower() == "wal"


def test_connect_sets_busy_timeout_and_synchronous(tmp_path: Path) -> None:
    """Each connection sets busy_timeout=30000 and synchronous=NORMAL."""
    tracker = _tracker(tmp_path)
    with tracker._connect() as conn:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
    assert busy_timeout == _SQLITE_BUSY_TIMEOUT_MS
    assert synchronous == 1  # NORMAL


def test_concurrent_record_experiment_across_threads(tmp_path: Path) -> None:
    """Multiple threads writing experiments simultaneously must not deadlock."""
    tracker = _tracker(tmp_path)
    competition_id = "concurrent-test"

    def _write(index: int) -> str:
        worker_tracker = ExperimentTracker(db_path=tracker.db_path)
        record = ExperimentRecord(competition_id=competition_id, phase="prototype", cv_score=float(index))
        stored = worker_tracker.record_experiment(record)
        return stored.record_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        record_ids = list(pool.map(_write, range(16)))

    assert len(set(record_ids)) == 16
    stored = tracker.list_experiments(competition_id, limit=32)
    assert {row.record_id for row in stored} == set(record_ids)


def test_concurrent_record_submission_across_threads(tmp_path: Path) -> None:
    """Concurrent submission writes should all succeed under WAL + busy_timeout."""
    tracker = _tracker(tmp_path)
    competition_id = "concurrent-submissions"

    def _write(index: int) -> str:
        worker_tracker = ExperimentTracker(db_path=tracker.db_path)
        record = KaggleSubmissionRecord(
            competition_id=competition_id,
            submission_id=f"submission-{index}",
            model_config_hash=f"hash-{index}",
            public_score=float(index),
        )
        stored = worker_tracker.record_submission(record)
        return stored.submission_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        submission_ids = list(pool.map(_write, range(16)))

    assert len(set(submission_ids)) == 16
    stored = tracker.list_submissions(competition_id, limit=32)
    assert {row.submission_id for row in stored} == set(submission_ids)
