"""Tests for experiment tracking persistence.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agent_k.core.tracking import ExperimentRecord, ExperimentTracker, KaggleSubmissionRecord, create_experiment_tracker

if TYPE_CHECKING:
    pass

__all__ = ()


class _CountingConnection(sqlite3.Connection):
    """Subclass that counts explicit close() calls."""

    close_count = 0

    def close(self) -> None:
        type(self).close_count += 1
        super().close()


class _ConnectionTracker:
    """Wraps sqlite3.connect so we can assert connections are explicitly closed."""

    def __init__(self) -> None:
        self.open_count: int = 0
        self._original = sqlite3.connect

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _CountingConnection.close_count = 0
        tracker = self

        def wrapped(*args: object, **kwargs: object) -> sqlite3.Connection:
            kwargs["factory"] = _CountingConnection  # type: ignore[assignment]
            conn = tracker._original(*args, **kwargs)
            tracker.open_count += 1
            return conn

        monkeypatch.setattr(sqlite3, "connect", wrapped)

    @property
    def close_count(self) -> int:
        return _CountingConnection.close_count


class TestExperimentTrackerConnectionLifecycle:
    """Regression tests for the SQLite connection lifecycle.

    `with sqlite3.connect(...) as conn` commits/rolls back the transaction
    but does NOT close the connection. CPython's refcount GC happens to
    close it when conn goes out of scope, but PyPy and explicit
    `gc.disable()` paths would leak. The tracker must explicitly close
    every connection it opens.
    """

    def test_every_opened_connection_is_explicitly_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every sqlite3.connect() inside the tracker must be paired with a close()."""
        connections = _ConnectionTracker()
        connections.install(monkeypatch)

        db_path = tmp_path / "experiments.sqlite"
        tracker = ExperimentTracker(db_path=db_path)
        for _ in range(10):
            tracker.record_experiment(ExperimentRecord(competition_id="comp-test", phase="prototype", cv_score=0.1))
            tracker.list_experiments("comp-test", limit=5)
            tracker.best_experiment("comp-test", metric="cv_score", direction="maximize")

        assert connections.open_count > 0, "Test sanity: expected connections to be opened"
        assert connections.close_count == connections.open_count, (
            f"Connection leak: opened {connections.open_count}, closed only {connections.close_count}"
        )

    def test_construction_closes_schema_connection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Schema initialization must close the connection it opens."""
        connections = _ConnectionTracker()
        connections.install(monkeypatch)

        ExperimentTracker(db_path=tmp_path / "experiments.sqlite")

        assert connections.open_count >= 1
        assert connections.close_count == connections.open_count


class TestExperimentTrackerRoundTrip:
    """Smoke tests for tracker persistence round-trip."""

    def test_record_and_retrieve_experiment(self, tmp_path: Path) -> None:
        """Recorded experiments must be retrievable in newest-first order."""
        db_path = tmp_path / "experiments.sqlite"
        tracker = create_experiment_tracker(db_path=db_path)
        record = ExperimentRecord(competition_id="comp-x", phase="prototype", model_name="LGBMRegressor", cv_score=0.42)
        stored = tracker.record_experiment(record)
        assert stored.config_signature is not None

        results = tracker.list_experiments("comp-x")
        assert len(results) == 1
        assert results[0].record_id == record.record_id
        assert results[0].cv_score == 0.42

    def test_best_experiment_respects_direction(self, tmp_path: Path) -> None:
        """best_experiment should return the highest score when maximizing."""
        db_path = tmp_path / "experiments.sqlite"
        tracker = create_experiment_tracker(db_path=db_path)
        for score in (0.1, 0.5, 0.3):
            tracker.record_experiment(ExperimentRecord(competition_id="comp-y", phase="prototype", public_score=score))

        best = tracker.best_experiment("comp-y", metric="public_score", direction="maximize")
        assert best is not None
        assert best.public_score == 0.5

        worst = tracker.best_experiment("comp-y", metric="public_score", direction="minimize")
        assert worst is not None
        assert worst.public_score == 0.1

    def test_find_duplicate_config(self, tmp_path: Path) -> None:
        """Submissions with matching config hashes must be found by duplicate lookup."""
        db_path = tmp_path / "experiments.sqlite"
        tracker = create_experiment_tracker(db_path=db_path)
        submission = KaggleSubmissionRecord(
            competition_id="comp-z", submission_id="sub-1", model_config_hash="abc123", feature_set_hash="feat1"
        )
        tracker.record_submission(submission)

        match = tracker.find_duplicate_config("comp-z", model_config_hash="abc123", feature_set_hash="feat1")
        assert match is not None
        assert match.submission_id == "sub-1"

        miss = tracker.find_duplicate_config("comp-z", model_config_hash="def456", feature_set_hash="feat1")
        assert miss is None
