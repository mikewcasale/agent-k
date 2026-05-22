"""Tests for hint effectiveness tracking persistence.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_k.core.tracking import ExperimentTracker, HintAttemptRecord, HintEffectivenessTracker

__all__ = ()


@pytest.fixture
def tracker_db(tmp_path: Path) -> ExperimentTracker:
    """Return an ExperimentTracker pointing at a fresh SQLite file."""
    return ExperimentTracker(db_path=tmp_path / "experiments.sqlite")


@pytest.fixture
def hint_tracker_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the hint tracker JSON to a temporary path."""
    path = tmp_path / "hint_tracker.json"
    monkeypatch.setattr(HintEffectivenessTracker, "_HINT_TRACKER_PATH", path)
    return path


def _attempt(
    *,
    hint_id: str,
    competition_id: str,
    delta: float,
    generation: int = 0,
    applied: bool = True,
    when: datetime | None = None,
) -> HintAttemptRecord:
    return HintAttemptRecord(
        hint_id=hint_id,
        competition_id=competition_id,
        delta=delta,
        generation=generation,
        applied=applied,
        created_at=when or datetime.now(UTC),
    )


class TestHintEffectivenessTrackerPersistence:
    """HintEffectivenessTracker must survive process restarts."""

    def test_success_rate_persists_across_restart(self, tracker_db: ExperimentTracker, hint_tracker_path: Path) -> None:
        """Success rate must reflect persisted counts, not just in-memory records."""
        hint = HintEffectivenessTracker(experiment_tracker=tracker_db)
        hint.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=0.05, generation=0))
        hint.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=0.02, generation=1))
        hint.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=-0.01, generation=2))

        # Pre-restart sanity check.
        assert hint.get_success_rate("scale", "comp") == pytest.approx(2 / 3)

        # Simulate a fresh process: new tracker instances read from disk only.
        fresh_db = ExperimentTracker(db_path=tracker_db.db_path)
        restarted = HintEffectivenessTracker(experiment_tracker=fresh_db)
        assert restarted._records == []
        assert restarted.get_success_rate("scale", "comp") == pytest.approx(2 / 3)

    def test_success_rate_falls_back_to_sqlite_when_json_missing(
        self, tracker_db: ExperimentTracker, hint_tracker_path: Path
    ) -> None:
        """Even if the JSON sidecar is deleted, SQLite counts hydrate the rate."""
        seeded = HintEffectivenessTracker(experiment_tracker=tracker_db)
        seeded.record_attempt(_attempt(hint_id="impute", competition_id="comp", delta=0.10))
        seeded.record_attempt(_attempt(hint_id="impute", competition_id="comp", delta=-0.05))

        hint_tracker_path.unlink()

        restarted = HintEffectivenessTracker(experiment_tracker=tracker_db)
        assert restarted.get_success_rate("impute", "comp") == pytest.approx(0.5)

    def test_zero_delta_attempts_are_ignored(self, tracker_db: ExperimentTracker, hint_tracker_path: Path) -> None:
        """Zero-delta attempts contribute neither success nor failure."""
        hint = HintEffectivenessTracker(experiment_tracker=tracker_db)
        hint.record_attempt(_attempt(hint_id="encode", competition_id="comp", delta=0.0))
        assert hint.get_success_rate("encode", "comp") == 0.0

        hint.record_attempt(_attempt(hint_id="encode", competition_id="comp", delta=0.01))
        assert hint.get_success_rate("encode", "comp") == 1.0

    def test_unapplied_attempts_excluded_from_rate(
        self, tracker_db: ExperimentTracker, hint_tracker_path: Path
    ) -> None:
        """Attempts marked applied=False must not skew the success rate."""
        hint = HintEffectivenessTracker(experiment_tracker=tracker_db)
        hint.record_attempt(_attempt(hint_id="bin", competition_id="comp", delta=-0.5, applied=False))
        hint.record_attempt(_attempt(hint_id="bin", competition_id="comp", delta=0.10, applied=True))

        # Pre-restart: counts reflect only the applied attempt.
        assert hint.get_success_rate("bin", "comp") == 1.0

        # Force SQLite fallback by clearing the JSON sidecar.
        hint_tracker_path.unlink()
        restarted = HintEffectivenessTracker(experiment_tracker=tracker_db)
        assert restarted.get_success_rate("bin", "comp") == 1.0

    def test_get_last_attempt_persists_across_restart(
        self, tracker_db: ExperimentTracker, hint_tracker_path: Path
    ) -> None:
        """get_last_attempt must return the persisted record after a restart."""
        seeded = HintEffectivenessTracker(experiment_tracker=tracker_db)
        earlier = datetime.now(UTC) - timedelta(minutes=10)
        later = datetime.now(UTC)
        seeded.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=0.05, generation=1, when=earlier))
        seeded.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=0.07, generation=4, when=later))

        restarted = HintEffectivenessTracker(experiment_tracker=tracker_db)
        assert restarted._records == []
        last = restarted.get_last_attempt("scale", "comp")
        assert last is not None
        assert last.generation == 4
        assert last.delta == pytest.approx(0.07)
        assert last.hint_id == "scale"

    def test_get_last_attempt_without_tracker_uses_records(self, hint_tracker_path: Path) -> None:
        """Without an experiment tracker, fall back to in-memory records."""
        hint = HintEffectivenessTracker(experiment_tracker=None)
        hint.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=0.01, generation=2))
        last = hint.get_last_attempt("scale", "comp")
        assert last is not None
        assert last.generation == 2

    def test_get_last_attempt_prefers_newest_record(
        self, tracker_db: ExperimentTracker, hint_tracker_path: Path
    ) -> None:
        """A fresh in-memory attempt wins over an older SQLite row."""
        seeded = HintEffectivenessTracker(experiment_tracker=tracker_db)
        old_time = datetime.now(UTC) - timedelta(hours=1)
        seeded.record_attempt(_attempt(hint_id="scale", competition_id="comp", delta=0.02, generation=1, when=old_time))

        live = HintEffectivenessTracker(experiment_tracker=tracker_db)
        new_attempt = _attempt(hint_id="scale", competition_id="comp", delta=0.04, generation=5)
        live.record_attempt(new_attempt)

        last = live.get_last_attempt("scale", "comp")
        assert last is not None
        assert last.generation == 5

    def test_suppression_threshold_triggers_after_repeated_failures(
        self, tracker_db: ExperimentTracker, hint_tracker_path: Path
    ) -> None:
        """Three persisted failures must mark the hint as suppressed."""
        hint = HintEffectivenessTracker(experiment_tracker=tracker_db)
        for _ in range(3):
            hint.record_attempt(_attempt(hint_id="impute", competition_id="comp", delta=-0.01))
        assert hint.is_suppressed("impute", "comp") is True

        restarted = HintEffectivenessTracker(experiment_tracker=tracker_db)
        assert restarted.is_suppressed("impute", "comp") is True
