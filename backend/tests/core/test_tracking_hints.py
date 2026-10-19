"""Tests for hint effectiveness persistence and replay.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from agent_k.core.tracking import ExperimentTracker, HintAttemptRecord, HintEffectivenessTracker

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

_COMPETITION = "sample-competition"
_OTHER_COMPETITION = "other-competition"
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _attempt(hint_id: str, *, delta: float, generation: int, competition_id: str = _COMPETITION) -> HintAttemptRecord:
    """Build an applied attempt whose timestamp orders with its generation."""
    return HintAttemptRecord(
        hint_id=hint_id,
        competition_id=competition_id,
        generation=generation,
        applied=True,
        cv_score_before=1.0,
        cv_score_after=1.0 + delta,
        delta=delta,
        created_at=_EPOCH + timedelta(minutes=generation),
    )


def _hint_tracker(tmp_path: Path, *, experiment_tracker: ExperimentTracker | None = None) -> HintEffectivenessTracker:
    return HintEffectivenessTracker(
        experiment_tracker=experiment_tracker, competition_id=_COMPETITION, storage_path=tmp_path / "hint_tracker.json"
    )


class TestHintStatisticsPersistence:
    """Tests for the JSON sidecar holding aggregate hint statistics."""

    def test_failure_only_hint_survives_a_restart(self, tmp_path: Path) -> None:
        """A hint that never succeeded keeps its failure count across instances."""
        tracker = _hint_tracker(tmp_path)
        tracker.record_attempt(_attempt("scale_numeric", delta=-0.3, generation=1))

        reloaded = _hint_tracker(tmp_path)

        assert reloaded.get_success_rate("scale_numeric", _COMPETITION) == 0.0
        assert json.loads((tmp_path / "hint_tracker.json").read_text(encoding="utf-8")) == {
            _COMPETITION: {"scale_numeric": {"success": 0, "failure": 1, "suppressed": False, "amplified": False}}
        }

    def test_suppression_survives_a_restart(self, tmp_path: Path) -> None:
        """Three failures suppress a hint and the decision is persisted."""
        tracker = _hint_tracker(tmp_path)
        for generation in range(3):
            tracker.record_attempt(_attempt("target_encode", delta=-0.1, generation=generation))
        assert tracker.is_suppressed("target_encode", _COMPETITION)

        reloaded = _hint_tracker(tmp_path)

        assert reloaded.is_suppressed("target_encode", _COMPETITION)
        assert reloaded.suppressed_hints == {"target_encode"}

    def test_amplification_survives_a_restart(self, tmp_path: Path) -> None:
        """Amplified hints are written out and restored."""
        tracker = _hint_tracker(tmp_path)
        tracker.amplify_hint("log_target", _COMPETITION)
        tracker.save()

        reloaded = _hint_tracker(tmp_path)

        assert reloaded.is_amplified("log_target", _COMPETITION)

    def test_success_rate_falls_back_to_persisted_counts(self, tmp_path: Path) -> None:
        """Aggregate counts drive the success rate when no attempts ran this session."""
        tracker = _hint_tracker(tmp_path)
        tracker.record_attempt(_attempt("impute_median", delta=0.2, generation=1))
        tracker.record_attempt(_attempt("impute_median", delta=-0.1, generation=2))

        reloaded = _hint_tracker(tmp_path)

        assert reloaded.get_success_rate("impute_median", _COMPETITION) == 0.5

    def test_save_leaves_no_temporary_files(self, tmp_path: Path) -> None:
        """The staged write is renamed into place rather than left behind."""
        tracker = _hint_tracker(tmp_path)
        tracker.record_attempt(_attempt("clip_outliers", delta=0.4, generation=1))

        assert sorted(path.name for path in tmp_path.iterdir()) == ["hint_tracker.json"]

    def test_corrupt_sidecar_is_ignored(self, tmp_path: Path) -> None:
        """A malformed payload degrades to empty statistics instead of raising."""
        (tmp_path / "hint_tracker.json").write_text("[]", encoding="utf-8")

        tracker = _hint_tracker(tmp_path)

        assert tracker.suppressed_hints == set()
        assert tracker.get_success_rate("impute_median", _COMPETITION) == 0.0


class TestHintAttemptReplay:
    """Tests for replaying the durable hint attempt log."""

    def test_list_hint_attempts_returns_chronological_history(self, tmp_path: Path) -> None:
        """Attempts come back oldest first and can be filtered per competition."""
        experiment_tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
        experiment_tracker.record_hint_attempt(_attempt("a", delta=0.1, generation=1))
        experiment_tracker.record_hint_attempt(_attempt("b", delta=-0.1, generation=2))
        experiment_tracker.record_hint_attempt(
            _attempt("c", delta=0.3, generation=3, competition_id=_OTHER_COMPETITION)
        )

        scoped = experiment_tracker.list_hint_attempts(_COMPETITION)
        every = experiment_tracker.list_hint_attempts()

        assert [record.generation for record in scoped] == [1, 2]
        assert len(every) == 3

    def test_last_attempt_is_visible_after_a_restart(self, tmp_path: Path) -> None:
        """Recency comes from the attempt log, not only the in-session list."""
        experiment_tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
        tracker = _hint_tracker(tmp_path, experiment_tracker=experiment_tracker)
        tracker.record_attempt(_attempt("knn_features", delta=0.1, generation=1))
        tracker.record_attempt(_attempt("knn_features", delta=0.2, generation=2))

        reloaded = _hint_tracker(tmp_path, experiment_tracker=experiment_tracker)
        last_attempt = reloaded.get_last_attempt("knn_features", _COMPETITION)

        assert last_attempt is not None
        assert last_attempt.generation == 2
        assert reloaded.get_success_rate("knn_features", _COMPETITION) == 1.0

    def test_replay_does_not_double_count_attempts(self, tmp_path: Path) -> None:
        """Counts from the sidecar are rebuilt from the log rather than added to it."""
        experiment_tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
        tracker = _hint_tracker(tmp_path, experiment_tracker=experiment_tracker)
        tracker.record_attempt(_attempt("bin_numeric", delta=0.1, generation=1))

        reloaded = _hint_tracker(tmp_path, experiment_tracker=experiment_tracker)
        reloaded.save()

        assert json.loads((tmp_path / "hint_tracker.json").read_text(encoding="utf-8")) == {
            _COMPETITION: {"bin_numeric": {"success": 1, "failure": 0, "suppressed": False, "amplified": False}}
        }

    def test_replay_restores_suppression_without_a_sidecar(self, tmp_path: Path) -> None:
        """A lost sidecar is rebuilt from the attempt log, suppression included."""
        experiment_tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
        for generation in range(3):
            experiment_tracker.record_hint_attempt(_attempt("geo_distance", delta=-0.2, generation=generation))

        tracker = _hint_tracker(tmp_path, experiment_tracker=experiment_tracker)

        assert tracker.is_suppressed("geo_distance", _COMPETITION)

    def test_replay_window_is_bounded(self, tmp_path: Path) -> None:
        """Only the newest attempts are replayed when the history exceeds the limit."""
        experiment_tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
        for generation in range(5):
            experiment_tracker.record_hint_attempt(_attempt("ratio_features", delta=0.1, generation=generation))

        tracker = HintEffectivenessTracker(
            experiment_tracker=experiment_tracker,
            competition_id=_COMPETITION,
            storage_path=tmp_path / "hint_tracker.json",
            history_limit=2,
        )
        last_attempt = tracker.get_last_attempt("ratio_features", _COMPETITION)

        assert last_attempt is not None
        assert last_attempt.generation == 4
