"""Tests for experiment tracking lookups.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from agent_k.core.tracking import ExperimentRecord, ExperimentTracker

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()

_COMPETITION_ID = "generic-tabular-regression"
_SIGNATURE = "a1b2c3d4e5f6"


def _record(phase: str, *, created_at: datetime, **fields: Any) -> ExperimentRecord:
    return ExperimentRecord(
        competition_id=_COMPETITION_ID, phase=phase, code_signature=_SIGNATURE, created_at=created_at, **fields
    )


class TestFindLatestByCodeSignature:
    """Tests for signature lookups across mission phases."""

    def test_returns_latest_record_without_phase_filter(self, tmp_path: Path) -> None:
        """Without a filter the newest record for the signature wins."""
        tracker = ExperimentTracker(db_path=tmp_path / "experiments.db")
        now = datetime.now(UTC)
        tracker.record_experiment(_record("evolution", created_at=now - timedelta(minutes=5), cv_score=0.42))
        tracker.record_experiment(_record("submission", created_at=now, submission_id="sub-1"))

        found = tracker.find_latest_by_code_signature(_COMPETITION_ID, _SIGNATURE)

        assert found is not None
        assert found.phase == "submission"

    def test_phase_filter_skips_later_records_from_other_phases(self, tmp_path: Path) -> None:
        """A later submission row must not shadow the evaluation it came from."""
        tracker = ExperimentTracker(db_path=tmp_path / "experiments.db")
        now = datetime.now(UTC)
        tracker.record_experiment(
            _record("evolution", created_at=now - timedelta(minutes=5), cv_score=0.42, metrics={"stage": "full"})
        )
        tracker.record_experiment(_record("submission", created_at=now, submission_id="sub-1"))

        found = tracker.find_latest_by_code_signature(_COMPETITION_ID, _SIGNATURE, phase="evolution")

        assert found is not None
        assert found.phase == "evolution"
        assert found.metrics["stage"] == "full"

    def test_phase_filter_returns_none_when_phase_absent(self, tmp_path: Path) -> None:
        """Filtering on a phase that never recorded the signature yields nothing."""
        tracker = ExperimentTracker(db_path=tmp_path / "experiments.db")
        tracker.record_experiment(_record("prototype", created_at=datetime.now(UTC), cv_score=0.5))

        assert tracker.find_latest_by_code_signature(_COMPETITION_ID, _SIGNATURE, phase="evolution") is None
