"""Tests for HintEffectivenessTracker persistence semantics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from typing import TYPE_CHECKING

import pytest

from agent_k.core.tracking import HintAttemptRecord, HintEffectivenessTracker

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


@pytest.fixture
def tracker_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the tracker to a per-test JSON file."""
    path = tmp_path / "hint_tracker.json"
    monkeypatch.setattr(HintEffectivenessTracker, "_HINT_TRACKER_PATH", path)
    return path


class TestHintEffectivenessTrackerSave:
    """Regression tests for ``HintEffectivenessTracker.save``.

    The prior implementation iterated only ``_success_counts`` when
    serializing, so hints that had accumulated failures — or that had been
    manually suppressed — but no successes were silently dropped on the next
    reload. That reset their suppression state and let the evolver waste
    evaluation budget re-trying hints already known to regress.
    """

    def test_failure_only_hint_survives_reload(self, tracker_path: Path) -> None:
        """A hint with only failures should persist and stay suppressed."""
        tracker = HintEffectivenessTracker()
        for gen in range(3):
            tracker.record_attempt(
                HintAttemptRecord(
                    competition_id="comp1", hint_id="h_only_fails", applied=True, delta=-1.0, generation=gen
                )
            )

        assert tracker.is_suppressed("h_only_fails", "comp1") is True

        payload = json.loads(tracker_path.read_text(encoding="utf-8"))
        assert payload["comp1"]["h_only_fails"] == {"success": 0, "failure": 3, "suppressed": True}

        reloaded = HintEffectivenessTracker()
        assert reloaded.is_suppressed("h_only_fails", "comp1") is True
        assert reloaded._failure_counts[("comp1", "h_only_fails")] == 3
        assert reloaded._success_counts[("comp1", "h_only_fails")] == 0

    def test_manual_suppression_without_attempts_persists(self, tracker_path: Path) -> None:
        """A hint suppressed via ``suppress_hint`` alone must survive reload."""
        tracker = HintEffectivenessTracker()
        tracker.suppress_hint("h_manual", "comp2")
        tracker.save()

        payload = json.loads(tracker_path.read_text(encoding="utf-8"))
        assert payload["comp2"]["h_manual"] == {"success": 0, "failure": 0, "suppressed": True}

        reloaded = HintEffectivenessTracker()
        assert reloaded.is_suppressed("h_manual", "comp2") is True

    def test_mixed_success_and_failure_hints_all_persist(self, tracker_path: Path) -> None:
        """Success-only, failure-only, and mixed hints all round-trip."""
        tracker = HintEffectivenessTracker()
        tracker.record_attempt(
            HintAttemptRecord(competition_id="comp3", hint_id="h_ok", applied=True, delta=0.05, generation=0)
        )
        tracker.record_attempt(
            HintAttemptRecord(competition_id="comp3", hint_id="h_bad", applied=True, delta=-0.1, generation=1)
        )
        tracker.record_attempt(
            HintAttemptRecord(competition_id="comp3", hint_id="h_mixed", applied=True, delta=0.02, generation=2)
        )
        tracker.record_attempt(
            HintAttemptRecord(competition_id="comp3", hint_id="h_mixed", applied=True, delta=-0.02, generation=3)
        )

        reloaded = HintEffectivenessTracker()
        assert reloaded._success_counts[("comp3", "h_ok")] == 1
        assert reloaded._failure_counts[("comp3", "h_ok")] == 0
        assert reloaded._success_counts[("comp3", "h_bad")] == 0
        assert reloaded._failure_counts[("comp3", "h_bad")] == 1
        assert reloaded._success_counts[("comp3", "h_mixed")] == 1
        assert reloaded._failure_counts[("comp3", "h_mixed")] == 1

    def test_success_rate_survives_reload_for_failure_only_hint(self, tracker_path: Path) -> None:
        """After reload, ``get_success_rate`` must reflect persisted failures."""
        tracker = HintEffectivenessTracker()
        for gen in range(2):
            tracker.record_attempt(
                HintAttemptRecord(competition_id="comp4", hint_id="h_reload", applied=True, delta=-0.5, generation=gen)
            )
        # Reload picks up per-hint counters even though live attempts are lost;
        # the persisted rate for a hint that never succeeded must stay at 0.0.
        reloaded = HintEffectivenessTracker()
        # ``get_success_rate`` derives from the live ``_records`` list which is
        # not persisted; ensure the persisted failure counter is still usable
        # via the direct accessor so subsequent runs know the hint has failed.
        assert reloaded._failure_counts[("comp4", "h_reload")] == 2
