"""Tests for the submission feedback orchestrator.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path

import pytest

from agent_k.core.feedback import SubmissionCandidate, SubmissionOrchestrator
from agent_k.core.tracking import ExperimentTracker

__all__ = ()


@pytest.fixture
def tracker(tmp_path: Path) -> ExperimentTracker:
    """Tracker backed by a temporary SQLite database."""
    return ExperimentTracker(db_path=tmp_path / "feedback.db")


@pytest.fixture
def candidate() -> SubmissionCandidate:
    """Baseline candidate with one mutation tag."""
    return SubmissionCandidate(model_config_hash="cfg-1", mutations=["point"])


def _submit(
    orchestrator: SubmissionOrchestrator,
    candidate: SubmissionCandidate,
    *,
    submission_id: str,
    public_score: float | None = None,
    private_score: float | None = None,
    rank: int | None = None,
) -> None:
    orchestrator.process_submission(
        candidate, submission_id=submission_id, public_score=public_score, private_score=private_score, rank=rank
    )


class TestBaselineEstablishment:
    """First scored submission must establish a baseline without faking stagnation."""

    def test_first_scored_submission_sets_baseline(
        self, tracker: ExperimentTracker, candidate: SubmissionCandidate
    ) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-1", public_score=0.5, private_score=None, rank=None
        )

        assert feedback.improved is None
        # Stagnation must not advance on the very first scored submission.
        assert orchestrator._stagnant_generations == 0
        assert orchestrator._last_score == 0.5

    def test_baseline_does_not_trigger_false_convergence(
        self, tracker: ExperimentTracker, candidate: SubmissionCandidate
    ) -> None:
        """Convergence must not fire on the very first scored submission.

        Regression guard for a bug where the baseline was never set, so every
        scored submission counted as stagnation and the loop falsely converged
        after ``convergence_generations`` calls.
        """
        orchestrator = SubmissionOrchestrator(
            tracker, competition_id="comp-1", metric_direction="maximize", convergence_generations=2
        )

        first = orchestrator.process_submission(
            candidate, submission_id="sub-1", public_score=0.5, private_score=None, rank=None
        )
        second = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=0.5, private_score=None, rank=None
        )

        assert first.convergence_reason is None
        # One real stagnation step after the baseline; below the threshold of 2.
        assert second.convergence_reason is None
        assert orchestrator._stagnant_generations == 1

    def test_missing_score_does_not_increment_stagnation(
        self, tracker: ExperimentTracker, candidate: SubmissionCandidate
    ) -> None:
        """Submissions with no leaderboard score yet carry no signal in either direction."""
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5)
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=None, private_score=None, rank=None
        )

        assert feedback.improved is None
        assert orchestrator._stagnant_generations == 0
        assert orchestrator._last_score == 0.5


class TestImprovementTracking:
    """Improvement / regression handling across metric directions."""

    def test_maximize_improvement(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5)
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=0.7, private_score=None, rank=None
        )

        assert feedback.improved is True
        assert orchestrator._last_score == 0.7
        assert orchestrator._stagnant_generations == 0

    def test_maximize_regression(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.7)
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=0.5, private_score=None, rank=None
        )

        assert feedback.improved is False
        assert orchestrator._last_score == 0.7
        assert orchestrator._stagnant_generations == 1

    def test_minimize_improvement(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="minimize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=1.0)
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=0.5, private_score=None, rank=None
        )

        assert feedback.improved is True
        assert orchestrator._last_score == 0.5

    def test_private_score_overrides_public(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5, private_score=0.9)
        # Even though public_score regressed, private_score keeps improving.
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=0.3, private_score=0.95, rank=None
        )

        assert feedback.improved is True
        assert orchestrator._last_score == 0.95


class TestStopConditions:
    """Convergence, budget, and target-rank stop reasons."""

    def test_convergence_after_stagnant_generations(
        self, tracker: ExperimentTracker, candidate: SubmissionCandidate
    ) -> None:
        orchestrator = SubmissionOrchestrator(
            tracker, competition_id="comp-1", metric_direction="maximize", convergence_generations=2
        )

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5)
        _submit(orchestrator, candidate, submission_id="sub-2", public_score=0.5)
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-3", public_score=0.5, private_score=None, rank=None
        )

        assert feedback.convergence_reason == "convergence"

    def test_target_rank_stop_takes_priority(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(
            tracker, competition_id="comp-1", metric_direction="maximize", target_rank=3
        )

        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-1", public_score=0.5, private_score=None, rank=2
        )

        assert feedback.convergence_reason == "target_rank"

    def test_budget_exhausted_stops_loop(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(
            tracker,
            competition_id="comp-1",
            metric_direction="maximize",
            iteration_budget=2,
            convergence_generations=100,
        )

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5)
        feedback = orchestrator.process_submission(
            candidate, submission_id="sub-2", public_score=0.6, private_score=None, rank=None
        )

        assert feedback.convergence_reason == "budget_exhausted"


class TestMutationWeights:
    """Mutation weight updates respond only to a real improvement signal."""

    def test_baseline_does_not_adjust_weights(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")
        before = orchestrator.mutation_weights["point"]

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5)

        assert orchestrator.mutation_weights["point"] == before

    def test_improvement_boosts_weight(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.5)
        before = orchestrator.mutation_weights["point"]
        _submit(orchestrator, candidate, submission_id="sub-2", public_score=0.7)

        assert orchestrator.mutation_weights["point"] > before

    def test_regression_penalises_weight(self, tracker: ExperimentTracker, candidate: SubmissionCandidate) -> None:
        orchestrator = SubmissionOrchestrator(tracker, competition_id="comp-1", metric_direction="maximize")

        _submit(orchestrator, candidate, submission_id="sub-1", public_score=0.7)
        before = orchestrator.mutation_weights["point"]
        _submit(orchestrator, candidate, submission_id="sub-2", public_score=0.5)

        assert orchestrator.mutation_weights["point"] < before
