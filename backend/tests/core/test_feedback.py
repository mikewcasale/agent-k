"""Tests for the submission feedback orchestrator.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

from agent_k.core.feedback import SubmissionCandidate, SubmissionOrchestrator
from agent_k.core.tracking import ExperimentTracker

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


_CANDIDATE = SubmissionCandidate(
    model_config_hash="cfg-hash",
    feature_set_hash="feat-hash",
    hyperparameters={"n_estimators": 100},
    mutations=["hyperparameter"],
)


def _orchestrator(
    tmp_path: Path,
    *,
    metric_direction: str = "maximize",
    convergence_generations: int = 4,
    improvement_threshold: float = 1e-4,
    iteration_budget: int = 25,
    target_rank: int = 3,
) -> SubmissionOrchestrator:
    tracker = ExperimentTracker(db_path=tmp_path / "experiments.sqlite")
    return SubmissionOrchestrator(
        tracker,
        competition_id="comp-1",
        metric_direction=metric_direction,
        iteration_budget=iteration_budget,
        target_rank=target_rank,
        convergence_generations=convergence_generations,
        improvement_threshold=improvement_threshold,
    )


def test_first_submission_seeds_baseline_without_incrementing_stagnation(tmp_path: Path) -> None:
    """The first valid score must seed `_last_score` and leave stagnation at zero.

    Previously the first submission always fell through to the `else` branch and
    incremented `_stagnant_generations`, so the loop tripped the "convergence"
    stop after `convergence_generations` submissions regardless of whether the
    scores were actually improving.
    """
    orchestrator = _orchestrator(tmp_path)

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-1", public_score=0.85, private_score=None, rank=None
    )

    assert feedback.improved is None
    assert feedback.convergence_reason is None
    assert orchestrator._last_score == 0.85
    assert orchestrator._stagnant_generations == 0


def test_missing_score_does_not_count_as_stagnation(tmp_path: Path) -> None:
    """Submissions without any usable score must not touch the stagnation counter.

    Neither improvement nor regression can be judged when both public and
    private scores are missing.
    """
    orchestrator = _orchestrator(tmp_path)
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-1", public_score=0.85, private_score=None, rank=None)
    assert orchestrator._stagnant_generations == 0

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-2", public_score=None, private_score=None, rank=None
    )

    assert feedback.improved is None
    assert orchestrator._stagnant_generations == 0
    assert orchestrator._last_score == 0.85


def test_improvement_resets_stagnation_and_updates_baseline(tmp_path: Path) -> None:
    """An actual improvement resets stagnation and updates the baseline."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-1", public_score=0.80, private_score=None, rank=None)
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-2", public_score=0.79, private_score=None, rank=None)
    assert orchestrator._stagnant_generations == 1

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-3", public_score=0.90, private_score=None, rank=None
    )

    assert feedback.improved is True
    assert orchestrator._stagnant_generations == 0
    assert orchestrator._last_score == 0.90


def test_regression_increments_stagnation(tmp_path: Path) -> None:
    """A regression (below-threshold delta) increments stagnation and leaves the baseline."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-1", public_score=0.85, private_score=None, rank=None)

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-2", public_score=0.80, private_score=None, rank=None
    )

    assert feedback.improved is False
    assert orchestrator._stagnant_generations == 1
    assert orchestrator._last_score == 0.85


def test_minimize_direction_detects_improvement(tmp_path: Path) -> None:
    """For minimize metrics, lower scores are improvements."""
    orchestrator = _orchestrator(tmp_path, metric_direction="minimize")
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-1", public_score=0.30, private_score=None, rank=None)

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-2", public_score=0.20, private_score=None, rank=None
    )

    assert feedback.improved is True
    assert orchestrator._last_score == 0.20


def test_steadily_improving_run_does_not_trigger_convergence_stop(tmp_path: Path) -> None:
    """Strictly improving runs must never report a `convergence` stop.

    Regression guard: without the baseline-seeding fix the loop counted the
    first submission as stagnation and tripped `convergence` after
    `convergence_generations` iterations regardless of actual improvement.
    """
    orchestrator = _orchestrator(tmp_path, convergence_generations=3, iteration_budget=100)

    reasons: list[str | None] = []
    for idx, score in enumerate([0.60, 0.65, 0.70, 0.75, 0.80], start=1):
        feedback = orchestrator.process_submission(
            _CANDIDATE, submission_id=f"sub-{idx}", public_score=score, private_score=None, rank=None
        )
        reasons.append(feedback.convergence_reason)

    assert reasons == [None, None, None, None, None]
    assert orchestrator._stagnant_generations == 0
    assert orchestrator._last_score == 0.80


def test_convergence_stop_fires_after_true_stagnation(tmp_path: Path) -> None:
    """After enough non-improving submissions, the loop should stop with `convergence`."""
    orchestrator = _orchestrator(tmp_path, convergence_generations=2, iteration_budget=100)
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-1", public_score=0.85, private_score=None, rank=None)
    first = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-2", public_score=0.80, private_score=None, rank=None
    )
    assert first.convergence_reason is None

    second = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-3", public_score=0.79, private_score=None, rank=None
    )

    assert second.convergence_reason == "convergence"
    assert orchestrator._stagnant_generations == 2


def test_target_rank_stop_short_circuits_convergence_check(tmp_path: Path) -> None:
    """Hitting the target rank stops the loop even before convergence would fire."""
    orchestrator = _orchestrator(tmp_path, target_rank=3, convergence_generations=10)

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-1", public_score=0.85, private_score=None, rank=2
    )

    assert feedback.convergence_reason == "target_rank"


def test_private_score_wins_over_public_for_comparison(tmp_path: Path) -> None:
    """`_select_score` prefers private scores; ensure orchestrator honors that."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator.process_submission(_CANDIDATE, submission_id="sub-1", public_score=0.5, private_score=0.9, rank=None)
    assert orchestrator._last_score == 0.9

    feedback = orchestrator.process_submission(
        _CANDIDATE, submission_id="sub-2", public_score=0.99, private_score=0.85, rank=None
    )

    assert feedback.improved is False
    assert orchestrator._last_score == 0.9
