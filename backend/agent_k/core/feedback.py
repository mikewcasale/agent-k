"""Submission feedback loop utilities for AGENT-K.

@notice: |
    Submission feedback loop utilities for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.feedback
    provides:
        - agent_k.core.feedback
    pattern: feedback-models

@agent-guidance:
    do:
        - "Use agent_k.core.feedback as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

import logfire

from agent_k.core.tracking import ExperimentTracker, KaggleSubmissionRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_k.core.types import MetricDirection

type MutationWeights = dict[str, float]
"""Weight mapping for mutation tags."""

__all__ = ("SubmissionCandidate", "SubmissionFeedback", "SubmissionOrchestrator")


@dataclass(slots=True)
class SubmissionCandidate:
    """Candidate configuration for submission feedback loops.

    @pattern:
        name: value-object
        rationale: "Dataclass for candidate submission configuration."
    """

    model_config_hash: str
    feature_set_hash: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    mutations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubmissionFeedback:
    """Feedback summary for a processed submission.

    @pattern:
        name: value-object
        rationale: "Dataclass for submission result feedback."
    """

    submission_id: str
    public_score: float | None
    private_score: float | None
    rank: int | None
    improved: bool | None
    convergence_reason: str | None = None


class SubmissionOrchestrator:
    """Manage the submission feedback loop across iterations.

    @pattern:
        name: orchestrator
        rationale: "Coordinates submission feedback loops with tracker."
    """

    _default_mutations: Final[tuple[str, ...]] = (
        "point",
        "structural",
        "hyperparameter",
        "crossover",
        "feature_engineering",
        "feature_selection",
        "loss_function",
    )

    def __init__(
        self,
        tracker: ExperimentTracker,
        *,
        competition_id: str,
        metric_direction: MetricDirection = "maximize",
        iteration_budget: int = 25,
        target_rank: int = 3,
        convergence_generations: int = 4,
        improvement_threshold: float = 1e-4,
    ) -> None:
        self._tracker = tracker
        self._competition_id = competition_id
        self._metric_direction = metric_direction
        self._iteration_budget = iteration_budget
        self._target_rank = target_rank
        self._convergence_generations = convergence_generations
        self._improvement_threshold = improvement_threshold
        self._iteration = 0
        self._mutation_weights: MutationWeights = {name: 1.0 for name in self._default_mutations}
        self._last_score: float | None = None
        self._stagnant_generations = 0

    @property
    def iteration(self) -> int:
        """Return the current iteration counter."""
        return self._iteration

    @property
    def mutation_weights(self) -> MutationWeights:
        """Return the current mutation weight map."""
        return dict(self._mutation_weights)

    def record_submission(
        self,
        candidate: SubmissionCandidate,
        *,
        submission_id: str,
        public_score: float | None,
        private_score: float | None,
        rank: int | None,
    ) -> KaggleSubmissionRecord:
        """Persist a submission and return the stored record."""
        record = KaggleSubmissionRecord(
            competition_id=self._competition_id,
            submission_id=submission_id,
            public_score=public_score,
            private_score=private_score,
            rank=rank,
            model_config_hash=candidate.model_config_hash,
            feature_set_hash=candidate.feature_set_hash,
            hyperparameters=candidate.hyperparameters,
            created_at=datetime.now(UTC),
        )
        return self._tracker.record_submission(record)

    def process_submission(
        self,
        candidate: SubmissionCandidate,
        *,
        submission_id: str,
        public_score: float | None,
        private_score: float | None,
        rank: int | None,
    ) -> SubmissionFeedback:
        """Process a submission outcome and update evolution weights."""
        record = self.record_submission(
            candidate, submission_id=submission_id, public_score=public_score, private_score=private_score, rank=rank
        )
        score = _select_score(record)
        improved = _is_improvement(score, self._last_score, self._metric_direction, self._improvement_threshold)
        self._update_mutation_weights(candidate.mutations, improved)
        self._iteration += 1

        if improved:
            self._last_score = score
            self._stagnant_generations = 0
        else:
            self._stagnant_generations += 1

        stop, reason = self._should_stop(rank=rank)
        if stop:
            logfire.info("submission_loop_stopped", reason=reason, iteration=self._iteration)

        return SubmissionFeedback(
            submission_id=record.submission_id,
            public_score=record.public_score,
            private_score=record.private_score,
            rank=record.rank,
            improved=improved,
            convergence_reason=reason if stop else None,
        )

    def recommend_next_batch(self, *, batch_size: int = 5) -> list[dict[str, Any]]:
        """Recommend the next candidate batch using current mutation weights."""
        ranked = sorted(self._mutation_weights.items(), key=lambda item: item[1], reverse=True)
        picks = ranked[: max(1, min(batch_size, len(ranked)))]
        return [{"mutation": name, "weight": weight, "priority": idx + 1} for idx, (name, weight) in enumerate(picks)]

    def analyze_changes(self, candidates: Iterable[SubmissionCandidate]) -> dict[str, Any]:
        """Analyze which mutation tags are producing improvements."""
        history = self._tracker.get_improvement_history(self._competition_id, direction=self._metric_direction)
        improvements = {item["model_config_hash"] for item in history if item.get("improved")}
        tag_stats: dict[str, int] = {}
        for candidate in candidates:
            if candidate.model_config_hash not in improvements:
                continue
            for mutation in candidate.mutations:
                tag_stats[mutation] = tag_stats.get(mutation, 0) + 1
        return {"improved_configs": len(improvements), "mutation_hits": tag_stats}

    def _should_stop(self, *, rank: int | None) -> tuple[bool, str | None]:
        if rank is not None and rank <= self._target_rank:
            return True, "target_rank"
        if self._iteration >= self._iteration_budget:
            return True, "budget_exhausted"
        if self._stagnant_generations >= self._convergence_generations:
            return True, "convergence"
        return False, None

    def _update_mutation_weights(self, mutations: list[str], improved: bool | None) -> None:
        if improved is None or not mutations:
            return
        factor = 1.15 if improved else 0.9
        for mutation in mutations:
            if mutation not in self._mutation_weights:
                self._mutation_weights[mutation] = 1.0
            self._mutation_weights[mutation] *= factor


def _select_score(record: KaggleSubmissionRecord) -> float | None:
    return record.private_score if record.private_score is not None else record.public_score


def _is_improvement(
    score: float | None, previous: float | None, direction: MetricDirection, threshold: float
) -> bool | None:
    if score is None or previous is None:
        return None
    delta = score - previous
    if direction == "maximize":
        return delta > threshold
    return delta < -threshold
