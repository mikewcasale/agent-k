"""Pydantic Evals integration for reproducible trusted candidate scoring.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence

from pydantic_evals import Case, Dataset
from pydantic_evals.reporting import EvaluationReport

from .evaluator import EvaluationRequest, EvaluationResult, evaluate_candidate

__all__ = (
    "build_evaluation_dataset",
    "run_evaluation_dataset",
)


def build_evaluation_dataset(
    requests: Sequence[EvaluationRequest],
    *,
    name: str = "trusted-candidate-evaluations",
) -> Dataset[EvaluationRequest, EvaluationResult, dict[str, str]]:
    """Build typed Pydantic Evals cases from trusted evaluation requests."""
    cases = [
        Case[EvaluationRequest, EvaluationResult, dict[str, str]](
            name=request.candidate_result.candidate_id,
            inputs=request,
            metadata={
                "metric_id": request.metric_id.value,
                "candidate_id": request.candidate_result.candidate_id,
            },
        )
        for request in requests
    ]
    return Dataset(name=name, cases=cases)


def run_evaluation_dataset(
    dataset: Dataset[EvaluationRequest, EvaluationResult, dict[str, str]],
    *,
    max_concurrency: int = 1,
) -> EvaluationReport[EvaluationRequest, EvaluationResult, dict[str, str]]:
    """Run trusted host-side scoring through Pydantic Evals instrumentation."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    return dataset.evaluate_sync(
        evaluate_candidate,
        max_concurrency=max_concurrency,
        progress=False,
        metadata={"trust_boundary": "host-only-hidden-labels"},
    )
