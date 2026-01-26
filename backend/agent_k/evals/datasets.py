"""Evaluation datasets.

@notice: |
    Evaluation datasets.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evals.datasets
    provides:
        - agent_k.evals.datasets
    pattern: evaluation-datasets

@agent-guidance:
    do:
        - "Use agent_k.evals.datasets as the canonical home for this capability."
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

from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance, LLMJudge, MaxDuration

__all__ = ("discovery_dataset", "evolution_dataset", "load_dataset")


def load_dataset(name: str) -> Dataset:
    """Load dataset from YAML file.

    Args:
            name: Dataset name (without extension).

    Returns:
            Loaded Dataset instance.

    @notice: |
        Load dataset from YAML file.

    @dev: |
        See module for behavior details and invariants.
    """
    path = Path(__file__).parent / f"{name}.yaml"
    return Dataset.from_file(path)


discovery_dataset = Dataset(
    cases=[
        Case(
            name="featured_competition",
            inputs="Find featured competitions with >$10k prize",
            expected_output="Competition selected with prize pool",
            metadata={"category": "discovery"},
        ),
        Case(
            name="research_competition",
            inputs="Find research competitions about NLP",
            expected_output="NLP competition selected",
            metadata={"category": "discovery"},
        ),
    ],
    evaluators=[
        IsInstance("dict"),
        Contains("competition"),
        MaxDuration(seconds=30),
        LLMJudge(rubric="Response contains a valid competition selection with reasoning", model="openai:gpt-4o-mini"),
    ],
)


evolution_dataset = Dataset(
    cases=[
        Case(
            name="simple_optimization",
            inputs="Optimize this solution: def predict(x): return 0",
            expected_output="Improved solution with better logic",
            metadata={"difficulty": "easy"},
        )
    ],
    evaluators=[
        Contains("def predict"),
        LLMJudge(rubric="Solution shows improvement over baseline with valid Python syntax"),
    ],
)
