"""Evaluation datasets.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from pathlib import Path

# Third-party (alphabetical)
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance, LLMJudge, MaxDuration

__all__ = ("discovery_dataset", "evolution_dataset", "load_dataset")


def load_dataset(name: str) -> Dataset:
    """Load dataset from YAML file.

    Args:
        name: Dataset name (without extension).

    Returns:
        Loaded Dataset instance.
    """
    path = Path(__file__).parent / f"{name}.yaml"
    return Dataset.from_file(path)


# =============================================================================
# Dataset Definitions
# =============================================================================
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
        LLMJudge(
            rubric="Response contains a valid competition selection with reasoning",
            model="openai:gpt-4o-mini",
        ),
    ],
)


evolution_dataset = Dataset(
    cases=[
        Case(
            name="simple_optimization",
            inputs="Optimize this solution: def predict(x): return 0",
            expected_output="Improved solution with better logic",
            metadata={"difficulty": "easy"},
        ),
    ],
    evaluators=[
        Contains("def predict"),
        LLMJudge(
            rubric="Solution shows improvement over baseline with valid Python syntax",
        ),
    ],
)
