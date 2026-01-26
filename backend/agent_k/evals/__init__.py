"""Evaluation utilities for Agent-K.

@notice: |
    Evaluation utilities for Agent-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evals
    provides:
        - agent_k.evals
    pattern: evals-package

@agent-guidance:
    do:
        - "Use agent_k.evals as the canonical home for this capability."
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

from .datasets import discovery_dataset, evolution_dataset, load_dataset
from .evaluators import CompetitionSelected, FitnessImprovement, ValidPython

__all__ = (
    "CompetitionSelected",
    "FitnessImprovement",
    "ValidPython",
    "discovery_dataset",
    "evolution_dataset",
    "load_dataset",
)
