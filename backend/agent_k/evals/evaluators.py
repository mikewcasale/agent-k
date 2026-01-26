"""Custom evaluators for Agent-K.

@notice: |
    Custom evaluators for Agent-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evals.evaluators
    provides:
        - agent_k.evals.evaluators
    pattern: evaluation

@agent-guidance:
    do:
        - "Use agent_k.evals.evaluators as the canonical home for this capability."
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

import ast
from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

__all__ = ("CompetitionSelected", "FitnessImprovement", "ValidPython")


@dataclass
class ValidPython(Evaluator[str, str]):
    """Evaluate if output is valid Python code.

    @pattern:
        name: evaluator
        rationale: "Encapsulates evaluation logic for pydantic-evals."
        violations: "Inline checks are harder to reuse and test."
    """

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, bool]:
        """Check if output parses as valid Python."""
        try:
            ast.parse(ctx.output)
            return {"valid_python": True}
        except SyntaxError:
            return {"valid_python": False}


@dataclass
class FitnessImprovement(Evaluator[str, dict[str, Any]]):
    """Evaluate if fitness improved over baseline.

    @pattern:
        name: evaluator
        rationale: "Encapsulates fitness evaluation logic."
        violations: "Inline comparisons fragment evaluator reuse."
    """

    baseline_fitness: float = 0.0

    def evaluate(self, ctx: EvaluatorContext[str, dict[str, Any]]) -> dict[str, bool | float]:
        """Check fitness improvement."""
        fitness = ctx.output.get("best_fitness", 0.0)
        improvement = fitness - self.baseline_fitness
        return {"fitness_improved": improvement > 0, "improvement_amount": improvement, "final_fitness": fitness}


@dataclass
class CompetitionSelected(Evaluator[str, dict[str, Any]]):
    """Evaluate if a valid competition was selected.

    @pattern:
        name: evaluator
        rationale: "Encapsulates competition selection checks."
        violations: "Repeated checks drift across evals."
    """

    def evaluate(self, ctx: EvaluatorContext[str, dict[str, Any]]) -> dict[str, Any]:
        """Check competition selection output."""
        competition = ctx.output.get("competition")
        return {"has_competition": competition is not None, "competition_type": (competition or {}).get("type")}
