"""Custom evaluators for Agent-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import ast
from dataclasses import dataclass
from typing import Any

# Third-party (alphabetical)
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

__all__ = ("CompetitionSelected", "FitnessImprovement", "ValidPython")


@dataclass
class ValidPython(Evaluator[str, str]):
    """Evaluate if output is valid Python code."""

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, bool]:
        """Check if output parses as valid Python."""
        try:
            ast.parse(ctx.output)
            return {"valid_python": True}
        except SyntaxError:
            return {"valid_python": False}


@dataclass
class FitnessImprovement(Evaluator[str, dict[str, Any]]):
    """Evaluate if fitness improved over baseline."""

    baseline_fitness: float = 0.0

    def evaluate(self, ctx: EvaluatorContext[str, dict[str, Any]]) -> dict[str, bool | float]:
        """Check fitness improvement."""
        fitness = ctx.output.get("best_fitness", 0.0)
        improvement = fitness - self.baseline_fitness
        return {
            "fitness_improved": improvement > 0,
            "improvement_amount": improvement,
            "final_fitness": fitness,
        }


@dataclass
class CompetitionSelected(Evaluator[str, dict[str, Any]]):
    """Evaluate if a valid competition was selected."""

    def evaluate(self, ctx: EvaluatorContext[str, dict[str, Any]]) -> dict[str, Any]:
        """Check competition selection output."""
        competition = ctx.output.get("competition")
        return {
            "has_competition": competition is not None,
            "competition_type": (competition or {}).get("type"),
        }
