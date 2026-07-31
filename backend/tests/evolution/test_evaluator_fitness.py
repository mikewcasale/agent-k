"""Tests for OpenEvolve evaluator fitness mapping.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math

import pytest

from agent_k.evolution.evaluator import _fitness_from_score

__all__ = ()


class TestFitnessFromScore:
    """Regression tests for ``_fitness_from_score``.

    Ensures the evaluator's fitness mapping matches the canonical
    formula used in ``agents/evolver.py`` and ``mission/nodes.py`` so
    OpenEvolve's ``combined_score`` stays comparable across the stack
    and failed evaluations never rank above valid solutions.
    """

    @pytest.mark.parametrize(
        ("cv_score", "direction", "expected"),
        [
            (0.0, "minimize", 1.0),
            (3.0, "minimize", 0.25),
            (1.0, "minimize", 0.5),
            (-2.0, "minimize", 1.0),
            (0.25, "maximize", 0.25),
            (0.85, "maximize", 0.85),
            (-1.0, "maximize", 0.0),
        ],
    )
    def test_canonical_mapping(self, cv_score: float, direction: str, expected: float) -> None:
        """Fitness mapping matches the shared formula used elsewhere."""
        assert _fitness_from_score(cv_score, direction) == pytest.approx(expected)

    def test_none_cv_score_is_worst_for_minimize(self) -> None:
        """Missing cv_score must not beat any valid minimize solution."""
        failure = _fitness_from_score(None, "minimize")
        # Any valid RMSE-like score, no matter how large, still produces
        # positive fitness under the canonical formula.
        worst_valid = _fitness_from_score(1e9, "minimize")

        assert failure == 0.0
        assert worst_valid > failure

    def test_none_cv_score_is_worst_for_maximize(self) -> None:
        """Missing cv_score must not beat any valid maximize solution."""
        failure = _fitness_from_score(None, "maximize")
        best_valid = _fitness_from_score(1.0, "maximize")

        assert failure == 0.0
        assert best_valid > failure

    def test_lower_minimize_score_yields_higher_fitness(self) -> None:
        """Order-preserving for the minimize direction (lower is better)."""
        assert _fitness_from_score(0.1, "minimize") > _fitness_from_score(0.5, "minimize")
        assert _fitness_from_score(0.5, "minimize") > _fitness_from_score(5.0, "minimize")

    def test_higher_maximize_score_yields_higher_fitness(self) -> None:
        """Order-preserving for the maximize direction (higher is better)."""
        assert _fitness_from_score(0.9, "maximize") > _fitness_from_score(0.5, "maximize")

    def test_minimize_fitness_is_bounded(self) -> None:
        """Minimize fitness must stay within (0, 1] for all non-negative scores."""
        for score in (0.0, 0.5, 1.0, 10.0, 1e6):
            fitness = _fitness_from_score(score, "minimize")
            assert 0.0 < fitness <= 1.0
            assert math.isfinite(fitness)
