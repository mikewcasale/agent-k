"""Tests for OpenEvolve evaluator fitness-from-score transform.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.evolution.evaluator import _failure_metrics, _fitness_from_score

__all__ = ()


class TestFitnessFromScore:
    """Verify fitness transform ranks success above failure for both directions."""

    @pytest.mark.parametrize(
        ("cv_score", "direction", "expected"),
        [
            (0.85, "maximize", 0.85),
            (0.0, "maximize", 0.0),
            (-0.1, "maximize", 0.0),
            (0.0, "minimize", 1.0),
            (0.5, "minimize", pytest.approx(1.0 / 1.5)),
            (3.0, "minimize", 0.25),
            (-2.0, "minimize", 1.0),
        ],
    )
    def test_known_values(self, cv_score: float, direction: str, expected: float) -> None:
        assert _fitness_from_score(cv_score, direction) == expected

    @pytest.mark.parametrize("direction", ["maximize", "minimize"])
    def test_failure_returns_zero(self, direction: str) -> None:
        assert _fitness_from_score(None, direction) == 0.0

    @pytest.mark.parametrize(
        ("cv_score", "direction"), [(0.85, "maximize"), (0.5, "minimize"), (3.0, "minimize"), (1e-6, "minimize")]
    )
    def test_success_outranks_failure(self, cv_score: float, direction: str) -> None:
        success = _fitness_from_score(cv_score, direction)
        failure = _failure_metrics()["fitness"]
        assert success > failure

    def test_minimize_lower_score_higher_fitness(self) -> None:
        better = _fitness_from_score(0.1, "minimize")
        worse = _fitness_from_score(1.0, "minimize")
        assert better > worse

    def test_maximize_higher_score_higher_fitness(self) -> None:
        better = _fitness_from_score(0.9, "maximize")
        worse = _fitness_from_score(0.1, "maximize")
        assert better > worse

    def test_minimize_fitness_bounded_unit_interval(self) -> None:
        for score in (0.0, 0.001, 1.0, 1_000.0, 1e9):
            value = _fitness_from_score(score, "minimize")
            assert 0.0 < value <= 1.0


class TestEvolverAgentInverse:
    """The agent's `_score_from_fitness` must invert the evaluator transform."""

    @pytest.fixture(scope="class")
    def evolver(self):  # type: ignore[no-untyped-def]
        try:
            from agent_k.agents.evolver import evolver_agent_instance
        except TypeError as exc:
            if "MCPServerTool" in str(exc):
                pytest.skip(f"MCPServerTool API issue: {exc}")
            raise
        return evolver_agent_instance

    @pytest.mark.parametrize(
        ("cv_score", "direction"),
        [(0.85, "maximize"), (0.1, "maximize"), (0.5, "minimize"), (3.0, "minimize"), (0.001, "minimize")],
    )
    def test_round_trip(self, evolver, cv_score: float, direction: str) -> None:  # type: ignore[no-untyped-def]
        fitness = _fitness_from_score(cv_score, direction)
        recovered = evolver._score_from_fitness(fitness, direction)
        assert recovered == pytest.approx(cv_score, rel=1e-9, abs=1e-9)
