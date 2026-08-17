"""Tests for canonical score/fitness conversions.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.fitness import FITNESS_FLOOR, coerce_metric_direction, fitness_to_score, score_to_fitness

__all__ = ()


class TestCoerceMetricDirection:
    """Tests for direction normalization."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("minimize", "minimize"), ("MINIMIZE", "minimize"), (" minimize ", "minimize"), ("maximize", "maximize")],
    )
    def test_known_directions(self, raw: str, expected: str) -> None:
        """Known directions normalize to their canonical spelling."""
        assert coerce_metric_direction(raw) == expected

    def test_unknown_direction_defaults_to_maximize(self) -> None:
        """Unrecognized directions default to maximize."""
        assert coerce_metric_direction("descending") == "maximize"


class TestScoreToFitness:
    """Tests for score to fitness conversion."""

    @pytest.mark.parametrize(
        ("score", "direction", "expected"),
        [(0.25, "maximize", 0.25), (-1.0, "maximize", 0.0), (3.0, "minimize", 0.25), (-2.0, "minimize", 1.0)],
    )
    def test_conversion(self, score: float, direction: str, expected: float) -> None:
        """Fitness reflects metric direction."""
        assert score_to_fitness(score, coerce_metric_direction(direction)) == pytest.approx(expected)

    def test_missing_score_reports_floor(self) -> None:
        """A missing score reports the failure floor."""
        assert score_to_fitness(None, "minimize") == FITNESS_FLOOR
        assert score_to_fitness(None, "maximize") == FITNESS_FLOOR

    def test_minimize_fitness_is_strictly_above_floor(self) -> None:
        """Any real minimize score outranks a failed evaluation."""
        for score in (0.0, 0.5, 12.5, 1_000_000.0):
            assert score_to_fitness(score, "minimize") > FITNESS_FLOOR

    def test_minimize_fitness_is_monotonically_decreasing(self) -> None:
        """Lower minimize scores yield higher fitness."""
        scores = [0.1, 0.5, 1.0, 4.0, 20.0]
        fitnesses = [score_to_fitness(score, "minimize") for score in scores]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_maximize_fitness_is_monotonically_increasing(self) -> None:
        """Higher maximize scores yield higher fitness."""
        scores = [0.1, 0.5, 0.8, 0.95]
        fitnesses = [score_to_fitness(score, "maximize") for score in scores]
        assert fitnesses == sorted(fitnesses)


class TestFitnessToScore:
    """Tests for the inverse conversion."""

    @pytest.mark.parametrize("score", [0.0, 0.25, 1.0, 7.5, 133.25])
    def test_minimize_round_trip(self, score: float) -> None:
        """Minimize scores survive a fitness round trip."""
        fitness = score_to_fitness(score, "minimize")
        assert fitness_to_score(fitness, "minimize") == pytest.approx(score)

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.9, 3.0])
    def test_maximize_round_trip(self, score: float) -> None:
        """Maximize scores survive a fitness round trip."""
        fitness = score_to_fitness(score, "maximize")
        assert fitness_to_score(fitness, "maximize") == pytest.approx(score)

    def test_none_fitness_returns_none(self) -> None:
        """A missing fitness has no recoverable score."""
        assert fitness_to_score(None, "minimize") is None
        assert fitness_to_score(None, "maximize") is None

    def test_floor_fitness_has_no_minimize_score(self) -> None:
        """The failure floor does not decode to a minimize score."""
        assert fitness_to_score(FITNESS_FLOOR, "minimize") is None
        assert fitness_to_score(-1.0, "minimize") is None
