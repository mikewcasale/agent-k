"""Tests for Evolver convergence and adaptive-magnitude progress measurement.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest

__all__ = ()

try:
    from agent_k.agents.evolver import (
        _ADAPTIVE_MAGNITUDE_DEFAULT,
        _ADAPTIVE_MAGNITUDE_EXPLOIT,
        _ADAPTIVE_MAGNITUDE_EXPLORE,
        _ADAPTIVE_MAGNITUDE_STEADY,
        _adaptive_mutation_magnitude,
        _best_fitness_series,
        _window_progress,
    )
except TypeError as exc:
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise


@dataclass(frozen=True, slots=True)
class _GenerationMetrics:
    """Attribute-style generation record, as mission state supplies."""

    best_fitness: float


def _history(*values: float) -> list[dict[str, float]]:
    return [{"generation": idx, "best_fitness": value} for idx, value in enumerate(values, start=1)]


class TestBestFitnessSeries:
    """Tests for best-fitness extraction from generation history."""

    def test_reads_dict_entries_in_order(self) -> None:
        """Dict records should keep their recorded order."""
        assert _best_fitness_series(_history(0.1, 0.2, 0.3)) == [0.1, 0.2, 0.3]

    def test_reads_attribute_entries(self) -> None:
        """Attribute-style records should be supported too."""
        history = [_GenerationMetrics(0.4), _GenerationMetrics(0.5)]

        assert _best_fitness_series(history) == [0.4, 0.5]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None, "0.5", True])
    def test_skips_unusable_values(self, value: object) -> None:
        """Non-finite, non-numeric, and boolean values should be dropped."""
        history = [{"best_fitness": 0.2}, {"best_fitness": value}, {"best_fitness": 0.3}]

        assert _best_fitness_series(history) == [0.2, 0.3]

    def test_skips_missing_key(self) -> None:
        """Records without a best_fitness key should be dropped."""
        assert _best_fitness_series([{"generation": 1}, {"best_fitness": 0.7}]) == [0.7]


class TestWindowProgress:
    """Tests for trailing-window improvement measurement."""

    def test_empty_series(self) -> None:
        """An empty series has no progress and no best."""
        assert _window_progress([], 5) == (0.0, 0.0)

    def test_monotonic_window_measures_full_gain(self) -> None:
        """A rising window spanning all history gains from its first value."""
        improvement, best = _window_progress([0.10, 0.20, 0.30], 3)

        assert improvement == pytest.approx(0.20)
        assert best == pytest.approx(0.30)

    def test_declining_window_is_not_progress(self) -> None:
        """A declining window must report zero improvement, not its spread."""
        improvement, best = _window_progress([0.90, 0.80, 0.70, 0.60, 0.50], 5)

        assert improvement == pytest.approx(0.0)
        assert best == pytest.approx(0.90)

    def test_spike_before_window_then_plateau_is_not_progress(self) -> None:
        """An early spike followed by a plateau must not read as progress."""
        improvement, best = _window_progress([0.50, 0.90, 0.50, 0.50, 0.50, 0.50], 4)

        assert improvement == pytest.approx(0.0)
        assert best == pytest.approx(0.90)

    def test_flatline_after_a_single_win_is_not_progress(self) -> None:
        """A win when the window opened does not count as progress inside it."""
        improvement, best = _window_progress([0.10, 0.60, 0.60, 0.60, 0.60, 0.60], 5)

        assert improvement == pytest.approx(0.0)
        assert best == pytest.approx(0.60)

    def test_full_span_window_references_the_first_generation(self) -> None:
        """Without history before the window, the gain over its start still counts."""
        improvement, best = _window_progress([0.50, 0.90, 0.50, 0.50, 0.50], 5)

        assert improvement == pytest.approx(0.40)
        assert best == pytest.approx(0.90)

    def test_improvement_is_measured_against_history_before_window(self) -> None:
        """A new best inside the window counts only the gain over the prior best."""
        improvement, best = _window_progress([0.10, 0.60, 0.60, 0.60, 0.60, 0.65], 3)

        assert improvement == pytest.approx(0.05)
        assert best == pytest.approx(0.65)

    def test_no_new_best_inside_window(self) -> None:
        """A window that never beats the prior best reports zero improvement."""
        improvement, best = _window_progress([0.10, 0.80, 0.40, 0.40, 0.40], 3)

        assert improvement == pytest.approx(0.0)
        assert best == pytest.approx(0.80)

    def test_window_larger_than_series_is_clamped(self) -> None:
        """An oversized window falls back to the whole series."""
        improvement, best = _window_progress([0.20, 0.50], 10)

        assert improvement == pytest.approx(0.30)
        assert best == pytest.approx(0.50)

    def test_single_entry_series(self) -> None:
        """One recorded generation cannot show improvement."""
        improvement, best = _window_progress([0.42], 5)

        assert improvement == pytest.approx(0.0)
        assert best == pytest.approx(0.42)


class TestAdaptiveMutationMagnitude:
    """Tests for progress-driven mutation magnitude selection."""

    def test_defaults_without_enough_history(self) -> None:
        """A single generation keeps the default magnitude."""
        assert _adaptive_mutation_magnitude(_history(0.5)) == pytest.approx(_ADAPTIVE_MAGNITUDE_DEFAULT)

    def test_strong_gain_exploits(self) -> None:
        """A large recent gain narrows the mutation magnitude."""
        history = _history(0.10, 0.20, 0.30, 0.40, 0.50)

        assert _adaptive_mutation_magnitude(history) == pytest.approx(_ADAPTIVE_MAGNITUDE_EXPLOIT)

    def test_small_gain_stays_steady(self) -> None:
        """A modest gain keeps the steady magnitude."""
        history = _history(0.500, 0.505, 0.510, 0.515, 0.520)

        assert _adaptive_mutation_magnitude(history) == pytest.approx(_ADAPTIVE_MAGNITUDE_STEADY)

    def test_plateau_explores(self) -> None:
        """A plateau widens the mutation magnitude."""
        history = _history(0.50, 0.50, 0.50, 0.50, 0.50)

        assert _adaptive_mutation_magnitude(history) == pytest.approx(_ADAPTIVE_MAGNITUDE_EXPLORE)

    def test_regression_explores_instead_of_exploiting(self) -> None:
        """Losing ground must widen, not narrow, the mutation magnitude."""
        history = _history(0.90, 0.80, 0.70, 0.60, 0.50)

        assert _adaptive_mutation_magnitude(history) == pytest.approx(_ADAPTIVE_MAGNITUDE_EXPLORE)

    def test_only_recent_window_is_considered(self) -> None:
        """Generations older than the window must not drive the magnitude."""
        history = _history(0.01, 0.99, 0.50, 0.50, 0.50, 0.50, 0.50)

        assert _adaptive_mutation_magnitude(history) == pytest.approx(_ADAPTIVE_MAGNITUDE_EXPLORE)
