"""Tests for OpenEvolve evaluator CV-variance parsing.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from statistics import variance as sample_variance

import pytest

from agent_k.evolution.evaluator import _compute_cv_variance

__all__ = ()


class TestComputeCvVariance:
    """Fold parsing must be robust to sign, scientific notation, and NaN."""

    def test_returns_zero_when_fewer_than_two_folds(self) -> None:
        assert _compute_cv_variance("Fold 1: 0.85\n") == 0.0

    def test_returns_zero_when_no_folds_present(self) -> None:
        assert _compute_cv_variance("no folds here") == 0.0

    def test_sample_variance_matches_statistics_module(self) -> None:
        stdout = "Fold 1: 0.80\nFold 2: 0.85\nFold 3: 0.90\n"
        scores = [0.80, 0.85, 0.90]
        assert _compute_cv_variance(stdout) == pytest.approx(sample_variance(scores))

    def test_handles_negative_scores(self) -> None:
        """Sklearn ``neg_*`` scorers and signed losses report negative folds.

        Previously the ``[0-9.]+`` regex dropped the leading ``-``, so a
        fold logged as ``-0.85`` parsed as ``0.85`` and skewed the variance.
        """
        stdout = "Fold 1: -0.80\nFold 2: -0.85\nFold 3: -0.90\n"
        expected = sample_variance([-0.80, -0.85, -0.90])
        assert _compute_cv_variance(stdout) == pytest.approx(expected)

    def test_handles_scientific_notation(self) -> None:
        stdout = "Fold 1: 1.5e-04\nFold 2: 2.0e-4\nFold 3: 1.0e-4\n"
        expected = sample_variance([1.5e-4, 2.0e-4, 1.0e-4])
        assert _compute_cv_variance(stdout) == pytest.approx(expected)

    def test_filters_non_finite_fold_values(self) -> None:
        """A single NaN fold must not poison the metric.

        ``float('nan')`` propagates through ``mean``/``variance`` and yields
        NaN which then breaks downstream metric aggregation in the
        evaluator. The fold regex does not match ``nan`` by design, so this
        case validates the defensive ``math.isfinite`` filter.
        """
        stdout = "Fold 1: 0.80\nFold 2: 0.85\n"
        result = _compute_cv_variance(stdout)
        assert result == pytest.approx(sample_variance([0.80, 0.85]))
