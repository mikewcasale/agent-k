"""Tests for OpenEvolve evaluator CV-variance parsing.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.evolution.evaluator import _compute_cv_variance

__all__ = ()


def test_cv_variance_zero_when_fewer_than_two_folds() -> None:
    assert _compute_cv_variance("") == 0.0
    assert _compute_cv_variance("Fold 1: 0.5") == 0.0


def test_cv_variance_matches_decimal_folds() -> None:
    stdout = "Fold 1: 0.80\nFold 2: 0.82\nFold 3: 0.78\n"
    result = _compute_cv_variance(stdout)
    scores = [0.80, 0.82, 0.78]
    mean = sum(scores) / len(scores)
    expected = sum((s - mean) ** 2 for s in scores) / len(scores)
    assert result == pytest.approx(expected)


def test_cv_variance_supports_scientific_notation() -> None:
    stdout = "Fold 1: 1.0e-3\nFold 2: 2.0e-3\n"
    result = _compute_cv_variance(stdout)
    scores = [1.0e-3, 2.0e-3]
    mean = sum(scores) / len(scores)
    expected = sum((s - mean) ** 2 for s in scores) / len(scores)
    assert result == pytest.approx(expected)
    assert result > 0.0


def test_cv_variance_supports_negative_fold_scores() -> None:
    stdout = "Fold 1: -0.5\nFold 2: -0.7\nFold 3: -0.6\n"
    result = _compute_cv_variance(stdout)
    scores = [-0.5, -0.7, -0.6]
    mean = sum(scores) / len(scores)
    expected = sum((s - mean) ** 2 for s in scores) / len(scores)
    assert result == pytest.approx(expected)
    assert result > 0.0


def test_cv_variance_skips_non_finite_folds() -> None:
    stdout = "Fold 1: nan\nFold 2: 0.5\nFold 3: 0.6\n"
    result = _compute_cv_variance(stdout)
    # Only the two finite scores contribute.
    scores = [0.5, 0.6]
    mean = sum(scores) / len(scores)
    expected = sum((s - mean) ** 2 for s in scores) / len(scores)
    assert result == pytest.approx(expected)


def test_cv_variance_mixed_notation() -> None:
    stdout = "Fold 1: 0.9\nFold 2: 9e-1\nFold 3: .90\n"
    result = _compute_cv_variance(stdout)
    assert result == pytest.approx(0.0)
