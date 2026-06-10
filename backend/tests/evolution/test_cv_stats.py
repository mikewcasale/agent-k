"""Tests for CV fold-score parsing and stability metrics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math

import pytest

from agent_k.evolution.evaluator import _compute_cv_stats, _parse_fold_scores

__all__ = ()


def test_parse_fold_scores_basic_decimals() -> None:
    stdout = "Fold 1: 0.85\nFold 2: 0.83\nFold 3: 0.86\n"
    assert _parse_fold_scores(stdout) == [0.85, 0.83, 0.86]


def test_parse_fold_scores_handles_negative_values() -> None:
    stdout = "Fold 1: -0.512\nFold 2: -0.498\nFold 3: -0.503\n"
    assert _parse_fold_scores(stdout) == [-0.512, -0.498, -0.503]


def test_parse_fold_scores_handles_scientific_notation() -> None:
    stdout = "Fold 1: 1.5e-3\nFold 2: 2.0E-3\nFold 3: 1.75e-3\n"
    assert _parse_fold_scores(stdout) == [0.0015, 0.002, 0.00175]


def test_parse_fold_scores_handles_explicit_positive_sign() -> None:
    stdout = "Fold 1: +0.5\nFold 2: 0.6\n"
    assert _parse_fold_scores(stdout) == [0.5, 0.6]


def test_parse_fold_scores_handles_multi_digit_fold_numbers() -> None:
    stdout = "Fold 9: 0.5\nFold 10: 0.6\nFold 11: 0.55\n"
    assert _parse_fold_scores(stdout) == [0.5, 0.6, 0.55]


def test_parse_fold_scores_handles_tab_separator() -> None:
    stdout = "Fold 1\t0.5\nFold 2\t0.6\n"
    assert _parse_fold_scores(stdout) == [0.5, 0.6]


def test_parse_fold_scores_handles_leading_dot_format() -> None:
    stdout = "Fold 1: .5\nFold 2: .6\n"
    assert _parse_fold_scores(stdout) == [0.5, 0.6]


def test_parse_fold_scores_ignores_unrelated_lines() -> None:
    stdout = "Training...\nFold 1: 0.85\nSome noise here\nFold 2: 0.82\nDone.\n"
    assert _parse_fold_scores(stdout) == [0.85, 0.82]


def test_parse_fold_scores_returns_empty_for_no_matches() -> None:
    stdout = "Training complete.\nNo fold output here.\n"
    assert _parse_fold_scores(stdout) == []


def test_compute_cv_stats_returns_zero_for_no_folds() -> None:
    assert _compute_cv_stats("no folds here") == (0.0, 0.0)


def test_compute_cv_stats_returns_zero_for_single_fold() -> None:
    assert _compute_cv_stats("Fold 1: 0.85") == (0.0, 0.0)


def test_compute_cv_stats_computes_variance_and_stddev() -> None:
    stdout = "Fold 1: 0.80\nFold 2: 0.90\n"
    variance, stddev = _compute_cv_stats(stdout)
    expected_variance = ((0.80 - 0.85) ** 2 + (0.90 - 0.85) ** 2) / 2
    assert variance == pytest.approx(expected_variance)
    assert stddev == pytest.approx(math.sqrt(expected_variance))


def test_compute_cv_stats_handles_negative_scores() -> None:
    stdout = "Fold 1: -0.5\nFold 2: -0.4\nFold 3: -0.6\n"
    variance, stddev = _compute_cv_stats(stdout)
    mean = -0.5
    expected_variance = sum((s - mean) ** 2 for s in (-0.5, -0.4, -0.6)) / 3
    assert variance == pytest.approx(expected_variance)
    assert stddev == pytest.approx(math.sqrt(expected_variance))


def test_compute_cv_stats_handles_scientific_notation() -> None:
    stdout = "Fold 1: 1.0e-3\nFold 2: 3.0e-3\n"
    variance, stddev = _compute_cv_stats(stdout)
    expected_variance = ((1e-3 - 2e-3) ** 2 + (3e-3 - 2e-3) ** 2) / 2
    assert variance == pytest.approx(expected_variance)
    assert stddev == pytest.approx(math.sqrt(expected_variance))


def test_compute_cv_stats_identical_folds_gives_zero_spread() -> None:
    stdout = "Fold 1: 0.5\nFold 2: 0.5\nFold 3: 0.5\n"
    assert _compute_cv_stats(stdout) == (0.0, 0.0)


def test_compute_cv_stats_stddev_in_score_units() -> None:
    stdout = "Fold 1: 1.0\nFold 2: 2.0\nFold 3: 3.0\n"
    variance, stddev = _compute_cv_stats(stdout)
    assert variance == pytest.approx(2.0 / 3)
    assert stddev == pytest.approx(math.sqrt(2.0 / 3))
