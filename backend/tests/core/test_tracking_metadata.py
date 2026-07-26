"""Tests for hyperparameter extraction in agent_k.core.tracking.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.tracking import (
    ExperimentRecord,
    _compute_config_signature,
    _extract_hyperparameters,
    _parse_hyperparam_value,
    extract_solution_metadata,
)

__all__ = ()


class TestParseHyperparamValue:
    """Regression tests for `_parse_hyperparam_value`.

    The previous implementation only routed to `float()` when a `.` was
    present, so scientific-notation strings like `1e-3` slipped through the
    integer branch, raised `ValueError`, and were returned as raw strings.
    """

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1e-3", 0.001),
            ("1E-4", 0.0001),
            ("2.5e-2", 0.025),
            ("1e3", 1000.0),
            ("1E+3", 1000.0),
            ("-0.5", -0.5),
            ("-1", -1),
            ("-2.5e-3", -0.0025),
            (".05", 0.05),
            ("0.05", 0.05),
            ("1000", 1000),
        ],
    )
    def test_returns_numeric_types(self, raw: str, expected: float | int) -> None:
        result = _parse_hyperparam_value(raw)
        assert result == expected
        assert type(result) is type(expected)


class TestExtractHyperparameters:
    r"""Regression tests for `_extract_hyperparameters`.

    Scientific notation and negative-signed values are common for
    LightGBM/XGBoost/sklearn regularization terms. The old `[\d\.]+`
    character class silently truncated `learning_rate=1e-3` to just `1`
    and dropped `alpha=-0.5` entirely, corrupting config signatures used
    for duplicate-detection and best-experiment lookup.
    """

    def test_captures_scientific_notation_learning_rate(self) -> None:
        code = "model = lgb.LGBMRegressor(learning_rate=1e-3, n_estimators=1000)"
        assert _extract_hyperparameters(code) == {"learning_rate": 0.001, "n_estimators": 1000}

    def test_captures_uppercase_exponent(self) -> None:
        code = "model = lgb.LGBMRegressor(learning_rate=1E-3, lambda_l1=2.5E-4)"
        assert _extract_hyperparameters(code) == {"learning_rate": 0.001, "lambda_l1": 0.00025}

    def test_captures_negative_regularization(self) -> None:
        code = "model = lgb.LGBMRegressor(learning_rate=0.05, alpha=-0.5, num_leaves=31)"
        result = _extract_hyperparameters(code)
        assert result == {"learning_rate": 0.05, "alpha": -0.5, "num_leaves": 31}

    def test_captures_leading_dot_float(self) -> None:
        code = "model = Ridge(alpha=.001)"
        assert _extract_hyperparameters(code) == {"alpha": 0.001}

    def test_captures_positive_exponent(self) -> None:
        code = "model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=5e-2, max_bin=255)"
        assert _extract_hyperparameters(code) == {"n_estimators": 1000, "learning_rate": 0.05, "max_bin": 255}

    def test_distinct_scientific_values_produce_distinct_signatures(self) -> None:
        """Two configs differing only in exponent must not hash identically."""
        code_a = "model = lgb.LGBMRegressor(learning_rate=1e-3, lambda_l1=1e-5, n_estimators=1000)"
        code_b = "model = lgb.LGBMRegressor(learning_rate=1e-2, lambda_l1=1e-4, n_estimators=1000)"
        record_a = ExperimentRecord(
            competition_id="c",
            phase="prototype",
            model_name="LGBMRegressor",
            hyperparameters=_extract_hyperparameters(code_a),
        )
        record_b = ExperimentRecord(
            competition_id="c",
            phase="prototype",
            model_name="LGBMRegressor",
            hyperparameters=_extract_hyperparameters(code_b),
        )
        assert record_a.hyperparameters != record_b.hyperparameters
        assert _compute_config_signature(record_a) != _compute_config_signature(record_b)


class TestExtractSolutionMetadata:
    """End-to-end check that metadata extraction preserves numeric fidelity."""

    def test_scientific_notation_flows_through(self) -> None:
        code = (
            "import lightgbm as lgb\nmodel = lgb.LGBMRegressor(learning_rate=1e-3, lambda_l1=1e-5, n_estimators=1000)\n"
        )
        metadata = extract_solution_metadata(code)
        assert metadata.model_name == "LGBMRegressor"
        assert metadata.model_family == "lightgbm"
        assert metadata.hyperparameters == {"learning_rate": 0.001, "lambda_l1": 1e-05, "n_estimators": 1000}
