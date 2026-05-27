"""Tests for experiment tracking metadata extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.tracking import _parse_hyperparam_value, extract_solution_metadata

__all__ = ()


class TestParseHyperparamValue:
    """Tests for the hyperparameter value parser."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("100", 100),
            ("1.5", 1.5),
            ("0.05", 0.05),
            (".05", 0.05),
            ("'distance'", "distance"),
            ('"euclidean"', "euclidean"),
            ("None", None),
            ("null", None),
            # Scientific notation: previously ``int("1e-3")`` raised and the raw
            # string leaked into the hyperparameter dict.
            ("1e-3", 1e-3),
            ("1E-3", 1e-3),
            ("5e-2", 0.05),
            ("1.5e-04", 1.5e-4),
            ("2e+10", 2e10),
            ("+1e-3", 1e-3),
            ("-3.5e-2", -0.035),
        ],
    )
    def test_parse_hyperparam_value(self, raw: str, expected: object) -> None:
        assert _parse_hyperparam_value(raw) == expected


class TestExtractSolutionMetadataScientificNotation:
    """Hyperparameter regex must capture scientific notation in full."""

    def test_learning_rate_scientific_notation_lowercase(self) -> None:
        code = "model = LGBMRegressor(n_estimators=500, learning_rate=1e-3)"
        metadata = extract_solution_metadata(code)
        assert metadata.hyperparameters["learning_rate"] == pytest.approx(1e-3)
        assert metadata.hyperparameters["n_estimators"] == 500

    def test_learning_rate_scientific_notation_uppercase(self) -> None:
        code = "model = LGBMRegressor(learning_rate=5.0E-2)"
        metadata = extract_solution_metadata(code)
        assert metadata.hyperparameters["learning_rate"] == pytest.approx(0.05)

    def test_distinct_scientific_values_do_not_collapse(self) -> None:
        r"""Two distinct exponents must yield distinct parsed values.

        Previously the ``[\d\.]+`` regex captured only the mantissa, so
        ``1e-3`` and ``1e-5`` both stored as ``1``, collapsing different
        configurations into the same config signature.
        """
        meta_a = extract_solution_metadata("model = LGBMRegressor(learning_rate=1e-3)")
        meta_b = extract_solution_metadata("model = LGBMRegressor(learning_rate=1e-5)")
        assert meta_a.hyperparameters["learning_rate"] != meta_b.hyperparameters["learning_rate"]
        assert meta_a.hyperparameters["learning_rate"] == pytest.approx(1e-3)
        assert meta_b.hyperparameters["learning_rate"] == pytest.approx(1e-5)

    def test_leading_dot_float(self) -> None:
        code = "model = LGBMRegressor(subsample=.8, colsample_bytree=.5)"
        metadata = extract_solution_metadata(code)
        assert metadata.hyperparameters["subsample"] == pytest.approx(0.8)
        assert metadata.hyperparameters["colsample_bytree"] == pytest.approx(0.5)
