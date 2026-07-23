"""Tests for experiment metadata extraction.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.tracking import extract_solution_metadata

__all__ = ()


class TestExtractHyperparameters:
    """Regression tests for hyperparameter regex boundary behavior.

    The `_HYPERPARAM_PATTERNS` table lacked word-boundary anchors, so short
    names like `alpha`, `subsample`, `weights`, and `metric` matched as
    substrings of longer identifiers (`quantile_alpha`, `feature_subsample`,
    `sample_weights`, `main_metric`). That polluted the recorded
    `hyperparameters` dict with phantom entries, corrupted the config
    signature used for dedup / best-experiment lookups, and misled the
    evolver agent when it inspected historical experiment metadata.
    """

    def test_quantile_alpha_does_not_bleed_into_alpha(self) -> None:
        """`quantile_alpha=0.5` must not surface as `alpha=0.5`."""
        metadata = extract_solution_metadata("model = LGBMRegressor(quantile_alpha=0.5)")
        assert metadata.hyperparameters.get("quantile_alpha") == 0.5
        assert "alpha" not in metadata.hyperparameters

    def test_alpha_alone_is_still_captured(self) -> None:
        """The `alpha` regex must still match when the identifier is unqualified."""
        metadata = extract_solution_metadata("model = Ridge(alpha=0.5)")
        assert metadata.hyperparameters.get("alpha") == 0.5

    def test_alpha_and_quantile_alpha_both_captured_when_both_present(self) -> None:
        """Both parameters must land in the dict when the code sets both."""
        code = "model = LGBMRegressor(alpha=0.7, quantile_alpha=0.3)"
        metadata = extract_solution_metadata(code)
        assert metadata.hyperparameters.get("alpha") == 0.7
        assert metadata.hyperparameters.get("quantile_alpha") == 0.3

    def test_feature_subsample_does_not_bleed_into_subsample(self) -> None:
        """`feature_subsample=0.8` must not surface as `subsample=0.8`."""
        metadata = extract_solution_metadata("model = LGBMRegressor(feature_subsample=0.8)")
        assert "subsample" not in metadata.hyperparameters

    def test_subsample_alone_is_still_captured(self) -> None:
        """Bare `subsample=` should still be captured."""
        metadata = extract_solution_metadata("model = LGBMRegressor(subsample=0.75)")
        assert metadata.hyperparameters.get("subsample") == 0.75

    def test_sample_weights_does_not_bleed_into_weights(self) -> None:
        """`sample_weights=w` must not surface as `weights=w`."""
        metadata = extract_solution_metadata("model.fit(X, y, sample_weights=w)")
        assert "weights" not in metadata.hyperparameters

    def test_weights_alone_is_still_captured(self) -> None:
        """Bare `weights='distance'` should still be captured."""
        metadata = extract_solution_metadata('KNeighborsRegressor(weights="distance")')
        assert metadata.hyperparameters.get("weights") == "distance"

    def test_main_metric_does_not_bleed_into_metric(self) -> None:
        """`main_metric='rmse'` must not surface as `metric='rmse'`."""
        metadata = extract_solution_metadata('params = {"main_metric": "rmse"}\nmain_metric = "rmse"')
        assert "metric" not in metadata.hyperparameters

    def test_metric_alone_is_still_captured(self) -> None:
        """Bare `metric='mae'` should still be captured."""
        metadata = extract_solution_metadata('LGBMRegressor(metric="mae")')
        assert metadata.hyperparameters.get("metric") == "mae"

    def test_no_false_positive_from_shared_suffix(self) -> None:
        """A parameter name buried inside a longer identifier must not be captured."""
        metadata = extract_solution_metadata("do_thing(subsample_alpha=0.1, foo_max_depth=7)")
        assert "subsample" not in metadata.hyperparameters
        assert "alpha" not in metadata.hyperparameters
        assert "max_depth" not in metadata.hyperparameters

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (
                "LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8)",
                {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8},
            ),
            ("LGBMRegressor(objective='huber', huber_delta=0.9)", {"objective": "huber", "huber_delta": 0.9}),
            (
                "KNeighborsRegressor(n_neighbors=15, p=2, weights='distance')",
                {"n_neighbors": 15, "p": 2, "weights": "distance"},
            ),
        ],
    )
    def test_common_lightgbm_and_sklearn_configs(self, code: str, expected: dict[str, float | int | str]) -> None:
        """Common realistic parameter combinations should round-trip."""
        metadata = extract_solution_metadata(code)
        for key, value in expected.items():
            assert metadata.hyperparameters.get(key) == value, (
                f"expected {key}={value!r}, got {metadata.hyperparameters.get(key)!r}"
            )


class TestModelDetection:
    """Guard the primary model/family detection surface used by the tracker."""

    def test_lightgbm_regressor_detected(self) -> None:
        metadata = extract_solution_metadata("from lightgbm import LGBMRegressor\nm = LGBMRegressor()")
        assert metadata.model_name == "LGBMRegressor"
        assert metadata.model_family == "lightgbm"

    def test_random_forest_regressor_detected(self) -> None:
        metadata = extract_solution_metadata(
            "from sklearn.ensemble import RandomForestRegressor\nm = RandomForestRegressor()"
        )
        assert metadata.model_name == "RandomForestRegressor"
        assert metadata.model_family == "random_forest"

    def test_no_model_returns_none(self) -> None:
        metadata = extract_solution_metadata("import pandas as pd\nprint('hello')")
        assert metadata.model_name is None
        assert metadata.model_family is None
