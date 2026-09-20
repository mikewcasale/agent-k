"""Tests for the custom LightGBM objective hints emitted by the hint generator.

The snippets in these hints are injected verbatim into evolving candidate
solutions, so they are executed here against real LightGBM rather than
inspected as strings.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import numpy as np
import pytest

from agent_k.core.hints import (
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    MissingPattern,
    PreprocessingHint,
    generate_preprocessing_hints,
)

RNG_SEED = 0
N_ROWS = 300
N_FEATURES = 4


def _numeric_column(name: str, *, unique_count: int = 200) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        column_type=ColumnType.NUMERIC_CONTINUOUS,
        missing_rate=0.0,
        unique_count=unique_count,
        unique_ratio=unique_count / N_ROWS,
        mean=0.0,
        std=1.0,
        min_value=-3.0,
        max_value=3.0,
        skewness=0.0,
        average_length=None,
        sample_values=("0.1", "0.2"),
    )


def _profile(target: ColumnProfile) -> DatasetProfile:
    columns = {f"feat_{idx}": _numeric_column(f"feat_{idx}") for idx in range(N_FEATURES)}
    columns[target.name] = target
    return DatasetProfile(
        columns=columns,
        row_count=N_ROWS,
        missing_pattern=MissingPattern.MCAR,
        has_temporal_features=False,
        has_geographic_features=False,
        has_text_features=False,
        has_price_features=False,
        target_distribution=None,
        feature_correlations={},
        target_columns=(target.name,),
        id_column=None,
    )


def _regression_profile() -> DatasetProfile:
    return _profile(_numeric_column("target"))


def _binary_profile() -> DatasetProfile:
    target = ColumnProfile(
        name="target",
        dtype="int64",
        column_type=ColumnType.BINARY,
        missing_rate=0.0,
        unique_count=2,
        unique_ratio=2 / N_ROWS,
        mean=0.5,
        std=0.5,
        min_value=0.0,
        max_value=1.0,
        skewness=0.0,
        average_length=None,
        sample_values=("0", "1"),
    )
    return _profile(target)


def _multiclass_profile() -> DatasetProfile:
    target = ColumnProfile(
        name="target",
        dtype="int64",
        column_type=ColumnType.CATEGORICAL_LOW_CARDINALITY,
        missing_rate=0.0,
        unique_count=5,
        unique_ratio=5 / N_ROWS,
        mean=2.0,
        std=1.4,
        min_value=0.0,
        max_value=4.0,
        skewness=0.0,
        average_length=None,
        sample_values=("0", "3"),
    )
    return _profile(target)


def _find_hint(hints: list[PreprocessingHint], hint_id: str) -> PreprocessingHint | None:
    return next((hint for hint in hints if hint.id == hint_id), None)


def _run_snippet(snippet: str, namespace: dict[str, Any]) -> dict[str, Any]:
    exec(compile(snippet, "<hint-snippet>", "exec"), namespace)
    return namespace


@pytest.fixture(name="features")
def features_fixture() -> np.ndarray:
    """Return a deterministic feature matrix shared by the snippet tests."""
    rng = np.random.default_rng(RNG_SEED)
    return rng.normal(size=(N_ROWS, N_FEATURES))


class TestRegressionCustomObjective:
    """The RMSLE hint must be runnable LightGBM 4.x code."""

    def test_hint_is_emitted_for_regression(self) -> None:
        hints = generate_preprocessing_hints(_regression_profile())
        assert _find_hint(hints, "lightgbm_custom_rmsle") is not None

    def test_snippet_does_not_use_the_removed_fobj_argument(self) -> None:
        hint = _find_hint(generate_preprocessing_hints(_regression_profile()), "lightgbm_custom_rmsle")
        assert hint is not None
        assert "fobj" not in hint.code_snippet

    def test_snippet_trains_a_booster(self, features: np.ndarray) -> None:
        hint = _find_hint(generate_preprocessing_hints(_regression_profile()), "lightgbm_custom_rmsle")
        assert hint is not None
        rng = np.random.default_rng(RNG_SEED)
        targets = np.abs(rng.normal(loc=10.0, scale=3.0, size=N_ROWS))
        namespace = _run_snippet(hint.code_snippet, {"X_train": features, "y_train": targets})
        predictions = namespace["model"].predict(features)
        assert predictions.shape == (N_ROWS,)
        assert np.isfinite(predictions).all()

    def test_objective_gradient_matches_finite_differences(self, features: np.ndarray) -> None:
        hint = _find_hint(generate_preprocessing_hints(_regression_profile()), "lightgbm_custom_rmsle")
        assert hint is not None
        rng = np.random.default_rng(RNG_SEED)
        targets = np.abs(rng.normal(loc=10.0, scale=3.0, size=N_ROWS))
        namespace = _run_snippet(hint.code_snippet, {"X_train": features, "y_train": targets})
        objective = namespace["rmsle_objective"]
        dataset = namespace["lgb"].Dataset(features, label=targets).construct()

        raw = np.abs(rng.normal(loc=10.0, scale=2.0, size=N_ROWS))
        epsilon = 1e-5

        def loss(values: np.ndarray) -> np.ndarray:
            return 0.5 * (np.log1p(np.maximum(values, 0.0)) - np.log1p(targets)) ** 2

        numeric_gradient = (loss(raw + epsilon) - loss(raw - epsilon)) / (2 * epsilon)
        gradient, hessian = objective(raw, dataset)
        assert np.abs(gradient - numeric_gradient).max() < 1e-5
        assert (hessian > 0).all(), "LightGBM requires strictly positive hessians"


class TestBinaryClassificationCustomObjective:
    """Binary classification must also get a custom-objective hint."""

    def test_hint_is_emitted_for_binary_targets(self) -> None:
        hints = generate_preprocessing_hints(_binary_profile())
        assert _find_hint(hints, "lightgbm_custom_focal") is not None

    def test_hint_is_not_emitted_for_regression_or_multiclass(self) -> None:
        assert _find_hint(generate_preprocessing_hints(_regression_profile()), "lightgbm_custom_focal") is None
        assert _find_hint(generate_preprocessing_hints(_multiclass_profile()), "lightgbm_custom_focal") is None

    def test_snippet_trains_and_yields_probabilities(self, features: np.ndarray) -> None:
        hint = _find_hint(generate_preprocessing_hints(_binary_profile()), "lightgbm_custom_focal")
        assert hint is not None
        rng = np.random.default_rng(RNG_SEED)
        labels = (rng.random(N_ROWS) < 0.2).astype(float)
        namespace = _run_snippet(hint.code_snippet, {"X_train": features, "y_train": labels})
        probabilities = namespace["focal_train_proba"]
        assert probabilities.shape == (N_ROWS,)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    def test_objective_derivatives_match_finite_differences(self, features: np.ndarray) -> None:
        hint = _find_hint(generate_preprocessing_hints(_binary_profile()), "lightgbm_custom_focal")
        assert hint is not None
        rng = np.random.default_rng(RNG_SEED)
        labels = (rng.random(N_ROWS) < 0.2).astype(float)
        namespace = _run_snippet(hint.code_snippet, {"X_train": features, "y_train": labels})
        objective = namespace["focal_objective"]
        alpha = namespace["FOCAL_ALPHA"]
        gamma = namespace["FOCAL_GAMMA"]
        dataset = namespace["lgb"].Dataset(features, label=labels).construct()

        raw = rng.normal(scale=2.0, size=N_ROWS)
        epsilon = 1e-5

        def loss(values: np.ndarray) -> np.ndarray:
            prob = 1.0 / (1.0 + np.exp(-values))
            p_t = np.clip(np.where(labels == 1, prob, 1.0 - prob), 1e-9, 1.0 - 1e-9)
            alpha_t = np.where(labels == 1, alpha, 1.0 - alpha)
            return -alpha_t * (1.0 - p_t) ** gamma * np.log(p_t)

        numeric_gradient = (loss(raw + epsilon) - loss(raw - epsilon)) / (2 * epsilon)
        gradient, hessian = objective(raw, dataset)
        assert np.abs(gradient - numeric_gradient).max() < 1e-5

        def analytic_gradient(values: np.ndarray) -> np.ndarray:
            return objective(values, dataset)[0]

        step = 1e-4
        numeric_hessian = (analytic_gradient(raw + step) - analytic_gradient(raw - step)) / (2 * step)
        # The emitted hessian is floored for LightGBM, so only compare where the floor is inactive.
        unfloored = numeric_hessian > 1e-5
        assert np.abs(hessian[unfloored] - numeric_hessian[unfloored]).max() < 1e-5
        assert (hessian > 0).all(), "LightGBM requires strictly positive hessians"
