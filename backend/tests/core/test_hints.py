"""Tests for the LightGBM hint snippets in ``agent_k.core.hints``.

These tests execute the snippet text embedded in generated hints so the agent
never quotes broken code at the LLM (e.g. the ``fobj=`` keyword that
``lightgbm.train`` removed in 4.0).

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from agent_k.core.hints import (
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    DistributionStats,
    MissingPattern,
    PreprocessingHint,
    generate_preprocessing_hints,
)

__all__ = ()


def _build_tabular_regression_profile(*, skewed: bool = True) -> DatasetProfile:
    skewness = 2.5 if skewed else 0.1
    columns = {
        "feat_num": ColumnProfile(
            name="feat_num",
            dtype="float64",
            column_type=ColumnType.NUMERIC_CONTINUOUS,
            missing_rate=0.0,
            unique_count=128,
            unique_ratio=0.64,
            mean=0.0,
            std=1.0,
            min_value=-3.0,
            max_value=3.0,
            skewness=0.0,
            sample_values=("0.1",),
        ),
        "target": ColumnProfile(
            name="target",
            dtype="float64",
            column_type=ColumnType.NUMERIC_CONTINUOUS,
            missing_rate=0.0,
            unique_count=128,
            unique_ratio=0.64,
            mean=10.0,
            std=1.0,
            min_value=0.0,
            max_value=20.0,
            skewness=skewness,
            sample_values=("1.0",),
        ),
    }
    return DatasetProfile(
        columns=columns,
        row_count=200,
        missing_pattern=MissingPattern.MCAR,
        has_temporal_features=False,
        has_geographic_features=False,
        has_text_features=False,
        has_price_features=False,
        target_distribution=DistributionStats(
            column_name="target",
            count=200,
            mean=10.0,
            std=1.0,
            min_value=0.0,
            max_value=20.0,
            median=10.0,
            skewness=skewness,
        ),
        feature_correlations={"feat_num": 0.5},
        target_columns=("target",),
        id_column=None,
    )


def _get_lightgbm_custom_rmsle_hint() -> PreprocessingHint:
    profile = _build_tabular_regression_profile()
    hints = generate_preprocessing_hints(profile, competition_id="test-comp")
    matching = [h for h in hints if h.id == "lightgbm_custom_rmsle"]
    assert matching, "Expected `lightgbm_custom_rmsle` hint for a skewed regression target"
    return matching[0]


def test_lightgbm_custom_rmsle_snippet_does_not_use_removed_fobj_keyword() -> None:
    """`fobj` was removed in LightGBM 4.0; passing it raises TypeError."""
    snippet = _get_lightgbm_custom_rmsle_hint().code_snippet
    assert "fobj=" not in snippet
    assert "fobj =" not in snippet


def test_lightgbm_custom_rmsle_snippet_executes_against_real_lightgbm() -> None:
    """The full snippet must train under the LightGBM pinned in pyproject.toml."""
    hint = _get_lightgbm_custom_rmsle_hint()

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.standard_normal((64, 3)), columns=["a", "b", "c"])
    X_test = pd.DataFrame(rng.standard_normal((16, 3)), columns=["a", "b", "c"])
    y_train = pd.Series(np.abs(rng.standard_normal(64)) + 0.1, name="target")

    namespace: dict[str, Any] = {"X_train": X_train, "y_train": y_train, "X_test": X_test}
    exec(hint.code_snippet, namespace)

    booster = namespace["model"]
    predictions = namespace["predictions"]

    assert isinstance(booster, lgb.Booster)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_test.shape[0],)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
