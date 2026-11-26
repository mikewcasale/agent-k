"""Tests for strict trusted metric semantics.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import numpy as np
import pytest

from agent_k.metrics import MetricId, prediction_columns, score_predictions

__all__ = ()


def test_binary_auc_uses_declared_positive_class() -> None:
    labels = np.array(["benign", "malignant", "benign", "malignant"])
    probabilities = np.array([0.05, 0.9, 0.1, 0.8])

    score = score_predictions(
        MetricId.AUC,
        labels,
        probabilities,
        class_labels=("benign", "malignant"),
    )

    assert score == pytest.approx(1.0)


def test_multiclass_probabilities_must_be_normalized() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        score_predictions(
            MetricId.LOG_LOSS,
            np.array(["a", "b"]),
            np.array([[0.8, 0.4, 0.0], [0.1, 0.2, 0.4]]),
            class_labels=("a", "b", "c"),
        )


def test_non_finite_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        score_predictions(
            MetricId.RMSE,
            np.array([1.0, 2.0]),
            np.array([1.0, np.nan]),
        )


def test_prediction_columns_encode_probability_semantics() -> None:
    assert prediction_columns(
        MetricId.AUC,
        ("target",),
        ("no", "yes"),
    ) == ("target__probability",)
    assert prediction_columns(
        MetricId.LOG_LOSS,
        ("target",),
        ("a", "b", "c"),
    ) == (
        "target__probability__a",
        "target__probability__b",
        "target__probability__c",
    )
