"""Tests for autonomous dataset discovery and trusted split preparation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from agent_k.datasets import ProblemType, discover_dataset
from agent_k.evaluator import materialize_trusted_splits

__all__ = ()


def _write_competition(root: Path, frame: pd.DataFrame, target: str) -> None:
    root.mkdir()
    data = frame.copy()
    data.insert(0, "row_id", [f"row-{index}" for index in range(len(data))])
    split = int(len(data) * 0.8)
    data.iloc[:split].to_csv(root / "train.csv", index=False)
    data.iloc[split:].drop(columns=target).to_csv(root / "test.csv", index=False)
    data.iloc[split:][["row_id", target]].to_csv(
        root / "sample_submission.csv",
        index=False,
    )


def test_discovers_real_classification_data_and_separates_hidden_labels(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "competition"
    _write_competition(
        dataset_root,
        load_breast_cancer(as_frame=True).frame,
        "target",
    )

    manifest = discover_dataset(dataset_root, metric_id="auc")

    assert manifest.problem_type is ProblemType.TABULAR_CLASSIFICATION
    assert manifest.id_columns == ("row_id",)
    assert manifest.target_columns == ("target",)
    assert manifest.class_labels == ("0", "1")

    splits = materialize_trusted_splits(
        manifest,
        dataset_root,
        tmp_path / "candidate-inputs",
        tmp_path / "trusted-labels",
        validation_fraction=0.2,
        acceptance_fraction=0.2,
    )
    validation_predict = pd.read_csv(Path(splits.validation.candidate_input_root) / "predict.csv")
    acceptance_predict = pd.read_csv(Path(splits.acceptance.candidate_input_root) / "predict.csv")
    validation_labels = pd.read_csv(splits.validation.trusted_labels_path)
    acceptance_labels = pd.read_csv(splits.acceptance.trusted_labels_path)

    assert "target" not in validation_predict
    assert "target" not in acceptance_predict
    assert set(validation_predict["row_id"]) == set(validation_labels["row_id"])
    assert set(acceptance_predict["row_id"]) == set(acceptance_labels["row_id"])
    assert set(validation_predict["row_id"]).isdisjoint(acceptance_predict["row_id"])


def test_discovers_real_regression_data(tmp_path: Path) -> None:
    dataset_root = tmp_path / "competition"
    _write_competition(
        dataset_root,
        load_diabetes(as_frame=True).frame,
        "target",
    )

    manifest = discover_dataset(dataset_root, metric_id="rmse")

    assert manifest.problem_type is ProblemType.TABULAR_REGRESSION
    assert manifest.target_columns == ("target",)
    assert manifest.class_labels == ()
