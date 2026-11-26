"""End-to-end tests for real tabular candidate engines.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

from agent_k.datasets import DatasetManifest, discover_dataset
from agent_k.evaluator import (
    CandidateEngine,
    EvaluationRequest,
    build_submission_spec,
    evaluate_candidate,
)
from agent_k.metrics import MetricId
from agent_k.tabular import run_candidate

__all__ = ()


def _write_data(root: Path, frame: pd.DataFrame) -> tuple[DatasetManifest, Path]:
    root.mkdir()
    data = frame.copy()
    data.insert(0, "row_id", [f"row-{index}" for index in range(len(data))])
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    predict = data.iloc[split:].drop(columns="target")
    labels = data.iloc[split:][["row_id", "target"]]
    train.to_csv(root / "train.csv", index=False)
    predict.to_csv(root / "test.csv", index=False)
    labels.to_csv(root / "sample_submission.csv", index=False)
    labels_path = root.parent / f"{root.name}-hidden.csv"
    labels.to_csv(labels_path, index=False)
    metric = "auc" if frame["target"].nunique() == 2 else "rmse"
    return discover_dataset(root, metric_id=metric), labels_path


def test_lightgbm_binary_custom_loss_scores_real_breast_cancer(tmp_path: Path) -> None:
    input_root = tmp_path / "classification"
    manifest, labels_path = _write_data(
        input_root,
        load_breast_cancer(as_frame=True).frame,
    )
    spec = build_submission_spec(
        manifest,
        candidate_id="lightgbm-binary",
        engine=CandidateEngine.LIGHTGBM,
        metric_id=MetricId.AUC,
        hyperparameters={
            "n_estimators": 30,
            "loss": {"loss_id": "binary_log_loss"},
        },
    )

    result = run_candidate(spec, input_root, tmp_path / "classification-output")
    evaluation = evaluate_candidate(
        EvaluationRequest(
            candidate_result=result,
            metric_id=MetricId.AUC,
            prediction_root=str(tmp_path / "classification-output"),
            trusted_labels_path=str(labels_path),
            id_columns=manifest.id_columns,
            target_columns=manifest.target_columns,
            class_labels=manifest.class_labels,
        )
    )

    assert evaluation.row_count == manifest.predict.row_count
    assert 0.5 <= evaluation.score <= 1.0


def test_lightgbm_regression_custom_loss_scores_real_diabetes(tmp_path: Path) -> None:
    input_root = tmp_path / "regression"
    manifest, labels_path = _write_data(
        input_root,
        load_diabetes(as_frame=True).frame,
    )
    spec = build_submission_spec(
        manifest,
        candidate_id="lightgbm-regression",
        engine=CandidateEngine.LIGHTGBM,
        metric_id=MetricId.RMSE,
        hyperparameters={
            "n_estimators": 30,
            "loss": {"loss_id": "squared_error"},
        },
    )

    result = run_candidate(spec, input_root, tmp_path / "regression-output")
    evaluation = evaluate_candidate(
        EvaluationRequest(
            candidate_result=result,
            metric_id=MetricId.RMSE,
            prediction_root=str(tmp_path / "regression-output"),
            trusted_labels_path=str(labels_path),
            id_columns=manifest.id_columns,
            target_columns=manifest.target_columns,
        )
    )

    assert 0.0 < evaluation.score < 100.0


def test_kumo_small_runs_on_cpu_with_real_breast_cancer_when_assets_available(
    tmp_path: Path,
) -> None:
    asset_source = os.environ.get("AGENT_K_KUMO_ASSETS")
    if asset_source is None:
        pytest.skip("set AGENT_K_KUMO_ASSETS to the provisioned small asset directory")
    input_root = tmp_path / "kumo"
    manifest, _ = _write_data(
        input_root,
        load_breast_cancer(as_frame=True).frame,
    )
    shutil.copytree(asset_source, input_root / "assets" / "kumo", symlinks=False)
    spec = build_submission_spec(
        manifest,
        candidate_id="kumo-cpu",
        engine=CandidateEngine.KUMO,
        metric_id=MetricId.AUC,
        hyperparameters={
            "context_fraction": 0.5,
            "lightgbm_parameters": {"n_estimators": 10},
            "lightgbm_weight": 0.25,
            "device_policy": "cpu_allowed",
            "num_estimators": 1,
            "size": "small",
        },
        offline_assets={
            "hf_cache": "assets/kumo/hf-cache",
            "kumo_manifest": "assets/kumo/manifest.json",
        },
    )

    result = run_candidate(spec, input_root, tmp_path / "kumo-output")

    assert result.row_count == manifest.predict.row_count
