"""Tests for host-only evaluation and final submission materialization.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from agent_k.datasets import discover_dataset
from agent_k.evals import build_evaluation_dataset, run_evaluation_dataset
from agent_k.evaluator import (
    CandidateResult,
    CandidateStatus,
    EvaluationRequest,
    evaluate_candidate,
    write_submission,
)
from agent_k.metrics import MetricId

__all__ = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_fixture(tmp_path: Path) -> tuple[EvaluationRequest, Path, Path]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    frame = load_breast_cancer(as_frame=True).frame
    frame.insert(0, "row_id", [f"row-{index}" for index in range(len(frame))])
    split = 450
    train = frame.iloc[:split]
    test = frame.iloc[split:]
    train.to_csv(dataset_root / "train.csv", index=False)
    test.drop(columns="target").to_csv(dataset_root / "test.csv", index=False)
    test[["row_id", "target"]].to_csv(
        dataset_root / "sample_submission.csv",
        index=False,
    )
    manifest = discover_dataset(dataset_root, metric_id="auc")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2_000))
    model.fit(train.loc[:, list(manifest.feature_columns)], train["target"])
    probability = model.predict_proba(test.loc[:, list(manifest.feature_columns)])[:, 1]
    output_root = tmp_path / "candidate-output"
    output_root.mkdir()
    prediction_path = output_root / "predictions.csv"
    predictions = pd.DataFrame(
        {
            "row_id": test["row_id"].iloc[::-1].to_numpy(),
            "target__probability": probability[::-1],
        }
    )
    predictions.to_csv(prediction_path, index=False)
    labels_path = tmp_path / "trusted-labels.csv"
    test[["row_id", "target"]].to_csv(labels_path, index=False)
    result = CandidateResult(
        candidate_id="trusted-evaluation",
        status=CandidateStatus.SUCCEEDED,
        predictions_path="predictions.csv",
        predictions_sha256=_sha256(prediction_path),
        row_count=len(predictions),
        prediction_columns=("target__probability",),
        duration_seconds=0.1,
    )
    request = EvaluationRequest(
        candidate_result=result,
        metric_id=MetricId.AUC,
        prediction_root=str(output_root),
        trusted_labels_path=str(labels_path),
        id_columns=("row_id",),
        target_columns=("target",),
        class_labels=("0", "1"),
    )
    return request, dataset_root, prediction_path


def test_evaluator_aligns_exact_ids_instead_of_row_position(tmp_path: Path) -> None:
    request, _, _ = _evaluation_fixture(tmp_path)

    result = evaluate_candidate(request)

    assert result.row_count == 119
    assert 0.9 <= result.score <= 1.0


def test_evaluator_rejects_missing_candidate_ids(tmp_path: Path) -> None:
    request, _, prediction_path = _evaluation_fixture(tmp_path)
    predictions = pd.read_csv(prediction_path).iloc[:-1]
    predictions.to_csv(prediction_path, index=False)
    result = request.candidate_result.model_copy(
        update={
            "predictions_sha256": _sha256(prediction_path),
            "row_count": len(predictions),
        }
    )

    with pytest.raises(ValueError, match="exact match"):
        evaluate_candidate(request.model_copy(update={"candidate_result": result}))


def test_evaluator_rejects_false_candidate_metadata(tmp_path: Path) -> None:
    request, _, _ = _evaluation_fixture(tmp_path)
    assert request.candidate_result.row_count is not None
    false_result = request.candidate_result.model_copy(
        update={"row_count": request.candidate_result.row_count + 1}
    )

    with pytest.raises(ValueError, match="row count differs"):
        evaluate_candidate(request.model_copy(update={"candidate_result": false_result}))


def test_write_submission_restores_sample_order_and_schema(tmp_path: Path) -> None:
    request, dataset_root, prediction_path = _evaluation_fixture(tmp_path)
    manifest = discover_dataset(dataset_root, metric_id="auc")
    destination = tmp_path / "submission.csv"

    artifact = write_submission(
        prediction_path,
        manifest,
        dataset_root,
        destination,
    )
    submission = pd.read_csv(destination, dtype={"row_id": str})
    sample = pd.read_csv(dataset_root / "sample_submission.csv", dtype={"row_id": str})

    assert artifact.row_count == len(sample)
    assert tuple(submission.columns) == tuple(sample.columns)
    assert submission["row_id"].tolist() == sample["row_id"].tolist()
    assert request.candidate_result.predictions_sha256 == _sha256(prediction_path)


def test_write_submission_rejects_invalid_full_dataset_predictions(tmp_path: Path) -> None:
    _, dataset_root, prediction_path = _evaluation_fixture(tmp_path)
    predictions = pd.read_csv(prediction_path)
    predictions.loc[0, "target__probability"] = 1.5
    predictions.to_csv(prediction_path, index=False)
    manifest = discover_dataset(dataset_root, metric_id="auc")

    with pytest.raises(ValueError, match="probabilities must be within"):
        write_submission(
            prediction_path,
            manifest,
            dataset_root,
            tmp_path / "invalid-submission.csv",
        )


def test_pydantic_evals_runs_trusted_requests(tmp_path: Path) -> None:
    request, _, _ = _evaluation_fixture(tmp_path)
    dataset = build_evaluation_dataset([request])

    report = run_evaluation_dataset(dataset)

    assert len(report.cases) == 1
    assert report.failures == []
    assert report.cases[0].output.score >= 0.9
