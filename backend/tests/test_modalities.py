"""Real-data and explicit contract tests for vision and text candidates.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.datasets import load_digits

from agent_k.datasets import DatasetManifest, ProblemType, discover_dataset
from agent_k.evaluator import (
    CandidateEngine,
    EvaluationRequest,
    build_submission_spec,
    evaluate_candidate,
)
from agent_k.metrics import MetricId
from agent_k.text import run_candidate as run_text_candidate
from agent_k.vision import run_candidate as run_vision_candidate

__all__ = ()

_GLUE_SHA256 = {
    "SST-2/train.tsv": "48eac26399f8456935f81dd9e3a745c1430adb6c62e233ccfe22db4a0d15d1d1",
    "SST-2/dev.tsv": "54a9c12c77e402fc14d6a66a506f24d999f596b0176eb777c972cccc2f9cc4b0",
    "STS-B/train.tsv": "41c4d978b6162abdcada7e49a982211954cca5459c228db478af333124b257a2",
    "STS-B/dev.tsv": "fe7dc0632a5033562f3cadff991f6e8d9b9b6311861660e78c3c82eb7db67f59",
}
_TRANSFORMER_REPOSITORY = "google/bert_uncased_L-2_H-128_A-2"
_TRANSFORMER_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
_TRANSFORMER_SHA256 = {
    "README.md": "d13a54d652a2a54fbbc50e063b465fb273295870867821e1b06baba1d49d68a9",
    "config.json": "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    "model.safetensors": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def _write_image_data(root: Path, *, metric_id: str) -> DatasetManifest:
    root.mkdir()
    train_image_root = root / "train_images"
    predict_image_root = root / "test_images"
    train_image_root.mkdir()
    predict_image_root.mkdir()
    digits = load_digits()
    rows: list[dict[str, object]] = []
    for index, (pixels, target) in enumerate(zip(digits.images[:70], digits.target[:70])):
        relative = f"digit-{index}.png"
        image = Image.fromarray(np.asarray(pixels / 16.0 * 255.0, dtype=np.uint8))
        image_root = train_image_root if index < 50 else predict_image_root
        image.save(image_root / relative)
        rows.append({"row_id": f"digit-{index}", "image": relative, "target": int(target)})
    frame = pd.DataFrame(rows)
    frame.iloc[:50].to_csv(root / "train.csv", index=False)
    frame.iloc[50:].drop(columns="target").to_csv(root / "test.csv", index=False)
    frame.iloc[50:][["row_id", "target"]].to_csv(
        root / "sample_submission.csv",
        index=False,
    )
    return discover_dataset(root, metric_id=metric_id)


def _write_text_data(
    root: Path,
    train_text: pd.Series,
    train_target: pd.Series,
    predict_text: pd.Series,
    predict_target: pd.Series,
    *,
    metric_id: str,
) -> DatasetManifest:
    root.mkdir()
    train = pd.DataFrame(
        {
            "row_id": [f"train-{index}" for index in range(len(train_text))],
            "text": train_text.reset_index(drop=True),
            "target": train_target.reset_index(drop=True),
        }
    )
    predict = pd.DataFrame(
        {
            "row_id": [f"dev-{index}" for index in range(len(predict_text))],
            "text": predict_text.reset_index(drop=True),
            "target": predict_target.reset_index(drop=True),
        }
    )
    train.to_csv(root / "train.csv", index=False)
    predict.drop(columns="target").to_csv(root / "test.csv", index=False)
    predict[["row_id", "target"]].to_csv(
        root / "sample_submission.csv",
        index=False,
    )
    return discover_dataset(root, metric_id=metric_id)


def _glue_root() -> Path:
    value = os.environ.get("AGENT_K_GLUE_FIXTURES")
    if value is None:
        pytest.skip("set AGENT_K_GLUE_FIXTURES to extracted official GLUE data")
    root = Path(value)
    for relative, expected in _GLUE_SHA256.items():
        path = root / relative
        if not path.is_file():
            pytest.fail(f"official GLUE fixture is missing {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            pytest.fail(f"official GLUE fixture checksum differs for {relative}")
    return root


def _sst2_manifest(
    root: Path,
    fixture_root: Path,
    *,
    training_rows: int = 5_000,
    development_rows: int | None = None,
) -> DatasetManifest:
    training = pd.read_csv(fixture_root / "SST-2/train.tsv", sep="\t").iloc[:training_rows]
    development = pd.read_csv(fixture_root / "SST-2/dev.tsv", sep="\t").iloc[:development_rows]
    return _write_text_data(
        root,
        training["sentence"],
        training["label"],
        development["sentence"],
        development["label"],
        metric_id=MetricId.ACCURACY.value,
    )


def _stsb_manifest(root: Path, fixture_root: Path) -> DatasetManifest:
    training = pd.read_csv(
        fixture_root / "STS-B/train.tsv",
        sep="\t",
        quoting=csv.QUOTE_NONE,
    )
    development = pd.read_csv(
        fixture_root / "STS-B/dev.tsv",
        sep="\t",
        quoting=csv.QUOTE_NONE,
    )
    train_text = training["sentence1"] + " [SEP] " + training["sentence2"]
    development_text = development["sentence1"] + " [SEP] " + development["sentence2"]
    return _write_text_data(
        root,
        train_text,
        training["score"],
        development_text,
        development["score"],
        metric_id=MetricId.RMSE.value,
    )


def test_resnet18_executes_real_digit_classification(tmp_path: Path) -> None:
    input_root = tmp_path / "digit-classification"
    manifest = _write_image_data(input_root, metric_id=MetricId.ACCURACY.value)
    spec = build_submission_spec(
        manifest,
        candidate_id="resnet-digit-classification",
        engine=CandidateEngine.RESNET18,
        metric_id=MetricId.ACCURACY,
        hyperparameters={
            "batch_size": 10,
            "device_policy": "cpu",
            "epochs": 1,
            "freeze_backbone": True,
            "image_size": 32,
        },
    )

    result = run_vision_candidate(spec, input_root, tmp_path / "out-digit-classification")

    assert manifest.problem_type is ProblemType.VISION_CLASSIFICATION
    assert manifest.train_image_root == "train_images"
    assert manifest.predict_image_root == "test_images"
    assert result.row_count == 20


def test_resnet18_regression_output_contract(tmp_path: Path) -> None:
    """Cover continuous-output mechanics only; the MOS fixture gate is authoritative."""
    input_root = tmp_path / "regression-contract"
    manifest = _write_image_data(input_root, metric_id=MetricId.RMSE.value)
    spec = build_submission_spec(
        manifest,
        candidate_id="resnet-regression-contract",
        engine=CandidateEngine.RESNET18,
        metric_id=MetricId.RMSE,
        hyperparameters={
            "batch_size": 10,
            "device_policy": "cpu",
            "epochs": 2,
            "freeze_backbone": True,
            "image_size": 32,
        },
    )

    result = run_vision_candidate(spec, input_root, tmp_path / "out-regression-contract")

    assert manifest.problem_type is ProblemType.VISION_REGRESSION
    assert result.row_count == 20
    assert result.diagnostics["batch_norm_calibrated"] is True
    assert result.diagnostics["batch_norm_calibration_batches"] == 5


def test_resnet18_executes_real_mos_regression_when_fixture_is_available(
    tmp_path: Path,
) -> None:
    fixture_value = os.environ.get("AGENT_K_IMAGE_REGRESSION_FIXTURE")
    if fixture_value is None:
        pytest.skip("set AGENT_K_IMAGE_REGRESSION_FIXTURE to a verified MOS corpus subset")
    input_root = Path(fixture_value)
    provenance_path = input_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["task"] == "image_regression"
    assert provenance["dataset"]
    assert provenance["dataset_doi"]
    assert provenance["dataset_version"]
    assert provenance["source_url"].startswith("https://")
    assert provenance["license"]
    assert provenance["label_provenance"]
    for relative, expected in provenance["files"].items():
        assert hashlib.sha256((input_root / relative).read_bytes()).hexdigest() == expected
    manifest = discover_dataset(input_root, metric_id=MetricId.RMSE.value)
    train = pd.read_csv(input_root / manifest.train.path)
    predict = pd.read_csv(input_root / manifest.predict.path)
    assert manifest.image_column is not None
    referenced = {
        str(Path(manifest.train_image_root or "") / value)
        for value in train[manifest.image_column].astype(str)
    } | {
        str(Path(manifest.predict_image_root or "") / value)
        for value in predict[manifest.image_column].astype(str)
    }
    assert referenced <= set(provenance["files"])
    candidate_root = tmp_path / "candidate-real-mos"
    candidate_root.mkdir()
    shutil.copy2(input_root / manifest.train.path, candidate_root / manifest.train.path)
    shutil.copy2(input_root / manifest.predict.path, candidate_root / manifest.predict.path)
    assert manifest.train_image_root is not None
    assert manifest.predict_image_root is not None
    shutil.copytree(
        input_root / manifest.train_image_root,
        candidate_root / manifest.train_image_root,
    )
    shutil.copytree(
        input_root / manifest.predict_image_root,
        candidate_root / manifest.predict_image_root,
    )
    assert not (candidate_root / "sample_submission.csv").exists()
    assert not (candidate_root / "source").exists()
    spec = build_submission_spec(
        manifest,
        candidate_id="resnet-real-mos",
        engine=CandidateEngine.RESNET18,
        metric_id=MetricId.RMSE,
        hyperparameters={
            "batch_size": 16,
            "device_policy": "cpu",
            "epochs": 1,
            "image_size": 64,
        },
    )

    output_root = tmp_path / "out-real-mos"
    result = run_vision_candidate(spec, candidate_root, output_root)
    evaluation = evaluate_candidate(
        EvaluationRequest(
            candidate_result=result,
            metric_id=MetricId.RMSE,
            prediction_root=str(output_root),
            trusted_labels_path=str(input_root / "sample_submission.csv"),
            id_columns=manifest.id_columns,
            target_columns=manifest.target_columns,
        )
    )

    assert manifest.problem_type is ProblemType.VISION_REGRESSION
    assert result.row_count == manifest.predict.row_count
    assert evaluation.row_count == manifest.predict.row_count
    assert np.isfinite(evaluation.score)


def test_tfidf_executes_real_sst2_classification(tmp_path: Path) -> None:
    input_root = tmp_path / "sst2"
    manifest = _sst2_manifest(input_root, _glue_root())
    spec = build_submission_spec(
        manifest,
        candidate_id="tfidf-sst2",
        engine=CandidateEngine.TFIDF_LINEAR,
        metric_id=MetricId.ACCURACY,
        hyperparameters={"min_df": 2, "max_features": 20_000},
    )

    result = run_text_candidate(spec, input_root, tmp_path / "out-sst2")

    assert manifest.problem_type is ProblemType.TEXT_CLASSIFICATION
    assert result.row_count == 872


def test_tfidf_executes_real_stsb_regression(tmp_path: Path) -> None:
    input_root = tmp_path / "stsb"
    manifest = _stsb_manifest(input_root, _glue_root())
    spec = build_submission_spec(
        manifest,
        candidate_id="tfidf-stsb",
        engine=CandidateEngine.TFIDF_LINEAR,
        metric_id=MetricId.RMSE,
        hyperparameters={"min_df": 2, "max_features": 20_000},
    )

    result = run_text_candidate(spec, input_root, tmp_path / "out-stsb")

    assert manifest.problem_type is ProblemType.TEXT_REGRESSION
    assert result.row_count == 1_500


def test_transformer_executes_offline_when_assets_are_available(tmp_path: Path) -> None:
    asset_source = os.environ.get("AGENT_K_TRANSFORMER_ASSETS")
    if asset_source is None:
        pytest.skip("set AGENT_K_TRANSFORMER_ASSETS to a local sequence model")
    asset_root = Path(asset_source)
    provenance = json.loads((asset_root / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["repository"] == _TRANSFORMER_REPOSITORY
    assert provenance["revision"] == _TRANSFORMER_REVISION
    assert provenance["files"] == _TRANSFORMER_SHA256
    for relative, expected in _TRANSFORMER_SHA256.items():
        actual = hashlib.sha256((asset_root / relative).read_bytes()).hexdigest()
        assert actual == expected
    input_root = tmp_path / "transformer"
    manifest = _sst2_manifest(
        input_root,
        _glue_root(),
        training_rows=512,
        development_rows=128,
    )
    shutil.copytree(asset_root, input_root / "assets" / "transformer")
    spec = build_submission_spec(
        manifest,
        candidate_id="offline-transformer",
        engine=CandidateEngine.TRANSFORMER,
        metric_id=MetricId.ACCURACY,
        hyperparameters={
            "batch_size": 16,
            "device_policy": "cpu",
            "epochs": 1,
            "max_length": 64,
        },
        offline_assets={"transformer_model": "assets/transformer"},
    )

    result = run_text_candidate(spec, input_root, tmp_path / "out-transformer")

    assert result.row_count == 128
