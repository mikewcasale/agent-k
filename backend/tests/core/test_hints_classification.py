"""Tests for classification-aware preprocessing hints.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from typing import TYPE_CHECKING, Any

import numpy as np

from agent_k.core.hints import (
    CLASS_IMBALANCE_RATIO_THRESHOLD,
    DatasetProfile,
    HintCategory,
    build_dataset_profile,
    detect_applied_hints,
    generate_preprocessing_hints,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agent_k.core.hints import PreprocessingHint

__all__ = ()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_competition(tmp_path: Path, targets: list[str]) -> tuple[Path, Path, Path]:
    """Write a two-feature competition whose target column cycles through ``targets``."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    sample_path = tmp_path / "sample_submission.csv"

    rows = [
        [str(index), f"{index % 17 / 17:.4f}", f"{index % 11 / 11:.4f}", targets[index % len(targets)]]
        for index in range(len(targets) * 30)
    ]
    _write_csv(train_path, ["id", "feature_a", "feature_b", "target"], rows)
    _write_csv(test_path, ["id", "feature_a", "feature_b"], [row[:3] for row in rows[:20]])
    _write_csv(sample_path, ["id", "target"], [[row[0], targets[0]] for row in rows[:20]])
    return train_path, test_path, sample_path


def _hints_for(tmp_path: Path, targets: list[str]) -> tuple[DatasetProfile, dict[str, PreprocessingHint]]:
    train_path, test_path, sample_path = _write_competition(tmp_path, targets)
    profile = build_dataset_profile(train_path, test_path, sample_path)
    return profile, {hint.id: hint for hint in generate_preprocessing_hints(profile)}


def test_binary_target_class_balance_is_profiled(tmp_path: Path) -> None:
    """Ensure a rare positive class is measured rather than assumed."""
    profile, _ = _hints_for(tmp_path, ["0"] * 19 + ["1"])

    balance = profile.target_class_balance
    assert balance is not None
    assert balance.column_name == "target"
    assert balance.class_count == 2
    assert balance.is_binary is True
    assert balance.minority_ratio == 0.05
    assert balance.imbalance_ratio == 19.0
    assert balance.is_imbalanced is True
    assert balance.is_severely_imbalanced is True
    assert balance.class_counts == (("0", 570), ("1", 30))


def test_balanced_binary_target_skips_imbalance_weighting(tmp_path: Path) -> None:
    """Ensure balanced targets still get stratified folds but no reweighting hint."""
    profile, hints = _hints_for(tmp_path, ["0", "1"])

    balance = profile.target_class_balance
    assert balance is not None
    assert balance.imbalance_ratio == 1.0
    assert balance.is_imbalanced is False
    assert "class_imbalance_weighting" not in hints
    assert "multiclass_objective" not in hints
    assert hints["stratified_cv"].category == HintCategory.VALIDATION
    assert hints["stratified_cv"].priority == 0.8


def test_imbalanced_binary_target_hints_weighting_above_threshold(tmp_path: Path) -> None:
    """Ensure imbalance past the threshold raises the stratified and weighting priorities."""
    profile, hints = _hints_for(tmp_path, ["0"] * 19 + ["1"])

    balance = profile.target_class_balance
    assert balance is not None
    assert balance.imbalance_ratio >= CLASS_IMBALANCE_RATIO_THRESHOLD
    assert hints["stratified_cv"].priority == 0.9
    weighting = hints["class_imbalance_weighting"]
    assert weighting.category == HintCategory.MODEL_OPTIMIZATION
    assert weighting.priority == 0.9
    assert weighting.applicable_columns == ["target"]
    assert "compute_sample_weight" in weighting.code_snippet
    assert "19.00:1" in weighting.description


def test_multiclass_target_hints_concrete_num_class(tmp_path: Path) -> None:
    """Ensure multiclass targets get a LightGBM multiclass objective with the measured class count."""
    _, hints = _hints_for(tmp_path, ["low", "mid", "high", "top"])

    objective = hints["multiclass_objective"]
    assert objective.category == HintCategory.MODEL_SELECTION
    assert '"num_class": 4' in objective.code_snippet
    assert '"objective": "multiclass"' in objective.code_snippet
    assert "4-class target" in objective.description


def test_regression_target_gets_no_classification_hints(tmp_path: Path) -> None:
    """Ensure continuous targets are left to the regression hints."""
    targets = [f"{value * 137 % 9973 / 7:.3f}" for value in range(1, 121)]
    profile, hints = _hints_for(tmp_path, targets)

    assert profile.target_class_balance is None
    assert "stratified_cv" not in hints
    assert "class_imbalance_weighting" not in hints
    assert "multiclass_objective" not in hints


def test_classification_hint_snippets_are_detected_as_applied(tmp_path: Path) -> None:
    """Ensure applied-hint tracking recognises the snippets these hints suggest."""
    _, hints = _hints_for(tmp_path, ["a"] * 19 + ["b"] + ["c"] * 5)

    classification_ids = {"stratified_cv", "class_imbalance_weighting", "multiclass_objective"}
    assert classification_ids <= set(hints)
    for hint_id in classification_ids:
        detected = detect_applied_hints(hints[hint_id].code_snippet, [hints[hint_id]])
        assert hint_id in detected


def test_target_class_balance_survives_serialization(tmp_path: Path) -> None:
    """Ensure the class balance survives the research-to-evolution mission-state round trip."""
    profile, _ = _hints_for(tmp_path, ["0"] * 9 + ["1"])

    restored = DatasetProfile.from_dict(profile.to_dict())

    assert profile.target_class_balance is not None
    assert restored.target_class_balance == profile.target_class_balance


def test_classification_hint_snippets_run_against_real_data(tmp_path: Path) -> None:
    """Ensure every suggested snippet executes against the real sklearn and LightGBM APIs."""
    train_path, test_path, _ = _write_competition(tmp_path, ["0"] * 9 + ["1"] + ["2"] * 3)
    profile = build_dataset_profile(train_path, test_path, tmp_path / "sample_submission.csv")
    hints = {hint.id: hint for hint in generate_preprocessing_hints(profile)}

    features, labels = _load_features_and_labels(train_path)
    for hint_id in ("stratified_cv", "multiclass_objective", "class_imbalance_weighting"):
        namespace: dict[str, Any] = {"X_train": features, "y_train": labels, "X_test": features[:20]}
        exec(hints[hint_id].code_snippet, namespace)


def _load_features_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = np.array([[float(row["feature_a"]), float(row["feature_b"])] for row in rows], dtype=float)
    labels = np.array([row["target"] for row in rows])
    return features, labels
