"""Tests for Evolver validation-split stratification.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

__all__ = ()

try:
    from agent_k.agents.evolver import (
        _prepare_validation_split,
        _resolve_stratify_labels,
        _shrink_split,
        _subsample_indices,
    )
except TypeError as exc:  # pragma: no cover - optional dependency shim
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise

_ROW_COUNT = 200
_POSITIVE_ROWS = 3


def _write_rare_positive_csv(tmp_path: Path) -> Path:
    """Write a binary-target CSV where the positive class is rare."""
    target = np.zeros(_ROW_COUNT, dtype=np.int64)
    target[:_POSITIVE_ROWS] = 1
    frame = pd.DataFrame(
        {
            "id": np.arange(_ROW_COUNT, dtype=np.int64),
            "feature": np.arange(_ROW_COUNT, dtype=float) / _ROW_COUNT,
            "target": target,
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return path


def _write_continuous_csv(tmp_path: Path) -> Path:
    """Write a regression-target CSV with an all-distinct target."""
    frame = pd.DataFrame(
        {
            "id": np.arange(_ROW_COUNT, dtype=np.int64),
            "feature": np.arange(_ROW_COUNT, dtype=float),
            "target": np.linspace(10.0, 500.0, _ROW_COUNT),
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return path


class TestResolveStratifyLabels:
    """Tests for stratification eligibility."""

    def test_single_discrete_target_yields_codes(self) -> None:
        """A low-cardinality single target should produce class codes."""
        frame = pd.DataFrame({"target": ["a", "b", "a", "b", "a", "b"]})

        labels = _resolve_stratify_labels(frame, ["target"])

        assert labels is not None
        assert len(np.unique(labels)) == 2

    def test_missing_values_form_their_own_class(self) -> None:
        """Null targets must not be dropped from the label codes."""
        frame = pd.DataFrame({"target": [1.0, None, 1.0, None, 1.0, None]})

        labels = _resolve_stratify_labels(frame, ["target"])

        assert labels is not None
        assert len(labels) == len(frame)
        assert len(np.unique(labels)) == 2

    def test_continuous_target_is_rejected(self) -> None:
        """An all-distinct target exceeds the cardinality cap."""
        frame = pd.DataFrame({"target": np.linspace(0.0, 1.0, 50)})

        assert _resolve_stratify_labels(frame, ["target"]) is None

    def test_multi_target_is_rejected(self) -> None:
        """Multi-output problems have no single label to stratify on."""
        frame = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})

        assert _resolve_stratify_labels(frame, ["a", "b"]) is None

    def test_missing_column_is_rejected(self) -> None:
        """An absent target column disables stratification."""
        frame = pd.DataFrame({"other": [0, 1, 0, 1]})

        assert _resolve_stratify_labels(frame, ["target"]) is None


class TestPrepareValidationSplit:
    """Tests for train/validation split construction."""

    def test_stratified_split_keeps_rare_class_on_both_sides(self, tmp_path: Path) -> None:
        """Every seed must leave the rare class represented in train and validation."""
        train_path = _write_rare_positive_csv(tmp_path)

        for seed in range(10):
            train_df, _val_features, y_val, _id_column = _prepare_validation_split(
                train_path=train_path,
                id_column="id",
                target_columns=["target"],
                validation_split=0.2,
                stratify=True,
                seed=seed,
            )

            assert int(y_val["target"].sum()) >= 1
            assert int(train_df["target"].sum()) >= 1

    def test_unstratified_split_can_lose_the_rare_class(self, tmp_path: Path) -> None:
        """The unstratified path is what strands single-class validation folds."""
        train_path = _write_rare_positive_csv(tmp_path)

        positives_per_seed = []
        for seed in range(10):
            _train_df, _val_features, y_val, _id_column = _prepare_validation_split(
                train_path=train_path,
                id_column="id",
                target_columns=["target"],
                validation_split=0.2,
                stratify=False,
                seed=seed,
            )
            positives_per_seed.append(int(y_val["target"].sum()))

        assert 0 in positives_per_seed

    def test_stratified_split_preserves_every_row(self, tmp_path: Path) -> None:
        """Train and validation together must cover the full frame exactly once."""
        train_path = _write_rare_positive_csv(tmp_path)

        train_df, val_features, y_val, _id_column = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.25, stratify=True
        )

        assert len(val_features) == len(y_val)
        assert len(train_df) + len(y_val) == _ROW_COUNT
        assert set(train_df["id"]).isdisjoint(set(y_val["id"]))
        assert set(train_df["id"]) | set(y_val["id"]) == set(range(_ROW_COUNT))

    def test_stratified_split_is_deterministic(self, tmp_path: Path) -> None:
        """Repeat calls with the same seed must produce identical rows."""
        train_path = _write_rare_positive_csv(tmp_path)

        first_train, _first_features, first_y, _first_id = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, stratify=True
        )
        second_train, _second_features, second_y, _second_id = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, stratify=True
        )

        assert list(first_train["id"]) == list(second_train["id"])
        assert list(first_y["id"]) == list(second_y["id"])

    def test_continuous_target_falls_back_to_shuffle_split(self, tmp_path: Path) -> None:
        """Regression targets keep the plain split and its proportions."""
        train_path = _write_continuous_csv(tmp_path)

        train_df, _val_features, y_val, _id_column = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, stratify=True
        )

        assert len(y_val) == int(round(_ROW_COUNT * 0.2))
        assert len(train_df) == _ROW_COUNT - len(y_val)

    def test_validation_features_exclude_the_target(self, tmp_path: Path) -> None:
        """Stratifying must not leak the target into validation features."""
        train_path = _write_rare_positive_csv(tmp_path)

        _train_df, val_features, _y_val, _id_column = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, stratify=True
        )

        assert "target" not in val_features.columns


class TestSubsampleIndices:
    """Tests for bounded row selection."""

    def test_without_labels_keeps_leading_rows(self) -> None:
        """The unlabelled path matches the previous head() behaviour."""
        rng = np.random.default_rng(0)

        indices = _subsample_indices(None, row_count=100, max_rows=10, rng=rng)

        assert list(indices) == list(range(10))

    def test_max_rows_at_or_above_row_count_selects_everything(self) -> None:
        """No truncation happens when the budget covers the frame."""
        rng = np.random.default_rng(0)

        indices = _subsample_indices(None, row_count=5, max_rows=5, rng=rng)

        assert list(indices) == list(range(5))

    def test_rare_class_survives_truncation(self) -> None:
        """A class with a single member must still be selected."""
        labels = np.zeros(500, dtype=np.int64)
        labels[0] = 1
        rng = np.random.default_rng(0)

        indices = _subsample_indices(labels, row_count=500, max_rows=50, rng=rng)

        assert len(indices) <= 50
        assert 1 in set(labels[indices])
        assert 0 in set(labels[indices])

    def test_selection_respects_the_row_budget(self) -> None:
        """Per-class allocation must never exceed max_rows in total."""
        labels = np.repeat(np.arange(8, dtype=np.int64), 40)
        rng = np.random.default_rng(0)

        indices = _subsample_indices(labels, row_count=len(labels), max_rows=33, rng=rng)

        assert len(indices) <= 33
        assert len(set(indices)) == len(indices)


class TestShrinkSplit:
    """Tests for shrinking a prepared split."""

    def test_validation_frames_stay_aligned(self) -> None:
        """Features and labels must be sliced with the same positions."""
        train_df = pd.DataFrame({"id": range(100), "feature": range(100), "target": [0, 1] * 50})
        val_features = pd.DataFrame({"id": range(100, 140), "feature": range(40)})
        y_val = pd.DataFrame({"id": range(100, 140), "target": [0] * 38 + [1, 1]})

        shrunk_train, shrunk_features, shrunk_y = _shrink_split(
            train_df, val_features, y_val, target_columns=["target"], max_rows=20, stratify=True
        )

        assert list(shrunk_features["id"]) == list(shrunk_y["id"])
        assert len(shrunk_train) <= 20
        assert len(shrunk_y) <= 20
        assert int(shrunk_y["target"].sum()) >= 1

    def test_unstratified_shrink_keeps_leading_rows(self) -> None:
        """Regression problems keep the deterministic leading-row subset."""
        train_df = pd.DataFrame({"id": range(100), "target": np.linspace(0.0, 1.0, 100)})
        val_features = pd.DataFrame({"id": range(100, 140)})
        y_val = pd.DataFrame({"id": range(100, 140), "target": np.linspace(0.0, 1.0, 40)})

        shrunk_train, shrunk_features, shrunk_y = _shrink_split(
            train_df, val_features, y_val, target_columns=["target"], max_rows=10, stratify=False
        )

        assert list(shrunk_train["id"]) == list(range(10))
        assert list(shrunk_features["id"]) == list(range(100, 110))
        assert list(shrunk_y["id"]) == list(range(100, 110))
