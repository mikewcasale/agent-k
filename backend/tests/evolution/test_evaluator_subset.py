"""Tests for the OpenEvolve stage 2 data subset builder.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pandas as pd

from agent_k.evolution.evaluator import _create_subset_data, _infer_target_column, _stratified_head

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _write_competition_data(
    directory: Path, *, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None = None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / "train.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
    if sample is not None:
        sample.to_csv(directory / "sample_submission.csv", index=False)


def test_sample_submission_is_trimmed_to_the_test_subset(tmp_path: Path) -> None:
    """The generated prototype assigns test predictions into the sample frame.

    A full-length sample submission beside a truncated test set makes that
    assignment raise, which failed every candidate at stage 2.
    """
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    rows = 50
    _write_competition_data(
        source,
        train=pd.DataFrame({"Id": range(rows), "feature": range(rows), "target": [0.5] * rows}),
        test=pd.DataFrame({"Id": range(rows, 2 * rows), "feature": range(rows)}),
        sample=pd.DataFrame({"Id": range(rows, 2 * rows), "target": [0.0] * rows}),
    )

    _create_subset_data(source, target, max_rows=10)

    test_subset = pd.read_csv(target / "test.csv")
    sample_subset = pd.read_csv(target / "sample_submission.csv")

    assert len(test_subset) == 10
    assert len(sample_subset) == len(test_subset)

    # The exact pattern emitted by PrototypeNode._generate_prototype.
    sample_subset["target"] = [0.1] * len(test_subset)


def test_sample_submission_rows_follow_test_row_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    _write_competition_data(
        source,
        train=pd.DataFrame({"Id": [1, 2, 3], "feature": [1, 2, 3], "target": [0, 1, 0]}),
        test=pd.DataFrame({"Id": [30, 10, 20, 40], "feature": [3, 1, 2, 4]}),
        sample=pd.DataFrame({"Id": [10, 20, 30, 40], "target": [0, 0, 0, 0]}),
    )

    _create_subset_data(source, target, max_rows=3)

    test_subset = pd.read_csv(target / "test.csv")
    sample_subset = pd.read_csv(target / "sample_submission.csv")

    assert list(test_subset["Id"]) == [30, 10, 20]
    assert list(sample_subset["Id"]) == [30, 10, 20]


def test_sample_submission_falls_back_to_positional_alignment(tmp_path: Path) -> None:
    """Disjoint identifiers still yield equal row counts."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    _write_competition_data(
        source,
        train=pd.DataFrame({"row": [1, 2, 3], "feature": [1, 2, 3], "target": [0, 1, 0]}),
        test=pd.DataFrame({"row": [1, 2, 3, 4, 5], "feature": [1, 2, 3, 4, 5]}),
        sample=pd.DataFrame({"Id": ["a", "b", "c", "d", "e"], "target": [0, 0, 0, 0, 0]}),
    )

    _create_subset_data(source, target, max_rows=2)

    test_subset = pd.read_csv(target / "test.csv")
    sample_subset = pd.read_csv(target / "sample_submission.csv")

    assert len(test_subset) == 2
    assert len(sample_subset) == 2


def test_missing_sample_submission_is_tolerated(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    _write_competition_data(
        source,
        train=pd.DataFrame({"Id": [1, 2, 3], "feature": [1, 2, 3], "target": [0.1, 0.2, 0.3]}),
        test=pd.DataFrame({"Id": [4, 5, 6], "feature": [4, 5, 6]}),
    )

    _create_subset_data(source, target, max_rows=2)

    assert len(pd.read_csv(target / "test.csv")) == 2
    assert not (target / "sample_submission.csv").exists()


def test_train_subset_widens_when_leading_rows_collapse_the_target(tmp_path: Path) -> None:
    """A label-sorted train file must not hand stage 2 a single-class subset."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    rows = 400
    labels = [0] * (rows // 2) + [1] * (rows // 2)
    _write_competition_data(
        source,
        train=pd.DataFrame({"Id": range(rows), "feature": range(rows), "target": labels}),
        test=pd.DataFrame({"Id": range(rows, rows + 20), "feature": range(20)}),
        sample=pd.DataFrame({"Id": range(rows, rows + 20), "target": [0] * 20}),
    )

    _create_subset_data(source, target, max_rows=20)

    train_subset = pd.read_csv(target / "train.csv")

    assert set(train_subset["target"]) == {0, 1}
    assert len(train_subset) <= 20


def test_train_subset_keeps_head_for_continuous_targets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    rows = 100
    _write_competition_data(
        source,
        train=pd.DataFrame({"Id": range(rows), "feature": range(rows), "target": [i * 0.5 for i in range(rows)]}),
        test=pd.DataFrame({"Id": range(rows, rows + 10), "feature": range(10)}),
        sample=pd.DataFrame({"Id": range(rows, rows + 10), "target": [0.0] * 10}),
    )

    _create_subset_data(source, target, max_rows=10)

    train_subset = pd.read_csv(target / "train.csv")

    assert list(train_subset["Id"]) == list(range(10))


def test_train_subset_kept_when_target_is_genuinely_constant(tmp_path: Path) -> None:
    """Scanning stops at end of file instead of loading the whole train set."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()

    rows = 40
    _write_competition_data(
        source,
        train=pd.DataFrame({"Id": range(rows), "feature": range(rows), "target": [7] * rows}),
        test=pd.DataFrame({"Id": range(rows, rows + 5), "feature": range(5)}),
        sample=pd.DataFrame({"Id": range(rows, rows + 5), "target": [0] * 5}),
    )

    _create_subset_data(source, target, max_rows=5)

    train_subset = pd.read_csv(target / "train.csv")

    assert len(train_subset) == 5
    assert set(train_subset["target"]) == {7}


def test_infer_target_column_requires_exactly_one_train_only_column() -> None:
    assert _infer_target_column(["Id", "feature", "target"], ["Id", "feature"]) == "target"
    assert _infer_target_column(["Id", "feature", "a", "b"], ["Id", "feature"]) is None
    assert _infer_target_column(["Id", "feature"], ["Id", "feature"]) is None
    assert _infer_target_column(["Id", "feature", "target"], None) is None


def test_stratified_head_covers_every_class_within_the_row_budget() -> None:
    frame = pd.DataFrame({"feature": range(30), "target": [0] * 10 + [1] * 10 + [2] * 10})

    sampled = _stratified_head(frame, "target", max_rows=6)

    assert len(sampled) == 6
    assert set(sampled["target"]) == {0, 1, 2}


def test_stratified_head_falls_back_to_head_for_high_cardinality_targets() -> None:
    frame = pd.DataFrame({"feature": range(20), "target": range(20)})

    sampled = _stratified_head(frame, "target", max_rows=5)

    assert list(sampled["target"]) == [0, 1, 2, 3, 4]
