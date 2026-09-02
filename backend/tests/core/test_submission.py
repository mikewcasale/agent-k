"""Tests for submission validation and repair.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from typing import TYPE_CHECKING

from agent_k.core.submission import repair_submission, validate_submission

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, list(reader)


def _sample(path: Path, ids: list[str], targets: tuple[str, ...] = ("target",)) -> None:
    _write_csv(path, ["id", *targets], [[row_id, *["0.0"] * len(targets)] for row_id in ids])


def test_valid_submission_passes(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2", "3"])
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"], ["2", "0.25"], ["3", "1.5"]])

    validation = validate_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert validation.is_valid
    assert validation.row_count == 3
    assert validation.errors == ()


def test_missing_target_column_is_an_error(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2"])
    _write_csv(submission_path, ["id", "prediction"], [["1", "0.5"], ["2", "0.25"]])

    validation = validate_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert not validation.is_valid
    assert "missing columns: target" in validation.summary


def test_non_finite_values_are_errors(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2", "3"])
    _write_csv(submission_path, ["id", "target"], [["1", "nan"], ["2", "inf"], ["3", ""]])

    validation = validate_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert not validation.is_valid
    assert "3 non-finite value(s)" in validation.summary


def test_duplicate_and_unknown_ids_are_errors(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2"])
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"], ["1", "0.6"], ["9", "0.7"]])

    validation = validate_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert not validation.is_valid
    assert "duplicate ids: 1" in validation.summary
    assert "missing ids: 2" in validation.summary
    assert "unknown ids: 9" in validation.summary


def test_repair_reorders_columns_and_rows(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2", "3"])
    _write_csv(
        submission_path, ["target", "extra", "ID"], [["0.3", "junk", "3"], ["0.1", "junk", "1"], ["0.2", "junk", "2"]]
    )

    repaired = repair_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert repaired.is_valid, repaired.errors
    header, rows = _read_csv(submission_path)
    assert header == ["id", "target"]
    assert rows == [["1", "0.1"], ["2", "0.2"], ["3", "0.3"]]


def test_repair_drops_duplicates_and_unknown_ids(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2"])
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"], ["1", "0.9"], ["2", "0.6"], ["9", "0.7"]])

    repaired = repair_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert repaired.is_valid, repaired.errors
    _, rows = _read_csv(submission_path)
    assert rows == [["1", "0.5"], ["2", "0.6"]]
    assert any("duplicate" in repair for repair in repaired.repairs)
    assert any("absent from the sample" in repair for repair in repaired.repairs)


def test_repair_replaces_non_finite_numeric_values_with_median(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1", "2", "3", "4", "5"])
    _write_csv(
        submission_path, ["id", "target"], [["1", "1.0"], ["2", "nan"], ["3", "3.0"], ["4", "5.0"], ["5", "-inf"]]
    )

    repaired = repair_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert repaired.is_valid, repaired.errors
    _, rows = _read_csv(submission_path)
    assert [row[1] for row in rows] == ["1.0", "3.0", "3.0", "5.0", "3.0"]


def test_repair_replaces_blank_categorical_values_with_mode(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_csv(sample_path, ["id", "label"], [["1", "cat"], ["2", "cat"], ["3", "cat"]])
    _write_csv(submission_path, ["id", "label"], [["1", "dog"], ["2", "dog"], ["3", ""]])

    repaired = repair_submission(submission_path, sample_path, id_column="id", target_columns=["label"])

    assert repaired.is_valid, repaired.errors
    _, rows = _read_csv(submission_path)
    assert [row[1] for row in rows] == ["dog", "dog", "dog"]


def test_repair_backfills_only_within_the_limit(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    ids = [str(index) for index in range(200)]
    _sample(sample_path, ids)
    _write_csv(submission_path, ["id", "target"], [[row_id, "0.5"] for row_id in ids[:-1]])

    repaired = repair_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert repaired.is_valid, repaired.errors
    assert any("back-filled 1 row" in repair for repair in repaired.repairs)
    _, rows = _read_csv(submission_path)
    assert len(rows) == 200
    assert rows[-1] == ["199", "0.0"]


def test_repair_refuses_to_backfill_beyond_the_limit(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    ids = [str(index) for index in range(200)]
    _sample(sample_path, ids)
    _write_csv(submission_path, ["id", "target"], [[row_id, "0.5"] for row_id in ids[:100]])

    repaired = repair_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert not repaired.is_valid
    assert "back-fill limit" in repaired.summary
    _, rows = _read_csv(submission_path)
    assert len(rows) == 100


def test_multi_target_submission_round_trip(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["a", "b"], targets=("class_0", "class_1", "class_2"))
    _write_csv(
        submission_path,
        ["id", "class_2", "class_0", "class_1"],
        [["b", "0.1", "0.2", "0.7"], ["a", "0.3", "0.3", "0.4"]],
    )

    repaired = repair_submission(
        submission_path, sample_path, id_column="id", target_columns=["class_0", "class_1", "class_2"]
    )

    assert repaired.is_valid, repaired.errors
    header, rows = _read_csv(submission_path)
    assert header == ["id", "class_0", "class_1", "class_2"]
    assert rows == [["a", "0.3", "0.4", "0.3"], ["b", "0.2", "0.7", "0.1"]]


def test_empty_submission_is_invalid(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _sample(sample_path, ["1"])
    submission_path.write_text("", encoding="utf-8")

    validation = validate_submission(submission_path, sample_path, id_column="id", target_columns=["target"])

    assert not validation.is_valid
    assert "empty" in validation.summary
