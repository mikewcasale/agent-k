"""Tests for submission file validation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from typing import TYPE_CHECKING

from agent_k.core.submission import validate_submission

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sample(path: Path) -> None:
    _write_csv(path, ["id", "target"], [["1", "0.0"], ["2", "0.0"], ["3", "0.0"]])


def test_matching_submission_is_valid(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"], ["2", "1.5"], ["3", "2.5"]])

    validation = validate_submission(submission_path, sample_path)

    assert validation.valid
    assert validation.issues == ()
    assert validation.row_count == 3
    assert validation.expected_row_count == 3
    assert validation.summary == "submission matches the sample submission layout"


def test_missing_submission_file_is_invalid(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    _write_sample(sample_path)

    validation = validate_submission(tmp_path / "submission.csv", sample_path)

    assert not validation.valid
    assert validation.issues == ("submission.csv was not created",)


def test_header_only_submission_is_invalid(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [])

    validation = validate_submission(submission_path, sample_path)

    assert not validation.valid
    assert "no prediction rows" in validation.summary


def test_column_mismatch_is_reported(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(
        submission_path, ["id", "prediction", "extra"], [["1", "0.5", "x"], ["2", "1.5", "y"], ["3", "2.5", "z"]]
    )

    validation = validate_submission(submission_path, sample_path)

    assert not validation.valid
    assert "missing columns ['target']" in validation.summary
    assert "unexpected columns ['extra', 'prediction']" in validation.summary


def test_row_count_mismatch_is_reported(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"], ["2", "1.5"]])

    validation = validate_submission(submission_path, sample_path)

    assert not validation.valid
    assert "row count 2 does not match the sample submission (3)" in validation.summary
    assert "1 sample submission 'id' values are missing" in validation.summary


def test_duplicate_and_unknown_ids_are_reported(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"], ["1", "1.5"], ["9", "2.5"]])

    validation = validate_submission(submission_path, sample_path)

    assert not validation.valid
    assert "1 duplicate 'id' values" in validation.summary
    assert "1 'id' values are absent from the sample submission" in validation.summary


def test_non_finite_predictions_are_reported(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["1", ""], ["2", "nan"], ["3", "inf"]])

    validation = validate_submission(submission_path, sample_path)

    assert not validation.valid
    assert "column 'target' has 3 empty, non-numeric, or non-finite values" in validation.summary


def test_numeric_ids_compare_across_formats(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["1.0", "0.5"], ["2", "1.5"], [" 3 ", "2.5"]])

    validation = validate_submission(submission_path, sample_path)

    assert validation.valid, validation.summary


def test_categorical_label_columns_are_not_numeric_checked(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_csv(sample_path, ["id", "label"], [["1", "cat"], ["2", "cat"]])
    _write_csv(submission_path, ["id", "label"], [["1", "dog"], ["2", "cat"]])

    validation = validate_submission(submission_path, sample_path)

    assert validation.valid, validation.summary


def test_lenient_mode_allows_subset_rows(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"]])

    assert validate_submission(submission_path, sample_path, strict=False).valid
    assert not validate_submission(submission_path, sample_path).valid


def test_lenient_mode_still_rejects_unknown_ids(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_submission.csv"
    submission_path = tmp_path / "submission.csv"
    _write_sample(sample_path)
    _write_csv(submission_path, ["id", "target"], [["42", "0.5"]])

    validation = validate_submission(submission_path, sample_path, strict=False)

    assert not validation.valid
    assert "absent from the sample submission" in validation.summary


def test_missing_sample_skips_validation(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.csv"
    _write_csv(submission_path, ["id", "target"], [["1", "0.5"]])

    validation = validate_submission(submission_path, tmp_path / "sample_submission.csv")

    assert validation.valid
    assert validation.issues == ("sample submission unavailable: structural checks skipped",)
