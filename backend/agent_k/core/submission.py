"""Submission file validation and repair for AGENT-K.

@notice: |
    Structural validation and best-effort repair of generated submission files.

@dev: |
    Kaggle competitions cap the number of daily submissions, so uploading a
    structurally broken ``submission.csv`` wastes a scarce slot and ends the
    mission with a rejection instead of a score. These helpers compare a
    generated submission against the competition's ``sample_submission.csv``
    (the authoritative shape contract) and repair the mismatches that are
    recoverable without inventing predictions.

    All checks are generic across ML problem types: column identity comes from
    the inferred schema and value expectations come from the sample file, so no
    competition-specific knowledge is encoded here.

@graph:
    id: agent_k.core.submission
    provides:
        - agent_k.core.submission:SubmissionValidation
        - agent_k.core.submission:validate_submission
        - agent_k.core.submission:repair_submission
    pattern: validation

@agent-guidance:
    do:
        - "Use agent_k.core.submission before uploading any submission file."
        - "Derive column expectations from CompetitionSchema and the sample file."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Add competition-specific column or value rules."

@human-review:
    last-verified: 2026-09-02
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

from .sage import Doc, Range

__all__ = ("MAX_FILL_RATIO", "SubmissionValidation", "repair_submission", "validate_submission")

MAX_FILL_RATIO: Final[float] = 0.01
"""Largest share of sample rows that may be back-filled from the sample file."""

MAX_REPORTED_EXAMPLES: Final[int] = 5
"""Number of offending identifiers or columns quoted in an issue message."""


@dataclass(frozen=True, slots=True)
class SubmissionValidation:
    """Outcome of validating a submission file against a sample submission.

    @notice: |
        Outcome of validating a submission file against a sample submission.

    @dev: |
        ``errors`` block an upload, ``warnings`` are informational, and
        ``repairs`` records the fixes applied by :func:`repair_submission`.

        @pattern:
            name: output-model
            rationale: "Stable schema for submission validation reporting."
            violations: "Bare booleans hide why an upload was rejected."
    """

    is_valid: bool
    row_count: int
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        """Return a single-line summary of the blocking errors."""
        return "; ".join(self.errors) if self.errors else "ok"


def validate_submission(
    submission_path: Annotated[Path, Doc("Generated submission file to validate.")],
    sample_path: Annotated[Path, Doc("Competition sample submission defining the expected shape.")],
    *,
    id_column: Annotated[str, Doc("Identifier column name from the inferred schema.")],
    target_columns: Annotated[list[str], Doc("Target column names from the inferred schema.")],
) -> SubmissionValidation:
    """Validate a submission file against the competition sample submission.

    @notice: |
        Checks columns, row identity, and value finiteness without mutating files.

    @dev: |
        Column expectations come from the inferred schema; per-column value
        expectations (numeric vs categorical) are learned from the sample file
        so probability, regression, and label submissions all validate.
    """
    try:
        sample_header, sample_rows = _load_rows(sample_path)
    except (OSError, ValueError) as exc:
        return SubmissionValidation(is_valid=False, row_count=0, errors=(f"sample submission unreadable: {exc}",))

    try:
        header, rows = _load_rows(submission_path)
    except (OSError, ValueError) as exc:
        return SubmissionValidation(is_valid=False, row_count=0, errors=(f"submission unreadable: {exc}",))

    expected = [id_column, *target_columns]
    errors: list[str] = []
    warnings: list[str] = []

    column_index = _resolve_columns(header, expected)
    missing_columns = [name for name in expected if name not in column_index]
    if missing_columns:
        errors.append(f"missing columns: {_format_examples(missing_columns)}")
        return SubmissionValidation(is_valid=False, row_count=len(rows), errors=tuple(errors))

    if len(header) != len(expected) or list(header) != expected:
        warnings.append(f"column layout {header} differs from expected {expected}")

    sample_index = _resolve_columns(sample_header, expected)
    sample_ids = [_cell(row, sample_index[id_column]) for row in sample_rows if sample_index[id_column] < len(row)]
    submission_ids = [_cell(row, column_index[id_column]) for row in rows if column_index[id_column] < len(row)]

    errors.extend(_check_ids(sample_ids, submission_ids))
    errors.extend(
        _check_values(
            rows=rows,
            column_index=column_index,
            sample_rows=sample_rows,
            sample_index=sample_index,
            target_columns=target_columns,
        )
    )

    if not errors and submission_ids != sample_ids:
        warnings.append("row order differs from sample submission")

    return SubmissionValidation(
        is_valid=not errors, row_count=len(rows), errors=tuple(errors), warnings=tuple(warnings)
    )


def repair_submission(
    submission_path: Annotated[Path, Doc("Submission file to repair in place.")],
    sample_path: Annotated[Path, Doc("Competition sample submission defining the expected shape.")],
    *,
    id_column: Annotated[str, Doc("Identifier column name from the inferred schema.")],
    target_columns: Annotated[list[str], Doc("Target column names from the inferred schema.")],
    max_fill_ratio: Annotated[
        float, Doc("Largest share of rows that may be back-filled from the sample."), Range(0.0, 1.0)
    ] = MAX_FILL_RATIO,
) -> SubmissionValidation:
    """Repair recoverable submission defects and rewrite the file atomically.

    @notice: |
        Fixes column layout, row identity, and non-finite values where possible.

    @dev: |
        Repairs never invent predictions beyond a bounded back-fill: rows absent
        from the submission are taken from the sample file only while they stay
        under ``max_fill_ratio`` of the expected rows. Non-finite or blank cells
        are replaced by the column's median (numeric) or modal (categorical)
        value so a single bad row cannot void an otherwise usable submission.
        Returns the validation of the rewritten file.
    """
    try:
        sample_header, sample_rows = _load_rows(sample_path)
        header, rows = _load_rows(submission_path)
    except (OSError, ValueError) as exc:
        return SubmissionValidation(is_valid=False, row_count=0, errors=(f"submission unreadable: {exc}",))

    expected = [id_column, *target_columns]
    column_index = _resolve_columns(header, expected)
    missing_columns = [name for name in expected if name not in column_index]
    if missing_columns:
        return SubmissionValidation(
            is_valid=False, row_count=len(rows), errors=(f"missing columns: {_format_examples(missing_columns)}",)
        )

    repairs: list[str] = []
    if list(header) != expected:
        repairs.append(f"normalized columns to {expected}")

    sample_index = _resolve_columns(sample_header, expected)
    sample_by_id: dict[str, list[str]] = {}
    for row in sample_rows:
        if sample_index[id_column] < len(row):
            sample_by_id.setdefault(row[sample_index[id_column]], [_cell(row, sample_index[name]) for name in expected])

    projected: dict[str, list[str]] = {}
    duplicates = 0
    for row in rows:
        if column_index[id_column] >= len(row):
            continue
        row_id = _cell(row, column_index[id_column])
        if row_id in projected:
            duplicates += 1
            continue
        projected[row_id] = [_cell(row, column_index[name]) for name in expected]
    if duplicates:
        repairs.append(f"dropped {duplicates} duplicate id row(s)")

    extra_ids = [row_id for row_id in projected if row_id not in sample_by_id]
    if extra_ids:
        repairs.append(f"dropped {len(extra_ids)} row(s) with ids absent from the sample submission")

    missing_ids = [row_id for row_id in sample_by_id if row_id not in projected]
    allowed_fill = int(len(sample_by_id) * max_fill_ratio)
    if len(missing_ids) > allowed_fill:
        return SubmissionValidation(
            is_valid=False,
            row_count=len(projected),
            errors=(
                f"{len(missing_ids)} of {len(sample_by_id)} sample ids are missing, "
                f"above the {max_fill_ratio:.1%} back-fill limit",
            ),
            repairs=tuple(repairs),
        )
    if missing_ids:
        repairs.append(f"back-filled {len(missing_ids)} row(s) from the sample submission")

    ordered = [projected.get(row_id) or list(sample_by_id[row_id]) for row_id in sample_by_id]
    filled = _fill_invalid_cells(
        ordered=ordered,
        expected=expected,
        target_columns=target_columns,
        sample_by_id=sample_by_id,
        numeric_targets=_numeric_targets(sample_rows, sample_index, target_columns),
    )
    if filled:
        repairs.append(f"replaced {filled} non-finite or blank value(s)")

    _write_rows(submission_path, expected, ordered)
    validation = validate_submission(submission_path, sample_path, id_column=id_column, target_columns=target_columns)
    return SubmissionValidation(
        is_valid=validation.is_valid,
        row_count=validation.row_count,
        errors=validation.errors,
        warnings=validation.warnings,
        repairs=tuple(repairs),
    )


def _check_ids(sample_ids: list[str], submission_ids: list[str]) -> list[str]:
    errors: list[str] = []
    if len(submission_ids) != len(sample_ids):
        errors.append(f"row count {len(submission_ids)} does not match sample {len(sample_ids)}")

    duplicates = [row_id for row_id, count in Counter(submission_ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate ids: {_format_examples(duplicates)}")

    submission_set = set(submission_ids)
    missing = [row_id for row_id in sample_ids if row_id not in submission_set]
    if missing:
        errors.append(f"missing ids: {_format_examples(missing)}")

    sample_set = set(sample_ids)
    unknown = [row_id for row_id in submission_ids if row_id not in sample_set]
    if unknown:
        errors.append(f"unknown ids: {_format_examples(unknown)}")
    return errors


def _check_values(
    *,
    rows: list[list[str]],
    column_index: dict[str, int],
    sample_rows: list[list[str]],
    sample_index: dict[str, int],
    target_columns: list[str],
) -> list[str]:
    numeric_targets = _numeric_targets(sample_rows, sample_index, target_columns)
    errors: list[str] = []
    for column in target_columns:
        index = column_index[column]
        invalid = sum(1 for row in rows if not _is_valid_cell(_cell(row, index), numeric=column in numeric_targets))
        if invalid:
            kind = "non-finite" if column in numeric_targets else "blank"
            errors.append(f"column '{column}' has {invalid} {kind} value(s)")
    return errors


def _fill_invalid_cells(
    *,
    ordered: list[list[str]],
    expected: list[str],
    target_columns: list[str],
    sample_by_id: dict[str, list[str]],
    numeric_targets: set[str],
) -> int:
    filled = 0
    for column in target_columns:
        position = expected.index(column)
        numeric = column in numeric_targets
        invalid_rows = [
            index for index, row in enumerate(ordered) if not _is_valid_cell(row[position], numeric=numeric)
        ]
        if not invalid_rows:
            continue
        replacement = _replacement_value(
            [row[position] for row in ordered if _is_valid_cell(row[position], numeric=numeric)], numeric=numeric
        )
        for index in invalid_rows:
            row = ordered[index]
            fallback = sample_by_id.get(row[0], [""] * len(expected))[position]
            row[position] = replacement if replacement is not None else fallback
            filled += 1
    return filled


def _replacement_value(values: list[str], *, numeric: bool) -> str | None:
    if not values:
        return None
    if not numeric:
        return Counter(values).most_common(1)[0][0]
    numbers = sorted(float(value) for value in values)
    middle = len(numbers) // 2
    median = numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2.0
    return repr(median)


def _numeric_targets(sample_rows: list[list[str]], sample_index: dict[str, int], target_columns: list[str]) -> set[str]:
    numeric: set[str] = set()
    for column in target_columns:
        index = sample_index.get(column)
        if index is None:
            continue
        values = [_cell(row, index) for row in sample_rows]
        present = [value for value in values if value != ""]
        if present and all(_parse_float(value) is not None for value in present):
            numeric.add(column)
    return numeric


def _is_valid_cell(value: str, *, numeric: bool) -> bool:
    if value == "":
        return False
    if not numeric:
        return True
    parsed = _parse_float(value)
    return parsed is not None and math.isfinite(parsed)


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_columns(header: list[str], expected: list[str]) -> dict[str, int]:
    by_exact = {name: index for index, name in enumerate(header)}
    by_normalized: dict[str, int] = {}
    for index, name in enumerate(header):
        by_normalized.setdefault(_normalize(name), index)

    resolved: dict[str, int] = {}
    for name in expected:
        if name in by_exact:
            resolved[name] = by_exact[name]
            continue
        normalized_index = by_normalized.get(_normalize(name))
        if normalized_index is not None:
            resolved[name] = normalized_index
    return resolved


def _normalize(name: str) -> str:
    return name.strip().casefold().replace(" ", "_")


def _cell(row: list[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def _format_examples(values: list[str]) -> str:
    head = ", ".join(values[:MAX_REPORTED_EXAMPLES])
    remaining = len(values) - MAX_REPORTED_EXAMPLES
    return f"{head} (+{remaining} more)" if remaining > 0 else head


def _load_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path.name} is empty") from None
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not header:
        raise ValueError(f"{path.name} has no header")
    return [name.strip() for name in header], rows


def _write_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    temp_path = path.with_name(f"{path.name}.repair")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temp_path, path)
