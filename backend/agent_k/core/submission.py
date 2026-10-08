"""Structural validation for competition submission files.

@notice: |
    Structural validation for competition submission files.

@dev: |
    Compares a generated ``submission.csv`` against the competition's
    ``sample_submission.csv`` so malformed predictions are caught locally
    instead of consuming a platform submission slot. Checks are generic
    across ML problem types: column layout, row count, identifier
    coverage, and finiteness of numeric prediction cells.

@graph:
    id: agent_k.core.submission
    provides:
        - agent_k.core.submission
    pattern: validation

@agent-guidance:
    do:
        - "Use agent_k.core.submission as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Add competition-specific submission rules; keep checks generic."

@human-review:
    last-verified: 2026-10-08
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ("SubmissionValidation", "validate_submission")

_MAX_REPORTED_EXAMPLES: Final[int] = 3
"""Number of offending values quoted in an issue message."""

_MAX_TRACKED_IDS: Final[int] = 2_000_000
"""Upper bound on identifiers held in memory for set comparisons."""

_NUMERIC_SNIFF_ROWS: Final[int] = 500
"""Sample submission rows inspected to decide whether a column is numeric."""


@dataclass(frozen=True, slots=True)
class SubmissionValidation:
    """Outcome of validating a submission file against the sample submission.

    @notice: |
        Outcome of validating a submission file against the sample submission.

    @dev: |
        ``valid`` is False only for layouts a competition platform rejects.
        When the sample submission itself cannot be read the validation is
        reported as valid with a single explanatory issue, so a missing
        reference file never blocks an otherwise usable submission.

        @pattern:
            name: validation-result
            rationale: "Bundles the verdict with human-readable issues."
            violations: "Bare booleans lose the reason a submission failed."
    """

    valid: bool
    issues: tuple[str, ...]
    row_count: int
    expected_row_count: int
    columns: tuple[str, ...]
    expected_columns: tuple[str, ...]

    @property
    def summary(self) -> str:
        """Render the issues as a single line for logs and agent feedback."""
        if not self.issues:
            return "submission matches the sample submission layout"
        return "; ".join(self.issues)


@dataclass(slots=True)
class _Tally:
    """Bounded counter that keeps a few offending values for reporting.

    @notice: |
        Bounded counter that keeps a few offending values for reporting.

    @dev: |
        Keeps memory flat on large submissions: only the count grows while the
        retained examples stop at ``_MAX_REPORTED_EXAMPLES``.

        @pattern:
            name: bounded-accumulator
            rationale: "Counts every offender while retaining only a few examples."
            violations: "Unbounded example lists blow up on large submissions."
    """

    count: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, value: str) -> None:
        """Record an offending value, retaining only the first few examples."""
        self.count += 1
        if len(self.examples) < _MAX_REPORTED_EXAMPLES:
            self.examples.append(value)


@dataclass(frozen=True, slots=True)
class _SampleLayout:
    """Reference layout read from the sample submission.

    @notice: |
        Reference layout read from the sample submission.

    @dev: |
        ``ids_tracked`` is False when the sample exceeds ``_MAX_TRACKED_IDS``
        rows, in which case identifier comparisons are skipped to bound memory.

        @pattern:
            name: reference-layout
            rationale: "Captures the expected submission shape in one value."
            violations: "Re-reading the sample per check multiplies IO cost."
    """

    columns: tuple[str, ...]
    id_column: str
    row_count: int
    ids: frozenset[str]
    ids_tracked: bool
    numeric_columns: frozenset[str]


def validate_submission(submission_path: Path, sample_path: Path, *, strict: bool = True) -> SubmissionValidation:
    """Validate a submission file against the competition sample submission.

    @notice: |
        Checks submission layout, row coverage, identifiers, and numeric cells.

    @dev: |
        ``strict`` requires an exact row count and identifier match, which is
        what a platform enforces for a real submission. Pass ``strict=False``
        when the submission was produced from a data subset: the row count is
        then unconstrained and identifiers only have to be a duplicate-free
        subset of the sample submission.

    Args:
        submission_path: Path to the generated submission CSV.
        sample_path: Path to the competition ``sample_submission.csv``.
        strict: Require exact row and identifier parity with the sample.

    Returns:
        A :class:`SubmissionValidation` describing the verdict and issues.
    """
    layout = _read_sample_layout(sample_path)
    if layout is None:
        return SubmissionValidation(
            valid=True,
            issues=("sample submission unavailable: structural checks skipped",),
            row_count=0,
            expected_row_count=0,
            columns=(),
            expected_columns=(),
        )

    if not submission_path.exists():
        return SubmissionValidation(
            valid=False,
            issues=("submission.csv was not created",),
            row_count=0,
            expected_row_count=layout.row_count,
            columns=(),
            expected_columns=layout.columns,
        )

    try:
        header, row_count, ids, duplicates, unknown_ids, bad_cells = _scan_submission(submission_path, layout)
    except (OSError, UnicodeError, csv.Error) as exc:
        return SubmissionValidation(
            valid=False,
            issues=(f"submission.csv could not be read: {exc}",),
            row_count=0,
            expected_row_count=layout.row_count,
            columns=(),
            expected_columns=layout.columns,
        )

    issues: list[str] = []
    issues.extend(_column_issues(header, layout))
    if row_count == 0:
        issues.append("submission.csv has no prediction rows")
    elif strict and row_count != layout.row_count:
        issues.append(f"row count {row_count} does not match the sample submission ({layout.row_count})")

    issues.extend(_identifier_issues(ids, duplicates, unknown_ids, layout, strict=strict))
    issues.extend(_cell_issues(bad_cells))

    return SubmissionValidation(
        valid=not issues,
        issues=tuple(issues),
        row_count=row_count,
        expected_row_count=layout.row_count,
        columns=tuple(header),
        expected_columns=layout.columns,
    )


def _column_issues(header: list[str], layout: _SampleLayout) -> list[str]:
    expected = set(layout.columns)
    found = set(header)
    issues: list[str] = []
    missing = sorted(expected - found)
    if missing:
        issues.append(f"missing columns {_display_columns(missing)}")
    unexpected = sorted(found - expected)
    if unexpected:
        issues.append(f"unexpected columns {_display_columns(unexpected)}")
    return issues


def _display_columns(columns: list[str]) -> list[str]:
    return [column if column.strip() else "<unnamed>" for column in columns]


def _identifier_issues(
    ids: set[str], duplicates: _Tally, unknown_ids: _Tally, layout: _SampleLayout, *, strict: bool
) -> list[str]:
    if not layout.ids_tracked or layout.id_column not in layout.columns:
        return []

    issues: list[str] = []
    if duplicates.count:
        issues.append(
            f"{duplicates.count} duplicate '{layout.id_column}' values (e.g. {_examples(duplicates.examples)})"
        )
    if unknown_ids.count:
        issues.append(
            f"{unknown_ids.count} '{layout.id_column}' values are absent from the sample "
            f"submission (e.g. {_examples(unknown_ids.examples)})"
        )
    if strict and ids:
        missing = sorted(layout.ids - ids)
        if missing:
            issues.append(
                f"{len(missing)} sample submission '{layout.id_column}' values are missing (e.g. {_examples(missing)})"
            )
    return issues


def _cell_issues(bad_cells: dict[str, _Tally]) -> list[str]:
    issues: list[str] = []
    for column in sorted(bad_cells):
        offenders = bad_cells[column]
        issues.append(
            f"column '{column}' has {offenders.count} empty, non-numeric, or non-finite "
            f"values (e.g. {_examples(offenders.examples)})"
        )
    return issues


def _scan_submission(
    submission_path: Path, layout: _SampleLayout
) -> tuple[list[str], int, set[str], _Tally, _Tally, dict[str, _Tally]]:
    ids: set[str] = set()
    duplicates = _Tally()
    unknown_ids = _Tally()
    bad_cells: dict[str, _Tally] = {}
    row_count = 0

    with submission_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if not header:
            return [], 0, ids, duplicates, unknown_ids, bad_cells

        positions = {column: index for index, column in enumerate(header)}
        id_index = positions.get(layout.id_column)
        numeric_indexes = {column: index for column, index in positions.items() if column in layout.numeric_columns}

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            row_count += 1
            if id_index is not None and layout.ids_tracked:
                raw_id = row[id_index] if id_index < len(row) else ""
                identifier = _normalize_id(raw_id)
                if identifier in ids:
                    duplicates.add(raw_id)
                else:
                    ids.add(identifier)
                    if identifier not in layout.ids:
                        unknown_ids.add(raw_id)
            for column, index in numeric_indexes.items():
                value = row[index] if index < len(row) else ""
                if not _is_finite_number(value):
                    bad_cells.setdefault(column, _Tally()).add(value)

    return header, row_count, ids, duplicates, unknown_ids, bad_cells


def _read_sample_layout(sample_path: Path) -> _SampleLayout | None:
    if not sample_path.exists():
        return None

    try:
        with sample_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if not header:
                return None

            ids: set[str] = set()
            ids_tracked = True
            row_count = 0
            numeric_candidates = {column: True for column in header[1:]}
            seen_numeric_values = {column: False for column in header[1:]}

            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                row_count += 1
                if ids_tracked:
                    ids.add(_normalize_id(row[0] if row else ""))
                    if len(ids) > _MAX_TRACKED_IDS:
                        ids.clear()
                        ids_tracked = False
                if row_count <= _NUMERIC_SNIFF_ROWS:
                    for index, column in enumerate(header[1:], start=1):
                        value = row[index].strip() if index < len(row) else ""
                        if not value:
                            continue
                        if _is_finite_number(value):
                            seen_numeric_values[column] = True
                        else:
                            numeric_candidates[column] = False
    except (OSError, UnicodeError, csv.Error):
        return None

    numeric_columns = frozenset(
        column for column, is_numeric in numeric_candidates.items() if is_numeric and seen_numeric_values[column]
    )
    return _SampleLayout(
        columns=tuple(header),
        id_column=header[0],
        row_count=row_count,
        ids=frozenset(ids),
        ids_tracked=ids_tracked,
        numeric_columns=numeric_columns,
    )


def _normalize_id(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.lstrip("+-").isdigit():
        return str(int(text))
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _is_finite_number(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number)


def _examples(values: list[str]) -> str:
    shown = [value if value.strip() else "<empty>" for value in values[:_MAX_REPORTED_EXAMPLES]]
    return ", ".join(shown)
