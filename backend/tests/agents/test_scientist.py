"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import (
    _CSV_SUMMARY_FIELD_LIMIT,
    _raise_field_size_limit,
    scientist_agent,
    scientist_agent_instance,
)

if TYPE_CHECKING:
    pass

__all__ = ()

pytestmark = pytest.mark.anyio


class TestScientistAgentSingleton:
    """Tests for the Scientist agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent("scientist") is scientist_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(scientist_agent, Agent)
        assert scientist_agent.name == "scientist"


class TestSummarizeCsv:
    """Tests for the streaming CSV summary helper."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty CSV must summarize to zero rows and columns."""
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        summary = scientist_agent_instance._summarize_csv(path)
        assert summary == {"row_count": 0, "column_count": 0}

    def test_header_only(self, tmp_path: Path) -> None:
        """A header-only CSV should yield zero rows and no missing values."""
        path = tmp_path / "header.csv"
        path.write_text("a,b,c\n", encoding="utf-8")
        summary = scientist_agent_instance._summarize_csv(path)
        assert summary["row_count"] == 0
        assert summary["column_count"] == 3
        assert summary["columns"] == ["a", "b", "c"]
        assert summary["missing_values"] == {}

    def test_missing_value_detection_within_sample(self, tmp_path: Path) -> None:
        """Missing-value counts must be accumulated for rows within the sample window."""
        path = tmp_path / "missing.csv"
        rows = ["a,b,c"] + ["1,,3", "1,NA,3", "1,2,null"]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        summary = scientist_agent_instance._summarize_csv(path)
        assert summary["row_count"] == 3
        assert summary["column_count"] == 3
        assert summary["missing_values"] == {"b": 2, "c": 1}

    def test_row_count_beyond_sample(self, tmp_path: Path) -> None:
        """Row count must reflect the full file even when sampling caps at 100."""
        path = tmp_path / "many_rows.csv"
        header = "a,b\n"
        body_rows = [f"{i},{i * 2}" for i in range(250)]
        path.write_text(header + "\n".join(body_rows) + "\n", encoding="utf-8")
        summary = scientist_agent_instance._summarize_csv(path)
        assert summary["row_count"] == 250
        assert summary["column_count"] == 2
        assert summary["missing_values"] == {}

    def test_short_row_does_not_error(self, tmp_path: Path) -> None:
        """Rows with fewer fields than the header are tolerated via strict=False zip."""
        path = tmp_path / "short.csv"
        path.write_text("a,b,c\n1,2\n1,2,3\n", encoding="utf-8")
        summary = scientist_agent_instance._summarize_csv(path)
        assert summary["row_count"] == 2
        assert summary["column_count"] == 3

    def test_field_size_limit_is_raised(self) -> None:
        """The helper must raise the csv field size limit toward the requested target."""
        original = csv.field_size_limit()
        try:
            csv.field_size_limit(16 * 1024)
            _raise_field_size_limit(_CSV_SUMMARY_FIELD_LIMIT)
            assert csv.field_size_limit() >= _CSV_SUMMARY_FIELD_LIMIT
        finally:
            csv.field_size_limit(original)

    def test_field_size_limit_no_downgrade(self) -> None:
        """The helper must not lower an already large field size limit."""
        original = csv.field_size_limit()
        try:
            higher = max(_CSV_SUMMARY_FIELD_LIMIT * 2, original)
            csv.field_size_limit(higher)
            _raise_field_size_limit(_CSV_SUMMARY_FIELD_LIMIT)
            assert csv.field_size_limit() == higher
        finally:
            csv.field_size_limit(original)

    def test_handles_non_utf8_bytes(self, tmp_path: Path) -> None:
        """Non-UTF-8 bytes must be replaced rather than raise UnicodeDecodeError."""
        path = tmp_path / "bad_encoding.csv"
        path.write_bytes(b"a,b\n1,\xff\n2,\xfe\n")
        summary = scientist_agent_instance._summarize_csv(path)
        assert summary["row_count"] == 2
        assert summary["column_count"] == 2

    def test_handles_wide_field(self, tmp_path: Path) -> None:
        """Fields larger than the csv default limit must be readable after raising it."""
        path = tmp_path / "wide_field.csv"
        big_cell = "x" * (200_000)
        path.write_text(f"a,b\n{big_cell},1\n", encoding="utf-8")
        original = csv.field_size_limit()
        try:
            csv.field_size_limit(16 * 1024)
            summary = scientist_agent_instance._summarize_csv(path)
        finally:
            csv.field_size_limit(original)
        assert summary["row_count"] == 1
        assert summary["column_count"] == 2
