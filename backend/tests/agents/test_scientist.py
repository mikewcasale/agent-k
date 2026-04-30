"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
from pathlib import Path

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import ScientistAgent, scientist_agent, scientist_agent_instance

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


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


class TestSummarizeCsv:
    """Tests for ScientistAgent._summarize_csv streaming behavior."""

    def test_empty_file_returns_zero_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        result = _scientist()._summarize_csv(path)
        assert result == {"row_count": 0, "column_count": 0}

    def test_header_only_file(self, tmp_path: Path) -> None:
        path = tmp_path / "header_only.csv"
        _write_csv(path, ["id", "feature_a", "target"], [])
        result = _scientist()._summarize_csv(path)
        assert result == {
            "row_count": 0,
            "column_count": 3,
            "columns": ["id", "feature_a", "target"],
            "missing_values": {},
        }

    def test_small_file_counts_rows_and_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "small.csv"
        _write_csv(
            path,
            ["id", "feature_a", "target"],
            [["1", "0.5", "1"], ["2", "", "0"], ["3", "NaN", "1"], ["4", " None ", "0"], ["5", "1.5", "1"]],
        )
        result = _scientist()._summarize_csv(path)
        assert result == {
            "row_count": 5,
            "column_count": 3,
            "columns": ["id", "feature_a", "target"],
            "missing_values": {"feature_a": 3},
        }

    def test_sample_window_caps_missing_counts_at_first_100_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "large.csv"
        rows = [[str(i), "" if i <= 100 else "value"] for i in range(1, 251)]
        _write_csv(path, ["id", "feature"], rows)
        result = _scientist()._summarize_csv(path)
        assert result["row_count"] == 250
        assert result["missing_values"] == {"feature": 100}
        assert result["column_count"] == 2
        assert result["columns"] == ["id", "feature"]

    def test_missing_beyond_sample_window_is_not_counted(self, tmp_path: Path) -> None:
        path = tmp_path / "tail_missing.csv"
        rows = [[str(i), "value" if i <= 100 else ""] for i in range(1, 251)]
        _write_csv(path, ["id", "feature"], rows)
        result = _scientist()._summarize_csv(path)
        assert result["row_count"] == 250
        assert result["missing_values"] == {}

    def test_uneven_row_widths_do_not_crash(self, tmp_path: Path) -> None:
        path = tmp_path / "ragged.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("a,b,c\n")
            handle.write("1,2\n")
            handle.write("3,4,5,6\n")
        result = _scientist()._summarize_csv(path)
        assert result["row_count"] == 2
        assert result["column_count"] == 3
        assert result["columns"] == ["a", "b", "c"]

    def test_summarizes_large_file_correctly(self, tmp_path: Path) -> None:
        # Regression guard: a row count well above the sample window must
        # surface the true size while keeping missing-value sampling capped.
        path = tmp_path / "wide.csv"
        sentinel_rows = 25_000
        rows = [[str(i), "" if i % 3 == 0 else "value"] for i in range(1, sentinel_rows + 1)]
        _write_csv(path, ["id", "feature"], rows)

        result = _scientist()._summarize_csv(path)

        assert result["row_count"] == sentinel_rows
        assert result["column_count"] == 2
        assert result["columns"] == ["id", "feature"]
        # First 100 sampled rows include indices 1..100; ones divisible by 3 are
        # missing — i.e. 33 entries (3, 6, ..., 99).
        assert result["missing_values"] == {"feature": 33}


def _scientist() -> ScientistAgent:
    """Return the module-level ScientistAgent for unit-testing private helpers."""
    return scientist_agent_instance
