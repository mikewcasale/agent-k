"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import scientist_agent, scientist_agent_instance

if TYPE_CHECKING:
    from pathlib import Path

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


class TestDatasetSummary:
    """Tests for the dataset summary handed to the research prompt."""

    def test_summarizes_parquet_files(self, tmp_path: Path) -> None:
        """Columnar data files should report columns and row counts."""
        path = tmp_path / "train.parquet"
        pd.DataFrame({"id": [1, 2, 3], "target": [0.5, None, 1.5]}).to_parquet(path, index=False)

        summary = scientist_agent_instance._summarize_dataset([str(path)])

        assert summary["files"][0]["row_count"] == 3
        assert summary["files"][0]["columns"] == ["id", "target"]
        assert summary["files"][0]["missing_values"] == {"target": 1}

    def test_skips_non_tabular_files(self, tmp_path: Path) -> None:
        """Binary payloads should be listed without column details."""
        path = tmp_path / "images.bin"
        path.write_bytes(b"\x89PNG\x00\x01\x02\x03")

        summary = scientist_agent_instance._summarize_dataset([str(path)])

        assert summary["files"][0]["name"] == "images.bin"
        assert "columns" not in summary["files"][0]
