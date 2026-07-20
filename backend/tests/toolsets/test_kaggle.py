"""Tests for the Kaggle toolset.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic_ai.toolsets import FunctionToolset

from agent_k.toolsets.kaggle import _parse_dataset_files, kaggle_list_datasets, kaggle_toolset

__all__ = ()

pytestmark = pytest.mark.anyio


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance."""
    assert isinstance(kaggle_toolset, FunctionToolset)


def test_toolset_id() -> None:
    """Toolset should have the expected id."""
    assert kaggle_toolset.id == "kaggle"


class TestParseDatasetFiles:
    """Tests for the private ``_parse_dataset_files`` normalizer."""

    def test_dict_payload_with_files_key(self) -> None:
        """Kaggle wrapper dict ``{"files": [...]}`` should be unwrapped."""
        payload = {"files": [{"name": "train.csv", "totalBytes": 1024, "description": "train set"}]}

        result = _parse_dataset_files(payload)

        assert result == [{"name": "train.csv", "size": 1024, "description": "train set"}]

    def test_list_payload(self) -> None:
        """Bare list payloads should be parsed directly."""
        payload = [{"name": "test.csv", "totalBytes": 2048, "description": None}]

        result = _parse_dataset_files(payload)

        assert result == [{"name": "test.csv", "size": 2048, "description": None}]

    def test_string_entries(self) -> None:
        """String entries should be surfaced as name-only records."""
        payload = ["sample_submission.csv"]

        result = _parse_dataset_files(payload)

        assert result == [{"name": "sample_submission.csv", "size": None, "description": None}]

    def test_name_nullable_variant(self) -> None:
        """``nameNullable`` and ``size`` fallbacks should be honored."""
        payload = {"files": [{"nameNullable": "hidden.parquet", "size": 512}]}

        result = _parse_dataset_files(payload)

        assert result == [{"name": "hidden.parquet", "size": 512, "description": None}]

    def test_total_bytes_takes_precedence_over_size(self) -> None:
        """``totalBytes`` should win when both are present."""
        payload = [{"name": "features.csv", "totalBytes": 999, "size": 111}]

        result = _parse_dataset_files(payload)

        assert result == [{"name": "features.csv", "size": 999, "description": None}]

    def test_malformed_entries_are_skipped(self) -> None:
        """Non-str/dict entries should be filtered out without raising."""
        payload = [{"name": "a.csv"}, 42, None, {"name": "b.csv"}]

        result = _parse_dataset_files(payload)

        assert [item["name"] for item in result] == ["a.csv", "b.csv"]

    def test_non_list_payload_returns_empty(self) -> None:
        """Unexpected payload shapes should yield an empty list, not crash."""
        assert _parse_dataset_files({"files": "not-a-list"}) == []
        assert _parse_dataset_files(42) == []
        assert _parse_dataset_files(None) == []


@dataclass
class _StubDeps:
    """Minimal ``ctx.deps`` payload for tool invocation tests."""

    kaggle_adapter: Any
    event_emitter: Any = None


class _StubRunContext:
    """Duck-typed ``RunContext`` sufficient for the Kaggle toolset."""

    def __init__(self, deps: _StubDeps) -> None:
        self.deps = deps


class TestKaggleListDatasets:
    """Tests for ``kaggle_list_datasets`` end-to-end response handling."""

    def _adapter_with_payload(self, payload: Any, status_code: int = 200) -> AsyncMock:
        response = httpx.Response(status_code, json=payload)
        adapter = AsyncMock()
        adapter._request = AsyncMock(return_value=response)
        return adapter

    async def test_dict_payload_end_to_end(self) -> None:
        """Dict-shaped Kaggle responses should not crash the tool."""
        adapter = self._adapter_with_payload(
            {"files": [{"name": "train.csv", "totalBytes": 1024, "description": "train"}]}
        )
        ctx = _StubRunContext(_StubDeps(kaggle_adapter=adapter))

        result = await kaggle_list_datasets(ctx, competition_id="titanic")  # type: ignore[arg-type]

        assert result["competition_id"] == "titanic"
        assert result["files"] == [{"name": "train.csv", "size": 1024, "description": "train"}]
        adapter._request.assert_awaited_once_with("GET", "/competitions/data/list/titanic")

    async def test_list_payload_end_to_end(self) -> None:
        """Legacy bare-list payloads still work."""
        adapter = self._adapter_with_payload([{"name": "test.csv", "totalBytes": 2048}])
        ctx = _StubRunContext(_StubDeps(kaggle_adapter=adapter))

        result = await kaggle_list_datasets(ctx, competition_id="titanic")  # type: ignore[arg-type]

        assert result["files"] == [{"name": "test.csv", "size": 2048, "description": None}]

    async def test_non_200_status_returns_error(self) -> None:
        """Non-200 status is surfaced by the telemetry error wrapper."""
        adapter = self._adapter_with_payload({}, status_code=500)
        ctx = _StubRunContext(_StubDeps(kaggle_adapter=adapter))

        result = await kaggle_list_datasets(ctx, competition_id="titanic")  # type: ignore[arg-type]

        assert "error" in result
        assert "500" in result["error"]
