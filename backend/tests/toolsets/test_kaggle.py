"""Tests for the Kaggle toolset.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.toolsets import FunctionToolset

from agent_k.toolsets.kaggle import kaggle_list_datasets, kaggle_toolset

__all__ = ()


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance."""
    assert isinstance(kaggle_toolset, FunctionToolset)


def test_toolset_id() -> None:
    """Toolset should have the expected id."""
    assert kaggle_toolset.id == "kaggle"


@dataclass
class _FakeResponse:
    """Minimal httpx.Response stand-in for tests."""

    status_code: int
    payload: Any

    def json(self) -> Any:
        return self.payload


class _FakeAdapter:
    """Adapter stub that returns a pre-baked response for /competitions/data/list."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._response = _FakeResponse(status_code=status_code, payload=payload)
        self.calls: list[tuple[str, str]] = []

    async def _request(self, method: str, path: str, **_: Any) -> _FakeResponse:
        self.calls.append((method, path))
        return self._response


def _make_ctx(adapter: _FakeAdapter) -> SimpleNamespace:
    deps = SimpleNamespace(kaggle_adapter=adapter, event_emitter=None)
    return SimpleNamespace(deps=deps)


@pytest.mark.asyncio
async def test_list_datasets_dict_payload_with_files_key() -> None:
    """Kaggle returns a dict-wrapped payload; the toolset must unwrap it."""
    adapter = _FakeAdapter(
        {
            "files": [
                {"name": "train.csv", "totalBytes": 100, "description": "train"},
                {"name": "test.csv", "totalBytes": 50, "description": "test"},
            ],
            "datasetVersionNumber": 1,
        }
    )

    result = await kaggle_list_datasets(_make_ctx(adapter), "comp-slug")

    assert result["competition_id"] == "comp-slug"
    assert result["files"] == [
        {"name": "train.csv", "size": 100, "description": "train"},
        {"name": "test.csv", "size": 50, "description": "test"},
    ]


@pytest.mark.asyncio
async def test_list_datasets_falls_back_to_name_nullable() -> None:
    """Older payloads expose nameNullable rather than name."""
    adapter = _FakeAdapter({"files": [{"nameNullable": "train.csv", "totalBytes": 1, "description": None}]})

    result = await kaggle_list_datasets(_make_ctx(adapter), "comp")

    assert result["files"] == [{"name": "train.csv", "size": 1, "description": None}]


@pytest.mark.asyncio
async def test_list_datasets_accepts_plain_list_payload() -> None:
    """Some endpoints return the file list directly without a wrapper dict."""
    adapter = _FakeAdapter([{"name": "train.csv", "totalBytes": 7, "description": "d"}])

    result = await kaggle_list_datasets(_make_ctx(adapter), "comp")

    assert result["files"] == [{"name": "train.csv", "size": 7, "description": "d"}]


@pytest.mark.asyncio
async def test_list_datasets_skips_unparseable_entries() -> None:
    """String entries become name-only and unknown shapes are skipped."""
    adapter = _FakeAdapter({"files": ["sample_submission.csv", 42, {"name": "train.csv"}]})

    result = await kaggle_list_datasets(_make_ctx(adapter), "comp")

    assert result["files"] == [
        {"name": "sample_submission.csv", "size": None, "description": None},
        {"name": "train.csv", "size": None, "description": None},
    ]
