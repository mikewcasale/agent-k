"""Tests for the Kaggle toolset.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from agent_k.toolsets.kaggle import kaggle_list_datasets, kaggle_toolset

__all__ = ()

pytestmark = pytest.mark.anyio


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance."""
    assert isinstance(kaggle_toolset, FunctionToolset)


def test_toolset_id() -> None:
    """Toolset should have the expected id."""
    assert kaggle_toolset.id == "kaggle"


@dataclass
class _DepsStub:
    """Lightweight deps stub for toolset tests."""

    kaggle_adapter: Any
    event_emitter: Any = None
    search_cache: dict[str, Any] = field(default_factory=dict)


def _build_run_context(deps: _DepsStub) -> RunContext[Any]:
    """Build a RunContext sufficient for invoking toolset functions directly."""
    return RunContext(deps=deps, model=MagicMock(), usage=MagicMock())


class TestKaggleListDatasets:
    """Tests for the kaggle_list_datasets toolset entry point."""

    async def test_delegates_to_adapter_and_normalizes(self) -> None:
        """The tool should call list_competition_files and project the result."""
        adapter = MagicMock()
        adapter.list_competition_files = AsyncMock(
            return_value=[
                {"name": "train.csv", "size": 5, "description": "rows", "url": "/u"},
                {"name": "test.csv", "size": None, "description": None, "url": ""},
            ]
        )
        ctx = _build_run_context(_DepsStub(kaggle_adapter=adapter))

        result = await kaggle_list_datasets(ctx, "titanic")

        adapter.list_competition_files.assert_awaited_once_with("titanic")
        assert result == {
            "competition_id": "titanic",
            "files": [
                {"name": "train.csv", "size": 5, "description": "rows"},
                {"name": "test.csv", "size": None, "description": None},
            ],
        }

    async def test_raises_when_adapter_lacks_method(self) -> None:
        """Adapters missing list_competition_files should surface a clear error."""

        @dataclass
        class _AdapterWithoutListing:
            pass

        ctx = _build_run_context(_DepsStub(kaggle_adapter=_AdapterWithoutListing()))

        result = await kaggle_list_datasets(ctx, "titanic")

        assert "error" in result
        assert "does not support listing datasets" in result["error"]
