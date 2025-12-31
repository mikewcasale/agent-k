"""Tests for the Kaggle toolset.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

from pydantic_ai.toolsets import FunctionToolset

from agent_k.toolsets.kaggle import kaggle_toolset


__all__ = ()


def test_toolset_is_function_toolset() -> None:
    """Toolset should be a FunctionToolset instance."""
    assert isinstance(kaggle_toolset, FunctionToolset)


def test_toolset_id() -> None:
    """Toolset should have the expected id."""
    assert kaggle_toolset.id == "kaggle"
