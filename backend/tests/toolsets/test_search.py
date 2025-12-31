"""Tests for the search tool helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

from agent_k.toolsets.search import (
    build_kaggle_search_query,
    build_scholarly_query,
    create_web_fetch_tool,
    create_web_search_tool,
)


__all__ = ()


def test_build_kaggle_search_query() -> None:
    assert build_kaggle_search_query("titanic") == "site:kaggle.com titanic"


def test_build_scholarly_query_all() -> None:
    assert build_scholarly_query("xgboost") == "site:arxiv.org OR site:paperswithcode.com xgboost"


def test_build_scholarly_query_arxiv() -> None:
    assert build_scholarly_query("xgboost", source="arxiv") == "site:arxiv.org xgboost"


def test_build_scholarly_query_papers_with_code() -> None:
    assert (
        build_scholarly_query("xgboost", source="paperswithcode")
        == "site:paperswithcode.com xgboost"
    )


def test_create_web_search_tool() -> None:
    tool = create_web_search_tool(search_context_size="high")
    assert tool.search_context_size == "high"


def test_create_web_fetch_tool() -> None:
    tool = create_web_fetch_tool(allowed_domains=["kaggle.com"])
    assert tool.allowed_domains == ["kaggle.com"]
