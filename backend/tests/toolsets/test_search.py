"""Tests for the search tool helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
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
    assert build_scholarly_query("lightgbm") == "(site:arxiv.org OR site:paperswithcode.com) lightgbm"


def test_build_scholarly_query_all_applies_topic_to_both_sites() -> None:
    """Both site restrictions must be grouped so the topic filters both.

    Without the parentheses, `site:arxiv.org OR site:paperswithcode.com lightgbm`
    binds the topic only to the right operand of OR — arxiv results come back
    unfiltered by topic. Assert both sites appear inside a parenthesised group
    that precedes the topic.
    """
    query = build_scholarly_query("time series forecasting")
    assert query.startswith("(")
    open_paren = query.index("(")
    close_paren = query.index(")")
    inside = query[open_paren + 1 : close_paren]
    assert "site:arxiv.org" in inside
    assert "site:paperswithcode.com" in inside
    assert query[close_paren + 1 :].strip() == "time series forecasting"


def test_build_scholarly_query_arxiv() -> None:
    assert build_scholarly_query("lightgbm", source="arxiv") == "site:arxiv.org lightgbm"


def test_build_scholarly_query_papers_with_code() -> None:
    assert build_scholarly_query("lightgbm", source="paperswithcode") == "site:paperswithcode.com lightgbm"


def test_create_web_search_tool() -> None:
    tool = create_web_search_tool(search_context_size="high")
    assert tool.search_context_size == "high"


def test_create_web_fetch_tool() -> None:
    tool = create_web_fetch_tool(allowed_domains=["kaggle.com"])
    assert tool.allowed_domains == ["kaggle.com"]
