"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import scientist_agent, scientist_agent_instance

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


class TestExtractKernelRef:
    """Tests for ScientistAgent._extract_kernel_ref."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.kaggle.com/code/some-user/some-notebook", "some-user/some-notebook"),
            ("https://kaggle.com/code/some-user/some-notebook", "some-user/some-notebook"),
            ("https://www.kaggle.com/some-user/some-notebook", "some-user/some-notebook"),
            ("http://www.kaggle.com/code/abc-123/def_456", "abc-123/def_456"),
            ("https://www.kaggle.com/code/me/my-nb?foo=bar", "me/my-nb"),
            ("https://www.kaggle.com/code/me/my-nb#anchor", "me/my-nb"),
            ("https://kaggle.com/code/owner/slug/", "owner/slug"),
            ("/code/owner/slug", "owner/slug"),
            ("/owner/slug", "owner/slug"),
        ],
    )
    def test_extracts_known_kaggle_url_shapes(self, url: str, expected: str) -> None:
        """The fallback extractor must recover owner/slug from real Kaggle URL formats."""
        assert scientist_agent_instance._extract_kernel_ref(url) == expected

    @pytest.mark.parametrize("url", ["", "not-a-url", "https://example.com/code/owner/slug", "kaggle.com"])
    def test_returns_none_for_unrecognized_urls(self, url: str) -> None:
        """Unrecognized inputs return None so the caller skips kernel code fetch."""
        assert scientist_agent_instance._extract_kernel_ref(url) is None
