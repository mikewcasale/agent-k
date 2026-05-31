"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import ScientistAgent, scientist_agent

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
    """Regression tests for the Kaggle kernel URL extractor."""

    def test_extracts_owner_and_slug_from_code_url(self) -> None:
        """Standard /code/<owner>/<slug> URLs should yield owner/slug."""
        ref = ScientistAgent._extract_kernel_ref(
            None,  # type: ignore[arg-type]
            "https://www.kaggle.com/code/owner/notebook-slug",
        )
        assert ref == "owner/notebook-slug"

    def test_extracts_when_code_segment_omitted(self) -> None:
        """Legacy URLs without /code/ should still resolve."""
        ref = ScientistAgent._extract_kernel_ref(
            None,  # type: ignore[arg-type]
            "https://www.kaggle.com/owner/notebook-slug",
        )
        assert ref == "owner/notebook-slug"

    def test_returns_none_for_unrelated_url(self) -> None:
        """Non-Kaggle URLs return None."""
        ref = ScientistAgent._extract_kernel_ref(
            None,  # type: ignore[arg-type]
            "https://example.com/owner/slug",
        )
        assert ref is None
