"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from agent_k.core.models import EvaluationMetric
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _evaluate_metric,
    _is_rate_limit_error,
)

__all__ = ()

pytestmark = pytest.mark.anyio


@dataclass
class _FakeResponse:
    status_code: int


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ResponseError(Exception):
    def __init__(self, message: str, response: Any) -> None:
        super().__init__(message)
        self.response = response


class _CodedError(Exception):
    def __init__(self, message: str, *, code: str | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.error_code = error_code


class TestDiscoveryNode:
    """Tests for the DiscoveryNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = DiscoveryNode()
        assert node is not None


class TestResearchNode:
    """Tests for the ResearchNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = ResearchNode()
        assert node is not None


class TestPrototypeNode:
    """Tests for the PrototypeNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = PrototypeNode()
        assert node is not None


class TestEvolutionNode:
    """Tests for the EvolutionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = EvolutionNode()
        assert node is not None


class TestSubmissionNode:
    """Tests for the SubmissionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = SubmissionNode()
        assert node is not None


class TestEvaluateMetric:
    """Tests for metric evaluation helpers."""

    def test_rmsle_ignores_negative_values(self) -> None:
        """RMSLE should ignore negative targets in the denominator."""
        score = _evaluate_metric(EvaluationMetric.RMSLE, [1.0, -1.0], prediction=0.0)
        assert score == pytest.approx(math.log1p(1.0))


class TestIsRateLimitError:
    """Tests for the _is_rate_limit_error heuristic."""

    @pytest.mark.parametrize(
        "message",
        [
            "OpenRouter rate limit exceeded",
            "Anthropic rate_limit_error",
            "Too many requests, slow down",
            "Quota exceeded for this model",
            "Insufficient credits to complete this request",
            "Out of credits — top up your account",
            "Daily limit reached for free tier",
            "Throttled by upstream provider",
            "tokens per minute limit hit",
            "HTTP 429 Too Many Requests",
            "Status code: 503",
            "HTTP 500 Internal Server Error",
            "Internal server error from provider",
            "Service unavailable, try again later",
        ],
    )
    def test_classifies_known_rate_limit_messages(self, message: str) -> None:
        """Known rate-limit and 5xx messages should be classified as rate-limited."""
        assert _is_rate_limit_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "context_length_exceeded: 8193 > 8192",
            "Context length exceeded for this model",
            "max_tokens exceeded for prompt",
            "Maximum tokens exceeded",
            "Token limit exceeded for output",
            "Tool call exceeded retry budget",
            "Iteration count exceeded the configured cap",
            "Expected shape (500, 10), got (250, 10)",
            "Got 5000 rows but expected 500",
            "Submission contains 1500 entries",
            "Credits to the open source community",
            "Mission credits earned: 5",
            "Validation failed: expected float",
            "Acknowledgments and credits section",
            "Convergence after 500 generations",
        ],
    )
    def test_does_not_classify_unrelated_messages(self, message: str) -> None:
        """Messages with rate-limit-adjacent words but unrelated intent should NOT match."""
        assert _is_rate_limit_error(message) is False

    def test_empty_or_none_inputs_return_false(self) -> None:
        """Empty or None inputs should not be classified as rate limited."""
        assert _is_rate_limit_error(None) is False
        assert _is_rate_limit_error("") is False

    def test_429_status_attribute_on_exception_matches(self) -> None:
        """An exception carrying status_code=429 should match."""
        exc = _StatusError("client error", status_code=429)
        assert _is_rate_limit_error(exc) is True

    def test_5xx_status_attribute_on_exception_matches(self) -> None:
        """An exception carrying status_code >= 500 should match."""
        exc = _StatusError("upstream failure", status_code=503)
        assert _is_rate_limit_error(exc) is True

    def test_429_response_status_matches(self) -> None:
        """An exception wrapping a response with status_code=429 should match."""
        exc = _ResponseError("api error", response=_FakeResponse(status_code=429))
        assert _is_rate_limit_error(exc) is True

    def test_4xx_status_code_does_not_match(self) -> None:
        """A 4xx status that is not 429 (e.g. 400, 404) should not be classified as rate limited."""
        exc = _StatusError("bad request", status_code=400)
        assert _is_rate_limit_error(exc) is False

    def test_known_rate_limit_error_code_matches(self) -> None:
        """A known rate-limit error_code attribute should match even with a neutral message."""
        exc = _CodedError("provider issue", code="insufficient_quota")
        assert _is_rate_limit_error(exc) is True

    def test_context_length_error_code_does_not_match(self) -> None:
        """context_length_exceeded is not a rate-limit condition."""
        exc = _CodedError("Context length exceeded", code="context_length_exceeded")
        assert _is_rate_limit_error(exc) is False

    def test_rate_limit_exceeded_error_code_matches(self) -> None:
        """An error_code containing both 'rate' and 'limit' should match."""
        exc = _CodedError("provider issue", code="rate_limit_exceeded")
        assert _is_rate_limit_error(exc) is True

    def test_bare_500_in_message_does_not_match(self) -> None:
        """A bare '500' anywhere in text should not be classified — only contextualized 5xx codes."""
        assert _is_rate_limit_error("Loaded 500 features") is False
        assert _is_rate_limit_error("dataframe shape (500, 12)") is False

    def test_bare_429_in_message_does_not_match(self) -> None:
        """A bare '429' should not match without HTTP/status/code context."""
        assert _is_rate_limit_error("processed 429 rows") is False
