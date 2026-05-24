"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import math

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


class _StatusCodeError(Exception):
    """Test exception carrying a status_code attribute."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ResponseCarrier:
    """Mimics an httpx-like response carrier on an exception."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _ResponseError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _ResponseCarrier(status_code)


class TestIsRateLimitError:
    """Tests for rate-limit classification."""

    @pytest.mark.parametrize(
        "message",
        [
            "Rate limit exceeded",
            "rate_limit_reached",
            "RATE-LIMIT exceeded for model",
            "request_limit hit",
            "Too many requests from your account",
            "limit reached for free tier",
            "HTTP 429 Too Many Requests",
            "Insufficient credits remaining",
            "credit limit reached",
            "credits exhausted",
            "Quota exceeded",
            "Internal Server Error",
            "Service Unavailable",
            "Bad Gateway",
            "Gateway Timeout",
            "HTTP 502 from upstream",
            "Status: 503",
            "500 server error returned",
        ],
    )
    def test_recognizes_rate_limit_messages(self, message: str) -> None:
        """Common rate-limit / 5xx phrasings should be classified as rate limit."""
        assert _is_rate_limit_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Processed 5000 rows successfully",
            "Loaded 500 records into memory",
            "credit_score column has missing values",
            "User has credit_card field",
            "max recursion depth exceeded",
            "time limit exceeded in worker",
            "Server returned malformed JSON",
            "Validation failed: required field missing",
            "ImportError: No module named foo",
            "",
        ],
    )
    def test_does_not_match_unrelated_messages(self, message: str) -> None:
        """Substrings like '500' or 'exceeded' must not trigger false positives."""
        assert _is_rate_limit_error(message) is False

    def test_handles_none(self) -> None:
        """None should be treated as not a rate-limit error."""
        assert _is_rate_limit_error(None) is False

    def test_handles_empty_exception(self) -> None:
        """An exception with no relevant signal should be False."""
        assert _is_rate_limit_error(ValueError("bad input")) is False

    def test_status_code_429_attribute(self) -> None:
        """An exception with status_code=429 should be classified as rate limit."""
        assert _is_rate_limit_error(_StatusCodeError("rejected", status_code=429)) is True

    def test_status_code_5xx_attribute(self) -> None:
        """An exception with status_code in 5xx range should be classified as rate limit."""
        assert _is_rate_limit_error(_StatusCodeError("upstream down", status_code=503)) is True

    def test_status_code_4xx_other_attribute(self) -> None:
        """Non-rate-limit 4xx codes should NOT trigger."""
        assert _is_rate_limit_error(_StatusCodeError("not found", status_code=404)) is False

    def test_response_status_429(self) -> None:
        """An exception carrying a response with status 429 should be classified."""
        assert _is_rate_limit_error(_ResponseError("rejected", status_code=429)) is True

    def test_response_status_500(self) -> None:
        """An exception carrying a response with 5xx status should be classified."""
        assert _is_rate_limit_error(_ResponseError("server fault", status_code=500)) is True

    def test_error_code_attribute_with_rate(self) -> None:
        """An exception with `code` attribute containing 'rate' should be classified."""

        class _CodeError(Exception):
            code = "rate_limited"

        assert _is_rate_limit_error(_CodeError("blocked")) is True

    def test_error_code_attribute_without_rate(self) -> None:
        """An exception with an unrelated `code` attribute should not be classified."""

        class _CodeError(Exception):
            code = "validation_failed"

        assert _is_rate_limit_error(_CodeError("blocked")) is False
