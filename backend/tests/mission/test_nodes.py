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


class _StatusExc(Exception):
    """Minimal exception carrying a ``status_code`` attribute."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ResponseWrapperExc(Exception):
    """Minimal exception carrying a ``response.status_code`` attribute."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)

        class _Response:
            pass

        response = _Response()
        response.status_code = status_code  # type: ignore[attr-defined]
        self.response = response


class _CodeExc(Exception):
    """Minimal exception carrying an ``error_code`` attribute (OpenAI-style)."""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class TestIsRateLimitError:
    """Tests for ``_is_rate_limit_error``.

    Guards the false-positive fixes (bare "exceeded", "credits", "quota",
    "server error", "500" no longer trigger on their own) and the true-positive
    coverage we rely on for provider-specific spellings.
    """

    def test_none_and_empty_are_not_rate_limits(self) -> None:
        assert _is_rate_limit_error(None) is False
        assert _is_rate_limit_error("") is False

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504, 599])
    def test_exception_status_code_triggers(self, status_code: int) -> None:
        assert _is_rate_limit_error(_StatusExc("boom", status_code)) is True

    @pytest.mark.parametrize("status_code", [200, 400, 401, 403, 404, 418])
    def test_non_retryable_status_codes_do_not_trigger(self, status_code: int) -> None:
        assert _is_rate_limit_error(_StatusExc("boom", status_code)) is False

    def test_response_wrapper_status_code_triggers(self) -> None:
        assert _is_rate_limit_error(_ResponseWrapperExc("upstream", 503)) is True

    def test_error_code_containing_rate_triggers(self) -> None:
        assert _is_rate_limit_error(_CodeExc("nope", "rate_limit_exceeded")) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Rate limit reached for gpt-4",
            "rate_limit_error",
            "RateLimitError",
            "ratelimit hit for org-abc",
            "Too Many Requests",
            "TooManyRequests",
            "quota exceeded for tokens per minute",
            "quota_reached",
            "quotaError",
            "Insufficient credits on OpenRouter",
            "insufficient_quota",
            "out of credits",
            "no credits remaining",
            "credit limit reached",
            "Retry after 60 seconds",
            "retry-after: 30",
            "Internal Server Error",
            "Bad Gateway from upstream provider",
            "service unavailable",
            "Gateway Timeout",
            "upstream_error while proxying",
            "HTTP 429: Too Many Requests",
            "Server responded with 502",
            "status 503 from provider",
        ],
    )
    def test_true_positive_messages(self, message: str) -> None:
        assert _is_rate_limit_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            # Bare "exceeded" no longer triggers on its own — this is the primary
            # regression the tightening fixes.
            "max_tokens exceeded for this request",
            "context length exceeded",
            "budget exceeded, please upgrade plan",
            # Bare "credits" no longer triggers on its own; we still have 500 in
            # message but as part of a larger number, so \b500\b does not match.
            "You have used 5000 credits so far",
            # Bare "500" inside another number or a unit doesn't trigger.
            "500ms elapsed while calling the model",
            "processed 5000 rows in stage1",
            # Bare "quota" alone without exceeded/reached/error doesn't trigger.
            "quota_remaining=42",
            # Bare "server error" without the "internal" qualifier doesn't
            # trigger — pattern requires the specific "internal server error"
            # phrase (or a 5xx status token elsewhere in the message).
            "custom-server error: parse failure",
            # Neutral messages.
            "connection refused",
            "invalid api key",
            "schema validation failed for field X",
        ],
    )
    def test_false_positive_messages_are_rejected(self, message: str) -> None:
        assert _is_rate_limit_error(message) is False

    def test_error_type_name_from_pydantic_ai_failure(self) -> None:
        # EvolutionFailure passes ``type(exc).__name__`` as ``error_type``; the
        # camel-case type name for the common rate-limit exception must be
        # detected without a status_code attribute.
        assert _is_rate_limit_error("RateLimitError") is True
        assert _is_rate_limit_error("TooManyRequestsError") is True

    def test_word_boundary_prevents_embedded_status_false_positive(self) -> None:
        # "500" as part of a larger number must not trigger.
        assert _is_rate_limit_error("iteration 5001 finished") is False
        # "429" as a whole word triggers; embedded in a larger number does not.
        assert _is_rate_limit_error("Received 429 response") is True
        assert _is_rate_limit_error("stat=4291") is False
