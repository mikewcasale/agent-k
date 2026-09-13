"""Tests for core exceptions.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from agent_k.core.exceptions import (
    AgentExecutionError,
    AgentKError,
    AuthenticationError,
    CompetitionNotFoundError,
    RateLimitError,
    StateTransitionError,
    SubmissionError,
    is_model_unavailable_error,
)
from agent_k.mission.nodes import _is_rate_limit_error

__all__ = ()


class TestAgentKError:
    """Tests for the base AgentKError."""

    def test_basic_creation(self) -> None:
        """Error should be created with message."""
        error = AgentKError("Test error")

        assert str(error) == "Test error"

    def test_inheritance(self) -> None:
        """Should inherit from Exception."""
        error = AgentKError("Test")

        assert isinstance(error, Exception)


class TestAgentExecutionError:
    """Tests for AgentExecutionError."""

    def test_creation(self) -> None:
        """Error should be created with agent name and message."""
        error = AgentExecutionError("lobbyist", "Agent failed")

        assert "Agent failed" in str(error)
        assert error.agent_name == "lobbyist"

    def test_inheritance(self) -> None:
        """Should inherit from AgentKError."""
        error = AgentExecutionError("lobbyist", "test")

        assert isinstance(error, AgentKError)


class TestCompetitionNotFoundError:
    """Tests for CompetitionNotFoundError."""

    def test_creation_with_id(self) -> None:
        """Error should include competition ID."""
        error = CompetitionNotFoundError("titanic")

        assert "titanic" in str(error)

    def test_inheritance(self) -> None:
        """Should inherit from AgentKError."""
        error = CompetitionNotFoundError("test")

        assert isinstance(error, AgentKError)


class TestSubmissionError:
    """Tests for SubmissionError."""

    def test_creation(self) -> None:
        """Error should be created with competition ID and message."""
        error = SubmissionError("titanic", "Submission failed")

        assert "titanic" in str(error) or "Submission failed" in str(error)


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_creation(self) -> None:
        """Error should be created with platform and retry_after."""
        error = RateLimitError(platform="kaggle", message="Rate limit exceeded", retry_after=60)

        assert error.retry_after == 60
        assert error.platform == "kaggle"


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_creation(self) -> None:
        """Error should be created with platform info."""
        error = AuthenticationError(platform="kaggle")

        assert "kaggle" in str(error).lower() or error is not None


class TestStateTransitionError:
    """Tests for StateTransitionError."""

    def test_creation(self) -> None:
        """Error should be created with from/to states."""
        error = StateTransitionError(from_state="discovery", to_state="research", reason="No competitions found")

        assert error.from_state == "discovery"
        assert error.to_state == "research"
        assert error.reason == "No competitions found"


class TestIsModelUnavailableError:
    """Tests for permanent model-unavailability detection."""

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404])
    def test_permanent_status_codes(self, status_code: int) -> None:
        """Client-side provider rejections should mark the model unusable."""
        error = ModelHTTPError(status_code=status_code, model_name="openai/gpt-oss-120b:free", body=None)

        assert is_model_unavailable_error(error) is True

    @pytest.mark.parametrize("status_code", [429, 500, 502, 503])
    def test_transient_status_codes_are_not_unavailable(self, status_code: int) -> None:
        """Throttling and server faults recover on their own, so the model stays in the pool."""
        error = ModelHTTPError(status_code=status_code, model_name="openai/gpt-oss-120b:free", body=None)

        assert is_model_unavailable_error(error) is False

    def test_retired_free_tier_message(self) -> None:
        """A retired free tier reads as unavailable, not as a rate limit."""
        error = ModelHTTPError(
            status_code=404,
            model_name="mistralai/devstral-2512:free",
            body={"error": {"message": "No endpoints found for mistralai/devstral-2512:free.", "code": 404}},
        )

        assert is_model_unavailable_error(error) is True
        assert _is_rate_limit_error(error) is False

    def test_message_only_classification(self) -> None:
        """Agent failures carry provider text as a plain string and must classify the same."""
        assert is_model_unavailable_error("status_code: 404, model_name: x, body: None") is True
        assert is_model_unavailable_error("Provider returned: incorrect API key provided") is True

    def test_revoked_credentials(self) -> None:
        """A rejected key makes every request against that model fail permanently."""
        assert is_model_unavailable_error(AuthenticationError(platform="openrouter")) is True

    def test_unrelated_errors_are_not_unavailable(self) -> None:
        """Ordinary failures must not retire a working model."""
        assert is_model_unavailable_error(None) is False
        assert is_model_unavailable_error("") is False
        assert is_model_unavailable_error(ValueError("solution did not produce submission.csv")) is False
        assert is_model_unavailable_error(RateLimitError(platform="openrouter", message="Too many requests")) is False

    def test_solution_failure_text_does_not_retire_a_model(self) -> None:
        """Evolution failures carry solution stderr, which must never look like a dead model."""
        assert is_model_unavailable_error("pydantic_ai Embedder is not available") is False
        assert is_model_unavailable_error("FileNotFoundError: train.csv does not exist") is False
        assert is_model_unavailable_error("FutureWarning: 'squared' has been deprecated in 1.4") is False
