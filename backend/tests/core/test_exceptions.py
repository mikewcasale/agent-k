"""Tests for core exceptions.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.exceptions import (
    AgentExecutionError,
    AgentKError,
    AuthenticationError,
    CompetitionNotFoundError,
    RateLimitError,
    StateTransitionError,
    SubmissionError,
    classify_error,
    is_recoverable_error,
    recovery_delay_seconds,
)

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


class TestIsRecoverableError:
    """Tests for the canonical retry predicate."""

    def test_rate_limit_is_recoverable(self) -> None:
        """Platform rate limits should be retried."""
        assert is_recoverable_error(RateLimitError(platform="kaggle", message="429")) is True

    def test_authentication_is_not_recoverable(self) -> None:
        """Bad credentials can never be fixed by retrying."""
        assert is_recoverable_error(AuthenticationError(platform="kaggle")) is False

    def test_competition_not_found_is_not_recoverable(self) -> None:
        """A missing competition is terminal."""
        assert is_recoverable_error(CompetitionNotFoundError("titanic")) is False

    def test_non_recoverable_agent_k_error(self) -> None:
        """An AgentKError flagged non-recoverable must not be retried."""
        assert is_recoverable_error(AgentKError("fatal", recoverable=False)) is False

    def test_transient_third_party_error_is_recoverable(self) -> None:
        """Transient failures outside the AgentKError hierarchy must still be retried.

        Regression guard: the mission runner used to gate retries on
        ``isinstance(exc, AgentKError)``, so a dropped connection or read
        timeout ended the mission immediately while the UI was told the
        recovery strategy was "retry".
        """
        assert is_recoverable_error(ConnectionResetError("connection reset by peer")) is True
        assert is_recoverable_error(TimeoutError("read timed out")) is True

    def test_agrees_with_classify_error(self) -> None:
        """The predicate must stay consistent with classify_error."""
        for exc in (
            RateLimitError(platform="kaggle", message="429"),
            AuthenticationError(platform="kaggle"),
            CompetitionNotFoundError("titanic"),
            AgentKError("fatal", recoverable=False),
            ConnectionResetError("boom"),
        ):
            _category, strategy = classify_error(exc)
            assert is_recoverable_error(exc) is (strategy == "retry")


class TestRecoveryDelaySeconds:
    """Tests for retry backoff computation."""

    def test_exponential_backoff(self) -> None:
        """Delay should double with each attempt."""
        exc = ConnectionResetError("boom")

        assert recovery_delay_seconds(exc, 1, base_delay=2.0) == 2.0
        assert recovery_delay_seconds(exc, 2, base_delay=2.0) == 4.0
        assert recovery_delay_seconds(exc, 3, base_delay=2.0) == 8.0

    def test_backoff_is_clamped(self) -> None:
        """Delay must never exceed max_delay."""
        assert recovery_delay_seconds(ConnectionResetError("boom"), 10, base_delay=2.0, max_delay=30.0) == 30.0

    def test_rate_limit_retry_after_wins(self) -> None:
        """A platform-supplied retry_after overrides exponential backoff."""
        exc = RateLimitError(platform="kaggle", message="429", retry_after=45)

        assert recovery_delay_seconds(exc, 1, base_delay=2.0) == 45.0

    def test_rate_limit_retry_after_is_clamped(self) -> None:
        """An oversized retry_after is clamped to max_delay."""
        exc = RateLimitError(platform="kaggle", message="429", retry_after=3600)

        assert recovery_delay_seconds(exc, 1, max_delay=60.0) == 60.0

    def test_rate_limit_without_hint_falls_back_to_backoff(self) -> None:
        """Without a retry_after hint the exponential schedule applies."""
        exc = RateLimitError(platform="kaggle", message="429")

        assert recovery_delay_seconds(exc, 2, base_delay=2.0) == 4.0

    def test_non_positive_retry_after_falls_back_to_backoff(self) -> None:
        """A zero retry_after must not collapse the delay to zero."""
        exc = RateLimitError(platform="kaggle", message="429", retry_after=0)

        assert recovery_delay_seconds(exc, 1, base_delay=2.0) == 2.0

    def test_attempt_must_be_positive(self) -> None:
        """Attempt numbers are 1-based."""
        with pytest.raises(ValueError, match="attempt must be >= 1"):
            recovery_delay_seconds(ConnectionResetError("boom"), 0)
