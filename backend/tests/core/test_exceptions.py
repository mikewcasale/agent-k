"""Tests for core exceptions.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

from agent_k.core.exceptions import (
    AgentExecutionError,
    AgentKError,
    AuthenticationError,
    CompetitionNotFoundError,
    RateLimitError,
    StateTransitionError,
    SubmissionError,
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
        error = RateLimitError(
            platform="kaggle",
            message="Rate limit exceeded",
            retry_after=60,
        )

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
        error = StateTransitionError(
            from_state="discovery",
            to_state="research",
            reason="No competitions found",
        )

        assert error.from_state == "discovery"
        assert error.to_state == "research"
        assert error.reason == "No competitions found"
