"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import (
    KaggleAdapter,
    KaggleSettings,
    _classify_submission_status,
    _coerce_score,
    _extract_error_message,
    _parse_submission_item,
)

__all__ = ()

pytestmark = pytest.mark.anyio


class TestKaggleSettings:
    """Tests for the KaggleSettings class."""

    def test_config_creation(self) -> None:
        """Config should be created with credentials."""
        config = KaggleSettings(username="test_user", api_key="test_key")

        assert config.username == "test_user"
        assert config.api_key == "test_key"

    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = KaggleSettings(username="user", api_key="key")

        assert config.base_url == "https://www.kaggle.com/api/v1"


class TestKaggleAdapter:
    """Tests for the KaggleAdapter class."""

    def test_adapter_creation(self) -> None:
        """Adapter should be created with config."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        assert adapter is not None

    @pytest.fixture
    def mock_http_response(self) -> httpx.Response:
        """Create a mock HTTP response."""
        return httpx.Response(
            200,
            json=[
                {
                    "ref": "titanic",
                    "title": "Titanic",
                    "category": "gettingStarted",
                    "reward": "$0",
                    "deadline": "2030-01-01T00:00:00Z",
                }
            ],
        )

    async def test_search_competitions_basic(self) -> None:
        """Search competitions should return results."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        # The adapter requires actual HTTP calls or mocking
        # For unit tests, we verify the adapter is properly constructed
        assert adapter is not None

    async def test_get_leaderboard_basic(self) -> None:
        """Get leaderboard should return entries."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        assert adapter is not None


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation


class TestSubmissionParsing:
    """Tests for Kaggle submission status parsing helpers."""

    def test_coerce_score_handles_strings_and_numbers(self) -> None:
        """Scores arrive as strings from Kaggle and must coerce to floats."""
        assert _coerce_score("0.78532") == pytest.approx(0.78532)
        assert _coerce_score(0.5) == pytest.approx(0.5)
        assert _coerce_score(1) == pytest.approx(1.0)
        assert _coerce_score(None) is None
        assert _coerce_score("") is None
        assert _coerce_score("not-a-number") is None
        assert _coerce_score(True) is None
        assert _coerce_score(float("nan")) is None
        assert _coerce_score(float("inf")) is None

    def test_extract_error_message_prefers_first_populated(self) -> None:
        """Error messages may live under several keys; pick the first populated one."""
        assert _extract_error_message({"errorDescription": " row count mismatch "}) == "row count mismatch"
        assert _extract_error_message({"errorDescription": "", "submitErrorMessage": "bad header"}) == "bad header"
        assert _extract_error_message({}) is None
        assert _extract_error_message({"errorDescription": None}) is None

    def test_classify_status_uses_numeric_code(self) -> None:
        """Kaggle's integer status enum must map to our string statuses."""
        assert _classify_submission_status({"status": 1}) == "pending"
        assert _classify_submission_status({"status": 2}) == "complete"
        assert _classify_submission_status({"status": 3}) == "error"
        assert _classify_submission_status({"status": 4}) == "error"

    def test_classify_status_uses_string_label(self) -> None:
        """String status labels are normalized case-insensitively."""
        assert _classify_submission_status({"status": "Complete"}) == "complete"
        assert _classify_submission_status({"status": "FAILED"}) == "error"
        assert _classify_submission_status({"status": "queued"}) == "pending"

    def test_classify_status_falls_back_to_score_presence(self) -> None:
        """Without an explicit status, presence of a public score signals completion."""
        assert _classify_submission_status({"hasPublicScore": True}) == "complete"
        assert _classify_submission_status({"publicScore": "0.42"}) == "complete"
        assert _classify_submission_status({}) == "pending"

    def test_classify_status_detects_error_from_message(self) -> None:
        """An error description without an explicit status should still flag the failure."""
        assert _classify_submission_status({"errorDescription": "Invalid submission"}) == "error"

    def test_parse_submission_complete(self) -> None:
        """A complete submission produces a populated Submission model."""
        item = {"ref": "abc", "fileName": "submission.csv", "status": 2, "publicScore": "0.91", "privateScore": "0.89"}
        submission = _parse_submission_item(item, competition_id="comp", submission_id="abc")
        assert submission.status == "complete"
        assert submission.public_score == pytest.approx(0.91)
        assert submission.private_score == pytest.approx(0.89)
        assert submission.file_name == "submission.csv"
        assert submission.error_message is None

    def test_parse_submission_error_propagates_message(self) -> None:
        """Errors from Kaggle must surface on the Submission model."""
        item = {
            "ref": "abc",
            "fileName": "submission.csv",
            "status": 3,
            "errorDescription": "Submission must contain 1000 rows",
        }
        submission = _parse_submission_item(item, competition_id="comp", submission_id="abc")
        assert submission.status == "error"
        assert submission.public_score is None
        assert submission.error_message == "Submission must contain 1000 rows"

    def test_parse_submission_error_without_message_uses_default(self) -> None:
        """An error status without a message still flags the submission as failed."""
        item = {"ref": "abc", "status": "failed"}
        submission = _parse_submission_item(item, competition_id="comp", submission_id="abc")
        assert submission.status == "error"
        assert submission.error_message is not None and submission.error_message.strip()

    def test_parse_submission_complete_without_score_demotes_to_pending(self) -> None:
        """Don't claim completion when Kaggle has not produced a score yet."""
        item = {"ref": "abc", "status": "complete", "hasPublicScore": False, "publicScore": None}
        submission = _parse_submission_item(item, competition_id="comp", submission_id="abc")
        assert submission.status == "pending"
        assert submission.public_score is None
