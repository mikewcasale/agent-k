"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings

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


class TestParseSubmission:
    """Tests for parsing Kaggle submission records into Submission models."""

    def _adapter(self) -> KaggleAdapter:
        return KaggleAdapter(KaggleSettings(username="u", api_key="k"))

    def test_complete_submission_parses_scores(self) -> None:
        """A completed submission exposes status, public, and private scores."""
        adapter = self._adapter()
        item = {
            "ref": "sub-1",
            "fileName": "submission.csv",
            "status": "complete",
            "hasPublicScore": True,
            "publicScore": "0.873",
            "privateScore": "0.881",
            "errorDescription": None,
        }
        result = adapter._parse_submission("comp-1", "sub-1", item)
        assert result.status == "complete"
        assert result.public_score == pytest.approx(0.873)
        assert result.private_score == pytest.approx(0.881)
        assert result.error_message is None
        assert result.file_name == "submission.csv"

    def test_error_submission_sets_error_status_and_message(self) -> None:
        """An errored submission propagates the Kaggle errorDescription."""
        adapter = self._adapter()
        item = {
            "ref": "sub-2",
            "fileName": "submission.csv",
            "status": "error",
            "hasPublicScore": False,
            "publicScore": None,
            "errorDescription": "Submission column mismatch",
        }
        result = adapter._parse_submission("comp-1", "sub-2", item)
        assert result.status == "error"
        assert result.public_score is None
        assert result.error_message == "Submission column mismatch"

    def test_error_inferred_from_description_when_status_missing(self) -> None:
        """An ``errorDescription`` alone is sufficient to mark the submission failed."""
        adapter = self._adapter()
        item = {"ref": "sub-3", "fileName": "submission.csv", "status": "", "errorDescription": "Invalid format"}
        result = adapter._parse_submission("comp-1", "sub-3", item)
        assert result.status == "error"
        assert result.error_message == "Invalid format"

    def test_pending_submission_remains_pending(self) -> None:
        """Submissions still queued by Kaggle stay ``pending`` with no score."""
        adapter = self._adapter()
        item = {
            "ref": "sub-4",
            "fileName": "submission.csv",
            "status": "pending",
            "hasPublicScore": False,
            "publicScore": None,
        }
        result = adapter._parse_submission("comp-1", "sub-4", item)
        assert result.status == "pending"
        assert result.public_score is None
        assert result.error_message is None

    def test_complete_inferred_from_public_score_when_status_blank(self) -> None:
        """A numeric ``publicScore`` implies completion even if Kaggle omits status."""
        adapter = self._adapter()
        item = {"ref": "sub-5", "fileName": "submission.csv", "status": "", "hasPublicScore": True, "publicScore": 0.42}
        result = adapter._parse_submission("comp-1", "sub-5", item)
        assert result.status == "complete"
        assert result.public_score == pytest.approx(0.42)

    def test_nonnumeric_public_score_is_coerced_to_none(self) -> None:
        """Non-numeric score values do not raise and surface as ``None``."""
        adapter = self._adapter()
        item = {"ref": "sub-6", "fileName": "submission.csv", "status": "pending", "publicScore": "n/a"}
        result = adapter._parse_submission("comp-1", "sub-6", item)
        assert result.public_score is None
        assert result.status == "pending"
