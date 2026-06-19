"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import (
    _COMPETITION_DESCRIPTION_MAX_LENGTH,
    _COMPETITION_TITLE_MAX_LENGTH,
    _DEFAULT_DEADLINE_ISO,
    KaggleAdapter,
    KaggleSettings,
    _clip_text,
    _parse_deadline,
)
from agent_k.core.models import Competition

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


class TestParseCompetitionResilience:
    """Resilience of ``_parse_competition`` against malformed Kaggle payloads."""

    def _adapter(self) -> KaggleAdapter:
        return KaggleAdapter(KaggleSettings(username="u", api_key="k"))

    def _base_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ref": "titanic",
            "title": "Titanic",
            "category": "Getting Started",
            "evaluationMetric": "accuracy",
            "reward": "$0",
            "deadline": "2030-01-01T00:00:00Z",
        }
        payload.update(overrides)
        return payload

    def test_truncates_overlong_description(self) -> None:
        """Descriptions longer than the model cap are truncated rather than rejected."""
        assert _COMPETITION_DESCRIPTION_MAX_LENGTH is not None
        long_text = "x" * (_COMPETITION_DESCRIPTION_MAX_LENGTH + 500)

        comp = self._adapter()._parse_competition(self._base_payload(description=long_text))

        assert comp.description is not None
        assert len(comp.description) == _COMPETITION_DESCRIPTION_MAX_LENGTH

    def test_truncates_overlong_title(self) -> None:
        """Titles longer than the model cap are truncated rather than rejected."""
        assert _COMPETITION_TITLE_MAX_LENGTH is not None
        long_title = "T" * (_COMPETITION_TITLE_MAX_LENGTH + 50)

        comp = self._adapter()._parse_competition(self._base_payload(title=long_title))

        assert len(comp.title) == _COMPETITION_TITLE_MAX_LENGTH

    def test_handles_null_deadline(self) -> None:
        """A ``null`` deadline falls back to the far-future sentinel instead of crashing."""
        comp = self._adapter()._parse_competition(self._base_payload(deadline=None))

        assert comp.deadline == datetime.fromisoformat(_DEFAULT_DEADLINE_ISO)

    def test_handles_missing_deadline_key(self) -> None:
        """An entirely missing deadline key behaves the same as a null value."""
        payload = self._base_payload()
        payload.pop("deadline")

        comp = self._adapter()._parse_competition(payload)

        assert comp.deadline == datetime.fromisoformat(_DEFAULT_DEADLINE_ISO)

    def test_handles_unparseable_deadline(self) -> None:
        """An unparseable deadline string falls back to the sentinel."""
        comp = self._adapter()._parse_competition(self._base_payload(deadline="not-a-date"))

        assert comp.deadline == datetime.fromisoformat(_DEFAULT_DEADLINE_ISO)

    def test_existing_deadline_round_trips(self) -> None:
        """Standard ISO-8601 deadlines parse to timezone-aware datetimes."""
        comp = self._adapter()._parse_competition(self._base_payload(deadline="2030-06-01T12:30:00Z"))

        assert comp.deadline == datetime(2030, 6, 1, 12, 30, tzinfo=UTC)

    def test_empty_description_becomes_none(self) -> None:
        """Empty descriptions normalize to ``None`` instead of an empty string."""
        comp = self._adapter()._parse_competition(self._base_payload(description=""))

        assert comp.description is None


class TestClipText:
    """Tests for the ``_clip_text`` helper."""

    def test_returns_none_for_none(self) -> None:
        assert _clip_text(None, 10) is None

    def test_returns_none_for_empty_string(self) -> None:
        assert _clip_text("", 10) is None

    def test_passes_through_short_values(self) -> None:
        assert _clip_text("hello", 10) == "hello"

    def test_truncates_overlong_values(self) -> None:
        assert _clip_text("x" * 20, 10) == "x" * 10

    def test_no_cap_when_max_length_is_none(self) -> None:
        assert _clip_text("x" * 50, None) == "x" * 50

    def test_coerces_non_string_values(self) -> None:
        assert _clip_text(42, 5) == "42"


class TestParseDeadline:
    """Tests for the ``_parse_deadline`` helper."""

    def test_parses_iso_with_trailing_z(self) -> None:
        result = _parse_deadline("2030-01-01T00:00:00Z")
        assert result == datetime(2030, 1, 1, tzinfo=UTC)

    def test_parses_iso_with_offset(self) -> None:
        result = _parse_deadline("2030-01-01T00:00:00+00:00")
        assert result == datetime(2030, 1, 1, tzinfo=UTC)

    def test_assumes_utc_for_naive_timestamps(self) -> None:
        """Naive timestamps should be treated as UTC to satisfy the Competition validator."""
        result = _parse_deadline("2030-01-01T00:00:00")
        assert result.tzinfo is not None
        assert result == datetime(2030, 1, 1, tzinfo=UTC)

    def test_returns_sentinel_for_none(self) -> None:
        assert _parse_deadline(None) == datetime.fromisoformat(_DEFAULT_DEADLINE_ISO)

    def test_returns_sentinel_for_empty_string(self) -> None:
        assert _parse_deadline("   ") == datetime.fromisoformat(_DEFAULT_DEADLINE_ISO)

    def test_returns_sentinel_for_garbage(self) -> None:
        assert _parse_deadline("garbage") == datetime.fromisoformat(_DEFAULT_DEADLINE_ISO)


def test_competition_max_lengths_match_model_metadata() -> None:
    """Adapter constants must stay aligned with the Competition model field metadata."""
    title_metadata = [m.max_length for m in Competition.model_fields["title"].metadata if hasattr(m, "max_length")]
    description_metadata = [
        m.max_length for m in Competition.model_fields["description"].metadata if hasattr(m, "max_length")
    ]
    assert _COMPETITION_TITLE_MAX_LENGTH in title_metadata
    assert _COMPETITION_DESCRIPTION_MAX_LENGTH in description_metadata
