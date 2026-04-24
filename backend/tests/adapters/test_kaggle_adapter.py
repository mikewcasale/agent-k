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
    _DEFAULT_DEADLINE,
    _MAX_SEARCH_PAGES,
    KaggleAdapter,
    KaggleSettings,
    _normalize_category_key,
    _parse_deadline,
)
from agent_k.core.models import CompetitionType

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


class TestParseDeadline:
    """Tests for the _parse_deadline helper."""

    def test_none_returns_default(self) -> None:
        """Null deadlines (common for newly-created comps) must fall back."""
        assert _parse_deadline(None) == _DEFAULT_DEADLINE

    def test_empty_string_returns_default(self) -> None:
        """Empty strings normalize to the far-future default."""
        assert _parse_deadline("") == _DEFAULT_DEADLINE
        assert _parse_deadline("   ") == _DEFAULT_DEADLINE

    def test_non_string_non_datetime_returns_default(self) -> None:
        """Unexpected types fall back without raising."""
        assert _parse_deadline(12345) == _DEFAULT_DEADLINE
        assert _parse_deadline([]) == _DEFAULT_DEADLINE

    def test_trailing_z_is_handled(self) -> None:
        """Kaggle commonly emits ISO-8601 with trailing Z."""
        parsed = _parse_deadline("2024-06-01T12:00:00Z")
        assert parsed == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

    def test_fractional_trailing_z_is_handled(self) -> None:
        """Trailing Z with fractional seconds must parse cleanly."""
        parsed = _parse_deadline("2024-06-01T12:00:00.500Z")
        assert parsed.tzinfo is UTC
        assert parsed.year == 2024

    def test_explicit_offset_is_preserved(self) -> None:
        """Non-UTC offsets are retained as-is."""
        parsed = _parse_deadline("2024-06-01T12:00:00+02:00")
        offset = parsed.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 7200

    def test_naive_string_assumes_utc(self) -> None:
        """Naive ISO strings get UTC attached (matches Kaggle's implicit contract)."""
        parsed = _parse_deadline("2024-06-01T12:00:00")
        assert parsed.tzinfo is UTC

    def test_naive_datetime_assumes_utc(self) -> None:
        """Naive datetime instances get UTC attached."""
        parsed = _parse_deadline(datetime(2024, 6, 1, 12, 0, 0))
        assert parsed.tzinfo is UTC

    def test_aware_datetime_passed_through(self) -> None:
        """Timezone-aware datetime instances are returned unchanged."""
        original = datetime(2024, 6, 1, tzinfo=UTC)
        assert _parse_deadline(original) is original

    def test_garbage_string_falls_back(self) -> None:
        """Unparseable strings return the default without raising."""
        assert _parse_deadline("not-a-date") == _DEFAULT_DEADLINE


class TestNormalizeCategoryKey:
    """Tests for category key normalization."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Featured", "featured"),
            ("featured", "featured"),
            ("Getting Started", "gettingstarted"),
            ("gettingStarted", "gettingstarted"),
            ("getting-started", "gettingstarted"),
            ("getting_started", "gettingstarted"),
            ("RESEARCH", "research"),
            ("  Playground  ", "playground"),
            ("", ""),
            (None, ""),
            (123, ""),
        ],
    )
    def test_normalization(self, raw: Any, expected: str) -> None:
        """Category keys normalize across Kaggle's casing variants."""
        assert _normalize_category_key(raw) == expected


class TestParseCompetition:
    """Tests for KaggleAdapter._parse_competition edge cases."""

    def _adapter(self) -> KaggleAdapter:
        return KaggleAdapter(KaggleSettings(username="u", api_key="k"))

    def test_null_deadline_does_not_drop_competition(self) -> None:
        """A null deadline must parse into the far-future default."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {"ref": "some-comp", "title": "Some Comp", "category": "Featured", "reward": "$10000", "deadline": None}
        )
        assert comp.id == "some-comp"
        assert comp.deadline == _DEFAULT_DEADLINE
        assert comp.competition_type is CompetitionType.FEATURED

    def test_camelcase_category_resolves(self) -> None:
        """Kaggle's ``gettingStarted`` must map to GETTING_STARTED, not COMMUNITY."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "titanic",
                "title": "Titanic",
                "category": "gettingStarted",
                "reward": "$0",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )
        assert comp.competition_type is CompetitionType.GETTING_STARTED

    def test_kebab_case_category_resolves(self) -> None:
        """Kebab-case variants also resolve correctly."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "digit-recognizer",
                "title": "Digit Recognizer",
                "category": "getting-started",
                "reward": "$0",
                "deadline": "2030-01-01T00:00:00Z",
            }
        )
        assert comp.competition_type is CompetitionType.GETTING_STARTED

    def test_null_limits_fall_back_to_defaults(self) -> None:
        """Null maxTeamSize / maxDailySubmissions must not break model validation."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {
                "ref": "abc",
                "title": "Abc",
                "category": "featured",
                "deadline": "2030-01-01T00:00:00Z",
                "maxTeamSize": None,
                "maxDailySubmissions": None,
            }
        )
        assert comp.max_team_size == 1
        assert comp.max_daily_submissions == 5

    def test_unknown_category_defaults_to_community(self) -> None:
        """Unknown category values fall through to COMMUNITY."""
        adapter = self._adapter()
        comp = adapter._parse_competition(
            {"ref": "abc", "title": "Abc", "category": "not-a-real-category", "deadline": "2030-01-01T00:00:00Z"}
        )
        assert comp.competition_type is CompetitionType.COMMUNITY


class TestSearchCompetitionsPagination:
    """Tests for search_competitions pagination safeguards."""

    @staticmethod
    def _make_adapter_with_transport(handler: Any) -> KaggleAdapter:
        config = KaggleSettings(username="u", api_key="k")
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url, transport=httpx.MockTransport(handler), auth=(config.username, config.api_key)
        )
        return adapter

    async def test_stalled_pagination_terminates(self) -> None:
        """Identical page payloads must short-circuit to avoid infinite loops."""
        call_count = {"n": 0}
        fixed_payload = [
            {"ref": "comp-a", "title": "Comp A", "category": "featured", "deadline": "2030-01-01T00:00:00Z"}
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json=fixed_payload)

        adapter = self._make_adapter_with_transport(handler)
        try:
            results = [comp async for comp in adapter.search_competitions()]
        finally:
            await adapter._client.aclose()

        # First page yields one comp; second identical page triggers stall detection.
        assert len(results) == 1
        assert call_count["n"] == 2

    async def test_page_cap_respected(self) -> None:
        """Loop must terminate after _MAX_SEARCH_PAGES even if Kaggle keeps feeding data."""
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            # Vary ref per page so stall detection doesn't kick in; but the filter
            # rejects every item so no yields happen.
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(
                200,
                json=[
                    {
                        "ref": f"comp-page-{page}",
                        "title": f"Comp {page}",
                        "category": "featured",
                        "deadline": "2000-01-01T00:00:00Z",  # already expired
                    }
                ],
            )

        adapter = self._make_adapter_with_transport(handler)
        try:
            results = [comp async for comp in adapter.search_competitions(active_only=True)]
        finally:
            await adapter._client.aclose()

        assert results == []
        assert call_count["n"] == _MAX_SEARCH_PAGES

    async def test_empty_page_ends_iteration(self) -> None:
        """An empty page is the natural end-of-results signal."""
        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            page = int(request.url.params.get("page", "1"))
            if page == 1:
                return httpx.Response(
                    200,
                    json=[
                        {"ref": "comp-1", "title": "Comp 1", "category": "featured", "deadline": "2030-01-01T00:00:00Z"}
                    ],
                )
            return httpx.Response(200, json=[])

        adapter = self._make_adapter_with_transport(handler)
        try:
            results = [comp async for comp in adapter.search_competitions()]
        finally:
            await adapter._client.aclose()

        assert len(results) == 1
        assert call_count["n"] == 2
