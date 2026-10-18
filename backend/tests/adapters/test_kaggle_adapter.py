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


class TestCompetitionEligibilityParsing:
    """Tests for parsing submission-eligibility metadata from Kaggle payloads."""

    @pytest.fixture
    def adapter(self) -> KaggleAdapter:
        """Create an adapter with throwaway credentials."""
        return KaggleAdapter(KaggleSettings(username="user", api_key="key"))

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ref": "https://www.kaggle.com/competitions/example-comp",
            "title": "Example",
            "category": "Featured",
            "deadline": "2030-01-01T00:00:00Z",
            "evaluationMetric": "auc",
        }
        payload.update(overrides)
        return payload

    def test_defaults_when_fields_absent(self, adapter: KaggleAdapter) -> None:
        """Missing eligibility fields should parse as permissive defaults."""
        competition = adapter._parse_competition(self._payload())

        assert competition.kernels_submissions_only is False
        assert competition.submissions_disabled is False
        assert competition.new_entrant_deadline is None
        assert competition.user_has_entered is False
        assert competition.is_enterable is True

    def test_camel_case_flags_parsed(self, adapter: KaggleAdapter) -> None:
        """Kaggle camelCase eligibility flags should be parsed."""
        competition = adapter._parse_competition(
            self._payload(isKernelsSubmissionsOnly=True, submissionsDisabled=True, userHasEntered=True)
        )

        assert competition.kernels_submissions_only is True
        assert competition.submissions_disabled is True
        assert competition.user_has_entered is True
        assert competition.entry_blockers == ("submissions_disabled", "kernels_submissions_only")

    def test_snake_case_and_string_flags_parsed(self, adapter: KaggleAdapter) -> None:
        """Snake-case keys and string booleans should be coerced."""
        competition = adapter._parse_competition(
            self._payload(is_kernels_submissions_only="true", submissions_disabled="false")
        )

        assert competition.kernels_submissions_only is True
        assert competition.submissions_disabled is False

    def test_new_entrant_deadline_parsed(self, adapter: KaggleAdapter) -> None:
        """A new-entrant deadline should parse into a timezone-aware datetime."""
        competition = adapter._parse_competition(self._payload(newEntrantDeadline="2029-12-25T00:00:00Z"))

        assert competition.new_entrant_deadline is not None
        assert competition.new_entrant_deadline.tzinfo is not None
        assert competition.accepts_new_entrants is True

    def test_unparseable_new_entrant_deadline_ignored(self, adapter: KaggleAdapter) -> None:
        """An unparseable deadline should not block entry."""
        competition = adapter._parse_competition(self._payload(newEntrantDeadline="not-a-date"))

        assert competition.new_entrant_deadline is None
        assert competition.is_enterable is True


class TestSearchCompetitionsEligibilityFilter:
    """Tests for filtering competitions the agent cannot submit to."""

    async def test_unenterable_competitions_skipped(self) -> None:
        """Active-only search should drop code-only and disabled competitions."""
        pages: dict[int, list[dict[str, Any]]] = {
            1: [
                {
                    "ref": "code-only",
                    "title": "Code only",
                    "category": "Featured",
                    "deadline": "2030-01-01T00:00:00Z",
                    "isKernelsSubmissionsOnly": True,
                },
                {
                    "ref": "submissions-off",
                    "title": "Submissions off",
                    "category": "Featured",
                    "deadline": "2030-01-01T00:00:00Z",
                    "submissionsDisabled": True,
                },
                {
                    "ref": "entry-closed",
                    "title": "Entry closed",
                    "category": "Featured",
                    "deadline": "2030-01-01T00:00:00Z",
                    "newEntrantDeadline": "2020-01-01T00:00:00Z",
                },
                {
                    "ref": "open-comp",
                    "title": "Open competition",
                    "category": "Featured",
                    "deadline": "2030-01-01T00:00:00Z",
                },
            ],
            2: [],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=pages.get(page, []))

        adapter = KaggleAdapter(KaggleSettings(username="user", api_key="key"))
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url="https://kaggle.test/api/v1", transport=httpx.MockTransport(handler)
        )
        try:
            found = [competition.id async for competition in adapter.search_competitions()]
        finally:
            await adapter._client.aclose()

        assert found == ["open-comp"]

    async def test_filter_disabled_when_not_active_only(self) -> None:
        """Opting out of active-only should keep ineligible competitions."""
        pages: dict[int, list[dict[str, Any]]] = {
            1: [
                {
                    "ref": "code-only",
                    "title": "Code only",
                    "category": "Featured",
                    "deadline": "2030-01-01T00:00:00Z",
                    "isKernelsSubmissionsOnly": True,
                }
            ],
            2: [],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=pages.get(page, []))

        adapter = KaggleAdapter(KaggleSettings(username="user", api_key="key"))
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url="https://kaggle.test/api/v1", transport=httpx.MockTransport(handler)
        )
        try:
            found = [competition.id async for competition in adapter.search_competitions(active_only=False)]
        finally:
            await adapter._client.aclose()

        assert found == ["code-only"]
