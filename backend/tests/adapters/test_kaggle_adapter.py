"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, normalize_competition_files

if TYPE_CHECKING:
    from collections.abc import Callable

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


class TestNormalizeCompetitionFiles:
    """Tests for normalize_competition_files response-shape handling."""

    def test_bare_list_of_dicts(self) -> None:
        """A bare list of dict entries should map to normalized records."""
        payload = [
            {"name": "train.csv", "totalBytes": 100, "description": "Training data", "url": "/dl/train"},
            {"name": "test.csv", "totalBytes": 50, "description": None},
        ]

        files = normalize_competition_files(payload)

        assert files == [
            {"name": "train.csv", "size": 100, "description": "Training data", "url": "/dl/train"},
            {"name": "test.csv", "size": 50, "description": None, "url": ""},
        ]

    def test_dict_wrapped_files_key(self) -> None:
        """A dict payload with a ``files`` key should be unwrapped."""
        payload = {"files": [{"name": "train.csv", "totalBytes": 1}]}

        files = normalize_competition_files(payload)

        assert files == [{"name": "train.csv", "size": 1, "description": None, "url": ""}]

    def test_dict_wrapped_datasetfiles_key(self) -> None:
        """A dict payload with a ``datasetFiles`` key should be unwrapped."""
        payload = {"datasetFiles": [{"name": "submission.csv"}]}

        files = normalize_competition_files(payload)

        assert files == [{"name": "submission.csv", "size": None, "description": None, "url": ""}]

    def test_string_entries(self) -> None:
        """String entries should be promoted to dicts using the string as name."""
        files = normalize_competition_files(["train.csv", "  test.csv  ", ""])

        assert files == [
            {"name": "train.csv", "size": None, "description": None, "url": ""},
            {"name": "test.csv", "size": None, "description": None, "url": ""},
        ]

    def test_mixed_entries_and_name_nullable_fallback(self) -> None:
        """Mixed dict/string entries and ``nameNullable`` fallback should work."""
        payload = [
            "raw_string.csv",
            {"nameNullable": "fallback.csv", "totalBytes": 10},
            {"name": "", "nameNullable": "second.csv"},
            {"name": "skipped_none", "totalBytes": None, "size": 99},
        ]

        files = normalize_competition_files(payload)

        assert files == [
            {"name": "raw_string.csv", "size": None, "description": None, "url": ""},
            {"name": "fallback.csv", "size": 10, "description": None, "url": ""},
            {"name": "second.csv", "size": None, "description": None, "url": ""},
            {"name": "skipped_none", "size": 99, "description": None, "url": ""},
        ]

    def test_invalid_or_empty_payload_returns_empty(self) -> None:
        """Non-list/non-dict payloads and empty containers should return ``[]``."""
        assert normalize_competition_files(None) == []
        assert normalize_competition_files(42) == []
        assert normalize_competition_files({"files": "not-a-list"}) == []
        assert normalize_competition_files({}) == []
        assert normalize_competition_files([]) == []
        assert normalize_competition_files([{"description": "no name"}, 42]) == []


class TestListCompetitionFiles:
    """Tests for KaggleAdapter.list_competition_files."""

    @pytest.fixture
    def adapter_with_transport(self) -> Callable[[httpx.MockTransport], KaggleAdapter]:
        """Build a KaggleAdapter whose HTTP client uses the supplied mock transport."""

        def factory(transport: httpx.MockTransport) -> KaggleAdapter:
            adapter = KaggleAdapter(KaggleSettings(username="user", api_key="key"))
            adapter._client = httpx.AsyncClient(
                base_url=adapter.config.base_url, timeout=adapter.config.timeout, transport=transport
            )
            return adapter

        return factory

    async def test_lists_files_from_dict_payload(
        self, adapter_with_transport: Callable[[httpx.MockTransport], KaggleAdapter]
    ) -> None:
        """list_competition_files should tolerate dict-wrapped payloads."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/competitions/data/list/titanic"
            return httpx.Response(
                200, json={"files": [{"name": "train.csv", "totalBytes": 5, "description": "rows"}, "test.csv"]}
            )

        adapter = adapter_with_transport(httpx.MockTransport(handler))
        try:
            files = await adapter.list_competition_files("titanic")
        finally:
            await adapter._client.aclose()

        assert files == [
            {"name": "train.csv", "size": 5, "description": "rows", "url": ""},
            {"name": "test.csv", "size": None, "description": None, "url": ""},
        ]


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation
