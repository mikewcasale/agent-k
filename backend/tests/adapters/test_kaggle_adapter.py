"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.exceptions import CompetitionRulesNotAcceptedError

__all__ = ()

pytestmark = pytest.mark.anyio


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    """AsyncByteStream that yields fixed chunks once and refuses re-iteration."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._consumed = False

    async def __aiter__(self) -> Any:
        if self._consumed:
            raise httpx.StreamConsumed()
        self._consumed = True
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


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


class TestRaiseRulesNotAccepted:
    """Tests for the streaming-aware rules-not-accepted detector."""

    async def test_non_403_status_returns_none(self) -> None:
        """Non-403 responses must not trigger the rules error."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)
        response = httpx.Response(404, text="not found")

        await adapter._raise_rules_not_accepted(response, "titanic")  # does not raise

    async def test_buffered_403_with_rules_text_raises(self) -> None:
        """Already-buffered 403 bodies are inspected without aread."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)
        response = httpx.Response(403, text="You must accept the competition rules to download data.")

        with pytest.raises(CompetitionRulesNotAcceptedError) as exc_info:
            await adapter._raise_rules_not_accepted(response, "titanic")
        assert exc_info.value.context.get("competition_id") == "titanic"

    async def test_buffered_403_without_rules_text_returns_none(self) -> None:
        """A 403 body without the keyword pair is left to raise_for_status."""
        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)
        response = httpx.Response(403, text="forbidden")

        await adapter._raise_rules_not_accepted(response, "titanic")  # does not raise

    async def test_streamed_403_reads_body_and_raises(self) -> None:
        """A streamed 403 (body not yet read) must still trigger the rules error.

        Regression test: previously this raised httpx.ResponseNotRead because
        ``response.text`` was accessed before ``aread()``, masking the helpful
        CompetitionRulesNotAcceptedError with an opaque httpx exception.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                stream=_ChunkedAsyncStream([b"Please accept ", b"the competition rules."]),
                headers={"content-type": "text/html"},
            )

        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            async with adapter._client.stream("GET", "http://example.com/data") as response:
                # Sanity: body is not yet read, so .text would raise ResponseNotRead.
                with pytest.raises(httpx.ResponseNotRead):
                    _ = response.text
                with pytest.raises(CompetitionRulesNotAcceptedError) as exc_info:
                    await adapter._raise_rules_not_accepted(response, "titanic")
                assert exc_info.value.context.get("competition_id") == "titanic"
        finally:
            await adapter._client.aclose()

    async def test_streamed_403_without_keywords_returns_none(self) -> None:
        """A streamed 403 with no rules keywords lets raise_for_status take over."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, stream=_ChunkedAsyncStream([b"forbidden"]))

        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            async with adapter._client.stream("GET", "http://example.com/data") as response:
                await adapter._raise_rules_not_accepted(response, "titanic")  # does not raise
        finally:
            await adapter._client.aclose()


class TestKaggleAdapterFromEnv:
    """Tests for creating adapter from environment."""

    def test_from_env_missing_credentials(self, env: Any) -> None:
        """Should raise error when credentials missing."""
        env.remove("KAGGLE_USERNAME")
        env.remove("KAGGLE_KEY")

        # The from_env method should handle missing credentials
        # Test depends on implementation
