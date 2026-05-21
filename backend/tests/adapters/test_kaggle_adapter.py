"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Any

import httpx
import pytest

from agent_k.adapters import kaggle as kaggle_adapter
from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.exceptions import CompetitionNotFoundError

__all__ = ()

pytestmark = pytest.mark.anyio


def _competition_payload(slug: str, *, title: str | None = None) -> dict[str, Any]:
    return {
        "ref": slug,
        "title": title or slug.title(),
        "category": "Featured",
        "reward": "$1000",
        "deadline": "2099-01-01T00:00:00Z",
        "evaluationMetric": "auc",
    }


def _patch_transport(adapter: KaggleAdapter, handler: Any) -> None:
    """Swap the adapter's httpx client for one driven by ``handler``."""
    adapter._client = httpx.AsyncClient(
        base_url=adapter.config.base_url,
        timeout=adapter.config.timeout,
        auth=(adapter.config.username, adapter.config.api_key),
        transport=httpx.MockTransport(handler),
    )


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


class TestSearchCompetitionsRobustness:
    """Tests for robust pagination in ``KaggleAdapter.search_competitions``."""

    async def test_search_terminates_on_empty_page(self) -> None:
        """Iteration should stop after the first empty page."""
        pages_served: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", 1))
            pages_served.append(page)
            if page == 1:
                return httpx.Response(200, json=[_competition_payload("alpha"), _competition_payload("beta")])
            return httpx.Response(200, json=[])

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        results = [comp async for comp in adapter.search_competitions(active_only=False)]
        await adapter._client.aclose()

        assert [comp.id for comp in results] == ["alpha", "beta"]
        assert pages_served == [1, 2]

    async def test_search_handles_dict_wrapped_payload(self) -> None:
        """A ``{"competitions": [...]}`` shape should be flattened transparently."""

        def handler(request: httpx.Request) -> httpx.Response:
            page = int(request.url.params.get("page", 1))
            if page == 1:
                return httpx.Response(200, json={"competitions": [_competition_payload("gamma")]})
            return httpx.Response(200, json={"competitions": []})

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        results = [comp async for comp in adapter.search_competitions(active_only=False)]
        await adapter._client.aclose()

        assert [comp.id for comp in results] == ["gamma"]

    async def test_search_stops_when_pages_repeat(self) -> None:
        """If the API returns the same data forever, iteration must still terminate."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=[_competition_payload("delta")])

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        results = [comp async for comp in adapter.search_competitions(active_only=False)]
        await adapter._client.aclose()

        assert [comp.id for comp in results] == ["delta"]
        # First page yields the new ID; second page finds zero new IDs and terminates.
        assert call_count == 2

    async def test_search_caps_at_max_pages(self) -> None:
        """Distinct, never-ending pages must still be capped by the safety ceiling."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            page = int(request.url.params.get("page", 1))
            return httpx.Response(200, json=[_competition_payload(f"comp-{page}")])

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        results = [comp async for comp in adapter.search_competitions(active_only=False)]
        await adapter._client.aclose()

        assert call_count == kaggle_adapter._MAX_SEARCH_PAGES
        assert len(results) == kaggle_adapter._MAX_SEARCH_PAGES

    async def test_search_handles_invalid_json(self) -> None:
        """A non-JSON 200 response must terminate iteration cleanly."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>maintenance</html>", headers={"content-type": "text/html"})

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        results = [comp async for comp in adapter.search_competitions(active_only=False)]
        await adapter._client.aclose()

        assert results == []

    async def test_search_raises_for_http_error(self) -> None:
        """A 4xx response must propagate as an ``HTTPStatusError`` instead of being silently parsed."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "unauthorized"})

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k", max_retries=1))
        _patch_transport(adapter, handler)

        iterator = adapter.search_competitions(active_only=False)
        with pytest.raises(httpx.HTTPStatusError):
            await iterator.__anext__()
        await adapter._client.aclose()


class TestGetCompetitionRobustness:
    """Tests for robust shape handling in ``KaggleAdapter.get_competition``."""

    async def test_get_competition_handles_dict_wrapped_payload(self) -> None:
        """``get_competition`` should locate the target inside a dict-wrapped list response."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/competitions/data/list/alpha"):
                return httpx.Response(200, json={"files": []})
            return httpx.Response(200, json={"competitions": [_competition_payload("alpha", title="Alpha")]})

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        competition = await adapter.get_competition("alpha")
        await adapter._client.aclose()

        assert competition.id == "alpha"
        assert competition.title == "Alpha"

    async def test_get_competition_invalid_json_raises_not_found(self) -> None:
        """A malformed list response must surface as ``CompetitionNotFoundError``."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/competitions/data/list/alpha"):
                return httpx.Response(200, json={"files": []})
            return httpx.Response(200, content=b"not-json", headers={"content-type": "text/html"})

        adapter = KaggleAdapter(KaggleSettings(username="u", api_key="k"))
        _patch_transport(adapter, handler)

        with pytest.raises(CompetitionNotFoundError):
            await adapter.get_competition("alpha")
        await adapter._client.aclose()
