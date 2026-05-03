"""Tests for the SCIENTIST research agent.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from pydantic_ai import Agent

from agent_k.agents import get_agent
from agent_k.agents.scientist import ScientistAgent, ScientistDeps, scientist_agent, scientist_agent_instance
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric, LeaderboardEntry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ()

pytestmark = pytest.mark.anyio


def _make_competition() -> Competition:
    return Competition(
        id="titanic",
        title="Titanic",
        competition_type=CompetitionType.GETTING_STARTED,
        metric=EvaluationMetric.ACCURACY,
        deadline=datetime(2099, 12, 31, tzinfo=UTC),
    )


def _make_leaderboard() -> list[LeaderboardEntry]:
    return [
        LeaderboardEntry(rank=1, team_name="alpha", score=0.99),
        LeaderboardEntry(rank=2, team_name="beta", score=0.95),
        LeaderboardEntry(rank=3, team_name="gamma", score=0.90),
    ]


def _make_ctx(deps: ScientistDeps) -> SimpleNamespace:
    """Build a minimal stand-in for ``RunContext`` with a ``deps`` attribute."""
    return SimpleNamespace(deps=deps)


def _make_deps(
    handler: Callable[[httpx.Request], httpx.Response], *, leaderboard: list[LeaderboardEntry] | None = None
) -> ScientistDeps:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    platform_adapter = SimpleNamespace(
        config=SimpleNamespace(username="user", api_key="key"),
        get_leaderboard=_async_return(leaderboard or _make_leaderboard()),
    )
    return ScientistDeps(
        http_client=client,
        platform_adapter=platform_adapter,  # type: ignore[arg-type]
        competition=_make_competition(),
        leaderboard=list(leaderboard or _make_leaderboard()),
    )


def _async_return(value: Any) -> Callable[..., Awaitable[Any]]:
    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _call


class TestScientistAgentSingleton:
    """Tests for the Scientist agent singleton."""

    def test_agent_is_registered(self) -> None:
        """Agent should be registered in the registry."""
        assert get_agent("scientist") is scientist_agent

    def test_agent_metadata(self) -> None:
        """Agent should be configured with a name."""
        assert isinstance(scientist_agent, Agent)
        assert scientist_agent.name == "scientist"


class TestGetKaggleNotebooks:
    """Verify ``get_kaggle_notebooks`` does not fabricate notebooks on API failure."""

    @pytest.fixture
    def agent(self) -> ScientistAgent:
        """Reuse the module singleton; the methods under test do not mutate it."""
        return scientist_agent_instance

    async def test_returns_real_notebooks_when_api_succeeds(self, agent: ScientistAgent) -> None:
        """When the kernels API returns a real list, it should be passed through verbatim."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json=[
                    {
                        "ref": "user/lightgbm-baseline",
                        "title": "LightGBM baseline",
                        "voteCount": 42,
                        "author": "user",
                        "url": "https://www.kaggle.com/code/user/lightgbm-baseline",
                    }
                ],
            )

        deps = _make_deps(handler)
        try:
            notebooks = await agent.get_kaggle_notebooks(_make_ctx(deps))
        finally:
            await deps.http_client.aclose()

        assert "kernels/list" in captured["url"]
        assert len(notebooks) == 1
        assert notebooks[0]["author"] == "user"
        assert notebooks[0]["votes"] == 42
        assert notebooks[0]["ref"] == "user/lightgbm-baseline"

    async def test_returns_empty_list_when_api_returns_non_200(self, agent: ScientistAgent) -> None:
        """A non-200 response must not be replaced by synthetic leaderboard-derived notebooks."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="unavailable")

        deps = _make_deps(handler)
        try:
            notebooks = await agent.get_kaggle_notebooks(_make_ctx(deps))
        finally:
            await deps.http_client.aclose()

        assert notebooks == []

    async def test_returns_empty_list_when_transport_raises(self, agent: ScientistAgent) -> None:
        """A network error must surface as an empty list, not fabricated data."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        deps = _make_deps(handler)
        try:
            notebooks = await agent.get_kaggle_notebooks(_make_ctx(deps))
        finally:
            await deps.http_client.aclose()

        assert notebooks == []

    async def test_returns_empty_list_when_payload_is_unexpected(self, agent: ScientistAgent) -> None:
        """A non-list JSON payload must not be coerced into fake notebook entries."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": "bad request"})

        deps = _make_deps(handler)
        try:
            notebooks = await agent.get_kaggle_notebooks(_make_ctx(deps))
        finally:
            await deps.http_client.aclose()

        assert notebooks == []

    async def test_no_synthetic_data_uses_leaderboard_team_names(self, agent: ScientistAgent) -> None:
        """Regression: leaderboard team names must never appear as fabricated notebook authors."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[])

        deps = _make_deps(handler)
        leaderboard_team_names = {entry.team_name for entry in deps.leaderboard}
        try:
            notebooks = await agent.get_kaggle_notebooks(_make_ctx(deps))
        finally:
            await deps.http_client.aclose()

        assert notebooks == []
        authors = {n.get("author") for n in notebooks}
        assert authors.isdisjoint(leaderboard_team_names)
