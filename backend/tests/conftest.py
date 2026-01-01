"""Shared test fixtures and helpers for AGENT-K tests.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# Ensure provider keys are present during test collection to avoid import errors.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

# Re-export dirty_equals for convenience
if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from datetime import datetime

    T = TypeVar("T")

    def IsInstance(arg: type[T]) -> T: ...
    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsFloat(*args: Any, **kwargs: Any) -> float: ...
    def IsInt(*args: Any, **kwargs: Any) -> int: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsBytes(*args: Any, **kwargs: Any) -> bytes: ...
else:
    from dirty_equals import IsBytes, IsDatetime, IsFloat, IsInstance, IsInt, IsStr
    from dirty_equals import IsNow as _IsNow

    def IsNow(*args: Any, **kwargs: Any):
        """IsNow with increased delta for test stability."""
        if "delta" not in kwargs:
            kwargs["delta"] = 10
        return _IsNow(*args, **kwargs)


__all__ = (
    "IsDatetime",
    "IsFloat",
    "IsNow",
    "IsStr",
    "IsBytes",
    "IsInt",
    "IsInstance",
    "TestEnv",
)


class TestEnv:
    """Helper for managing environment variables in tests."""

    __test__ = False  # Prevent pytest from collecting this class

    def __init__(self) -> None:
        self.envars: dict[str, str | None] = {}

    def set(self, name: str, value: str) -> None:
        """Set an environment variable, saving the original value."""
        self.envars[name] = os.getenv(name)
        os.environ[name] = value

    def remove(self, name: str) -> None:
        """Remove an environment variable, saving the original value."""
        self.envars[name] = os.environ.pop(name, None)

    def reset(self) -> None:
        """Reset all modified environment variables to original values."""
        for name, value in self.envars.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture
def env() -> Iterator[TestEnv]:
    """Fixture for managing environment variables in tests."""
    test_env = TestEnv()
    yield test_env
    test_env.reset()


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Create event loop for session-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
async def mock_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide a mock HTTP client for testing."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def mock_http_transport() -> httpx.MockTransport:
    """Create a mock HTTP transport for testing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    return httpx.MockTransport(handler)


@pytest.fixture
async def mock_async_client(
    mock_http_transport: httpx.MockTransport,
) -> AsyncIterator[httpx.AsyncClient]:
    """Provide a mock async HTTP client."""
    async with httpx.AsyncClient(transport=mock_http_transport) as client:
        yield client


@pytest.fixture
def mock_event_emitter() -> MagicMock:
    """Create a mock event emitter for testing."""
    emitter = MagicMock()
    emitter.emit_phase_start = AsyncMock()
    emitter.emit_phase_complete = AsyncMock()
    emitter.emit_progress = AsyncMock()
    emitter.emit_tool_call = AsyncMock()
    emitter.emit_error = AsyncMock()
    return emitter


@pytest.fixture(scope="session")
def kaggle_credentials() -> tuple[str, str]:
    """Get Kaggle credentials from environment or use mock values."""
    username = os.getenv("KAGGLE_USERNAME", "mock_user")
    api_key = os.getenv("KAGGLE_KEY", "mock_api_key")
    return username, api_key


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment or use mock value."""
    return os.getenv("ANTHROPIC_API_KEY", "mock-anthropic-key")


@pytest.fixture(scope="session")
def openrouter_api_key() -> str:
    """Get OpenRouter API key from environment or use mock value."""
    return os.getenv("OPENROUTER_API_KEY", "mock-openrouter-key")


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Get OpenAI API key from environment or use mock value."""
    return os.getenv("OPENAI_API_KEY", "mock-openai-key")


@pytest.fixture(scope="session")
def assets_path() -> Path:
    """Path to test assets directory."""
    return Path(__file__).parent / "assets"


@pytest.fixture
def temp_memory_path(tmp_path: Path) -> Path:
    """Provide a temporary path for memory storage tests."""
    return tmp_path / "test_memory.json"


# Mock data fixtures


@dataclass
class MockCompetition:
    """Mock competition data for testing."""

    id: str = "titanic"
    title: str = "Titanic - Machine Learning from Disaster"
    category: str = "Getting Started"
    reward: str = "$0"
    deadline: str = "2030-01-01T00:00:00Z"
    team_count: int = 50000
    kernel_count: int = 10000

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "reward": self.reward,
            "deadline": self.deadline,
            "team_count": self.team_count,
            "kernel_count": self.kernel_count,
        }


@dataclass
class MockLeaderboardEntry:
    """Mock leaderboard entry for testing."""

    rank: int = 1
    team_name: str = "top_team"
    score: float = 0.99999

    def model_dump(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "team_name": self.team_name,
            "score": self.score,
        }


@pytest.fixture
def mock_competition() -> MockCompetition:
    """Provide a mock competition for testing."""
    return MockCompetition()


@pytest.fixture
def mock_competitions() -> list[MockCompetition]:
    """Provide a list of mock competitions for testing."""
    return [
        MockCompetition(id="titanic", title="Titanic"),
        MockCompetition(id="house-prices", title="House Prices"),
        MockCompetition(id="digit-recognizer", title="Digit Recognizer"),
    ]


@pytest.fixture
def mock_leaderboard() -> list[MockLeaderboardEntry]:
    """Provide mock leaderboard data for testing."""
    return [
        MockLeaderboardEntry(rank=1, team_name="team_1", score=0.99999),
        MockLeaderboardEntry(rank=2, team_name="team_2", score=0.99998),
        MockLeaderboardEntry(rank=3, team_name="team_3", score=0.99997),
        MockLeaderboardEntry(rank=4, team_name="team_4", score=0.99990),
        MockLeaderboardEntry(rank=5, team_name="team_5", score=0.99980),
    ]


# Adapter mocks


@pytest.fixture
def mock_kaggle_adapter(
    mock_competitions: list[MockCompetition],
    mock_leaderboard: list[MockLeaderboardEntry],
) -> AsyncMock:
    """Create a mock Kaggle adapter for testing."""
    adapter = AsyncMock()
    adapter.search_competitions.return_value = [c.model_dump() for c in mock_competitions]
    adapter.get_competition.return_value = mock_competitions[0].model_dump()
    adapter.get_leaderboard.return_value = [e.model_dump() for e in mock_leaderboard]
    adapter.list_datasets.return_value = [
        {"name": "train.csv", "size": 59760},
        {"name": "test.csv", "size": 27960},
    ]
    return adapter


# Platform adapter mock


@pytest.fixture
def mock_platform_adapter() -> AsyncMock:
    """Create a mock platform adapter for testing."""
    adapter = AsyncMock()
    adapter.search_competitions = AsyncMock(return_value=[])
    adapter.get_competition = AsyncMock(return_value=None)
    adapter.get_leaderboard = AsyncMock(return_value=[])
    adapter.submit = AsyncMock(return_value={"submission_id": "test_123"})
    return adapter
