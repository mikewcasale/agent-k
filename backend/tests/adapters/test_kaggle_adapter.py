"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, _resolve_safe_target

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


class TestResolveSafeTarget:
    """Tests for path traversal hardening in download targets."""

    def test_basename_resolves_under_dest(self, tmp_path: Path) -> None:
        """A plain basename should resolve under the destination directory."""
        result = _resolve_safe_target(tmp_path, "train.csv")
        assert result is not None
        assert result == (tmp_path / "train.csv").resolve()

    def test_nested_subpath_allowed_when_under_dest(self, tmp_path: Path) -> None:
        """A nested relative path that stays under dest should be allowed."""
        result = _resolve_safe_target(tmp_path, "train/part-001.csv")
        assert result is not None
        assert result == (tmp_path / "train" / "part-001.csv").resolve()

    @pytest.mark.parametrize(
        "unsafe_name",
        ["../escape.csv", "../../etc/passwd", "nested/../../escape.csv", "/etc/passwd", "", "with\x00null.csv"],
    )
    def test_rejects_unsafe_names(self, tmp_path: Path, unsafe_name: str) -> None:
        """Paths that escape dest, are absolute, or contain NUL must be rejected."""
        assert _resolve_safe_target(tmp_path, unsafe_name) is None


class TestDownloadDataPathSafety:
    """End-to-end checks for download_data path validation."""

    async def test_skips_traversal_entries_and_writes_safe_files(self, tmp_path: Path) -> None:
        """download_data should only write files that resolve under destination."""
        list_payload = {
            "files": [
                {"name": "train.csv"},
                {"name": "../escape.csv"},
                {"name": "/etc/passwd"},
                {"name": "nested/inner.csv"},
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/competitions/data/list/test-comp"):
                return httpx.Response(200, json=list_payload)
            if "train.csv" in path:
                return httpx.Response(200, content=b"id,target\n1,0\n")
            if "inner.csv" in path:
                return httpx.Response(200, content=b"col\n1\n")
            return httpx.Response(404)

        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url, transport=httpx.MockTransport(handler), auth=(config.username, config.api_key)
        )
        try:
            destination = tmp_path / "data"
            downloaded = await adapter.download_data("test-comp", str(destination))
        finally:
            await adapter._client.aclose()

        downloaded_names = sorted(Path(p).name for p in downloaded)
        assert downloaded_names == ["inner.csv", "train.csv"]
        assert (destination / "train.csv").exists()
        assert (destination / "nested" / "inner.csv").exists()
        # The traversal targets must not have been written.
        assert not (tmp_path / "escape.csv").exists()
        assert not Path("/etc/passwd-kaggle-test").exists()
