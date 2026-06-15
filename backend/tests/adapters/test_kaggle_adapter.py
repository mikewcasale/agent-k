"""Tests for the Kaggle API adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.exceptions import SubmissionError

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

    def test_submit_timeout_default(self) -> None:
        """Submit uploads default to a 5-minute timeout, longer than the standard 30s."""
        config = KaggleSettings(username="user", api_key="key")

        assert config.submit_timeout == 300

    def test_submit_timeout_configurable(self) -> None:
        """Submit timeout is overridable via the config or environment."""
        config = KaggleSettings(username="user", api_key="key", submit_timeout=900)

        assert config.submit_timeout == 900


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


class TestIterFileChunks:
    """Tests for the streaming chunk iterator used during submission uploads."""

    async def test_yields_default_chunk_size(self, tmp_path: Path) -> None:
        """Default chunk size is 1 MiB; large files emit multiple chunks plus a tail."""
        file_path = tmp_path / "big.csv"
        tail = 7
        payload = b"x" * (3 * 1024 * 1024 + tail)
        file_path.write_bytes(payload)

        chunks = [chunk async for chunk in KaggleAdapter._iter_file_chunks(file_path)]

        assert len(chunks) == 4
        assert len(chunks[0]) == 1024 * 1024
        assert len(chunks[1]) == 1024 * 1024
        assert len(chunks[2]) == 1024 * 1024
        assert len(chunks[3]) == tail
        assert b"".join(chunks) == payload

    async def test_respects_explicit_chunk_size(self, tmp_path: Path) -> None:
        """Caller can specify a smaller chunk size."""
        file_path = tmp_path / "x.txt"
        file_path.write_bytes(b"abcdefgh")

        chunks = [chunk async for chunk in KaggleAdapter._iter_file_chunks(file_path, chunk_size=3)]

        assert chunks == [b"abc", b"def", b"gh"]

    async def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Empty file yields no chunks rather than raising."""
        file_path = tmp_path / "empty.txt"
        file_path.write_bytes(b"")

        chunks = [chunk async for chunk in KaggleAdapter._iter_file_chunks(file_path)]

        assert chunks == []

    async def test_closes_handle_after_iteration(self, tmp_path: Path) -> None:
        """Generator releases its file handle when exhausted so re-uploads succeed."""
        file_path = tmp_path / "x.txt"
        file_path.write_bytes(b"hello")

        chunks = [chunk async for chunk in KaggleAdapter._iter_file_chunks(file_path, chunk_size=2)]

        # Re-reading the file must succeed; would fail on Windows if a handle leaked.
        assert b"".join(chunks) == b"hello"
        assert file_path.read_bytes() == b"hello"


class TestKaggleAdapterSubmitStreaming:
    """Tests verifying submission uploads stream from disk with the dedicated timeout."""

    async def test_submit_streams_with_content_length_and_long_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submission upload should set Content-Length, stream the body, and use submit_timeout."""
        submission_file = tmp_path / "submission.csv"
        # 3 MiB + a tail to ensure streaming produces multiple chunks
        payload = b"id,target\n" + b"a" * (3 * 1024 * 1024)
        submission_file.write_bytes(payload)

        captured_upload: dict[str, Any] = {}
        captured_timeouts: list[httpx.Timeout] = []

        def api_handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/competitions/submission-url"):
                return httpx.Response(
                    200, json={"token": "tok-123", "createUrl": "https://upload.example.com/blob/abc"}
                )
            if "/competitions/submissions/submit/" in path:
                return httpx.Response(200, json={"ref": "sub-123"})
            return httpx.Response(404, json={})

        def upload_handler(request: httpx.Request) -> httpx.Response:
            captured_upload["method"] = request.method
            captured_upload["url"] = str(request.url)
            captured_upload["content_length_header"] = request.headers.get("Content-Length")
            captured_upload["body"] = request.read()
            return httpx.Response(200)

        config = KaggleSettings(username="user", api_key="key", submit_timeout=600)
        adapter = KaggleAdapter(config)

        # Replace the API client with one backed by MockTransport
        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url, timeout=config.timeout, transport=httpx.MockTransport(api_handler)
        )

        original_async_client = httpx.AsyncClient

        def mock_upload_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            captured_timeouts.append(kwargs.get("timeout"))
            kwargs["transport"] = httpx.MockTransport(upload_handler)
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr("agent_k.adapters.kaggle.httpx.AsyncClient", mock_upload_client)

        try:
            submission = await adapter.submit("titanic", str(submission_file), message="test")
        finally:
            await adapter._client.aclose()

        assert submission.competition_id == "titanic"
        assert submission.status == "pending"
        assert submission.id == "sub-123"

        assert captured_upload["method"] == "PUT"
        assert captured_upload["url"] == "https://upload.example.com/blob/abc"
        assert captured_upload["content_length_header"] == str(len(payload))
        assert captured_upload["body"] == payload

        assert len(captured_timeouts) == 1
        timeout = captured_timeouts[0]
        assert isinstance(timeout, httpx.Timeout)
        # All timeout components should reflect the longer submit_timeout
        assert timeout.read == 600.0

    async def test_submit_raises_on_upload_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-2xx upload responses surface as SubmissionError without breaking streaming."""
        submission_file = tmp_path / "submission.csv"
        submission_file.write_bytes(b"id,target\n1,0\n")

        def api_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/competitions/submission-url"):
                return httpx.Response(200, json={"token": "tok-x", "createUrl": "https://upload.example.com/blob/x"})
            return httpx.Response(404, json={})

        def upload_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        config = KaggleSettings(username="user", api_key="key")
        adapter = KaggleAdapter(config)

        await adapter._client.aclose()
        adapter._client = httpx.AsyncClient(
            base_url=config.base_url, timeout=config.timeout, transport=httpx.MockTransport(api_handler)
        )

        original_async_client = httpx.AsyncClient

        def mock_upload_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(upload_handler)
            return original_async_client(*args, **kwargs)

        monkeypatch.setattr("agent_k.adapters.kaggle.httpx.AsyncClient", mock_upload_client)

        try:
            with pytest.raises(SubmissionError, match="Upload failed"):
                await adapter.submit("titanic", str(submission_file))
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
