"""Tests for Kaggle adapter retry, rate-limit, and download integrity behaviour.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import gzip
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings, _partial_path
from agent_k.core.exceptions import CompetitionRulesNotAcceptedError, PlatformConnectionError, RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ()

pytestmark = pytest.mark.anyio


def _settings(**overrides: Any) -> KaggleSettings:
    values: dict[str, Any] = {
        "username": "user",
        "api_key": "key",
        "max_retries": 2,
        "rate_limit_delay": 0.01,
        "max_retry_delay": 5.0,
    }
    values.update(overrides)
    return KaggleSettings(**values)


def _adapter(handler: Callable[[httpx.Request], httpx.Response], **overrides: Any) -> KaggleAdapter:
    adapter = KaggleAdapter(_settings(**overrides))
    adapter._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://kaggle.test")
    return adapter


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Record backoff sleeps instead of waiting for them."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", fake_sleep)
    yield recorded


class TestRequestRetries:
    """Retry behaviour of the shared request path."""

    async def test_retries_server_error_then_succeeds(self, sleeps: list[float]) -> None:
        """A transient 5xx on an idempotent request is retried rather than surfaced."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return httpx.Response(503)
            return httpx.Response(200, json={"ok": True})

        adapter = _adapter(handler)
        response = await adapter._request("GET", "/competitions/list")

        assert response.status_code == 200
        assert len(calls) == 2
        assert len(sleeps) == 1

    async def test_exhausted_server_errors_return_last_response(self, sleeps: list[float]) -> None:
        """When retries run out the final response is returned for the caller to handle."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        adapter = _adapter(handler)
        response = await adapter._request("GET", "/competitions/list")

        assert response.status_code == 500
        assert len(calls) == 3
        assert len(sleeps) == 2

    async def test_post_server_error_is_not_retried(self, sleeps: list[float]) -> None:
        """Non-idempotent requests must not be replayed on a server error."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(500)

        adapter = _adapter(handler)
        response = await adapter._request("POST", "/competitions/submissions/submit/comp")

        assert response.status_code == 500
        assert len(calls) == 1
        assert sleeps == []

    async def test_post_transport_error_is_not_replayed(self, sleeps: list[float]) -> None:
        """A submission POST that fails in transit is reported, never resent."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            raise httpx.ConnectError("boom", request=request)

        adapter = _adapter(handler)
        with pytest.raises(PlatformConnectionError):
            await adapter._request("POST", "/competitions/submissions/submit/comp")

        assert len(calls) == 1
        assert sleeps == []

    async def test_transport_error_retried_for_get(self, sleeps: list[float]) -> None:
        """Idempotent reads survive a dropped connection."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                raise httpx.ReadError("reset", request=request)
            return httpx.Response(200, json=[])

        adapter = _adapter(handler)
        response = await adapter._request("GET", "/competitions/list")

        assert response.status_code == 200
        assert len(calls) == 3


class TestRateLimitHandling:
    """Rate-limit handling of the shared request path."""

    async def test_waits_for_retry_after_then_succeeds(self, sleeps: list[float]) -> None:
        """A 429 within the delay budget is waited out instead of aborting the mission."""
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return httpx.Response(200, json=[])

        adapter = _adapter(handler)
        response = await adapter._request("GET", "/competitions/list")

        assert response.status_code == 200
        assert sleeps == [pytest.approx(2.0)]

    async def test_raises_when_retry_after_exceeds_budget(self, sleeps: list[float]) -> None:
        """A long cool-off is surfaced with its hint rather than silently blocking."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "600"})

        adapter = _adapter(handler)
        with pytest.raises(RateLimitError) as exc_info:
            await adapter._request("GET", "/competitions/list")

        assert exc_info.value.retry_after == 600
        assert sleeps == []

    async def test_raises_after_exhausting_rate_limit_retries(self, sleeps: list[float]) -> None:
        """Persistent throttling still terminates with a rate-limit error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "1"})

        adapter = _adapter(handler)
        with pytest.raises(RateLimitError):
            await adapter._request("GET", "/competitions/list")

        assert len(sleeps) == 2

    def test_retry_after_accepts_http_date(self) -> None:
        """Retry-After expressed as an HTTP date is parsed instead of raising."""
        retry_at = datetime.now(UTC) + timedelta(seconds=30)
        response = httpx.Response(429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)})

        assert KaggleAdapter._retry_after_seconds(response) == pytest.approx(30.0, abs=2.0)

    def test_retry_after_falls_back_on_garbage(self) -> None:
        """An unparsable Retry-After falls back to the default cool-off."""
        response = httpx.Response(429, headers={"Retry-After": "soon"})

        assert KaggleAdapter._retry_after_seconds(response) == pytest.approx(60.0)

    def test_backoff_delay_is_bounded(self) -> None:
        """Exponential backoff never exceeds the configured ceiling."""
        adapter = KaggleAdapter(_settings(rate_limit_delay=1.0, max_retry_delay=4.0))

        assert adapter._backoff_delay(0) >= 1.0
        assert adapter._backoff_delay(10) == pytest.approx(4.0)


class TestDownloadIntegrity:
    """Download path integrity guarantees."""

    @staticmethod
    def _listing(names: list[str]) -> httpx.Response:
        return httpx.Response(200, json={"files": [{"name": name} for name in names]})

    async def test_truncated_body_is_retried(self, sleeps: list[float], tmp_path: Path) -> None:
        """A short body is discarded and refetched instead of being staged as data."""
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["train.csv"])
            attempts.append(1)
            if len(attempts) < 2:
                return httpx.Response(200, headers={"Content-Length": "11"}, content=b"id,x")
            return httpx.Response(200, content=b"id,x\n1,2\n3,4")

        adapter = _adapter(handler)
        downloaded = await adapter.download_data("comp", str(tmp_path))

        assert len(attempts) == 2
        assert downloaded == [str(tmp_path / "train.csv")]
        assert (tmp_path / "train.csv").read_bytes() == b"id,x\n1,2\n3,4"
        assert [entry.name for entry in tmp_path.iterdir()] == ["train.csv"]

    async def test_persistent_truncation_raises_and_leaves_no_file(self, sleeps: list[float], tmp_path: Path) -> None:
        """An always-short download fails loudly and stages nothing."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["train.csv"])
            return httpx.Response(200, headers={"Content-Length": "4096"}, content=b"id,x")

        adapter = _adapter(handler)
        with pytest.raises(PlatformConnectionError, match="Truncated download"):
            await adapter.download_data("comp", str(tmp_path))

        assert list(tmp_path.iterdir()) == []

    async def test_encoded_body_is_not_treated_as_truncated(self, sleeps: list[float], tmp_path: Path) -> None:
        """A compressed body whose decoded size differs from Content-Length is accepted."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["train.csv"])
            return httpx.Response(
                200,
                headers={"Content-Length": "9999", "Content-Encoding": "gzip"},
                content=gzip.compress(b"id,x\n1,2\n"),
            )

        adapter = _adapter(handler)
        downloaded = await adapter.download_data("comp", str(tmp_path))

        assert downloaded == [str(tmp_path / "train.csv")]
        assert (tmp_path / "train.csv").read_bytes() == b"id,x\n1,2\n"

    async def test_staging_name_cannot_pass_a_data_file_check(self, tmp_path: Path) -> None:
        """A partial left by a crashed run must not look like train/test/submission data."""
        partial = _partial_path(tmp_path / "train.csv")

        assert partial.name.startswith(".dl-")
        assert "train" not in partial.name
        assert partial.parent == tmp_path

    async def test_transport_failure_is_retried(self, sleeps: list[float], tmp_path: Path) -> None:
        """A dropped download connection is retried within the budget."""
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["train.csv"])
            attempts.append(1)
            if len(attempts) < 2:
                raise httpx.ReadError("reset", request=request)
            return httpx.Response(200, content=b"id,x\n1,2\n")

        adapter = _adapter(handler)
        downloaded = await adapter.download_data("comp", str(tmp_path))

        assert len(attempts) == 2
        assert downloaded == [str(tmp_path / "train.csv")]

    async def test_rules_not_accepted_is_reported(self, sleeps: list[float], tmp_path: Path) -> None:
        """A 403 during streaming raises the actionable rules error, not ResponseNotRead."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["train.csv"])
            return httpx.Response(403, text="You must accept the competition rules")

        adapter = _adapter(handler)
        with pytest.raises(CompetitionRulesNotAcceptedError):
            await adapter.download_data("comp", str(tmp_path))

    async def test_download_entry_cannot_escape_destination(self, sleeps: list[float], tmp_path: Path) -> None:
        """A traversal file name from the listing is rejected before any write."""
        destination = tmp_path / "data"

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["../escaped.csv"])
            return httpx.Response(200, content=b"id,x\n")

        adapter = _adapter(handler)
        with pytest.raises(ValueError, match="escapes destination"):
            await adapter.download_data("comp", str(destination))

        assert not (tmp_path / "escaped.csv").exists()

    async def test_nested_entry_is_staged_under_destination(self, sleeps: list[float], tmp_path: Path) -> None:
        """A listing entry inside a subdirectory is created under the destination."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "data/list" in request.url.path:
                return TestDownloadIntegrity._listing(["nested/train.csv"])
            return httpx.Response(200, content=b"id,x\n")

        adapter = _adapter(handler)
        downloaded = await adapter.download_data("comp", str(tmp_path))

        assert downloaded == [str(tmp_path / "nested" / "train.csv")]
        assert (tmp_path / "nested" / "train.csv").read_bytes() == b"id,x\n"
