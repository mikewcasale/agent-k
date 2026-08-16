"""Tests for ``KaggleAdapter._request`` retry/backoff behavior.

Exercises the transient-error handling that used to silently propagate 5xx
responses and immediately raise on the first 429. Every retry sleep here
uses a scripted MockTransport with ``rate_limit_delay=0`` and an
``asyncio.sleep`` patched to yield instantly, so the tests stay fast.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import httpx
import pytest

from agent_k.adapters.kaggle import _MAX_RETRY_SLEEP_SECONDS, KaggleAdapter, KaggleSettings, _parse_retry_after
from agent_k.core.exceptions import PlatformConnectionError, RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ()

pytestmark = pytest.mark.anyio


_PATH = "/competitions/list"


def _adapter(
    transport: httpx.AsyncBaseTransport, *, max_retries: int = 3, rate_limit_delay: float = 0.0
) -> KaggleAdapter:
    """Build a KaggleAdapter with a scripted transport and zero real sleep."""
    config = KaggleSettings(username="u", api_key="k", max_retries=max_retries, rate_limit_delay=rate_limit_delay)
    adapter = KaggleAdapter(config)
    adapter._client = httpx.AsyncClient(
        base_url=config.base_url, timeout=config.timeout, auth=(config.username, config.api_key), transport=transport
    )
    return adapter


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Deterministic transport that walks through preset responders."""

    def __init__(self, responders: list[Callable[[httpx.Request], httpx.Response]]) -> None:
        self._responders = responders
        self._index = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        responder = self._responders[min(self._index, len(self._responders) - 1)]
        self._index += 1
        return responder(request)


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real backoff sleeps so retry tests stay fast."""
    monkeypatch.setattr("agent_k.adapters.kaggle.asyncio.sleep", AsyncMock(return_value=None))


class TestParseRetryAfter:
    """Tests for the ``Retry-After`` header parser."""

    def test_none_and_empty(self) -> None:
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("   ") is None

    def test_integer_seconds(self) -> None:
        assert _parse_retry_after("30") == 30
        assert _parse_retry_after("  0  ") == 0

    def test_negative_rejected(self) -> None:
        assert _parse_retry_after("-5") is None

    def test_http_date_ignored(self) -> None:
        # HTTP-date form is intentionally rejected; caller falls back to
        # exponential backoff rather than misinterpreting a date as seconds.
        assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None

    def test_garbage_ignored(self) -> None:
        assert _parse_retry_after("soon") is None


class TestRequestRetry:
    """Retry behavior for ``KaggleAdapter._request``."""

    async def test_success_returns_immediately(self) -> None:
        transport = _ScriptedTransport([lambda _r: httpx.Response(200, json=[])])
        adapter = _adapter(transport)

        response = await adapter._request("GET", _PATH)

        assert response.status_code == 200
        assert len(transport.requests) == 1

    async def test_retries_transient_5xx_then_succeeds(self) -> None:
        transport = _ScriptedTransport(
            [
                lambda _r: httpx.Response(503),
                lambda _r: httpx.Response(502),
                lambda _r: httpx.Response(200, json={"ok": True}),
            ]
        )
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        response = await adapter._request("GET", _PATH)

        assert response.status_code == 200
        assert len(transport.requests) == 3

    async def test_5xx_exhaustion_raises_platform_error(self) -> None:
        transport = _ScriptedTransport(
            [lambda _r: httpx.Response(500), lambda _r: httpx.Response(500), lambda _r: httpx.Response(500)]
        )
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        with pytest.raises(PlatformConnectionError) as exc_info:
            await adapter._request("GET", _PATH)

        assert "500" in str(exc_info.value)
        assert len(transport.requests) == 3

    async def test_non_retryable_4xx_returned_without_retry(self) -> None:
        # 404 is not retryable — callers surface it as CompetitionNotFoundError
        # or similar, so _request must not swallow or retry it.
        transport = _ScriptedTransport([lambda _r: httpx.Response(404)])
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        response = await adapter._request("GET", _PATH)

        assert response.status_code == 404
        assert len(transport.requests) == 1

    async def test_403_returned_without_retry(self) -> None:
        # 403 (rules not accepted) is a permanent, actionable failure —
        # retrying wastes attempts and delays surfacing the real error.
        transport = _ScriptedTransport([lambda _r: httpx.Response(403, text="accept the rules")])
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        response = await adapter._request("GET", _PATH)

        assert response.status_code == 403
        assert len(transport.requests) == 1

    async def test_429_then_success_retries(self) -> None:
        transport = _ScriptedTransport(
            [lambda _r: httpx.Response(429, headers={"Retry-After": "1"}), lambda _r: httpx.Response(200, json=[])]
        )
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        response = await adapter._request("GET", _PATH)

        assert response.status_code == 200
        assert len(transport.requests) == 2

    async def test_429_exhaustion_raises_rate_limit_with_retry_after(self) -> None:
        transport = _ScriptedTransport(
            [
                lambda _r: httpx.Response(429, headers={"Retry-After": "5"}),
                lambda _r: httpx.Response(429, headers={"Retry-After": "5"}),
                lambda _r: httpx.Response(429, headers={"Retry-After": "5"}),
            ]
        )
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        with pytest.raises(RateLimitError) as exc_info:
            await adapter._request("GET", _PATH)

        assert exc_info.value.retry_after == 5
        assert len(transport.requests) == 3

    async def test_429_without_retry_after_falls_back_to_backoff(self) -> None:
        # Missing Retry-After must not raise and must not sleep for an
        # invented fixed number — the adapter falls back to exponential
        # backoff and eventually surfaces RateLimitError.
        transport = _ScriptedTransport(
            [lambda _r: httpx.Response(429), lambda _r: httpx.Response(429), lambda _r: httpx.Response(429)]
        )
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        with pytest.raises(RateLimitError) as exc_info:
            await adapter._request("GET", _PATH)

        assert exc_info.value.retry_after is None
        assert len(transport.requests) == 3

    async def test_httpx_error_retries_then_raises_platform_error(self) -> None:
        raise_count = 0

        def _raise(_request: httpx.Request) -> httpx.Response:
            nonlocal raise_count
            raise_count += 1
            raise httpx.ConnectError("boom")

        transport = _ScriptedTransport([_raise])
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        with pytest.raises(PlatformConnectionError):
            await adapter._request("GET", _PATH)

        assert raise_count == 3

    async def test_httpx_error_then_success_recovers(self) -> None:
        state = {"n": 0}

        def _handler(_request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                raise httpx.ReadError("transient")
            return httpx.Response(200, json={"ok": True})

        transport = _ScriptedTransport([_handler])
        adapter = _adapter(transport, max_retries=3, rate_limit_delay=0.01)

        response = await adapter._request("GET", _PATH)

        assert response.status_code == 200
        assert state["n"] == 2


class TestRetryDelay:
    """Backoff-delay helpers.

    These verify the numeric contract without waiting: exponential growth,
    jitter bounds, and the cap that keeps a hostile ``Retry-After`` header
    from stalling the mission.
    """

    def test_zero_base_returns_zero(self) -> None:
        config = KaggleSettings(username="u", api_key="k", rate_limit_delay=0.0)
        adapter = KaggleAdapter(config)
        assert adapter._retry_delay(1) == 0.0
        assert adapter._retry_delay(5) == 0.0

    def test_retry_delay_bounded_and_grows(self) -> None:
        config = KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)

        d1 = adapter._retry_delay(1)
        d2 = adapter._retry_delay(2)
        d3 = adapter._retry_delay(3)

        # Full-jitter: uniform(0.5*exp, exp) — verify each attempt's range.
        assert 0.5 <= d1 <= 1.0
        assert 1.0 <= d2 <= 2.0
        assert 2.0 <= d3 <= 4.0

    def test_retry_delay_capped(self) -> None:
        config = KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)

        # 2 ** 20 * 1s far exceeds the cap — must stay under it.
        delay = adapter._retry_delay(20)
        assert delay <= _MAX_RETRY_SLEEP_SECONDS

    def test_rate_limit_sleep_respects_retry_after(self) -> None:
        config = KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)

        assert adapter._rate_limit_sleep(1, 5) == 5.0

    def test_rate_limit_sleep_caps_large_retry_after(self) -> None:
        config = KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)

        # 3600s from the server must be clamped to the module cap.
        assert adapter._rate_limit_sleep(1, 3600) == _MAX_RETRY_SLEEP_SECONDS

    def test_rate_limit_sleep_falls_back_when_header_missing(self) -> None:
        config = KaggleSettings(username="u", api_key="k", rate_limit_delay=1.0)
        adapter = KaggleAdapter(config)

        # No header → exponential backoff. Bounded by the same jitter range.
        delay = adapter._rate_limit_sleep(1, None)
        assert 0.5 <= delay <= 1.0
