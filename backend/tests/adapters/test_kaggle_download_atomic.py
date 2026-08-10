"""Tests for atomic download behavior in ``KaggleAdapter.download_data``.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from httpx._content import AsyncIteratorByteStream

from agent_k.adapters.kaggle import KaggleAdapter, KaggleSettings
from agent_k.core.exceptions import PlatformConnectionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

__all__ = ()

pytestmark = pytest.mark.anyio


_LIST_URL = "/competitions/data/list/comp"
_DOWNLOAD_URL = "/competitions/data/download/comp/train.csv"


def _adapter_with_transport(transport: httpx.AsyncBaseTransport, *, max_retries: int = 3) -> KaggleAdapter:
    """Build an adapter and swap in a controlled HTTP transport."""
    config = KaggleSettings(username="u", api_key="k", max_retries=max_retries, rate_limit_delay=0.0)
    adapter = KaggleAdapter(config)
    adapter._client = httpx.AsyncClient(
        base_url=config.base_url, timeout=config.timeout, auth=(config.username, config.api_key), transport=transport
    )
    return adapter


def _list_response() -> httpx.Response:
    return httpx.Response(200, json={"files": [{"name": "train.csv", "url": _DOWNLOAD_URL}]})


def _stream_response(chunks: list[bytes], *, raise_after: int | None = None) -> httpx.Response:
    async def _iter() -> AsyncIterator[bytes]:
        for index, chunk in enumerate(chunks):
            if raise_after is not None and index >= raise_after:
                raise httpx.ReadError("simulated network drop")
            yield chunk

    return httpx.Response(200, stream=AsyncIteratorByteStream(_iter()))


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Transport that walks through a preset sequence of responders."""

    def __init__(self, responders: list[Callable[[httpx.Request], httpx.Response]]) -> None:
        self._responders = responders
        self._index = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        responder = self._responders[min(self._index, len(self._responders) - 1)]
        self._index += 1
        return responder(request)


async def test_download_success_writes_only_final_file(tmp_path: Path) -> None:
    payload = b"id,label\n1,a\n2,b\n"
    transport = _ScriptedTransport([lambda _r: _list_response(), lambda _r: _stream_response([payload])])
    adapter = _adapter_with_transport(transport)

    downloaded = await adapter.download_data("comp", str(tmp_path))

    assert downloaded == [str(tmp_path / "train.csv")]
    assert (tmp_path / "train.csv").read_bytes() == payload
    # No stray temp file left behind
    assert not (tmp_path / "train.csv.part").exists()


async def test_mid_stream_failure_leaves_no_partial_file(tmp_path: Path) -> None:
    # Always fails partway; retries exhaust and we should still see no files.
    transport = _ScriptedTransport(
        [
            lambda _r: _list_response(),
            lambda _r: _stream_response([b"header\n", b"row1"], raise_after=1),
            lambda _r: _stream_response([b"header\n", b"row1"], raise_after=1),
            lambda _r: _stream_response([b"header\n", b"row1"], raise_after=1),
        ]
    )
    adapter = _adapter_with_transport(transport, max_retries=3)

    with pytest.raises(PlatformConnectionError):
        await adapter.download_data("comp", str(tmp_path))

    assert not (tmp_path / "train.csv").exists()
    assert not (tmp_path / "train.csv.part").exists()


async def test_transient_failure_recovers_on_retry(tmp_path: Path) -> None:
    good_payload = b"id,label\n1,a\n2,b\n"
    transport = _ScriptedTransport(
        [
            lambda _r: _list_response(),
            lambda _r: _stream_response([b"header\n"], raise_after=0),
            lambda _r: _stream_response([good_payload]),
        ]
    )
    adapter = _adapter_with_transport(transport, max_retries=3)

    downloaded = await adapter.download_data("comp", str(tmp_path))

    assert downloaded == [str(tmp_path / "train.csv")]
    assert (tmp_path / "train.csv").read_bytes() == good_payload
    assert not (tmp_path / "train.csv.part").exists()
    # 1 list + 2 download attempts (1 failed, 1 successful)
    assert len(transport.requests) == 3


async def test_preexisting_temp_is_replaced(tmp_path: Path) -> None:
    # A leftover .part from a prior crashed run must not corrupt the new download.
    stale = tmp_path / "train.csv.part"
    stale.write_bytes(b"stale garbage from prior run")

    payload = b"clean,payload\n"
    transport = _ScriptedTransport([lambda _r: _list_response(), lambda _r: _stream_response([payload])])
    adapter = _adapter_with_transport(transport)

    await adapter.download_data("comp", str(tmp_path))

    assert (tmp_path / "train.csv").read_bytes() == payload
    assert not stale.exists()
