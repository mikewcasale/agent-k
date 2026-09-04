"""Tests for competition-cache staging and mission session data preparation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from agent_k.mission import nodes
from agent_k.mission.nodes import (
    _cache_exists,
    _cleanup_session_data,
    _get_competition_cache_dir,
    _get_session_data_dir,
    _get_session_root,
    _prepare_session_data,
    _promote_staging_dir,
)

if TYPE_CHECKING:
    from agent_k.core.protocols import PlatformAdapter

__all__ = ()

pytestmark = pytest.mark.anyio

_TRAIN_CSV = "Id,feature,target\n1,0.5,10\n2,1.5,20\n3,2.5,30\n"
_TEST_CSV = "Id,feature\n4,3.5\n5,4.5\n"
_SAMPLE_CSV = "Id,target\n4,0\n5,0\n"


class FakePlatformAdapter:
    """Platform adapter stand-in that writes real competition files to disk.

    Only ``download_data`` is exercised by the session-data path; the fake
    records every call so tests can assert a competition is fetched once.
    """

    def __init__(self, *, gate: asyncio.Event | None = None, fail_times: int = 0) -> None:
        self.calls: list[str] = []
        self.destinations: list[Path] = []
        self._gate = gate
        self._fail_times = fail_times
        self.started = asyncio.Event()

    async def download_data(self, competition_id: str, destination: str) -> list[str]:
        """Write the canonical competition files into ``destination``."""
        self.calls.append(competition_id)
        target = Path(destination)
        self.destinations.append(target)
        target.mkdir(parents=True, exist_ok=True)

        # Write a partial dataset first so a reader that peeks mid-download
        # would observe an incomplete competition.
        (target / "train.csv").write_text(_TRAIN_CSV, encoding="utf-8")
        self.started.set()
        if self._gate is not None:
            await self._gate.wait()
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("simulated download failure")

        (target / "test.csv").write_text(_TEST_CSV, encoding="utf-8")
        (target / "sample_submission.csv").write_text(_SAMPLE_CSV, encoding="utf-8")
        return [str(path) for path in sorted(target.iterdir())]


def _adapter(**kwargs: Any) -> PlatformAdapter:
    return cast("PlatformAdapter", FakePlatformAdapter(**kwargs))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``Path.home()`` at a temporary directory for cache isolation."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(nodes, "_COMPETITION_CACHE_LOCKS", {})
    return home


class TestPrepareSessionData:
    """Tests for ``_prepare_session_data``."""

    async def test_downloads_and_stages_competition_files(self) -> None:
        """The first mission downloads the competition and stages a local copy."""
        adapter = _adapter()

        train, test, sample = await _prepare_session_data(adapter, "mission-1", "demo-comp")

        assert train.read_text(encoding="utf-8") == _TRAIN_CSV
        assert test.read_text(encoding="utf-8") == _TEST_CSV
        assert sample.read_text(encoding="utf-8") == _SAMPLE_CSV
        assert train.parent == _get_session_data_dir("mission-1")
        assert cast("FakePlatformAdapter", adapter).calls == ["demo-comp"]

    async def test_second_mission_reuses_the_cache(self) -> None:
        """A later mission copies from the cache instead of downloading again."""
        adapter = _adapter()

        await _prepare_session_data(adapter, "mission-1", "demo-comp")
        train, _test, _sample = await _prepare_session_data(adapter, "mission-2", "demo-comp")

        assert cast("FakePlatformAdapter", adapter).calls == ["demo-comp"]
        assert train.read_text(encoding="utf-8") == _TRAIN_CSV
        assert train.parent == _get_session_data_dir("mission-2")

    async def test_session_copy_is_independent_of_the_cache(self) -> None:
        """A solution overwriting staged data must not corrupt the shared cache."""
        adapter = _adapter()

        train, _test, _sample = await _prepare_session_data(adapter, "mission-1", "demo-comp")
        train.write_text("Id,feature,target\n", encoding="utf-8")

        cached_train = _get_competition_cache_dir("demo-comp") / "train.csv"
        assert cached_train.read_text(encoding="utf-8") == _TRAIN_CSV

    async def test_writes_cache_metadata(self) -> None:
        """The published cache records the files it contains."""
        await _prepare_session_data(_adapter(), "mission-1", "demo-comp")

        metadata = json.loads((_get_competition_cache_dir("demo-comp") / "metadata.json").read_text(encoding="utf-8"))

        assert metadata["competition_id"] == "demo-comp"
        assert {entry["path"] for entry in metadata["files"]} == {"train.csv", "test.csv", "sample_submission.csv"}

    async def test_no_staging_directories_are_left_behind(self) -> None:
        """Staging directories are removed once the download is published."""
        await _prepare_session_data(_adapter(), "mission-1", "demo-comp")

        staging_root = Path.home() / ".agent_k" / "competitions" / ".staging"
        assert list(staging_root.iterdir()) == []

    async def test_filesystem_work_runs_off_the_event_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Copying the dataset must not block the loop other missions run on."""
        copy_threads: list[int] = []
        real_copy = nodes._copy_from_cache

        def _recording_copy(cache_dir: Path, session_dir: Path) -> None:
            copy_threads.append(threading.get_ident())
            real_copy(cache_dir, session_dir)

        monkeypatch.setattr(nodes, "_copy_from_cache", _recording_copy)

        await _prepare_session_data(_adapter(), "mission-1", "demo-comp")

        assert copy_threads
        assert threading.get_ident() not in copy_threads


class TestConcurrentMissions:
    """Tests for two missions racing on the same competition."""

    async def test_concurrent_missions_download_once(self) -> None:
        """Overlapping missions share a single download and both get full data."""
        gate = asyncio.Event()
        fake = FakePlatformAdapter(gate=gate)
        adapter = cast("PlatformAdapter", fake)

        first = asyncio.create_task(_prepare_session_data(adapter, "mission-1", "demo-comp"))
        await fake.started.wait()
        second = asyncio.create_task(_prepare_session_data(adapter, "mission-2", "demo-comp"))
        await asyncio.sleep(0)
        gate.set()

        results = await asyncio.gather(first, second)

        assert fake.calls == ["demo-comp"]
        for train, test, sample in results:
            assert train.read_text(encoding="utf-8") == _TRAIN_CSV
            assert test.read_text(encoding="utf-8") == _TEST_CSV
            assert sample.read_text(encoding="utf-8") == _SAMPLE_CSV

    async def test_partial_download_is_never_published(self) -> None:
        """An in-flight download stays invisible to the shared cache."""
        gate = asyncio.Event()
        fake = FakePlatformAdapter(gate=gate)
        adapter = cast("PlatformAdapter", fake)

        task = asyncio.create_task(_prepare_session_data(adapter, "mission-1", "demo-comp"))
        await fake.started.wait()

        cache_dir = _get_competition_cache_dir("demo-comp")
        assert not _cache_exists(cache_dir)
        assert not (cache_dir / "train.csv").exists()
        assert fake.destinations[0] != cache_dir

        gate.set()
        await task
        assert _cache_exists(cache_dir)


class TestFailedDownloads:
    """Tests for downloads that fail partway through."""

    async def test_failure_leaves_no_partial_cache(self) -> None:
        """A failed download publishes nothing for the next mission to read."""
        adapter = _adapter(fail_times=1)

        with pytest.raises(RuntimeError, match="simulated download failure"):
            await _prepare_session_data(adapter, "mission-1", "demo-comp")

        cache_dir = _get_competition_cache_dir("demo-comp")
        assert not _cache_exists(cache_dir)
        assert list(cache_dir.iterdir()) == []
        assert list((Path.home() / ".agent_k" / "competitions" / ".staging").iterdir()) == []

    async def test_retry_after_failure_succeeds(self) -> None:
        """The competition is re-downloaded after a failed attempt."""
        adapter = _adapter(fail_times=1)

        with pytest.raises(RuntimeError):
            await _prepare_session_data(adapter, "mission-1", "demo-comp")
        train, _test, _sample = await _prepare_session_data(adapter, "mission-1", "demo-comp")

        assert train.read_text(encoding="utf-8") == _TRAIN_CSV
        assert cast("FakePlatformAdapter", adapter).calls == ["demo-comp", "demo-comp"]

    async def test_incomplete_download_raises(self) -> None:
        """A download missing required files never becomes the cache."""

        class _MissingFilesAdapter:
            async def download_data(self, competition_id: str, destination: str) -> list[str]:
                path = Path(destination) / "train.csv"
                path.write_text(_TRAIN_CSV, encoding="utf-8")
                return [str(path)]

        with pytest.raises(RuntimeError, match="Download incomplete"):
            await _prepare_session_data(cast("PlatformAdapter", _MissingFilesAdapter()), "mission-1", "demo-comp")

        assert not _cache_exists(_get_competition_cache_dir("demo-comp"))


class TestPromoteStagingDir:
    """Tests for ``_promote_staging_dir``."""

    def _write_dataset(self, directory: Path, marker: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "train.csv").write_text(marker, encoding="utf-8")
        (directory / "test.csv").write_text(_TEST_CSV, encoding="utf-8")
        (directory / "sample_submission.csv").write_text(_SAMPLE_CSV, encoding="utf-8")

    def test_publishes_into_an_empty_cache_dir(self, tmp_path: Path) -> None:
        """A fresh cache directory is replaced by the completed download."""
        staging = tmp_path / "staging"
        cache = tmp_path / "cache"
        cache.mkdir()
        self._write_dataset(staging, _TRAIN_CSV)

        assert _promote_staging_dir(staging, cache, "demo-comp") is True
        assert (cache / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV
        assert not staging.exists()

    def test_replaces_an_incomplete_cache_dir(self, tmp_path: Path) -> None:
        """Leftovers from a crashed download are cleared before publishing."""
        staging = tmp_path / "staging"
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / "train.csv").write_text("truncated", encoding="utf-8")
        self._write_dataset(staging, _TRAIN_CSV)

        assert _promote_staging_dir(staging, cache, "demo-comp") is True
        assert (cache / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV
        assert (cache / "sample_submission.csv").exists()

    def test_keeps_a_complete_cache_written_by_another_process(self, tmp_path: Path) -> None:
        """A cache another writer already completed is left untouched."""
        staging = tmp_path / "staging"
        cache = tmp_path / "cache"
        self._write_dataset(cache, "winner")
        self._write_dataset(staging, _TRAIN_CSV)

        assert _promote_staging_dir(staging, cache, "demo-comp") is False
        assert (cache / "train.csv").read_text(encoding="utf-8") == "winner"


class TestCleanupSessionData:
    """Tests for ``_cleanup_session_data``."""

    async def test_removes_the_mission_session_tree(self) -> None:
        """Session data is deleted when a mission ends."""
        await _prepare_session_data(_adapter(), "mission-1", "demo-comp")
        session_root = _get_session_root("mission-1")
        assert session_root.exists()

        await _cleanup_session_data("mission-1")

        assert not session_root.exists()
        assert _cache_exists(_get_competition_cache_dir("demo-comp"))

    async def test_missing_session_tree_is_not_an_error(self) -> None:
        """Cleaning a mission that never staged data is a no-op."""
        await _cleanup_session_data("mission-unknown")

        assert not _get_session_root("mission-unknown").exists()
