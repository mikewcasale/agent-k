"""Tests for the OpenEvolve adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path

import pytest

from agent_k.adapters.openevolve import (
    _DEFAULT_OPENEVOLVE_MODEL,
    OpenEvolveAdapter,
    OpenEvolveJobState,
    OpenEvolveSettings,
)
from agent_k.infra.providers import OPENROUTER_FREE_MODELS

__all__ = ()

pytestmark = pytest.mark.anyio


class TestDefaultOpenEvolveModel:
    """Regression tests for the OpenEvolve default-model fallback.

    When callers omit ``model_specs``, the runner falls back to this constant.
    A stale value here previously caused live missions to 404 mid-evolution.
    """

    def test_default_is_a_known_free_model(self) -> None:
        """Default must resolve to an entry in OPENROUTER_FREE_MODELS."""
        assert _DEFAULT_OPENEVOLVE_MODEL in OPENROUTER_FREE_MODELS.values(), (
            f"_DEFAULT_OPENEVOLVE_MODEL {_DEFAULT_OPENEVOLVE_MODEL!r} must be one of "
            f"OPENROUTER_FREE_MODELS values {sorted(OPENROUTER_FREE_MODELS.values())!r}"
        )

    def test_default_is_openrouter_free_tier(self) -> None:
        """Default must be a free OpenRouter slug."""
        assert _DEFAULT_OPENEVOLVE_MODEL.startswith("openrouter:")
        assert _DEFAULT_OPENEVOLVE_MODEL.endswith(":free")


class TestOpenEvolveAdapter:
    """Tests for the OpenEvolveAdapter class."""

    def test_adapter_creation(self) -> None:
        """Adapter should be created as a dataclass."""
        adapter = OpenEvolveAdapter()

        assert adapter is not None
        assert adapter.platform_name == "openevolve"

    async def test_authenticate_returns_true(self) -> None:
        """Authenticate should return True (stub)."""
        adapter = OpenEvolveAdapter()

        result = await adapter.authenticate()
        assert result is True

    async def test_search_competitions_returns_catalog(self) -> None:
        """Search should yield the default catalog."""
        adapter = OpenEvolveAdapter()

        competitions = [comp async for comp in adapter.search_competitions()]
        assert competitions

    async def test_get_leaderboard_returns_baseline(self) -> None:
        """Leaderboard should return baseline entries when empty."""
        adapter = OpenEvolveAdapter()

        result = await adapter.get_leaderboard("oe-tabular-classification", limit=5)
        assert len(result) == 5
        assert result[0].rank == 1

    async def test_submit_and_status(self, tmp_path: Path) -> None:
        """Submission status should complete with a score."""
        adapter = OpenEvolveAdapter()

        file_path = tmp_path / "submission.csv"
        file_path.write_text("id,target\\n0,0.5\\n", encoding="utf-8")

        submission = await adapter.submit("oe-tabular-classification", str(file_path))
        assert submission.status == "pending"

        status = await adapter.get_submission_status("oe-tabular-classification", submission.id)
        assert status.status == "complete"
        assert status.public_score is not None

    async def test_download_data_writes_files(self, tmp_path: Path) -> None:
        """Download should write train/test/submission files."""
        adapter = OpenEvolveAdapter()

        files = await adapter.download_data("oe-tabular-classification", str(tmp_path))
        assert len(files) == 3
        for file_path in files:
            assert Path(file_path).exists()

    async def test_evolution_job_lifecycle(self) -> None:
        """Evolution job should complete and return a solution."""
        settings = OpenEvolveSettings(simulated_latency=0.0)
        adapter = OpenEvolveAdapter(config=settings)

        job_id = await adapter.submit_evolution(
            prototype='print("hello")', fitness_function=lambda _: 0.9, config={"population_size": 5}
        )

        status = await adapter.get_status(job_id)
        assert status.complete is True
        assert status.state == OpenEvolveJobState.COMPLETE

        solution = await adapter.get_best_solution(job_id)
        assert solution.fitness == pytest.approx(0.9)
        assert solution.solution_code == 'print("hello")'
