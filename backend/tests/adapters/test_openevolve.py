"""Tests for the OpenEvolve adapter.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from pathlib import Path

import pytest

from agent_k.adapters.openevolve import (
    OpenEvolveAdapter,
    OpenEvolveJobState,
    OpenEvolveSettings,
    _build_model_config,
    resolve_model_specs,
)
from agent_k.core.constants import DEFAULT_MODEL
from agent_k.infra.providers import OPENROUTER_FREE_MODELS, is_openai_compatible_spec

__all__ = ()

pytestmark = pytest.mark.anyio


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


class TestResolveModelSpecs:
    """Tests for resolve_model_specs function."""

    def test_keeps_openai_compatible_specs_in_order(self) -> None:
        """Compatible specs should pass through unchanged."""
        specs = ["openrouter:openai/gpt-oss-120b:free", "devstral:local"]

        assert resolve_model_specs(specs) == specs

    def test_trims_whitespace_and_skips_blanks(self) -> None:
        """Blank and padded entries should be normalised away."""
        resolved = resolve_model_specs(["  openai:gpt-4o  ", "", "   "])

        assert resolved == ["openai:gpt-4o"]

    def test_drops_native_sdk_specs_when_others_remain(self) -> None:
        """Specs OpenEvolve cannot call should be dropped, not substituted."""
        resolved = resolve_model_specs(["anthropic:claude-sonnet-4-5", "openai:gpt-4o"])

        assert resolved == ["openai:gpt-4o"]

    def test_falls_back_to_a_callable_model_when_all_specs_dropped(self) -> None:
        """An all-native-SDK request should still yield a callable model."""
        resolved = resolve_model_specs(["anthropic:claude-sonnet-4-5"])

        assert len(resolved) == 1
        assert is_openai_compatible_spec(resolved[0])

    def test_fallback_prefers_default_model_over_free_registry(self) -> None:
        """Fallback precedence is DEFAULT_MODEL, then the free-model registry."""
        expected = (
            DEFAULT_MODEL
            if is_openai_compatible_spec(DEFAULT_MODEL)
            else next(spec for spec in OPENROUTER_FREE_MODELS.values() if is_openai_compatible_spec(spec))
        )

        assert resolve_model_specs([]) == [expected]

    def test_fallback_is_callable_by_the_model_config_builder(self) -> None:
        """The fallback must produce a usable OpenEvolve model config."""
        config = _build_model_config(resolve_model_specs([])[0])

        assert config.name
        assert not config.name.startswith(("openrouter:", "openai:", "devstral:"))
        assert config.api_base
