"""Tests for evolver instruction stability and live-state delivery.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_k.core.hints import HintCategory, PreprocessingHint
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric

if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from agent_k.agents.evolver import EvolverDeps

try:
    from agent_k.agents.evolver import EvolutionArchiveEntry, EvolverDeps, EvolverSettings, evolver_agent_instance
except TypeError as exc:  # pragma: no cover - environment specific
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise

__all__ = ()

_evolver = evolver_agent_instance


def _competition() -> Competition:
    return Competition(
        id="sample-competition",
        title="Sample Competition",
        competition_type=CompetitionType.FEATURED,
        metric=EvaluationMetric.RMSE,
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _hint() -> PreprocessingHint:
    return PreprocessingHint(
        id="scale_numeric",
        category=HintCategory.SCALING,
        priority=0.6,
        applicable_columns=["feature_a"],
        description="Scale numeric features before training.",
        code_snippet="from sklearn.preprocessing import StandardScaler",
        success_rate=0.5,
        last_attempted=None,
        last_result=None,
    )


def _deps(tmp_path: Path, *, hints: list[PreprocessingHint] | None = None) -> EvolverDeps:
    emitter = MagicMock()
    emitter.emit_generation_complete = AsyncMock()
    return EvolverDeps(
        competition=_competition(),
        event_emitter=emitter,
        platform_adapter=AsyncMock(),
        data_dir=tmp_path,
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        sample_path=tmp_path / "sample_submission.csv",
        target_columns=["target"],
        train_target_columns=["target"],
        id_column="id",
        max_generations=4,
        preprocessing_hints=list(hints or []),
    )


def _archive_entry(fitness: float) -> EvolutionArchiveEntry:
    return EvolutionArchiveEntry(
        code="print('x')",
        fitness=fitness,
        cv_score=fitness,
        complexity=10,
        complexity_bin=1,
        model_family="lightgbm",
        signature=f"sig-{fitness}",
    )


def _context(deps: EvolverDeps) -> RunContext[EvolverDeps]:
    return cast("RunContext[EvolverDeps]", MagicMock(deps=deps))


class TestInstructionStability:
    """Instructions must stay byte-identical while evolution state mutates."""

    async def test_instructions_are_stable_across_state_changes(self, tmp_path: Path) -> None:
        """A mutating run must not change the cached prompt prefix."""
        deps = _deps(tmp_path, hints=[_hint()])
        ctx = _context(deps)

        first = await _evolver._add_evolution_context(ctx)

        deps.generation_history.append({"generation": 0, "best_fitness": 0.5, "mean_fitness": 0.4})
        deps.improvement_count = 3
        deps.best_solution = "print('solution')"
        deps.best_fitness = 0.9
        deps.elite_archive[(1, "lightgbm")] = _archive_entry(0.9)
        deps.failure_counts["timeout"] = 2
        deps.last_error_feedback = "Execution timed out."

        second = await _evolver._add_evolution_context(ctx)

        assert second == first

    async def test_instructions_carry_run_invariant_context(self, tmp_path: Path) -> None:
        """Static sections must describe the competition and the budget."""
        deps = _deps(tmp_path)
        snapshot = await _evolver._add_evolution_context(_context(deps))

        assert "COMPETITION CONTEXT:" in snapshot
        assert "sample-competition" in snapshot
        assert "EVOLUTION BUDGET:" in snapshot
        assert "Max Generations: 4" in snapshot

    async def test_instructions_exclude_mutating_state(self, tmp_path: Path) -> None:
        """Counters and failures must not be injected into the system block."""
        deps = _deps(tmp_path)
        deps.generation_history.append({"generation": 0, "best_fitness": 0.5, "mean_fitness": 0.4})
        deps.improvement_count = 2
        deps.elite_archive[(1, "lightgbm")] = _archive_entry(0.7)
        deps.failure_counts["timeout"] = 1
        deps.last_error_feedback = "Execution timed out."

        snapshot = await _evolver._add_evolution_context(_context(deps))

        assert "Generations Completed" not in snapshot
        assert "Improvements:" not in snapshot
        assert "LAST GENERATION" not in snapshot
        assert "EXECUTION FAILURES" not in snapshot
        assert "LAST EXECUTION ERROR" not in snapshot

    async def test_snapshot_is_cached_on_deps(self, tmp_path: Path) -> None:
        """The snapshot is computed once per run and reused."""
        deps = _deps(tmp_path)
        assert deps.instruction_snapshot is None

        snapshot = await _evolver._add_evolution_context(_context(deps))

        assert deps.instruction_snapshot == snapshot


class TestLiveStateDelivery:
    """Mutating state must reach the model through tool results."""

    async def test_record_generation_returns_live_state(self, tmp_path: Path) -> None:
        """record_generation reports counters, archive, and remaining budget."""
        deps = _deps(tmp_path, hints=[_hint()])
        deps.improvement_count = 2
        deps.best_fitness = 0.75
        deps.best_solution = "print('solution')"
        deps.elite_archive[(1, "lightgbm")] = _archive_entry(0.75)
        deps.failure_counts["timeout"] = 1
        deps.last_error_feedback = "Execution timed out."

        tool_return = await _evolver.record_generation(
            _context(deps), generation=0, best_fitness=0.75, mean_fitness=0.5, worst_fitness=0.1, mutations={"point": 1}
        )

        state = cast("dict[str, Any]", tool_return.return_value)
        assert state["generation"] == 0
        assert state["generations_completed"] == 1
        assert state["generations_remaining"] == 3
        assert state["improvements"] == 2
        assert state["best_fitness"] == pytest.approx(0.75)
        assert state["elite_cells"] == 1
        assert state["elite_families"] == ["lightgbm"]
        assert state["last_generation"]["best_fitness"] == pytest.approx(0.75)
        assert state["failure_summary"] == "timeout=1"
        assert state["last_error_feedback"] == "Execution timed out."
        assert "PREPROCESSING GUIDANCE" in state["preprocessing_guidance"]
        assert "improvements=2" in cast("str", tool_return.content)

    async def test_record_generation_applies_generation_offset(self, tmp_path: Path) -> None:
        """Segmented runs report globally numbered generations."""
        deps = _deps(tmp_path)
        deps.generation_offset = 5
        deps.generation_history.extend({"generation": index, "best_fitness": 0.1} for index in range(5))

        tool_return = await _evolver.record_generation(
            _context(deps), generation=0, best_fitness=0.2, mean_fitness=0.1, worst_fitness=0.0, mutations={}
        )

        state = cast("dict[str, Any]", tool_return.return_value)
        assert state["generation"] == 5
        assert state["generations_completed"] == 6
        assert state["generations_remaining"] == 3


class TestSettingsPromptCaching:
    """Agent settings must enable caching for providers that need it."""

    def test_anthropic_model_settings_enable_caching(self) -> None:
        """Anthropic evolution runs cache instructions and tool definitions."""
        settings = EvolverSettings(model="anthropic:claude-sonnet-4-20250514").model_settings

        assert settings.get("anthropic_cache_instructions") is True
        assert settings.get("anthropic_cache_tool_definitions") is True

    def test_openrouter_model_settings_stay_plain(self) -> None:
        """OpenRouter runs rely on automatic prefix caching."""
        settings = EvolverSettings(model="openrouter:openai/gpt-oss-120b:free").model_settings

        assert "anthropic_cache_instructions" not in settings
