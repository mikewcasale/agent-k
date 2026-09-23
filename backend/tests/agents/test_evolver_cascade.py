"""Tests for the Evolver cascade screening gate.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric

__all__ = ()

try:
    from agent_k.agents.evolver import EvolverAgent, EvolverDeps, EvolverSettings
except TypeError as exc:  # pragma: no cover - optional dependency mismatch
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise

pytestmark = pytest.mark.anyio


@dataclass
class _Ctx:
    """Minimal stand-in for the pydantic-ai RunContext used by cascade helpers."""

    deps: EvolverDeps


def _competition() -> Competition:
    return Competition(
        id="comp",
        title="Comp",
        competition_type=CompetitionType.FEATURED,
        metric=EvaluationMetric.RMSE,
        metric_direction="minimize",
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _deps(tmp_path: Path, **overrides: Any) -> EvolverDeps:
    return EvolverDeps(
        competition=_competition(),
        event_emitter=cast("Any", None),
        platform_adapter=cast("Any", None),
        data_dir=tmp_path,
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        sample_path=tmp_path / "sample_submission.csv",
        target_columns=["y"],
        train_target_columns=["y"],
        id_column="id",
        **overrides,
    )


def _agent() -> EvolverAgent:
    settings = EvolverSettings(
        cascade_evaluation=True, cascade_stage1_rows=300, cascade_relative_threshold=0.85, cascade_floor_threshold=0.05
    )
    return EvolverAgent(settings=settings, register=False)


class TestStage1Threshold:
    """Tests for the stage-1 promotion threshold."""

    def test_falls_back_to_floor_without_stage1_reference(self, tmp_path: Path) -> None:
        """A full-fidelity best fitness must not raise the stage-1 bar."""
        agent = _agent()
        deps = _deps(tmp_path, best_fitness=0.9)

        assert agent._stage1_threshold(deps) == pytest.approx(0.05)

    def test_anchors_on_best_stage1_fitness(self, tmp_path: Path) -> None:
        """The relative gate is measured against the best stage-1 fitness."""
        agent = _agent()
        deps = _deps(tmp_path, best_fitness=0.9, best_stage1_fitness=0.4)

        assert agent._stage1_threshold(deps) == pytest.approx(0.34)

    def test_floor_wins_for_weak_stage1_reference(self, tmp_path: Path) -> None:
        """The absolute floor still applies when the reference is tiny."""
        agent = _agent()
        deps = _deps(tmp_path, best_stage1_fitness=0.02)

        assert agent._stage1_threshold(deps) == pytest.approx(0.05)

    def test_update_reference_keeps_running_maximum(self, tmp_path: Path) -> None:
        """Only improvements move the stage-1 reference."""
        agent = _agent()
        deps = _deps(tmp_path)

        agent._update_stage1_reference(deps, 0.3)
        agent._update_stage1_reference(deps, 0.1)

        assert deps.best_stage1_fitness == pytest.approx(0.3)


class TestCascadePromotion:
    """Tests for cascade promotion in _run_evaluation."""

    @staticmethod
    def _patch_stages(
        monkeypatch: pytest.MonkeyPatch, agent: EvolverAgent, stage1_fitness: float, full_fitness: float
    ) -> list[str]:
        calls: list[str] = []

        async def fake_evaluate(
            _ctx: Any, _code: str, *, validation_split: float, stage: str | None = None, **_kwargs: Any
        ) -> dict[str, Any]:
            calls.append(stage or "full")
            fitness = stage1_fitness if stage == "stage1" else full_fitness
            return {
                "fitness": fitness,
                "cv_score": 1.0 / fitness - 1.0,
                "valid": True,
                "runtime_ms": 10,
                "timed_out": False,
                "returncode": 0,
                "error": None,
            }

        monkeypatch.setattr(agent, "_evaluate_solution", fake_evaluate)
        return calls

    async def test_promotes_when_only_full_fidelity_best_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A low-fidelity score must not be compared against a full-fidelity best."""
        agent = _agent()
        calls = self._patch_stages(monkeypatch, agent, stage1_fitness=0.4, full_fitness=0.92)
        deps = _deps(tmp_path, best_fitness=0.9)
        ctx = cast("Any", _Ctx(deps=deps))

        result = await agent._run_evaluation(ctx, "code = 1", validation_split=0.2)

        assert calls == ["stage1", "full"]
        assert result["stage"] == "full"
        assert result["fitness"] == pytest.approx(0.92)
        assert result["stage1_threshold"] == pytest.approx(0.05)
        assert deps.best_stage1_fitness == pytest.approx(0.4)

    async def test_gates_candidate_below_stage1_reference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A screening score well below the stage-1 reference stops before full evaluation."""
        agent = _agent()
        calls = self._patch_stages(monkeypatch, agent, stage1_fitness=0.1, full_fitness=0.92)
        deps = _deps(tmp_path, best_stage1_fitness=0.4)
        ctx = cast("Any", _Ctx(deps=deps))

        result = await agent._run_evaluation(ctx, "code = 1", validation_split=0.2)

        assert calls == ["stage1"]
        assert result["stage"] == "stage1"
        assert result["stage_threshold"] == pytest.approx(0.34)
        assert result["stage1_reference_fitness"] == pytest.approx(0.4)
        assert deps.best_stage1_fitness == pytest.approx(0.4)

    async def test_first_candidate_is_not_gated_by_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reference is updated after the gate so a candidate never screens itself out."""
        agent = _agent()
        calls = self._patch_stages(monkeypatch, agent, stage1_fitness=0.4, full_fitness=0.5)
        deps = _deps(tmp_path)
        ctx = cast("Any", _Ctx(deps=deps))

        result = await agent._run_evaluation(ctx, "code = 1", validation_split=0.2)

        assert calls == ["stage1", "full"]
        assert result["stage1_reference_fitness"] is None
        assert deps.best_stage1_fitness == pytest.approx(0.4)
