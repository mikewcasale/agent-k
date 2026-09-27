"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import math
from typing import Any, cast

import httpx
import pytest
from pydantic_graph import End, GraphRunContext

from agent_k.core.exceptions import PhaseTimeoutError
from agent_k.core.models import EvaluationMetric, MissionCriteria
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _await_within_phase,
    _evaluate_metric,
    _phase_deadline,
)
from agent_k.mission.state import GraphContext, MissionState
from agent_k.ui.agui import EventEmitter

__all__ = ()

pytestmark = pytest.mark.anyio


class TestDiscoveryNode:
    """Tests for the DiscoveryNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = DiscoveryNode()
        assert node is not None


class TestResearchNode:
    """Tests for the ResearchNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = ResearchNode()
        assert node is not None


class TestPrototypeNode:
    """Tests for the PrototypeNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = PrototypeNode()
        assert node is not None


class TestEvolutionNode:
    """Tests for the EvolutionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = EvolutionNode()
        assert node is not None


class TestSubmissionNode:
    """Tests for the SubmissionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = SubmissionNode()
        assert node is not None


class TestEvaluateMetric:
    """Tests for metric evaluation helpers."""

    def test_rmsle_ignores_negative_values(self) -> None:
        """RMSLE should ignore negative targets in the denominator."""
        score = _evaluate_metric(EvaluationMetric.RMSLE, [1.0, -1.0], prediction=0.0)
        assert score == pytest.approx(math.log1p(1.0))


class TestPhaseDeadline:
    """Tests for the per-phase timeout guard."""

    async def test_deadline_is_ahead_of_the_loop_clock(self) -> None:
        """The deadline should sit exactly one budget ahead of the loop clock."""
        loop = asyncio.get_running_loop()
        before = loop.time()
        deadline = _phase_deadline(30)
        assert deadline >= before + 30
        assert deadline <= loop.time() + 30

    async def test_result_is_returned_within_budget(self) -> None:
        """An operation that finishes inside the budget returns normally."""

        async def quick() -> str:
            return "done"

        result = await _await_within_phase(
            quick(), phase="discovery", operation="lobbyist_agent", deadline=_phase_deadline(30), timeout_seconds=30
        )
        assert result == "done"

    async def test_hanging_operation_raises_phase_timeout(self) -> None:
        """A hanging operation is cancelled once the deadline passes."""
        started = asyncio.Event()

        async def hangs() -> str:
            started.set()
            await asyncio.sleep(3600)
            return "never"

        loop = asyncio.get_running_loop()
        with pytest.raises(PhaseTimeoutError) as exc_info:
            await _await_within_phase(
                hangs(), phase="evolution", operation="evolver_agent", deadline=loop.time() + 0.05, timeout_seconds=7200
            )

        assert started.is_set()
        assert exc_info.value.phase == "evolution"
        assert exc_info.value.timeout_seconds == 7200
        assert exc_info.value.elapsed_seconds > 0

    async def test_exhausted_budget_skips_the_operation(self) -> None:
        """A spent budget fails fast without scheduling the operation."""
        entered = False

        async def never_runs() -> str:
            nonlocal entered
            entered = True
            return "unreachable"

        loop = asyncio.get_running_loop()
        with pytest.raises(PhaseTimeoutError):
            await _await_within_phase(
                never_runs(),
                phase="research",
                operation="scientist_agent",
                deadline=loop.time() - 1.0,
                timeout_seconds=600,
            )

        assert entered is False

    async def test_inner_timeout_error_is_not_swallowed(self) -> None:
        """A TimeoutError raised inside the operation keeps its own identity."""

        async def transport_timeout() -> str:
            raise TimeoutError("read timeout")

        with pytest.raises(TimeoutError) as exc_info:
            await _await_within_phase(
                transport_timeout(),
                phase="research",
                operation="scientist_agent",
                deadline=_phase_deadline(600),
                timeout_seconds=600,
            )

        assert not isinstance(exc_info.value, PhaseTimeoutError)


class _HangingAgent:
    """Agent stand-in whose run never completes."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, prompt: str, *, deps: Any) -> Any:
        """Block forever so the phase guard has to cancel the run."""
        self.started.set()
        await asyncio.sleep(3600)
        raise AssertionError("hanging agent should never complete")


class _InertAdapter:
    """Platform adapter stand-in that the discovery timeout path never calls."""

    async def get_competition(self, competition_id: str) -> Any:
        """Fail loudly if discovery reaches the platform in this scenario."""
        raise AssertionError("get_competition should not be called")


class TestDiscoveryTimeout:
    """Tests that an unresponsive model ends the discovery phase."""

    async def test_hanging_model_fails_discovery_instead_of_stalling(self) -> None:
        """Discovery ends with a recorded timeout rather than blocking forever."""
        agent = _HangingAgent()
        emitter = EventEmitter()
        async with httpx.AsyncClient() as http_client:
            context = GraphContext(
                event_emitter=emitter,
                http_client=http_client,
                platform_adapter=cast("Any", _InertAdapter()),
                agents={"lobbyist": cast("Any", agent)},
            )
            state = MissionState(mission_id="timeout-test", criteria=MissionCriteria())
            # The outer budget only exists so a regression fails the test instead of hanging CI.
            async with asyncio.timeout(30):
                outcome = await DiscoveryNode(timeout=1).run(GraphRunContext(state=state, deps=context))

        assert isinstance(outcome, End)
        assert outcome.data.success is False
        assert "timed out" in (outcome.data.error_message or "")
        assert agent.started.is_set()
        assert [error["error_type"] for error in state.errors] == ["PhaseTimeoutError"]
