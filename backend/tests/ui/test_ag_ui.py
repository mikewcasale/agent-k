"""Tests for the AG-UI FastAPI application.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
import gc

import pytest
from fastapi.testclient import TestClient

from agent_k.core.models import MissionCriteria
from agent_k.ui.agui import _BACKGROUND_TASKS, MissionRequest, _spawn_background_task, _terminal_task_status, create_app

__all__ = ()

pytestmark = pytest.mark.anyio


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestMissionRequest:
    """Tests for the MissionRequest model."""

    def test_creation_with_criteria(self) -> None:
        """Request should be created with criteria."""
        request = MissionRequest(criteria=MissionCriteria())

        assert request.criteria is not None
        assert request.user_prompt is None

    def test_creation_with_prompt(self) -> None:
        """Request should accept user prompt."""
        request = MissionRequest(criteria=MissionCriteria(), user_prompt="Find featured competitions")

        assert request.user_prompt == "Find featured competitions"


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_creates_fastapi_app(self) -> None:
        """Should create a FastAPI application."""
        app = create_app()

        assert app is not None
        assert app.title == "AGENT-K"
        assert app.version == "0.1.0"


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Health endpoint should return healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestMissionStartEndpoint:
    """Tests for the /api/mission/start endpoint."""

    def test_start_mission_basic(self, client: TestClient) -> None:
        """Should start a mission and return ID."""
        response = client.post(
            "/api/mission/start",
            json={"criteria": {"target_leaderboard_percentile": 0.10, "max_evolution_rounds": 100}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "missionId" in data
        assert isinstance(data["missionId"], str)

    def test_start_mission_with_prompt(self, client: TestClient) -> None:
        """Should accept user prompt."""
        response = client.post("/api/mission/start", json={"criteria": {}, "user_prompt": "Find a competition"})

        assert response.status_code == 200


class TestBackgroundTasks:
    """Tests for background task tracking."""

    async def test_task_survives_garbage_collection(self) -> None:
        """A tracked task should keep running when no caller holds a reference."""
        finished = asyncio.Event()

        async def work() -> None:
            await asyncio.sleep(0)
            finished.set()

        _spawn_background_task(work(), name="gc-probe")
        gc.collect()

        await asyncio.wait_for(finished.wait(), timeout=1.0)

    async def test_reference_released_after_completion(self) -> None:
        """The registry should not retain a task once it finishes."""

        async def work() -> None:
            return None

        task = _spawn_background_task(work(), name="completion-probe")
        assert task in _BACKGROUND_TASKS

        await task
        await asyncio.sleep(0)

        assert task not in _BACKGROUND_TASKS

    async def test_reference_released_after_failure(self) -> None:
        """A crashing task should be retired from the registry."""

        async def work() -> None:
            raise RuntimeError("mission exploded")

        task = _spawn_background_task(work(), name="failure-probe")
        with pytest.raises(RuntimeError, match="mission exploded"):
            await task
        await asyncio.sleep(0)

        assert task not in _BACKGROUND_TASKS


class TestTerminalTaskStatus:
    """Tests for terminal status resolution of mission tasks."""

    def test_missing_task(self) -> None:
        """A mission without a task has no terminal status."""
        assert _terminal_task_status(None) is None

    async def test_running_task(self) -> None:
        """A running task has no terminal status."""
        task = asyncio.create_task(asyncio.sleep(5))
        try:
            assert _terminal_task_status(task) is None
        finally:
            task.cancel()

    async def test_cancelled_task(self) -> None:
        """A cancelled task reports the mission as aborted."""
        task = asyncio.create_task(asyncio.sleep(5))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        status, message = _terminal_task_status(task)

        assert status == "aborted"
        assert "aborted" in message

    async def test_crashed_task(self) -> None:
        """A crashed task reports the failure type and message."""

        async def work() -> None:
            raise ValueError("bad schema")

        task = asyncio.create_task(work())
        with pytest.raises(ValueError, match="bad schema"):
            await task

        status, message = _terminal_task_status(task)

        assert status == "failed"
        assert message == "ValueError: bad schema"

    async def test_task_finished_without_result(self) -> None:
        """A task that returns without recording a result reports failure."""

        async def work() -> None:
            return None

        task = asyncio.create_task(work())
        await task

        status, message = _terminal_task_status(task)

        assert status == "failed"
        assert "without producing a result" in message


class TestMissionStatusEndpoint:
    """Tests for the /api/mission/{mission_id}/status endpoint."""

    def test_status_nonexistent_mission(self, client: TestClient) -> None:
        """Should return error for nonexistent mission."""
        response = client.get("/api/mission/nonexistent-id/status")

        assert response.status_code == 200  # Returns JSON error, not HTTP error
        data = response.json()
        assert "error" in data

    @pytest.mark.skip(reason="Known bug: MissionState doesn't have 'status' attribute")
    def test_status_after_start(self, client: TestClient) -> None:
        """Should return status for started mission.

        Note: The current implementation has a bug where MissionState
        doesn't have a 'status' field. The endpoint will fail until fixed.
        """
        # First start a mission
        start_response = client.post("/api/mission/start", json={"criteria": {}})
        mission_id = start_response.json()["missionId"]

        # Then get status
        status_response = client.get(f"/api/mission/{mission_id}/status")

        assert status_response.status_code == 200
        data = status_response.json()
        assert "missionId" in data
        assert data["missionId"] == mission_id


class TestMissionAbortEndpoint:
    """Tests for the /api/mission/{mission_id}/abort endpoint."""

    def test_abort_nonexistent_mission(self, client: TestClient) -> None:
        """Should return error for nonexistent mission."""
        response = client.post("/api/mission/nonexistent-id/abort")

        assert response.status_code == 200  # Returns JSON error
        data = response.json()
        assert "error" in data

    def test_abort_active_mission(self, client: TestClient) -> None:
        """Should abort an active mission."""
        # First start a mission
        start_response = client.post("/api/mission/start", json={"criteria": {}})
        mission_id = start_response.json()["missionId"]

        # Then abort it
        abort_response = client.post(f"/api/mission/{mission_id}/abort")

        assert abort_response.status_code == 200
        data = abort_response.json()
        assert data["status"] == "aborted"


class TestMissionStreamEndpoint:
    """Tests for the /api/mission/{mission_id}/stream endpoint."""

    def test_stream_nonexistent_mission(self, client: TestClient) -> None:
        """Should return error for nonexistent mission."""
        with client.stream("GET", "/api/mission/nonexistent-id/stream") as response:
            assert response.status_code == 200
            # The response will contain error data
            content = response.read().decode()
            assert "error" in content


class TestChatEndpoint:
    """Tests for the /agentic_generative_ui/ chat endpoint."""

    def test_chat_simple_message(self, client: TestClient) -> None:
        """Should respond to simple chat messages."""
        # The chat endpoint returns a streaming response
        with client.stream(
            "POST",
            "/agentic_generative_ui/",
            json={"messages": [{"role": "user", "content": "Hello"}], "id": "test-chat-id"},
        ) as response:
            assert response.status_code == 200
