"""Verify the authenticated API surface without substituting service dependencies.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os

import pytest
from fastapi.testclient import TestClient

from agent_k.api import create_app

__all__ = ()

REQUIRED_ROUTES = {
    "/health",
    "/ready",
    "/agentic_generative_ui/",
    "/api/competitions/fetch",
    "/api/competitions/search",
    "/api/mission/best-results",
    "/api/mission/start",
    "/api/mission/{mission_id}/abort",
    "/api/mission/{mission_id}/pause",
    "/api/mission/{mission_id}/resume",
    "/api/mission/{mission_id}/status",
    "/api/mission/{mission_id}/stream",
    "/api/worker/v1/jobs/claim",
    "/api/worker/v1/jobs/{job_id}/artifacts/upload-url",
    "/api/worker/v1/jobs/{job_id}/artifacts/{artifact_id}/complete",
    "/api/worker/v1/jobs/{job_id}/artifacts/{artifact_id}/download-url",
    "/api/worker/v1/jobs/{job_id}/complete",
    "/api/worker/v1/jobs/{job_id}/control",
    "/api/worker/v1/jobs/{job_id}/events",
    "/api/worker/v1/jobs/{job_id}/fail",
    "/api/worker/v1/jobs/{job_id}/heartbeat",
    "/api/worker/v1/jobs/{job_id}/snapshots",
    "/api/worker/v1/jobs/{job_id}/snapshots/{snapshot_id}",
    "/api/worker/v1/migrations",
    "/api/worker/v1/migrations/{migration_id}",
    "/api/worker/v1/migrations/{migration_id}/complete",
}


def test_openapi_exposes_the_versioned_authenticated_contract() -> None:
    """Keep browser and worker route names stable across the flat migration."""
    schema = create_app().openapi()
    paths = schema["paths"]
    assert REQUIRED_ROUTES <= paths.keys()
    assert paths["/api/mission/start"]["post"]["responses"].get("202") is not None
    assert paths["/api/worker/v1/migrations"]["post"]["responses"].get("201") is not None
    assert (
        paths["/api/worker/v1/migrations/{migration_id}/complete"]["post"]["responses"].get("202")
        is not None
    )


def test_openapi_uses_camel_case_wire_fields() -> None:
    """Protect the existing frontend and worker JSON field contract."""
    schemas = create_app().openapi()["components"]["schemas"]
    assert "missionId" in schemas["MissionStartResponse"]["properties"]
    assert "leaseId" in schemas["HeartbeatRequest"]["properties"]
    assert "finalEventSequence" in schemas["CompleteJobRequest"]["properties"]
    assert "evolutionIterations" in schemas["MissionResultPayload"]["properties"]
    criteria_fields = schemas["MissionCriteria"]["properties"]
    assert {
        "allowSubmission",
        "enableKumo",
        "evolutionModels",
        "maxEvolutionRounds",
        "minEvolutionIterations",
        "targetCompetitionTypes",
    } <= criteria_fields.keys()


@pytest.mark.skipif(
    not all(
        os.getenv(name)
        for name in (
            "AGENT_K_DATABASE_URL",
            "AGENT_K_GATEWAY_SECRET",
            "AGENT_K_WORKER_TOKEN",
        )
    ),
    reason="requires the configured real PostgreSQL API runtime",
)
def test_configured_runtime_starts_against_real_postgresql() -> None:
    """Exercise the actual lifespan and database migrations when configured."""
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
