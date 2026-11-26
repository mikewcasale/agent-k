"""Logfire instrumentation configured at application startup.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

__all__ = ("configure_observability",)


def configure_observability(*, service_name: str, environment: str) -> None:
    """Configure telemetry once in each API, worker, or mission process.

    Args:
        service_name: Deployment role included in telemetry resource attributes.
        environment: Runtime environment included in telemetry resource attributes.

    Agent instrumentation records timing and usage without prompts, completions, tool
    content, or binary payloads. A configured Logfire token enables remote export.
    """
    logfire.configure(
        service_name=service_name,
        environment=environment,
        send_to_logfire="if-token-present",
        console=False,
    )
    Agent.instrument_all(
        InstrumentationSettings(include_content=False, include_binary_content=False)
    )
