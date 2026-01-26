"""Infrastructure concerns for AGENT-K.

@notice: |
    Infrastructure concerns for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.infra
    provides:
        - agent_k.infra
    pattern: infra-package

@agent-guidance:
    do:
        - "Use agent_k.infra as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from .config import Settings, load_settings
from .instrumentation import Metrics, configure_instrumentation, get_logger, traced
from .providers import DEVSTRAL_BASE_URL, DEVSTRAL_MODEL_ID, create_devstral_model, get_model, is_devstral_model

__all__ = (
    "Settings",
    "load_settings",
    "configure_instrumentation",
    "get_logger",
    "traced",
    "Metrics",
    "DEVSTRAL_BASE_URL",
    "DEVSTRAL_MODEL_ID",
    "create_devstral_model",
    "get_model",
    "is_devstral_model",
)
