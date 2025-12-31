"""Infrastructure concerns for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Local imports (core first, then alphabetical)
from .config import Settings, load_settings
from .instrumentation import Metrics, configure_instrumentation, get_logger, traced
from .providers import (
    DEVSTRAL_BASE_URL,
    DEVSTRAL_MODEL_ID,
    create_devstral_model,
    get_model,
    is_devstral_model,
)

# =============================================================================
# Section 2: Module Exports
# =============================================================================
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
