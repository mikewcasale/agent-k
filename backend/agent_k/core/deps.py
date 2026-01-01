"""Shared dependency containers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from agent_k.adapters.kaggle import KaggleAdapter
    from agent_k.ui.ag_ui import EventEmitter

__all__ = ("BaseDeps", "KaggleDeps")


@dataclass(kw_only=True)
class BaseDeps:
    """Base dependencies shared across agents."""

    event_emitter: EventEmitter
    http_client: httpx.AsyncClient | None = None
    correlation_id: str | None = None


@dataclass(kw_only=True)
class KaggleDeps(BaseDeps):
    """Dependencies for Kaggle toolsets."""

    kaggle_adapter: KaggleAdapter
    max_results: int = 50

    @property
    def platform_adapter(self) -> KaggleAdapter:
        """Expose the Kaggle adapter as a platform adapter."""
        return self.kaggle_adapter
