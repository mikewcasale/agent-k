"""State persistence utilities for the mission graph."""
from __future__ import annotations

from ..core.models import MissionState

__all__ = ['save_state', 'load_state']


def save_state(state: MissionState) -> dict[str, str]:
    """Serialize mission state."""
    return state.model_dump()


def load_state(data: dict[str, str]) -> MissionState:
    """Deserialize mission state."""
    return MissionState.model_validate(data)
