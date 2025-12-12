"""Centralized logging utilities."""
from __future__ import annotations

import logfire

__all__ = ['get_logger']


def get_logger(component: str) -> logfire.Logfire:
    """Return a component-specific logger."""
    return logfire.with_settings(tags={'component': component})
