"""Configuration management for AGENT-K."""
from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ['Settings', 'load_settings']


@dataclass
class Settings:
    """Runtime configuration settings."""
    
    environment: str = os.getenv('ENVIRONMENT', 'development')


def load_settings() -> Settings:
    """Load settings from environment."""
    return Settings()
