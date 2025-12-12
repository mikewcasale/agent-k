"""Platform adapter implementations."""
from __future__ import annotations

from ._base import PlatformAdapter
from .kaggle import KaggleAdapter, KaggleConfig
from .openevolve import OpenEvolveAdapter

__all__ = [
    'PlatformAdapter',
    'KaggleAdapter',
    'KaggleConfig',
    'OpenEvolveAdapter',
]
