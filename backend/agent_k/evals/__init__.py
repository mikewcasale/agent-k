"""Evaluation utilities for Agent-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
See LICENSE file for details.
"""

from __future__ import annotations as _annotations

# =============================================================================
# Section 1: Imports
# =============================================================================
# Local imports (core first, then alphabetical)
from .datasets import discovery_dataset, evolution_dataset, load_dataset
from .evaluators import CompetitionSelected, FitnessImprovement, ValidPython

# =============================================================================
# Section 2: Module Exports
# =============================================================================
__all__ = (
    "CompetitionSelected",
    "FitnessImprovement",
    "ValidPython",
    "discovery_dataset",
    "evolution_dataset",
    "load_dataset",
)
