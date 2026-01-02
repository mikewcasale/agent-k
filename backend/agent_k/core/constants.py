"""Module-level constants for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
from typing import Final

__all__ = (
    'DISCOVERY_TIMEOUT_SECONDS',
    'RESEARCH_TIMEOUT_SECONDS',
    'PROTOTYPE_TIMEOUT_SECONDS',
    'EVOLUTION_TIMEOUT_SECONDS',
    'SUBMISSION_TIMEOUT_SECONDS',
    'SOLUTION_EXECUTION_TIMEOUT_SECONDS',
    'MAX_COMPETITIONS_PER_SEARCH',
    'MAX_CONCURRENT_SUBMISSIONS',
    'MAX_EVOLUTION_GENERATIONS',
    'MAX_MISSION_EVOLUTION_ROUNDS',
    'EVOLUTION_POPULATION_SIZE',
    'CONVERGENCE_THRESHOLD_GENERATIONS',
    'DEFAULT_MODEL',
    'DEVSTRAL_MODEL',
    'DEFAULT_KAGGLE_MCP_URL',
    'MEMORY_SESSION_TTL_SECONDS',
    'CHECKPOINT_INTERVAL_GENERATIONS',
    'MAX_RETRIES',
    'RETRY_DELAY_SECONDS',
    'VALID_COMPETITION_TYPES',
    'MUTATION_TYPES',
    'MISSION_PHASES',
    'KAGGLE_API_BASE_URL',
)

# Timeouts (seconds)
DISCOVERY_TIMEOUT_SECONDS: Final[int] = 300
RESEARCH_TIMEOUT_SECONDS: Final[int] = 600
PROTOTYPE_TIMEOUT_SECONDS: Final[int] = 900
EVOLUTION_TIMEOUT_SECONDS: Final[int] = 7200  # 2 hours
SUBMISSION_TIMEOUT_SECONDS: Final[int] = 120
SOLUTION_EXECUTION_TIMEOUT_SECONDS: Final[int] = 600

# Limits
MAX_COMPETITIONS_PER_SEARCH: Final[int] = 50
MAX_CONCURRENT_SUBMISSIONS: Final[int] = 5
MAX_EVOLUTION_GENERATIONS: Final[int] = 100
MAX_MISSION_EVOLUTION_ROUNDS: Final[int] = 200
EVOLUTION_POPULATION_SIZE: Final[int] = 50
CONVERGENCE_THRESHOLD_GENERATIONS: Final[int] = 5

# Models and platforms
DEFAULT_MODEL: Final[str] = 'anthropic:claude-sonnet-4-5'
# Devstral model for local LM Studio server
DEVSTRAL_MODEL: Final[str] = 'devstral:local'
DEFAULT_KAGGLE_MCP_URL: Final[str] = 'https://mcp.kaggle.com'
KAGGLE_API_BASE_URL: Final[str] = 'https://www.kaggle.com/api/v1'

# Memory
MEMORY_SESSION_TTL_SECONDS: Final[int] = 86400  # 24 hours
CHECKPOINT_INTERVAL_GENERATIONS: Final[int] = 10

# Retry
MAX_RETRIES: Final[int] = 3
RETRY_DELAY_SECONDS: Final[float] = 1.0

# Collections (immutable)
VALID_COMPETITION_TYPES: Final[frozenset[str]] = frozenset(
    {'featured', 'research', 'getting_started', 'playground', 'community'}
)

MUTATION_TYPES: Final[frozenset[str]] = frozenset({'point', 'structural', 'hyperparameter', 'crossover'})

MISSION_PHASES: Final[tuple[str, ...]] = ('discovery', 'research', 'prototype', 'evolution', 'submission')
