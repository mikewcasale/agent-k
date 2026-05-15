"""Evolution package exports for AGENT-K.

@notice: |
    Evolution package exports for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evolution
    provides:
        - agent_k.evolution
    pattern: evolution-package

@agent-guidance:
    do:
        - "Use agent_k.evolution as the canonical home for this capability."
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

from .features import (
    BinningGene,
    DomainFeatureGene,
    FeatureEvolver,
    FeatureGenome,
    FeatureSelectionIndividual,
    FeatureSelector,
    InteractionGene,
    RatioGene,
    TransformGene,
)
from .framework import (
    EvolutionaryFramework,
    HyperparamEvolver,
    HyperparamGenome,
    HyperparamSpace,
    Individual,
    MapElitesArchive,
    Population,
)
from .loss import (
    LightGBMObjective,
    LossFunctionEvolver,
    LossGenome,
    build_lightgbm_objective,
    build_lightgbm_objective_params,
)

__all__ = (
    "BinningGene",
    "DomainFeatureGene",
    "EvolutionaryFramework",
    "FeatureEvolver",
    "FeatureGenome",
    "FeatureSelector",
    "FeatureSelectionIndividual",
    "HyperparamEvolver",
    "HyperparamGenome",
    "HyperparamSpace",
    "Individual",
    "InteractionGene",
    "LightGBMObjective",
    "LossFunctionEvolver",
    "LossGenome",
    "MapElitesArchive",
    "Population",
    "RatioGene",
    "TransformGene",
    "build_lightgbm_objective",
    "build_lightgbm_objective_params",
)
