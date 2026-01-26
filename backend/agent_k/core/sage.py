"""SAGE docstring metadata helpers for AGENT-K.

@notice: |
    Typed metadata helpers for SAGE parameter annotations.

@dev: |
    These helpers are used with typing.Annotated to co-locate parameter docs
    and machine-readable constraints. Keep them lightweight and immutable.

@graph:
    id: agent_k.core.sage
    provides:
        - agent_k.core.sage:Doc
        - agent_k.core.sage:Constraint
        - agent_k.core.sage:Range
        - agent_k.core.sage:Default
    pattern: doc-metadata

@agent-guidance:
    do:
        - "Use Doc/Constraint/Range/Default in Annotated parameters."
    do_not:
        - "Add custom metadata helpers outside this module."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

__all__ = ("Constraint", "Default", "Doc", "Range")


@dataclass(frozen=True)
class Doc:
    """Human-facing parameter description.

    @notice: |
        Human-facing parameter description.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: doc-metadata
            rationale: "Co-locates parameter docs with type hints."
            violations: "Out-of-band docs drift from signatures."
    """

    text: str


@dataclass(frozen=True)
class Constraint:
    """Machine-readable parameter constraint.

    @notice: |
        Machine-readable parameter constraint.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: doc-metadata
            rationale: "Captures validation rules without custom parsers."
            violations: "Opaque constraints make automated checks brittle."
    """

    pattern: str


@dataclass(frozen=True)
class Range:
    """Numeric range constraint for parameters.

    @notice: |
        Numeric range constraint for parameters.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: doc-metadata
            rationale: "Encodes min/max bounds for tooling and agents."
            violations: "Hidden bounds lead to invalid parameter choices."
    """

    min: float | int
    max: float | int


@dataclass(frozen=True)
class Default:
    """Default-value metadata for parameters.

    @notice: |
        Default-value metadata for parameters.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: doc-metadata
            rationale: "Makes implicit defaults discoverable to agents."
            violations: "Silent defaults can mislead tool usage."
    """

    value: str
