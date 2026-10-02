"""Model configuration for AGENT-K.

@notice: |
    Model configuration for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.infra.providers
    provides:
        - agent_k.infra.providers:get_model
        - agent_k.infra.providers:create_devstral_model
        - agent_k.infra.providers:create_openrouter_model
        - agent_k.infra.providers:is_devstral_model
        - agent_k.infra.providers:DEVSTRAL_MODEL_ID
        - agent_k.infra.providers:DEVSTRAL_BASE_URL
    pattern: model-factory

@similar:
    - id: agent_k.infra.config
        when: "General configuration; this module builds model instances."

@agent-guidance:
    do:
        - "Use agent_k.infra.providers as the canonical home for this capability."
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

import os
from typing import TYPE_CHECKING, Annotated, Final

import logfire
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.models import Model

__all__ = (
    "get_model",
    "create_devstral_model",
    "create_fallback_model",
    "create_openrouter_model",
    "is_devstral_model",
    "split_model_specs",
    "DEVSTRAL_MODEL_ID",
    "DEVSTRAL_BASE_URL",
    "MODEL_SPEC_SEPARATOR",
    "OPENROUTER_FREE_MODELS",
    "ModelType",
)

DEVSTRAL_MODEL_ID: Final[str] = "mistralai/devstral-small-2-2512"
DEVSTRAL_BASE_URL: Final[str] = os.getenv("DEVSTRAL_BASE_URL", "http://192.168.105.1:1234/v1")
OPENROUTER_FREE_MODELS: Final[dict[str, str]] = {
    "devstral": "openrouter:mistralai/devstral-2512:free",
    "kat_coder": "openrouter:kwaipilot/kat-coder-pro:free",
    "qwen_coder": "openrouter:qwen/qwen3-coder:free",
}
# Free OpenRouter model identifiers for agentic coding tasks.
MODEL_SPEC_SEPARATOR: Final[str] = ","
# Separator used to express an ordered failover chain inside a single model spec.

type ModelType = str


def create_devstral_model(
    model_id: Annotated[str, Doc("Devstral model identifier.")] = DEVSTRAL_MODEL_ID,
    base_url: Annotated[str | None, Doc("Base URL for the Devstral endpoint.")] = None,
) -> OpenAIChatModel:
    """Create a Devstral model instance for local LM Studio server.

        This creates an OpenAI-compatible model that connects to a local
        LM Studio server running Devstral.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Creates an OpenAIChatModel configured for Devstral.

        @factory-for:
            id: pydantic_ai.models.openai:OpenAIChatModel
            rationale: "Centralizes local LM Studio configuration."
            singleton: false
            cache-key: model_id

        @canonical-home:
            for:
                - "devstral model creation"
            notes: "Use create_devstral_model for local Devstral endpoints."
    """
    url = base_url or DEVSTRAL_BASE_URL

    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(
            base_url=url,
            api_key="not-required",  # Local LM Studio doesn't require auth
        ),
    )


def create_openrouter_model(model_id: Annotated[str, Doc("OpenRouter model identifier.")]) -> OpenAIChatModel:
    """Create a model instance using OpenRouter.

        OpenRouter provides access to many models including Devstral, Claude,
        GPT-4, and more through a unified API.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Creates an OpenAIChatModel configured for OpenRouter.

        @factory-for:
            id: pydantic_ai.models.openai:OpenAIChatModel
            rationale: "Centralizes OpenRouter provider configuration."
            singleton: false
            cache-key: model_id

        @canonical-home:
            for:
                - "openrouter model creation"
            notes: "Use create_openrouter_model for OpenRouter endpoints."
    """
    return OpenAIChatModel(model_id, provider=OpenRouterProvider())


def split_model_specs(
    model_spec: Annotated[str, Doc("One or more model specs separated by MODEL_SPEC_SEPARATOR.")],
) -> tuple[str, ...]:
    """Split a model specification into its ordered failover chain.

    @notice: |
        Splits a separator-delimited model spec into individual specs.

    @dev: |
        Blank entries are dropped so padding and trailing separators are tolerated.
        Raises UserError when no spec remains, since an agent cannot run without a model.

    @canonical-home:
        for:
            - "model spec chain parsing"
        notes: "Use split_model_specs wherever a spec string may hold a chain."
    """
    specs = tuple(part.strip() for part in model_spec.split(MODEL_SPEC_SEPARATOR) if part.strip())
    if not specs:
        raise UserError("Model specification is empty; set at least one model spec.")
    return specs


def create_fallback_model(
    model_specs: Annotated[Sequence[str], Doc("Ordered model specs to fail over across.")],
) -> FallbackModel:
    """Create a model that fails over across the given specs in order.

        Specs that cannot be constructed - typically because the provider API key is
        absent - are skipped so a mixed-provider chain still works with partial
        credentials.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Creates a FallbackModel from an ordered list of model specs.

        @factory-for:
            id: pydantic_ai.models.fallback:FallbackModel
            rationale: "Centralizes multi-provider failover for every agent."
            singleton: false
            cache-key: model_specs

        @canonical-home:
            for:
                - "model failover chain creation"
            notes: "Use create_fallback_model instead of wiring FallbackModel directly."
    """
    resolved: list[Model] = []
    skipped: dict[str, str] = {}
    for spec in model_specs:
        try:
            resolved.append(infer_model(_resolve_model_spec(spec)))
        except Exception as exc:  # Provider construction fails per-spec when credentials are absent.
            skipped[spec] = str(exc)

    if not resolved:
        detail = "; ".join(f"{spec}: {reason}" for spec, reason in skipped.items())
        raise UserError(f"No model in the failover chain could be initialized ({detail}).")

    if skipped:
        logfire.warning(
            "model.fallback_chain_incomplete", usable=[model.model_name for model in resolved], skipped=sorted(skipped)
        )

    return FallbackModel(resolved[0], *resolved[1:])


def get_model(model_spec: Annotated[str, Doc("Model specification string.")]) -> Model | str:
    """Get a model instance based on specification string.

    Supports:
    - Standard pydantic-ai model strings (e.g., 'anthropic:claude-3-haiku-20240307')
    - Local Devstral (e.g., 'devstral:local')
    - OpenRouter models (e.g., 'openrouter:mistralai/devstral-small-2505')
    - Failover chains of any of the above (e.g., 'openrouter:a:free,openrouter:b:free')

    @notice: |
        Resolves model specs into provider-specific model objects when needed.

    @dev: |
        Returns a string for a single standard pydantic-ai model spec, and a
        FallbackModel when the spec holds more than one entry so a rate-limited
        or dead provider fails over instead of aborting the run.

    @factory-for:
        id: pydantic_ai.models:Model
        rationale: "Centralized model resolution for all agents."
        singleton: false
        cache-key: model_spec

    @canonical-home:
        for:
            - "model resolution"
        notes: "Use get_model to normalize model specs."
    """
    specs = split_model_specs(model_spec)
    if len(specs) > 1:
        return create_fallback_model(specs)
    return _resolve_model_spec(specs[0])


def _resolve_model_spec(model_spec: str) -> Model | str:
    """Resolve a single model spec into a model instance or a pydantic-ai spec string."""
    if model_spec.startswith("devstral:"):
        # Parse devstral model specification for local LM Studio
        suffix = model_spec[len("devstral:") :]

        if suffix == "local":
            # Use default local LM Studio configuration
            return create_devstral_model()
        elif suffix.startswith("http"):
            # Custom base URL provided
            return create_devstral_model(base_url=suffix)
        else:
            # Assume suffix is a model ID
            return create_devstral_model(model_id=suffix)

    if model_spec.startswith("openrouter:"):
        # Parse OpenRouter model specification
        model_id = model_spec[len("openrouter:") :]
        return create_openrouter_model(model_id)

    # Return string for standard pydantic-ai model resolution
    # (e.g., 'anthropic:claude-3-haiku-20240307', 'openai:gpt-4o')
    return model_spec


def is_devstral_model(model_spec: Annotated[str, Doc("Model specification string.")]) -> bool:
    """Check if a model specification refers to Devstral.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns true when any member of the spec chain targets Devstral.
    """
    if not model_spec.strip():
        return False
    return any(spec.startswith("devstral:") for spec in split_model_specs(model_spec))
