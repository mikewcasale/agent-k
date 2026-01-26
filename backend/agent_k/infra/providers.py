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

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from pydantic_ai.models import Model

__all__ = (
    "get_model",
    "create_devstral_model",
    "create_openrouter_model",
    "is_devstral_model",
    "DEVSTRAL_MODEL_ID",
    "DEVSTRAL_BASE_URL",
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


def get_model(model_spec: Annotated[str, Doc("Model specification string.")]) -> Model | str:
    """Get a model instance based on specification string.

    Supports:
    - Standard pydantic-ai model strings (e.g., 'anthropic:claude-3-haiku-20240307')
    - Local Devstral (e.g., 'devstral:local')
    - OpenRouter models (e.g., 'openrouter:mistralai/devstral-small-2505')

    @notice: |
        Resolves model specs into provider-specific model objects when needed.

    @dev: |
        Returns a string for standard pydantic-ai model specs.

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
            Returns true when the model spec targets Devstral.
    """
    return model_spec.startswith("devstral:")
