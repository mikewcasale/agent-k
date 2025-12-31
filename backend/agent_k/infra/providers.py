"""Model configuration for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import os
from typing import TYPE_CHECKING, Final, TypeAlias

# Third-party (alphabetical)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

__all__ = (
    'get_model',
    'create_devstral_model',
    'create_openrouter_model',
    'is_devstral_model',
    'DEVSTRAL_MODEL_ID',
    'DEVSTRAL_BASE_URL',
    'ModelType',
)

DEVSTRAL_MODEL_ID: Final[str] = 'mistralai/devstral-small-2-2512'
DEVSTRAL_BASE_URL: Final[str] = os.getenv('DEVSTRAL_BASE_URL', 'http://192.168.105.1:1234/v1')

ModelType: TypeAlias = str


def create_devstral_model(model_id: str = DEVSTRAL_MODEL_ID, base_url: str | None = None) -> OpenAIChatModel:
    """Create a Devstral model instance for local LM Studio server.

    This creates an OpenAI-compatible model that connects to a local
    LM Studio server running Devstral.

    Args:
        model_id: The model identifier to use (default: devstral-small-2-2512).
        base_url: Override the base URL for the LM Studio server.

    Returns:
        Configured OpenAIChatModel instance.

    Example:
        >>> model = create_devstral_model()
        >>> agent = Agent(model, deps_type=MyDeps)
    """
    url = base_url or DEVSTRAL_BASE_URL

    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(
            base_url=url,
            api_key='not-required',  # Local LM Studio doesn't require auth
        ),
    )


def create_openrouter_model(model_id: str) -> OpenAIChatModel:
    """Create a model instance using OpenRouter.

    OpenRouter provides access to many models including Devstral, Claude,
    GPT-4, and more through a unified API.

    Args:
        model_id: The OpenRouter model identifier (e.g., 'mistralai/devstral-small-2505').

    Returns:
        Configured OpenAIChatModel instance using OpenRouter.

    Example:
        >>> model = create_openrouter_model('mistralai/devstral-small-2505')
        >>> agent = Agent(model, deps_type=MyDeps)

    Note:
        Requires OPENROUTER_API_KEY environment variable to be set.
    """
    return OpenAIChatModel(model_id, provider=OpenRouterProvider())


def get_model(model_spec: str) -> Model | str:
    """Get a model instance based on specification string.

    Supports:
    - Standard pydantic-ai model strings (e.g., 'anthropic:claude-3-haiku-20240307')
    - Local Devstral (e.g., 'devstral:local')
    - OpenRouter models (e.g., 'openrouter:mistralai/devstral-small-2505')

    Args:
        model_spec: Model specification string.

    Returns:
        Either a Model instance (for custom models) or the string (for standard models).

    Examples:
        >>> get_model('anthropic:claude-3-haiku-20240307')  # Returns string for pydantic-ai
        'anthropic:claude-3-haiku-20240307'

        >>> get_model('devstral:local')  # Returns OpenAIChatModel for local LM Studio
        OpenAIChatModel(...)

        >>> get_model('openrouter:mistralai/devstral-small-2505')  # Returns OpenAIChatModel via OpenRouter
        OpenAIChatModel(...)
    """
    if model_spec.startswith('devstral:'):
        # Parse devstral model specification for local LM Studio
        suffix = model_spec[len('devstral:') :]

        if suffix == 'local':
            # Use default local LM Studio configuration
            return create_devstral_model()
        elif suffix.startswith('http'):
            # Custom base URL provided
            return create_devstral_model(base_url=suffix)
        else:
            # Assume suffix is a model ID
            return create_devstral_model(model_id=suffix)

    if model_spec.startswith('openrouter:'):
        # Parse OpenRouter model specification
        model_id = model_spec[len('openrouter:') :]
        return create_openrouter_model(model_id)

    # Return string for standard pydantic-ai model resolution
    # (e.g., 'anthropic:claude-3-haiku-20240307', 'openai:gpt-4o')
    return model_spec


def is_devstral_model(model_spec: str) -> bool:
    """Check if a model specification refers to Devstral.

    Args:
        model_spec: Model specification string.

    Returns:
        True if the model is a Devstral variant.
    """
    return model_spec.startswith('devstral:')
