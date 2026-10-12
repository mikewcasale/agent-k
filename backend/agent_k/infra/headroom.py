"""Adaptive output-token headroom for agent runs.

@notice: |
    Adaptive output-token headroom for agent runs.

@dev: |
    Structured agent outputs carry generated solution code, so a model that
    stops at its `max_tokens` ceiling emits a truncated tool call. Pydantic-AI
    retries that call until `output_retries` is exhausted and then raises, which
    wastes every retry on a request that cannot succeed and aborts the mission
    phase. This module detects the ceiling from the response finish reason and
    re-runs the agent with a larger `max_tokens` budget instead.

@graph:
    id: agent_k.infra.headroom
    provides:
        - agent_k.infra.headroom
    pattern: resilience

@agent-guidance:
    do:
        - "Use agent_k.infra.headroom as the canonical home for this capability."
        - "Run agents that emit code-bearing outputs through run_with_token_headroom."
    do_not:
        - "Create parallel modules without updating @similar or @graph."
        - "Raise per-agent max_tokens defaults past what small models accept."

@human-review:
    last-verified: 2026-10-12
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING, Annotated, Any, Final, cast

import logfire
from pydantic_ai import IncompleteToolCall, ModelAPIError, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.messages import ModelResponse

from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import Agent
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.settings import ModelSettings

__all__ = (
    "MAX_TOKENS_CEILING",
    "MAX_TOKENS_ESCALATIONS",
    "MAX_TOKENS_FACTOR",
    "MAX_TOKENS_UNSET_BASE",
    "escalate_max_tokens",
    "hit_token_ceiling",
    "run_with_token_headroom",
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_TOKENS_CEILING: Final[int] = _env_int("AGENT_K_MAX_TOKENS_CEILING", 32768)
"""Hard upper bound for an escalated `max_tokens` budget."""

MAX_TOKENS_FACTOR: Final[int] = _env_int("AGENT_K_MAX_TOKENS_FACTOR", 3)
"""Multiplier applied to the current budget on each escalation."""

MAX_TOKENS_ESCALATIONS: Final[int] = _env_int("AGENT_K_MAX_TOKENS_ESCALATIONS", 2)
"""Number of escalated re-runs allowed after the first truncated run."""

MAX_TOKENS_UNSET_BASE: Final[int] = _env_int("AGENT_K_MAX_TOKENS_UNSET_BASE", 8192)
"""Budget assumed when an agent relies on the provider default."""


def hit_token_ceiling(
    messages: Annotated[Sequence[ModelMessage], Doc("Messages exchanged during the failed agent run.")],
) -> bool:
    """Report whether any model response in the run stopped at its token ceiling.

    @notice: |
        Reports whether a run stopped because the output token budget ran out.

    @dev: |
        A `length` finish reason is the provider-independent signal that the
        response was cut short. Any occurrence counts, because output retries
        can follow a truncated response with a differently-shaped failure.
    """
    return any(message.finish_reason == "length" for message in messages if isinstance(message, ModelResponse))


def escalate_max_tokens(
    settings: Annotated[ModelSettings | None, Doc("Model settings used for the truncated run.")],
    *,
    ceiling: Annotated[int, Doc("Hard upper bound for the escalated budget.")] = MAX_TOKENS_CEILING,
    factor: Annotated[int, Doc("Multiplier applied to the current budget.")] = MAX_TOKENS_FACTOR,
    unset_base: Annotated[int, Doc("Budget assumed when max_tokens is unset.")] = MAX_TOKENS_UNSET_BASE,
) -> ModelSettings | None:
    """Return settings with a larger output token budget, or None at the ceiling.

    @notice: |
        Returns model settings carrying a larger max_tokens budget.

    @dev: |
        Returns None once the budget already reaches the ceiling so callers can
        surface the original error instead of re-running the same request. An
        unset budget escalates from unset_base, which stays above every
        provider default we target.
    """
    current = settings.get("max_tokens") if settings is not None else None
    base = int(current) if current else unset_base
    if base >= ceiling:
        return None

    escalated = min(ceiling, max(base + 1, base * max(factor, 2)))
    merged: dict[str, Any] = dict(settings) if settings is not None else {}
    merged["max_tokens"] = escalated
    return cast("ModelSettings", merged)


async def run_with_token_headroom(
    agent: Annotated[Agent[Any, Any], Doc("Agent to run.")],
    user_prompt: Annotated[str, Doc("Prompt for the agent run.")],
    *,
    deps: Annotated[Any, Doc("Dependency container for the agent run.")],
    phase: Annotated[str, Doc("Mission phase label used for observability.")],
    escalations: Annotated[int, Doc("Escalated re-runs allowed after a truncated run.")] = MAX_TOKENS_ESCALATIONS,
) -> AgentRunResult[Any]:
    """Run an agent, re-running with a larger token budget when output is truncated.

    @notice: |
        Runs an agent and retries with more output headroom on truncation.

    @dev: |
        Only truncation is retried: any other failure propagates untouched so
        existing rate-limit and fallback handling keeps working. Settings are
        merged at run level, which overrides the agent-level budget without
        mutating the shared agent singleton. Providers with a small output cap
        reject the larger budget, so a provider error raised by an escalated
        re-run surfaces the original truncation error instead.

    @effects:
        io:
            - model invocation
        state:
            - none

    @errors:
        terminal:
            - UnexpectedModelBehavior
    """
    settings: ModelSettings | None = agent.model_settings
    truncation: UnexpectedModelBehavior | None = None
    attempt = 0

    while True:
        with capture_run_messages() as messages:
            try:
                return await agent.run(user_prompt, deps=deps, model_settings=settings)
            except ModelAPIError as exc:
                if truncation is None:
                    raise
                logfire.warning(
                    "agent_headroom_rejected",
                    phase=phase,
                    agent=agent.name,
                    max_tokens=settings.get("max_tokens") if settings is not None else None,
                    error=str(exc),
                )
                raise truncation from exc
            except UnexpectedModelBehavior as exc:
                if attempt >= escalations:
                    raise
                if not isinstance(exc, IncompleteToolCall) and not hit_token_ceiling(messages):
                    raise
                escalated = escalate_max_tokens(settings)
                if escalated is None:
                    raise
                logfire.warning(
                    "agent_output_truncated",
                    phase=phase,
                    agent=agent.name,
                    attempt=attempt + 1,
                    previous_max_tokens=settings.get("max_tokens") if settings is not None else None,
                    max_tokens=escalated["max_tokens"],
                    error=str(exc),
                )
                settings = escalated
                truncation = exc
                attempt += 1
