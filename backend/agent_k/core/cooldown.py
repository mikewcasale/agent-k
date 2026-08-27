"""Rate-limit cooldown scheduling for model-backed work loops.

@notice: |
    Rate-limit cooldown scheduling for model-backed work loops.

@dev: |
    Free and shared model endpoints answer with HTTP 429 under load. Callers that
    treat a single 429 as terminal lose the remaining work budget even though the
    endpoint recovers within seconds. This module tracks per-model cooldowns so a
    caller can rotate to a ready model, sleep for a bounded interval, and only
    give up once every model is retired or the wait budget is spent.

@graph:
    id: agent_k.core.cooldown
    provides:
        - agent_k.core.cooldown:ModelLease
        - agent_k.core.cooldown:RateLimitScheduler
        - agent_k.core.cooldown:parse_retry_after
    pattern: scheduler

@similar:
    - id: agent_k.core.exceptions
        when: "Error taxonomy; this module schedules retries after rate limits."

@agent-guidance:
    do:
        - "Use agent_k.core.cooldown as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-08-27
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Annotated, Any, Final

from agent_k.core.sage import Doc, Range

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = (
    "DEFAULT_BASE_COOLDOWN_SECONDS",
    "DEFAULT_MAX_COOLDOWN_SECONDS",
    "DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS",
    "DEFAULT_WAIT_BUDGET_SECONDS",
    "ModelLease",
    "RateLimitScheduler",
    "parse_retry_after",
)

DEFAULT_BASE_COOLDOWN_SECONDS: Final[float] = 20.0
"""First cooldown applied to a model after a rate limit, in seconds."""

DEFAULT_MAX_COOLDOWN_SECONDS: Final[float] = 120.0
"""Upper bound for a single cooldown interval, in seconds."""

DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS: Final[int] = 3
"""Consecutive rate limits after which a model is retired for the run."""

DEFAULT_WAIT_BUDGET_SECONDS: Final[float] = 300.0
"""Total time a scheduler may spend sleeping on cooldowns, in seconds."""


@dataclass(frozen=True, slots=True)
class ModelLease:
    """Outcome of a scheduler acquisition attempt.

    @notice: |
        Outcome of a scheduler acquisition attempt.

    @dev: |
        Exactly one of the three states is reported: a ready ``model``, a
        ``wait_seconds`` hint when every live model is cooling down, or
        ``exhausted`` when no model can be used again.

        @pattern:
            name: result-object
            rationale: "Keeps the three scheduler outcomes explicit for callers."
            violations: "Tuple returns hide the exhausted case and get mishandled."
    """

    model: str | None
    wait_seconds: float
    exhausted: bool


@dataclass(slots=True)
class _ModelState:
    """Mutable cooldown bookkeeping for one model spec.

    @notice: |
        Mutable cooldown bookkeeping for one model spec.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: state-record
            rationale: "Keeps per-model cooldown counters in one place."
            violations: "Parallel dicts drift out of sync with each other."
    """

    consecutive_rate_limits: int = 0
    available_at: float = 0.0
    retired: bool = False


@dataclass(slots=True)
class RateLimitScheduler:
    """Round-robin scheduler that parks rate-limited models on a cooldown.

    @notice: |
        Round-robin scheduler that parks rate-limited models on a cooldown.

    @dev: |
        ``acquire`` returns the next model that is off cooldown. When every live
        model is cooling down it returns the bounded sleep the caller should
        perform, debiting the shared wait budget so a run cannot stall forever.
        A model is retired after ``max_consecutive_rate_limits`` rate limits with
        no successful call in between.

        @pattern:
            name: scheduler
            rationale: "Centralizes rate-limit backoff for model rotation loops."
            violations: "Ad-hoc eviction drops models that recover in seconds."

        @concurrency:
            model: single-task
            safe: false
            reason: "Mutates per-model state without locking."

        @invariants:
            - "Wait budget never goes negative."
            - "A retired model is never returned by acquire."
    """

    models: list[str]
    base_cooldown_seconds: float = DEFAULT_BASE_COOLDOWN_SECONDS
    max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS
    max_consecutive_rate_limits: int = DEFAULT_MAX_CONSECUTIVE_RATE_LIMITS
    wait_budget_seconds: float = DEFAULT_WAIT_BUDGET_SECONDS
    time_source: Callable[[], float] = time.monotonic
    _states: dict[str, _ModelState] = field(default_factory=dict, init=False)
    _cursor: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        deduped: list[str] = []
        for model in self.models:
            if model and model not in deduped:
                deduped.append(model)
        self.models = deduped
        self._states = {model: _ModelState() for model in self.models}

    @property
    def active_models(self) -> list[str]:
        """Model specs that have not been retired.

        @dev: |
            See module for behavior details and invariants.

            @notice: |
                Returns the model specs still eligible for scheduling.
        """
        return [model for model in self.models if not self._states[model].retired]

    def acquire(self) -> ModelLease:
        """Return the next usable model, a bounded wait, or exhaustion.

        @notice: |
            Picks the next model that is off cooldown, else a sleep hint.

        @dev: |
            Rotation starts from the internal cursor so consecutive calls spread
            load across models. When no model is ready the returned
            ``wait_seconds`` is debited from the shared wait budget; once the
            budget is spent the scheduler reports exhaustion instead.
        """
        active = self.active_models
        if not active:
            return ModelLease(model=None, wait_seconds=0.0, exhausted=True)

        now = self.time_source()
        for offset in range(len(self.models)):
            index = (self._cursor + offset) % len(self.models)
            model = self.models[index]
            state = self._states[model]
            if state.retired or state.available_at > now:
                continue
            self._cursor = (index + 1) % len(self.models)
            return ModelLease(model=model, wait_seconds=0.0, exhausted=False)

        wait = min(self._states[spec].available_at for spec in active) - now
        wait = max(wait, 0.0)
        if self.wait_budget_seconds <= 0.0:
            return ModelLease(model=None, wait_seconds=0.0, exhausted=True)

        granted = min(wait, self.wait_budget_seconds)
        self.wait_budget_seconds -= granted
        return ModelLease(model=None, wait_seconds=granted, exhausted=False)

    def record_success(self, model: Annotated[str, Doc("Model spec that completed a call.")]) -> None:
        """Clear the consecutive rate-limit streak for a model.

        @dev: |
            See module for behavior details and invariants.

            @notice: |
                Resets cooldown escalation after a successful call.
        """
        state = self._states.get(model)
        if state is None:
            return
        state.consecutive_rate_limits = 0
        state.available_at = 0.0

    def record_rate_limit(
        self,
        model: Annotated[str, Doc("Model spec that was rate limited.")],
        *,
        retry_after: Annotated[float | None, Doc("Server supplied Retry-After in seconds."), Range(0, 3600)] = None,
    ) -> float:
        """Park a model on an escalating cooldown and report the interval.

        @notice: |
            Applies exponential cooldown, honoring Retry-After when supplied.

        @dev: |
            The cooldown doubles per consecutive rate limit and is capped by
            ``max_cooldown_seconds``. A ``retry_after`` hint wins when it asks
            for a longer wait than the computed backoff. The model is retired
            once the streak reaches ``max_consecutive_rate_limits``.
        """
        state = self._states.get(model)
        if state is None:
            return 0.0

        state.consecutive_rate_limits += 1
        backoff = self.base_cooldown_seconds * (2.0 ** (state.consecutive_rate_limits - 1))
        cooldown = min(max(backoff, retry_after or 0.0), self.max_cooldown_seconds)
        state.available_at = self.time_source() + cooldown
        if state.consecutive_rate_limits >= self.max_consecutive_rate_limits:
            state.retired = True
        return cooldown

    def retire(self, model: Annotated[str, Doc("Model spec to remove from rotation.")]) -> None:
        """Remove a model from rotation for the remainder of the run.

        @dev: |
            See module for behavior details and invariants.

            @notice: |
                Marks a model as permanently unavailable.
        """
        state = self._states.get(model)
        if state is not None:
            state.retired = True


def parse_retry_after(error: Annotated[Any, Doc("Exception or header value to inspect.")]) -> float | None:
    """Extract a Retry-After delay in seconds from an error or header value.

    @notice: |
        Reads Retry-After from an exception, response headers, or raw value.

    @dev: |
        Accepts integer-second and HTTP-date header formats. Returns None when no
        usable hint is present so callers fall back to their own backoff.
    """
    candidate: Any = getattr(error, "retry_after", None)
    if candidate is None:
        headers = getattr(getattr(error, "response", None), "headers", None)
        if headers is not None:
            with suppress(AttributeError, TypeError, KeyError):
                candidate = headers.get("retry-after")
    if candidate is None and isinstance(error, int | float | str):
        candidate = error
    if candidate is None:
        return None

    if isinstance(candidate, int | float) and not isinstance(candidate, bool):
        return float(candidate) if candidate > 0 else None
    if not isinstance(candidate, str):
        return None

    text = candidate.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return _parse_http_date(text)
    return seconds if seconds > 0 else None


def _parse_http_date(value: str) -> float | None:
    try:
        retry_time = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_time is None:
        return None
    if retry_time.tzinfo is None:
        retry_time = retry_time.replace(tzinfo=UTC)
    seconds = (retry_time - datetime.now(UTC)).total_seconds()
    return seconds if seconds > 0 else None
