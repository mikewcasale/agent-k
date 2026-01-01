"""Centralized instrumentation for AGENT-K.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

# Standard library (alphabetical)
import asyncio
import os
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

# Third-party (alphabetical)
import logfire
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

P = ParamSpec("P")
"""Parameter specification for traced decorators."""

R = TypeVar("R")
"""Type variable for traced return values."""

__all__ = ("configure_instrumentation", "get_logger", "traced", "operation_span", "Metrics")


class Metrics:
    """Centralized metrics recording.

    Provides methods for recording various metric types consistently
    across the application.
    """

    @staticmethod
    def record_agent_run(agent_name: str, duration_ms: float, tokens_used: int, success: bool) -> None:
        """Record agent run metrics."""
        logfire.info("agent_run", agent=agent_name, duration_ms=duration_ms, tokens=tokens_used, success=success)

    @staticmethod
    def record_api_call(endpoint: str, status_code: int, duration_ms: float) -> None:
        """Record API call metrics."""
        logfire.info("api_call", endpoint=endpoint, status_code=status_code, duration_ms=duration_ms)

    @staticmethod
    def record_submission(competition_id: str, score: float | None, rank: int | None) -> None:
        """Record competition submission metrics."""
        logfire.info("submission", competition_id=competition_id, score=score, rank=rank)

    @staticmethod
    def record_evolution_generation(generation: int, best_fitness: float, mean_fitness: float, population_size: int) -> None:
        """Record evolution generation metrics."""
        logfire.info("evolution_generation", generation=generation, best_fitness=best_fitness, mean_fitness=mean_fitness, population_size=population_size)


def configure_instrumentation(*, service_name: str = "agent-k", environment: str | None = None, send_to_logfire: bool | Literal["if-token-present"] | None = "if-token-present") -> None:
    """Configure global instrumentation settings.

    This function should be called once at application startup.

    Args:
        service_name: Name of the service for tracing.
        environment: Deployment environment (dev, staging, prod).
        send_to_logfire: Whether to send telemetry to Logfire.
    """
    environment = environment or os.getenv("ENVIRONMENT", "development")

    logfire.configure(service_name=service_name, environment=environment, send_to_logfire=send_to_logfire)

    # Instrument common libraries
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()
    instrument_asyncio = getattr(logfire, "instrument_asyncio", None)
    if instrument_asyncio is not None:
        instrument_asyncio()


def get_logger(name: str) -> logfire.Logfire:
    """Get a logger with component-specific settings.

    Args:
        name: Component name (e.g., 'agents.lobbyist', 'adapters.kaggle').

    Returns:
        Configured Logfire instance.
    """
    return logfire.with_settings(tags=[f"component:{name}"])


# =============================================================================
# Span Decorators
# =============================================================================
def traced(name: str | None = None, *, record_args: bool = True, record_result: bool = True) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to add tracing to a function.

    Args:
        name: Span name (defaults to function name).
        record_args: Whether to record function arguments.
        record_result: Whether to record return value.

    Returns:
        Decorated function with tracing.

    Example:
        >>> @traced('my_operation')
        ... async def process_data(data: list[int]) -> int:
        ...     return sum(data)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attributes: dict[str, Any] = {}
            if record_args:
                attributes["args"] = _serialize_args(args, kwargs)

            with logfire.span(span_name, **attributes) as span:
                try:
                    async_func = cast("Callable[P, Awaitable[R]]", func)
                    result = await async_func(*args, **kwargs)
                    if record_result:
                        span.set_attribute("result", _serialize_result(result))
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.set_attribute("error_type", type(e).__name__)
                    raise

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attributes: dict[str, Any] = {}
            if record_args:
                attributes["args"] = _serialize_args(args, kwargs)

            with logfire.span(span_name, **attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    if record_result:
                        span.set_attribute("result", _serialize_result(result))
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.set_attribute("error_type", type(e).__name__)
                    raise

        if asyncio.iscoroutinefunction(func):
            return cast("Callable[P, R]", async_wrapper)
        return cast("Callable[P, R]", sync_wrapper)

    return decorator


@contextmanager
def operation_span(name: str, **attributes: Any) -> Iterator[Span]:
    """Context manager for custom operation spans.

    Args:
        name: Span name.
        **attributes: Additional span attributes.

    Yields:
        Active span for additional annotations.

    Example:
        >>> with operation_span('process_batch', batch_size=100) as span:
        ...     results = process(batch)
        ...     span.set_attribute('processed_count', len(results))
    """
    tracer = trace.get_tracer("agent-k")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, _serialize_value(value))
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


# =============================================================================
# Helper Functions
# =============================================================================
def _serialize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Serialize function arguments for logging."""
    parts = [repr(a)[:100] for a in args]
    parts.extend(f"{k}={repr(v)[:100]}" for k, v in kwargs.items())
    return ", ".join(parts)[:500]


def _serialize_result(result: Any) -> str:
    """Serialize function result for logging."""
    return repr(result)[:500]


def _serialize_value(value: Any) -> str | int | float | bool:
    """Serialize value for span attribute."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)[:500]
