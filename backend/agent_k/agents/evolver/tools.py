"""Domain-specific tools for the Evolver agent."""
from __future__ import annotations

__all__ = ['select_best_solution']


def select_best_solution(history: list[dict[str, float]]) -> dict[str, float] | None:
    """Return the best recorded generation metrics."""
    if not history:
        return None
    return max(history, key=lambda item: item.get('best_fitness', 0.0))
