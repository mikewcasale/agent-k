"""Tests for the feature-selection Pareto front dominance rule.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

from agent_k.evolution.features import FeatureSelectionIndividual, FeatureSelector, _dominates

__all__ = ()


def _make(mask: list[int], score: float | None) -> FeatureSelectionIndividual:
    return FeatureSelectionIndividual(mask=mask, score=score)


def test_dominates_returns_true_when_better_on_both_objectives() -> None:
    left = _make([1, 1, 0, 0], score=0.90)
    right = _make([1, 1, 1, 0], score=0.80)

    assert _dominates(left, right) is True
    assert _dominates(right, left) is False


def test_dominates_returns_true_when_equal_score_but_fewer_features() -> None:
    left = _make([1, 1, 0, 0], score=0.80)
    right = _make([1, 1, 1, 0], score=0.80)

    assert _dominates(left, right) is True
    assert _dominates(right, left) is False


def test_dominates_returns_true_when_higher_score_and_equal_features() -> None:
    left = _make([1, 1, 0, 0], score=0.85)
    right = _make([0, 1, 1, 0], score=0.80)

    assert _dominates(left, right) is True
    assert _dominates(right, left) is False


def test_dominates_returns_false_on_pareto_tie() -> None:
    left = _make([1, 1, 0, 0], score=0.80)
    right = _make([0, 0, 1, 1], score=0.80)

    assert _dominates(left, right) is False
    assert _dominates(right, left) is False


def test_dominates_returns_false_on_trade_off() -> None:
    left = _make([1, 1, 1, 0], score=0.90)
    right = _make([1, 0, 0, 0], score=0.70)

    assert _dominates(left, right) is False
    assert _dominates(right, left) is False


def test_dominates_returns_false_when_either_score_missing() -> None:
    scored = _make([1, 1, 0, 0], score=0.80)
    unscored = _make([1, 0, 0, 0], score=None)

    assert _dominates(scored, unscored) is False
    assert _dominates(unscored, scored) is False


def test_pareto_front_preserves_tied_non_dominated_individuals() -> None:
    features = ["a", "b", "c", "d"]
    selector = FeatureSelector(features, population_size=4, rng=random.Random(0))
    selector._population = [
        _make([1, 1, 0, 0], score=0.80),
        _make([0, 0, 1, 1], score=0.80),
        _make([1, 1, 1, 0], score=0.75),
        _make([1, 0, 0, 0], score=0.60),
    ]

    selector._update_pareto_front()

    front_masks = sorted(tuple(individual.mask) for individual in selector._pareto_front)
    assert front_masks == [(0, 0, 1, 1), (1, 0, 0, 0), (1, 1, 0, 0)]


def test_pareto_front_drops_strictly_dominated_individuals() -> None:
    features = ["a", "b", "c", "d"]
    selector = FeatureSelector(features, population_size=3, rng=random.Random(0))
    selector._population = [
        _make([1, 1, 0, 0], score=0.90),
        _make([1, 1, 1, 0], score=0.80),
        _make([1, 1, 1, 1], score=0.70),
    ]

    selector._update_pareto_front()

    front_masks = [tuple(individual.mask) for individual in selector._pareto_front]
    assert front_masks == [(1, 1, 0, 0)]


def test_pareto_front_skips_unscored_candidates() -> None:
    features = ["a", "b", "c", "d"]
    selector = FeatureSelector(features, population_size=3, rng=random.Random(0))
    selector._population = [
        _make([1, 1, 0, 0], score=0.80),
        _make([0, 1, 1, 0], score=None),
        _make([1, 0, 0, 0], score=0.60),
    ]

    selector._update_pareto_front()

    front_masks = sorted(tuple(individual.mask) for individual in selector._pareto_front)
    assert front_masks == [(1, 0, 0, 0), (1, 1, 0, 0)]


def test_pareto_front_keeps_duplicates_with_identical_signatures() -> None:
    features = ["a", "b", "c"]
    selector = FeatureSelector(features, population_size=2, rng=random.Random(0))
    duplicate_a = _make([1, 1, 0], score=0.75)
    duplicate_b = _make([1, 1, 0], score=0.75)
    selector._population = [duplicate_a, duplicate_b]

    selector._update_pareto_front()

    assert selector._pareto_front == [duplicate_a, duplicate_b]


def test_evolve_returns_non_empty_pareto_front_when_scores_tie() -> None:
    features = ["a", "b", "c", "d"]
    selector = FeatureSelector(features, population_size=6, rng=random.Random(1))

    def constant_score(_selected: list[str]) -> float:
        return 0.5

    result = selector.evolve(generations=2, fitness_fn=constant_score)

    assert result["pareto_front"], "constant fitness should still yield a non-empty Pareto front"
    for individual in result["pareto_front"]:
        assert individual.score == 0.5
