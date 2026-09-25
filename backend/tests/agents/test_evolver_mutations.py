"""Tests for Evolver mutation helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re

import pytest

__all__ = ()

try:
    from agent_k.agents.evolver import _MODEL_SWAPS, _OBJECTIVE_FAMILIES, evolver_agent_instance
except TypeError as exc:
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise

_evolver = evolver_agent_instance


class TestFitnessFromScore:
    """Tests for fitness calculation helper."""

    @pytest.mark.parametrize(
        ("score", "direction", "expected"),
        [(0.25, "maximize", 0.25), (-1.0, "maximize", 0.0), (3.0, "minimize", 0.25), (-2.0, "minimize", 1.0)],
    )
    def test_fitness_from_score(self, score: float, direction: str, expected: float) -> None:
        """Fitness should reflect metric direction."""
        assert _evolver._fitness_from_score(score, direction) == pytest.approx(expected)


class TestSeededRng:
    """Tests for deterministic RNG seeding."""

    def test_seeded_rng_is_deterministic(self) -> None:
        """Same inputs should yield identical RNG state."""
        rng_a = _evolver._seeded_rng("code", {"a": 1, "b": 2}, "salt")
        rng_b = _evolver._seeded_rng("code", {"a": 1, "b": 2}, "salt")
        rng_c = _evolver._seeded_rng("code", {"a": 1, "b": 2}, "other")

        assert rng_a.getstate() == rng_b.getstate()
        assert rng_a.getstate() != rng_c.getstate()


class TestPointMutation:
    """Tests for point mutation behavior."""

    def test_point_mutation_respects_max_changes(self) -> None:
        """Point mutation should stop after max_changes."""
        code = "a = 1.0\nb = 2.0\nc = 3.0\n"
        params = {"delta": 0.5, "max_changes": 1}

        mutated = _evolver._apply_point_mutation(code, params)
        numbers_before = re.findall(r"-?\d+\.?\d*", code)
        numbers_after = re.findall(r"-?\d+\.?\d*", mutated)

        differences = sum(1 for before, after in zip(numbers_before, numbers_after, strict=False) if before != after)
        assert differences == 1


class TestHyperparameterMutation:
    """Tests for hyperparameter mutation behavior."""

    def test_hyperparameter_mutation_keeps_integer(self) -> None:
        """Integer hyperparameters should remain integers."""
        code = "model = XGBClassifier(n_estimators=100)"
        mutated = _evolver._apply_hyperparameter_mutation(code, {"magnitude": 0.3})
        match = re.search(r"n_estimators\s*=\s*(\d+)", mutated)

        assert match is not None
        assert match.group(1).isdigit()

    def test_hyperparameter_mutation_updates_float(self) -> None:
        """Float hyperparameters should remain float-like."""
        code = "model = XGBClassifier(learning_rate=0.1)"
        mutated = _evolver._apply_hyperparameter_mutation(code, {"magnitude": 0.5})
        match = re.search(r"learning_rate\s*=\s*([\d\.]+)", mutated)

        assert match is not None
        assert match.group(1) != "0.1"
        assert "." in match.group(1)


class TestStructuralMutation:
    """Tests for structural mutation helpers."""

    @pytest.mark.parametrize(("source", "target"), list(_MODEL_SWAPS.items()))
    def test_structural_mutation_model_swap(self, source: str, target: str) -> None:
        """Structural mutation should swap model families."""
        code = f"model = {source}()"
        mutated = _evolver._apply_structural_mutation(code, {})

        assert target in mutated
        assert source not in mutated

    def test_structural_mutation_injects_fillna(self) -> None:
        """Structural mutation should inject fillna when applicable."""
        code = "import pandas as pd\n\ndata = pd.read_csv('train.csv')\n"
        mutated = _evolver._apply_structural_mutation(code, {})

        assert "data = data.fillna(0)" in mutated


class TestCrossover:
    """Tests for crossover helper."""

    def test_crossover_merges_imports(self) -> None:
        """Crossover should merge imports without duplicates."""
        code = "import os\nimport numpy as np\n\n\ndef foo():\n    return 1\n"
        other = "import os\nfrom math import sqrt\n\n\ndef bar():\n    return 2\n"

        merged = _evolver._apply_crossover(code, other, {})
        import_lines = [line for line in merged.splitlines() if line.startswith(("import ", "from "))]

        assert import_lines.count("import os") == 1
        assert "import numpy as np" in import_lines
        assert "from math import sqrt" in import_lines
        assert "def bar():" in merged


def _sweep_objectives(code: str, draws: int = 60) -> set[str]:
    """Return every objective value reachable by repeated objective mutation of ``code``."""
    values: set[str] = set()
    for draw in range(draws):
        mutated = _evolver._apply_hyperparameter_mutation(code, {"param": "objective", "draw": draw})
        match = re.search(r"objective\s*=\s*\"([^\"]+)\"", mutated)
        if match:
            values.add(match.group(1))
    return values


class TestObjectiveMutation:
    """Objective mutation must stay inside the problem family already implied by the code."""

    @pytest.mark.parametrize(
        ("current", "family"),
        [
            ("regression", "regression"),
            ("regression_l1", "regression"),
            ("huber", "regression"),
            ("binary", "binary"),
            ("cross_entropy", "binary"),
            ("multiclass", "multiclass"),
            ("multiclassova", "multiclass"),
        ],
    )
    def test_objective_stays_within_its_family(self, current: str, family: str) -> None:
        """Every reachable objective belongs to the family the code started in."""
        estimator = "LGBMRegressor" if family == "regression" else "LGBMClassifier"
        code = f'model = {estimator}(objective="{current}", learning_rate=0.05)\n'

        assert _sweep_objectives(code) <= set(_OBJECTIVE_FAMILIES[family])

    @pytest.mark.parametrize("current", ["binary", "cross_entropy"])
    def test_classification_never_becomes_regression(self, current: str) -> None:
        """Regression objectives silently destroy probability quality on binary targets."""
        code = f'model = LGBMClassifier(objective="{current}", learning_rate=0.05)\n'

        assert not _sweep_objectives(code) & set(_OBJECTIVE_FAMILIES["regression"])

    @pytest.mark.parametrize("current", ["multiclass", "multiclassova"])
    def test_multiclass_never_leaves_its_family(self, current: str) -> None:
        """Non-multiclass objectives are a fatal LightGBM error on multiclass targets."""
        code = f'model = LGBMClassifier(objective="{current}", num_class=5)\n'
        reachable = _sweep_objectives(code)

        assert reachable
        assert reachable <= set(_OBJECTIVE_FAMILIES["multiclass"])

    @pytest.mark.parametrize(
        ("alias", "family"), [("l2", "regression"), ("mae", "regression"), ("softmax", "multiclass")]
    )
    def test_objective_aliases_resolve_to_their_family(self, alias: str, family: str) -> None:
        """LightGBM objective aliases map onto the same family as their canonical name."""
        estimator = "LGBMRegressor" if family == "regression" else "LGBMClassifier"
        code = f'model = {estimator}(objective="{alias}", num_class=5)\n'

        assert _sweep_objectives(code) <= set(_OBJECTIVE_FAMILIES[family])

    def test_unrecognised_objective_on_a_classifier_is_left_alone(self) -> None:
        """An ambiguous classifier objective is skipped rather than guessed at."""
        code = 'model = SomeClassifier(objective="weird_custom", learning_rate=0.05)\n'

        assert _sweep_objectives(code) == {"weird_custom"}

    def test_unrecognised_objective_on_a_regressor_uses_regression_family(self) -> None:
        """An unambiguous regressor still explores regression objectives."""
        code = 'model = LGBMRegressor(objective="tweedie", learning_rate=0.05)\n'

        assert _sweep_objectives(code) <= set(_OBJECTIVE_FAMILIES["regression"]) | {"tweedie"}


class TestKnnParamScoping:
    """KNN-only parameters must not be rewritten in code that has no KNN estimator."""

    def test_lightgbm_metric_is_not_replaced_by_a_distance_metric(self) -> None:
        """``metric`` is a LightGBM parameter too; distance metrics are meaningless there."""
        code = 'model = LGBMClassifier(metric="auc", learning_rate=0.05)\n'

        for draw in range(60):
            mutated = _evolver._apply_hyperparameter_mutation(code, {"param": "metric", "draw": draw})
            assert 'metric="auc"' in mutated

    def test_knn_code_still_mutates_its_metric(self) -> None:
        """Gating must not disable KNN metric exploration."""
        code = 'model = KNeighborsRegressor(n_neighbors=5, metric="euclidean", p=2)\n'
        metrics = set()
        for draw in range(60):
            mutated = _evolver._apply_hyperparameter_mutation(code, {"param": "metric", "draw": draw})
            if match := re.search(r'metric\s*=\s*"([^"]+)"', mutated):
                metrics.add(match.group(1))

        assert len(metrics) > 1
