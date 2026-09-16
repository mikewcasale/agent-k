"""Tests for Evolver mutation helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re

import pytest

__all__ = ()

try:
    from agent_k.agents.evolver import (
        _LIGHTGBM_AVAILABLE,
        _MODEL_IMPORTS,
        _MODEL_SWAPS,
        _NO_PREDICT_PROBA,
        _PENALISED_CLASSIFIERS,
        _PENALISED_REGRESSORS,
        _TREE_CLASSIFIERS,
        _TREE_REGRESSORS,
        _is_importable_model,
        evolver_agent_instance,
    )
except TypeError as exc:
    if "MCPServerTool" in str(exc):
        pytest.skip(f"MCPServerTool API issue: {exc}", allow_module_level=True)
    raise

_evolver = evolver_agent_instance
_MODEL_SWAPS_SEED_SAMPLES = 40


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

    @pytest.mark.parametrize(("source", "targets"), list(_MODEL_SWAPS.items()))
    def test_structural_mutation_model_swap(self, source: str, targets: tuple[str, ...]) -> None:
        """Structural mutation should swap model families."""
        code = f"model = {source}()"
        mutated = _evolver._apply_structural_mutation(code, {})

        assert any(target in mutated for target in targets)
        assert source not in mutated

    def test_model_swap_never_targets_itself(self) -> None:
        """No estimator may be listed as its own replacement."""
        for source, targets in _MODEL_SWAPS.items():
            assert source not in targets

    def test_model_swap_reaches_more_than_two_families(self) -> None:
        """Repeated swaps should escape the two-model ping-pong of the old table."""
        reached = {
            _evolver._swap_model_family("model = GradientBoostingRegressor()", {"seed": seed})
            for seed in range(_MODEL_SWAPS_SEED_SAMPLES)
        }

        assert len(reached) > 2

    def test_no_swap_target_lacks_predict_proba(self) -> None:
        """A probability-metric solution must survive any swap, so targets keep predict_proba."""
        every_target = {target for targets in _MODEL_SWAPS.values() for target in targets}

        assert not every_target & _NO_PREDICT_PROBA

    def test_model_swap_keeps_predict_proba_available(self) -> None:
        """Swapping a probability classifier must not drop predict_proba."""
        code = "model = LogisticRegression()\npreds = model.predict_proba(X_test)[:, 1]\n"

        for seed in range(_MODEL_SWAPS_SEED_SAMPLES):
            mutated = _evolver._swap_model_family(code, {"seed": seed})

            assert not any(target in mutated for target in _NO_PREDICT_PROBA)

    @pytest.mark.skipif(not _LIGHTGBM_AVAILABLE, reason="lightgbm is not installed in this runtime")
    def test_model_swap_can_reach_lightgbm(self) -> None:
        """LightGBM must be reachable from both task families, per the repo preference."""
        classifier_hits = {
            _evolver._swap_model_family("model = LogisticRegression()", {"seed": seed})
            for seed in range(_MODEL_SWAPS_SEED_SAMPLES)
        }
        regressor_hits = {
            _evolver._swap_model_family("model = LinearRegression()", {"seed": seed})
            for seed in range(_MODEL_SWAPS_SEED_SAMPLES)
        }

        assert any("from lightgbm import LGBMClassifier" in code for code in classifier_hits)
        assert any("from lightgbm import LGBMRegressor" in code for code in regressor_hits)

    def test_lightgbm_targets_are_gated_on_availability(self) -> None:
        """LightGBM is only offered as a target when the runtime can import it."""
        assert _is_importable_model("LGBMRegressor") is _LIGHTGBM_AVAILABLE
        assert _is_importable_model("RandomForestRegressor") is True

    def test_model_swap_reroutes_the_rewritten_import(self) -> None:
        """The swap rewrites the source import too; it must not survive pointing at the wrong module."""
        code = "from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression()\n"

        for seed in range(_MODEL_SWAPS_SEED_SAMPLES):
            mutated = _evolver._swap_model_family(code, {"seed": seed})
            for line in mutated.splitlines():
                match = re.match(r"from (\S+) import (.+)", line)
                if match is None:
                    continue
                for symbol in (part.strip() for part in match.group(2).split(",")):
                    assert _MODEL_IMPORTS.get(symbol, match.group(1)) == match.group(1)

    def test_swap_target_carries_its_import(self) -> None:
        """A swap must add the target's import so the mutated program still runs."""
        mutated = _evolver._swap_model_family("model = LGBMRegressor()", {})
        target = next(name for name in _TREE_REGRESSORS if name in mutated)

        assert f"from {_MODEL_IMPORTS[target]} import {target}" in mutated

    def test_swap_targets_stay_within_the_task_family(self) -> None:
        """Classifier sources may only become classifiers, and regressors regressors."""
        classifier_sources = set(_TREE_CLASSIFIERS) | set(_PENALISED_CLASSIFIERS)
        regressor_sources = set(_TREE_REGRESSORS) | set(_PENALISED_REGRESSORS)

        assert not classifier_sources & regressor_sources
        assert set(_MODEL_SWAPS) == classifier_sources | regressor_sources

        for source, targets in _MODEL_SWAPS.items():
            family = _TREE_CLASSIFIERS if source in classifier_sources else _TREE_REGRESSORS
            assert set(targets) <= set(family)

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
