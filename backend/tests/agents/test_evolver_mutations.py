"""Tests for Evolver mutation helpers.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import re

import pytest

__all__ = ()

try:
    from agent_k.agents.evolver import _MODEL_SWAPS, evolver_agent_instance
except TypeError as exc:
    if 'MCPServerTool' in str(exc):
        pytest.skip(f'MCPServerTool API issue: {exc}', allow_module_level=True)
    raise

_evolver = evolver_agent_instance


class TestFitnessFromScore:
    """Tests for fitness calculation helper."""

    @pytest.mark.parametrize(
        ('score', 'direction', 'expected'),
        [(0.25, 'maximize', 0.25), (-1.0, 'maximize', 0.0), (3.0, 'minimize', 0.25), (-2.0, 'minimize', 1.0)],
    )
    def test_fitness_from_score(self, score: float, direction: str, expected: float) -> None:
        """Fitness should reflect metric direction."""
        assert _evolver._fitness_from_score(score, direction) == pytest.approx(expected)


class TestSeededRng:
    """Tests for deterministic RNG seeding."""

    def test_seeded_rng_is_deterministic(self) -> None:
        """Same inputs should yield identical RNG state."""
        rng_a = _evolver._seeded_rng('code', {'a': 1, 'b': 2}, 'salt')
        rng_b = _evolver._seeded_rng('code', {'a': 1, 'b': 2}, 'salt')
        rng_c = _evolver._seeded_rng('code', {'a': 1, 'b': 2}, 'other')

        assert rng_a.getstate() == rng_b.getstate()
        assert rng_a.getstate() != rng_c.getstate()


class TestPointMutation:
    """Tests for point mutation behavior."""

    def test_point_mutation_respects_max_changes(self) -> None:
        """Point mutation should stop after max_changes."""
        code = 'a = 1.0\nb = 2.0\nc = 3.0\n'
        params = {'delta': 0.5, 'max_changes': 1}

        mutated = _evolver._apply_point_mutation(code, params)
        numbers_before = re.findall(r'-?\d+\.?\d*', code)
        numbers_after = re.findall(r'-?\d+\.?\d*', mutated)

        differences = sum(1 for before, after in zip(numbers_before, numbers_after, strict=False) if before != after)
        assert differences == 1


class TestHyperparameterMutation:
    """Tests for hyperparameter mutation behavior."""

    def test_hyperparameter_mutation_keeps_integer(self) -> None:
        """Integer hyperparameters should remain integers."""
        code = 'model = XGBClassifier(n_estimators=100)'
        mutated = _evolver._apply_hyperparameter_mutation(code, {'magnitude': 0.3})
        match = re.search(r'n_estimators\s*=\s*(\d+)', mutated)

        assert match is not None
        assert match.group(1).isdigit()

    def test_hyperparameter_mutation_updates_float(self) -> None:
        """Float hyperparameters should remain float-like."""
        code = 'model = XGBClassifier(learning_rate=0.1)'
        mutated = _evolver._apply_hyperparameter_mutation(code, {'magnitude': 0.5})
        match = re.search(r'learning_rate\s*=\s*([\d\.]+)', mutated)

        assert match is not None
        assert match.group(1) != '0.1'
        assert '.' in match.group(1)


class TestStructuralMutation:
    """Tests for structural mutation helpers."""

    @pytest.mark.parametrize(('source', 'target'), list(_MODEL_SWAPS.items()))
    def test_structural_mutation_model_swap(self, source: str, target: str) -> None:
        """Structural mutation should swap model families."""
        code = f'model = {source}()'
        mutated = _evolver._apply_structural_mutation(code, {})

        assert target in mutated
        assert source not in mutated

    def test_structural_mutation_injects_fillna(self) -> None:
        """Structural mutation should inject fillna when applicable."""
        code = "import pandas as pd\n\ndata = pd.read_csv('train.csv')\n"
        mutated = _evolver._apply_structural_mutation(code, {})

        assert 'data = data.fillna(0)' in mutated


class TestCrossover:
    """Tests for crossover helper."""

    def test_crossover_merges_imports(self) -> None:
        """Crossover should merge imports without duplicates."""
        code = 'import os\nimport numpy as np\n\n\ndef foo():\n    return 1\n'
        other = 'import os\nfrom math import sqrt\n\n\ndef bar():\n    return 2\n'

        merged = _evolver._apply_crossover(code, other, {})
        import_lines = [line for line in merged.splitlines() if line.startswith(('import ', 'from '))]

        assert import_lines.count('import os') == 1
        assert 'import numpy as np' in import_lines
        assert 'from math import sqrt' in import_lines
        assert 'def bar():' in merged
