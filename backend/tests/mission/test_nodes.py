"""Tests for the graph nodes.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import ast
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.core.types import MetricDirection
from agent_k.mission.nodes import (
    DiscoveryNode,
    EvolutionNode,
    PrototypeNode,
    ResearchNode,
    SubmissionNode,
    _evaluate_metric,
)

__all__ = ()

pytestmark = pytest.mark.anyio

ID_COLUMN = "row_id"
TARGET_COLUMN = "target"
STRATEGY_HINTS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "Train a LightGBM model with early stopping."),
    ("lgbm", "An lgbm baseline is a strong starting point."),
    ("linear", "Start from a linear model baseline."),
    ("gradient", "Use gradient boosting over engineered features."),
    ("default", "Explore the data before modelling."),
)


@dataclass
class _Research:
    """Minimal research payload consumed by the prototype generator."""

    strategy_recommendations: list[str] = field(default_factory=list)


def _competition(metric: EvaluationMetric, direction: MetricDirection) -> Competition:
    return Competition(
        id="prototype-fixture",
        title="Prototype fixture",
        competition_type=CompetitionType.PLAYGROUND,
        metric=metric,
        metric_direction=direction,
        deadline=datetime.now(UTC) + timedelta(days=30),
    )


def _write_competition_data(
    root: Path, *, rows: int = 160, train_only_column: bool = False, labelled: bool = False
) -> None:
    """Write a small competition dataset with a high-cardinality string id column.

    When ``labelled`` is set the target is categorical and includes a class with a
    single member, which is the shape that breaks stratified splitting.
    """
    rng = random.Random(11)
    train_header = [ID_COLUMN, "feat_num", "feat_cat", TARGET_COLUMN]
    if train_only_column:
        train_header.insert(3, "leak_only_in_train")
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(train_header)
        for index in range(rows):
            numeric = rng.gauss(0.0, 1.0)
            category = rng.choice(["alpha", "beta", "gamma"])
            if labelled:
                target = "rare" if index == 0 else ("high" if numeric > 0.0 else "low")
            else:
                target = f"{3.0 * numeric + (1.0 if category == 'alpha' else 0.0) + rng.gauss(0.0, 0.1):.6f}"
            row = [f"id_{index:06d}", f"{numeric:.6f}", category, target]
            if train_only_column:
                row.insert(3, f"{rng.random():.6f}")
            writer.writerow(row)

    test_ids = [f"id_{rows + index:06d}" for index in range(rows // 4)]
    with (root / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([ID_COLUMN, "feat_num", "feat_cat"])
        for test_id in test_ids:
            writer.writerow([test_id, f"{rng.gauss(0.0, 1.0):.6f}", rng.choice(["alpha", "beta", "gamma"])])

    with (root / "sample_submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([ID_COLUMN, TARGET_COLUMN])
        for test_id in test_ids:
            writer.writerow([test_id, 0.0])


def _generate(
    strategy: str, metric: EvaluationMetric = EvaluationMetric.RMSE, direction: MetricDirection = "minimize"
) -> str:
    return PrototypeNode()._generate_prototype(
        _competition(metric, direction),
        _Research(strategy_recommendations=[strategy]),
        target_columns=[TARGET_COLUMN],
        train_target_columns=[TARGET_COLUMN],
        id_column=ID_COLUMN,
    )


def _run_prototype(root: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Execute generated prototype code and record the features the model was fitted on."""
    epilogue = dedent(
        """
        import json
        from pathlib import Path

        fitted = list(clf.named_steps["preprocessor"].feature_names_in_)
        Path("fitted_features.json").write_text(json.dumps(fitted), encoding="utf-8")
        """
    )
    (root / "solution.py").write_text(f"{code}\n{epilogue}", encoding="utf-8")
    return subprocess.run(
        [sys.executable, "solution.py"], cwd=root, capture_output=True, text=True, timeout=600, check=False
    )


class TestDiscoveryNode:
    """Tests for the DiscoveryNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = DiscoveryNode()
        assert node is not None


class TestResearchNode:
    """Tests for the ResearchNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = ResearchNode()
        assert node is not None


class TestPrototypeNode:
    """Tests for the PrototypeNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = PrototypeNode()
        assert node is not None


class TestEvolutionNode:
    """Tests for the EvolutionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = EvolutionNode()
        assert node is not None


class TestSubmissionNode:
    """Tests for the SubmissionNode."""

    def test_node_creation(self) -> None:
        """Node should be creatable."""
        node = SubmissionNode()
        assert node is not None


class TestGeneratePrototype:
    """Tests for the generated baseline prototype."""

    @pytest.mark.parametrize(("label", "strategy"), STRATEGY_HINTS, ids=[hint[0] for hint in STRATEGY_HINTS])
    def test_every_model_branch_is_valid_python(self, label: str, strategy: str) -> None:
        """Each model branch must emit a parseable module."""
        code = _generate(strategy)
        ast.parse(code)
        assert label

    @pytest.mark.parametrize(
        ("metric", "direction", "labelled"),
        [(EvaluationMetric.RMSE, "minimize", False), (EvaluationMetric.ACCURACY, "maximize", True)],
        ids=["regression", "classification"],
    )
    def test_lightgbm_branch_runs_end_to_end(
        self, tmp_path: Path, metric: EvaluationMetric, direction: MetricDirection, labelled: bool
    ) -> None:
        """The LightGBM branch must train and write a submission.

        The classification case carries a single-member class, which is the shape
        that made stratified splitting abort the whole prototype.
        """
        _write_competition_data(tmp_path, labelled=labelled)
        result = _run_prototype(tmp_path, _generate("Train a LightGBM model", metric, direction))

        assert result.returncode == 0, result.stderr
        submission = tmp_path / "submission.csv"
        assert submission.exists()

        with submission.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (tmp_path / "sample_submission.csv").open(newline="", encoding="utf-8") as handle:
            expected_ids = [row[ID_COLUMN] for row in csv.DictReader(handle)]
        assert [row[ID_COLUMN] for row in rows] == expected_ids
        assert all(row[TARGET_COLUMN] not in ("", None) for row in rows)

    def test_id_column_is_not_used_as_a_feature(self, tmp_path: Path) -> None:
        """The submission id must never reach the model as a feature."""
        _write_competition_data(tmp_path)
        result = _run_prototype(tmp_path, _generate("Use a linear model"))

        assert result.returncode == 0, result.stderr
        fitted = json.loads((tmp_path / "fitted_features.json").read_text(encoding="utf-8"))
        assert ID_COLUMN not in fitted
        assert set(fitted) == {"feat_num", "feat_cat"}

    def test_train_only_columns_are_dropped(self, tmp_path: Path) -> None:
        """Columns missing from test.csv must not be fitted on."""
        _write_competition_data(tmp_path, train_only_column=True)
        result = _run_prototype(tmp_path, _generate("Use gradient boosting"))

        assert result.returncode == 0, result.stderr
        fitted = json.loads((tmp_path / "fitted_features.json").read_text(encoding="utf-8"))
        assert "leak_only_in_train" not in fitted
        assert set(fitted) == {"feat_num", "feat_cat"}


class TestEvaluateMetric:
    """Tests for metric evaluation helpers."""

    def test_rmsle_ignores_negative_values(self) -> None:
        """RMSLE should ignore negative targets in the denominator."""
        score = _evaluate_metric(EvaluationMetric.RMSLE, [1.0, -1.0], prediction=0.0)
        assert score == pytest.approx(math.log1p(1.0))
