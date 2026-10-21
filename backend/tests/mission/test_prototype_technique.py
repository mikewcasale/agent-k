"""Tests for baseline technique resolution and prototype code generation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import csv
import subprocess
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from agent_k.agents.scientist import ResearchReport
from agent_k.core.models import Competition, CompetitionType, EvaluationMetric
from agent_k.mission.nodes import PrototypeNode, _BaselineTechnique, _resolve_baseline_technique

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


def _competition(metric: EvaluationMetric) -> Competition:
    return Competition(
        id="c-1",
        title="Test Competition",
        competition_type=CompetitionType.FEATURED,
        metric=metric,
        deadline=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _generate(metric: EvaluationMetric, approaches: list[str], *, target: str = "target") -> str:
    node = PrototypeNode()
    return node._generate_prototype(
        _competition(metric),
        ResearchReport(competition_id="c-1", recommended_approaches=approaches),
        target_columns=[target],
        train_target_columns=[target],
        id_column="id",
    )


class TestResolveBaselineTechnique:
    """Technique resolution from free-text research strategy."""

    @pytest.mark.parametrize(
        "text",
        [
            "Use tree ensembles to capture non-linear feature interactions",
            "Model the nonlinear structure in the features",
            "A deep network with non-linearities",
            "Account for non linear effects",
        ],
    )
    def test_negated_linear_does_not_select_linear(self, text: str) -> None:
        """A recommendation to model non-linear structure must not pick a linear model."""
        assert _resolve_baseline_technique(text) is not _BaselineTechnique.LINEAR

    @pytest.mark.parametrize(
        "text",
        [
            "Start with a linear baseline, then XGBoost",
            "Gradient boosting with regularization; avoid linear models",
            "LightGBM with early stopping",
            "Use LGBM and tune num_leaves",
            "CatBoost handles categoricals natively",
            "A GBDT is the strongest tabular learner here",
        ],
    )
    def test_boosting_wins_over_weaker_families(self, text: str) -> None:
        """When several families are mentioned the strongest tabular learner wins."""
        assert _resolve_baseline_technique(text) is _BaselineTechnique.GRADIENT_BOOSTING

    @pytest.mark.parametrize("text", ["Random forest and extra trees", "Bagging over decision trees"])
    def test_forest_selected_when_only_family(self, text: str) -> None:
        """Forest ensembles are selected when no boosting method is mentioned."""
        assert _resolve_baseline_technique(text) is _BaselineTechnique.RANDOM_FOREST

    @pytest.mark.parametrize(
        "text",
        ["Ridge regression on standardized features", "Logistic regression baseline", "Elastic net for sparse signals"],
    )
    def test_linear_selected_when_only_family(self, text: str) -> None:
        """Linear models are selected only when they are the sole recommendation."""
        assert _resolve_baseline_technique(text) is _BaselineTechnique.LINEAR

    @pytest.mark.parametrize("text", ["", "No specific guidance provided", "Explore the data carefully"])
    def test_defaults_to_gradient_boosting(self, text: str) -> None:
        """Absent guidance the baseline defaults to the preferred gradient-boosting learner."""
        assert _resolve_baseline_technique(text) is _BaselineTechnique.GRADIENT_BOOSTING


class TestGeneratedPrototypeModel:
    """The emitted prototype code reflects the resolved technique."""

    def test_default_emits_lightgbm(self) -> None:
        """With no guidance the prototype builds LightGBM, not a forest."""
        code = _generate(EvaluationMetric.RMSE, [])

        assert "LGBMRegressor" in code
        assert "RandomForestRegressor" not in code

    def test_non_linear_text_emits_lightgbm(self) -> None:
        """'non-linear' guidance must not emit a linear model."""
        code = _generate(EvaluationMetric.RMSE, ["Capture non-linear interactions between features"])

        assert "LGBMRegressor" in code
        assert "LinearRegression(" not in code

    def test_linear_regression_receives_no_random_state(self) -> None:
        """LinearRegression rejects random_state, so it must not be passed."""
        code = _generate(EvaluationMetric.RMSE, ["Ridge regression on standardized features"])

        assert "LinearRegression()" in code
        assert "LinearRegression(random_state" not in code

    def test_logistic_regression_keeps_random_state(self) -> None:
        """LogisticRegression accepts random_state and stays deterministic."""
        code = _generate(EvaluationMetric.ACCURACY, ["Logistic regression baseline"])

        assert "LogisticRegression(random_state=42)" in code


def _write_competition_data(directory: Path, *, target_values: list[str], target: str = "target") -> None:
    rows = len(target_values)
    train_header = ["id", "feature_num", "feature_cat", target]
    train_rows = [
        [str(idx), str(idx * 1.5 + (idx % 3)), ["a", "b", "c"][idx % 3], target_values[idx]] for idx in range(rows)
    ]
    test_rows = [[str(rows + idx), str(idx * 2.0), ["a", "b", "c"][idx % 3]] for idx in range(6)]

    for name, header, data in (
        ("train.csv", train_header, train_rows),
        ("test.csv", ["id", "feature_num", "feature_cat"], test_rows),
        ("sample_submission.csv", ["id", target], [[row[0], "0"] for row in test_rows]),
    ):
        with (directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(data)


@pytest.mark.parametrize(
    ("approaches", "metric", "target_values"),
    [
        ([], EvaluationMetric.RMSE, [f"{idx * 1.25 + 3:.3f}" for idx in range(40)]),
        (["Capture non-linear interactions"], EvaluationMetric.RMSE, [f"{idx * 0.75:.3f}" for idx in range(40)]),
        (
            ["Ridge regression on standardized features"],
            EvaluationMetric.RMSE,
            [f"{idx * 2.0:.3f}" for idx in range(40)],
        ),
        (["Random forest and extra trees"], EvaluationMetric.ACCURACY, [str(idx % 2) for idx in range(40)]),
        (["Logistic regression baseline"], EvaluationMetric.ACCURACY, [str(idx % 2) for idx in range(40)]),
        (["LightGBM with early stopping"], EvaluationMetric.AUC, [str(idx % 2) for idx in range(40)]),
    ],
)
def test_generated_prototype_runs_and_writes_submission(
    tmp_path: Path, approaches: list[str], metric: EvaluationMetric, target_values: list[str]
) -> None:
    """Every technique branch must emit code that runs and produces a submission."""
    _write_competition_data(tmp_path, target_values=target_values)
    code = _generate(metric, approaches)
    script = tmp_path / "solution.py"
    script.write_text(code, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "solution.py"], cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False
    )

    assert result.returncode == 0, f"prototype failed:\n{result.stdout}\n{result.stderr}"
    submission = tmp_path / "submission.csv"
    assert submission.exists()
    with submission.open("r", encoding="utf-8", newline="") as handle:
        written = list(csv.reader(handle))
    assert written[0] == ["id", "target"]
    assert len(written) == 7
