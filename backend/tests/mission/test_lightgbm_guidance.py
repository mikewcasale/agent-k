"""Tests for the LightGBM guidance injected into evolution prompts.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from agent_k.core.models import EvaluationMetric
from agent_k.core.strategy import ProblemProfile, ProblemType
from agent_k.mission.nodes import _build_lightgbm_guidance

__all__ = ()


def _profile(*, is_classification: bool) -> ProblemProfile:
    return ProblemProfile(
        problem_type=ProblemType.TABULAR_CLASSIFICATION if is_classification else ProblemType.TABULAR_REGRESSION,
        metric=EvaluationMetric.ACCURACY if is_classification else EvaluationMetric.RMSE,
        metric_direction="maximize" if is_classification else "minimize",
        target_columns=["target"],
        train_target_columns=["target"],
        id_column="id",
        uses_proba=is_classification,
        is_classification=is_classification,
    )


def test_regression_guidance_embeds_runnable_custom_objective() -> None:
    guidance = _build_lightgbm_guidance(_profile(is_classification=False))

    assert "def custom_objective(first, second):" in guidance
    assert "huber_delta" in guidance
    assert "boost_from_average" in guidance


def test_classification_guidance_omits_regression_objective() -> None:
    guidance = _build_lightgbm_guidance(_profile(is_classification=True))

    assert "def custom_objective" not in guidance
    assert "scale_pos_weight" in guidance
