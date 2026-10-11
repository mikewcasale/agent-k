"""Tests for LightGBM loss parameter evolution and normalization.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import random

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from agent_k.evolution.loss import (
    ALPHA_BOUNDS,
    CUSTOM_ASYMMETRIC_OBJECTIVE,
    LossGenome,
    apply_lightgbm_objective,
    build_lightgbm_objective,
    build_lightgbm_objective_params,
    normalize_lightgbm_loss_params,
    objective_alpha_bounds,
    sample_objective_alpha,
)

__all__ = ()

_BOOSTER_KWARGS = {"n_estimators": 40, "random_state": 0, "verbosity": -1}


@pytest.fixture(scope="module")
def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the bundled diabetes regression dataset."""
    features, target = load_diabetes(return_X_y=True)
    return features, target


class TestNormalizeLossParams:
    """Tests for making ignored LightGBM loss parameters effective."""

    def test_quantile_without_alpha_gets_explicit_alpha(self) -> None:
        """Quantile must not inherit LightGBM's 0.9 default."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="quantile", n_estimators=100)\n'
        updated, notes = normalize_lightgbm_loss_params(code)

        assert "alpha=0.5" in updated
        assert notes == ["set alpha=0.5 for objective=quantile"]

    def test_params_dict_gets_explicit_alpha(self) -> None:
        """Booster param dicts are normalized as well as estimator keywords."""
        code = 'import lightgbm as lgb\nparams = {"objective": "quantile", "num_leaves": 31}\n'
        updated, _ = normalize_lightgbm_loss_params(code)

        assert '"alpha": 0.5' in updated

    @pytest.mark.parametrize(("objective", "fictional"), [("huber", "huber_delta"), ("quantile", "quantile_alpha")])
    def test_fictional_keys_are_renamed_to_alpha(self, objective: str, fictional: str) -> None:
        """LightGBM has no huber_delta/quantile_alpha parameter and ignores both."""
        code = f'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="{objective}", {fictional}=0.4)\n'
        updated, notes = normalize_lightgbm_loss_params(code)

        assert "alpha=0.4" in updated
        assert fictional not in updated
        assert notes == [f"renamed {fictional} to alpha for objective={objective}"]

    def test_existing_alpha_is_preserved(self) -> None:
        """An explicit alpha is left alone."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="huber", alpha=3.0)\n'

        assert normalize_lightgbm_loss_params(code) == (code, [])

    def test_objectives_without_alpha_are_untouched(self) -> None:
        """Only huber and quantile read alpha."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="regression_l1")\n'

        assert normalize_lightgbm_loss_params(code) == (code, [])

    def test_callable_objective_is_untouched(self) -> None:
        """An evolved custom objective keeps its callable."""
        code = "import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective=my_loss)\n"

        assert normalize_lightgbm_loss_params(code) == (code, [])

    def test_non_lightgbm_alpha_is_untouched(self) -> None:
        """Ridge/Lasso alpha means regularization strength, not a loss parameter."""
        code = "from sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)\n"

        assert normalize_lightgbm_loss_params(code) == (code, [])

    def test_multiline_call_and_params_dict_in_one_file(self) -> None:
        """Generated solutions use both the estimator and the booster API."""
        code = (
            "import lightgbm as lgb\n\n"
            "model = lgb.LGBMRegressor(\n"
            "    n_estimators=100,\n"
            '    objective="quantile",\n'
            ")\n"
            "params = {\n"
            '    "objective": "huber",\n'
            '    "num_leaves": 31,\n'
            "}\n"
        )
        updated, notes = normalize_lightgbm_loss_params(code)

        assert "alpha=0.5" in updated
        assert '"alpha": 1.0' in updated
        assert len(notes) == 2
        compile(updated, "<candidate>", "exec")

    def test_unparseable_code_is_returned_unchanged(self) -> None:
        """Mid-mutation code may not parse; normalization must not raise."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="quantile"\n'

        assert normalize_lightgbm_loss_params(code) == (code, [])


class TestApplyObjective:
    """Tests for coupled objective/alpha rewrites."""

    def test_switching_objective_rewrites_stale_alpha(self) -> None:
        """A huber delta of 2.0 is an invalid quantile level and aborts training."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="huber", alpha=2.0)\n'
        updated, notes = apply_lightgbm_objective(code, "quantile", 0.35)

        assert 'objective="quantile"' in updated
        assert "alpha=0.35" in updated
        assert notes == ["objective=quantile alpha=0.35"]

    def test_switching_objective_injects_missing_alpha(self) -> None:
        """An objective that needs alpha gets one even when the source had none."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="regression")\n'
        updated, _ = apply_lightgbm_objective(code, "huber", 1.5)

        assert 'objective="huber", alpha=1.5' in updated

    def test_alpha_free_objective_leaves_alpha_alone(self) -> None:
        """Nothing to couple when the new objective ignores alpha."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="quantile", alpha=0.3)\n'
        updated, _ = apply_lightgbm_objective(code, "regression_l1", None)

        assert 'objective="regression_l1"' in updated
        assert "alpha=0.3" in updated

    def test_code_without_lightgbm_is_unchanged(self) -> None:
        """Nothing to rewrite outside a LightGBM parameter site."""
        code = "from sklearn.linear_model import Ridge\nmodel = Ridge()\n"

        assert apply_lightgbm_objective(code, "huber", 1.0) == (code, [])


class TestAlphaBounds:
    """Tests for objective-aware alpha ranges."""

    @pytest.mark.parametrize("objective", ["quantile", "huber"])
    def test_bounds_follow_the_declared_objective(self, objective: str) -> None:
        """Quantile and huber have different valid alpha ranges."""
        code = f'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="{objective}", alpha=0.5)\n'

        assert objective_alpha_bounds(code) == ALPHA_BOUNDS[objective]

    def test_no_bounds_without_an_alpha_objective(self) -> None:
        """Callers fall back to their generic bounds."""
        code = 'import lightgbm as lgb\nmodel = lgb.LGBMRegressor(objective="regression")\n'

        assert objective_alpha_bounds(code) is None

    def test_sampled_alpha_stays_inside_lightgbm_limits(self) -> None:
        """LightGBM hard-fails on a quantile alpha outside (0, 1)."""
        rng = random.Random(7)
        samples = [sample_objective_alpha("quantile", rng) for _ in range(50)]

        assert all(sample is not None and 0.0 < sample < 1.0 for sample in samples)
        assert sample_objective_alpha("regression", rng) is None


class TestObjectiveParams:
    """Tests that emitted params are parameters LightGBM actually reads."""

    def test_quantile_params_use_alpha(self) -> None:
        """quantile_alpha is not a LightGBM parameter."""
        params = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=0.3))

        assert params == {"objective": "quantile", "alpha": 0.3}

    def test_huber_params_use_alpha(self) -> None:
        """huber_delta is not a LightGBM parameter."""
        params = build_lightgbm_objective_params(LossGenome(objective="huber", huber_delta=2.5))

        assert params == {"objective": "huber", "alpha": 2.5}

    def test_plain_objective_emits_only_the_objective(self) -> None:
        """No alpha for objectives that ignore it."""
        assert build_lightgbm_objective_params(LossGenome(objective="regression_l1")) == {"objective": "regression_l1"}

    def test_quantile_alpha_is_clipped_into_range(self) -> None:
        """A drifted gene must not produce an alpha LightGBM rejects."""
        params = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=1.4))

        assert 0.0 < float(params["alpha"]) < 1.0

    def test_custom_objective_compiles_to_a_callable(self) -> None:
        """The asymmetric blend genes have no built-in LightGBM equivalent."""
        params = build_lightgbm_objective_params(LossGenome(objective=CUSTOM_ASYMMETRIC_OBJECTIVE))

        assert callable(params["objective"])
        assert "alpha" not in params


class TestLightGBMBehavior:
    """Tests that train real LightGBM models on the bundled diabetes dataset."""

    def test_explicit_alpha_beats_the_quantile_default(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Without alpha, quantile regression predicts the 90th percentile."""
        features, target = regression_data
        defaulted = lgb.LGBMRegressor(objective="quantile", **_BOOSTER_KWARGS).fit(features, target)
        normalized = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **_BOOSTER_KWARGS).fit(features, target)

        default_bias = float(np.mean(defaulted.predict(features) - target))
        normalized_bias = float(np.mean(normalized.predict(features) - target))

        assert default_bias > 20.0
        assert abs(normalized_bias) < abs(default_bias)

    @pytest.mark.parametrize("fictional", ["huber_delta", "quantile_alpha"])
    def test_fictional_parameters_are_silently_ignored(
        self, regression_data: tuple[np.ndarray, np.ndarray], fictional: str
    ) -> None:
        """This is why the normalizer renames them instead of tuning them."""
        features, target = regression_data
        baseline = lgb.LGBMRegressor(objective="quantile", **_BOOSTER_KWARGS).fit(features, target)
        with_fictional = lgb.LGBMRegressor(objective="quantile", **{fictional: 0.5}, **_BOOSTER_KWARGS).fit(
            features, target
        )

        assert np.allclose(baseline.predict(features), with_fictional.predict(features))

    def test_emitted_params_change_the_fitted_model(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """An evolved alpha has to move the fit, or the search is exploring nothing."""
        features, target = regression_data
        low = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=0.2))
        high = build_lightgbm_objective_params(LossGenome(objective="quantile", quantile_alpha=0.8))

        low_pred = lgb.LGBMRegressor(**low, **_BOOSTER_KWARGS).fit(features, target).predict(features)
        high_pred = lgb.LGBMRegressor(**high, **_BOOSTER_KWARGS).fit(features, target).predict(features)

        assert float(np.mean(high_pred)) > float(np.mean(low_pred))

    @pytest.mark.parametrize("blend", [0.0, 0.5, 1.0])
    def test_custom_objective_does_not_collapse_to_a_constant(
        self, regression_data: tuple[np.ndarray, np.ndarray], blend: float
    ) -> None:
        """Huber's exact hessian is 0 outside the delta band, which blocks every split."""
        features, target = regression_data
        genome = LossGenome(objective=CUSTOM_ASYMMETRIC_OBJECTIVE, mae_rmse_blend=blend, huber_delta=1.0)
        model = lgb.LGBMRegressor(objective=build_lightgbm_objective(genome), **_BOOSTER_KWARGS)

        predictions = model.fit(features, target).predict(features)

        assert float(np.std(predictions)) > 0.0
        assert float(np.corrcoef(predictions, target)[0, 1]) > 0.5

    def test_asymmetric_weight_penalizes_over_prediction(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Weighting the over-prediction side has to shift predictions down."""
        features, target = regression_data
        balanced = build_lightgbm_objective(LossGenome(objective=CUSTOM_ASYMMETRIC_OBJECTIVE, asymmetric_weight=1.0))
        skewed = build_lightgbm_objective(LossGenome(objective=CUSTOM_ASYMMETRIC_OBJECTIVE, asymmetric_weight=4.0))

        balanced_pred = lgb.LGBMRegressor(objective=balanced, **_BOOSTER_KWARGS).fit(features, target).predict(features)
        skewed_pred = lgb.LGBMRegressor(objective=skewed, **_BOOSTER_KWARGS).fit(features, target).predict(features)

        assert float(np.mean(skewed_pred)) < float(np.mean(balanced_pred))

    def test_custom_objective_returns_positive_hessians(self) -> None:
        """A non-positive hessian makes LightGBM emit a constant prediction."""
        objective = build_lightgbm_objective(LossGenome(objective=CUSTOM_ASYMMETRIC_OBJECTIVE, mae_rmse_blend=1.0))
        y_true = np.array([0.0, 10.0, -250.0, 1e6], dtype=np.float64)
        y_pred = np.array([0.0, 0.5, 300.0, -1e6], dtype=np.float64)

        gradient, hessian = objective(y_true, y_pred)

        assert np.all(hessian > 0.0)
        assert np.all(np.isfinite(gradient))
