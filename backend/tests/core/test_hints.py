"""Regression tests for hint detection regex patterns.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pytest

from agent_k.core.hints import HintCategory, PreprocessingHint, detect_applied_hints

__all__ = ()


def _make_hint(
    hint_id: str, snippet: str, *, category: HintCategory = HintCategory.FEATURE_ENGINEERING
) -> PreprocessingHint:
    return PreprocessingHint(
        id=hint_id,
        category=category,
        priority=0.5,
        applicable_columns=[],
        description="test hint",
        code_snippet=snippet,
        success_rate=0.0,
        last_attempted=None,
        last_result=None,
    )


class TestDetectAppliedHintsSignatures:
    """Signature-based detection must find class names in real code (word-boundary regex)."""

    def test_signature_matches_encoder_class(self) -> None:
        hint = _make_hint(
            "onehot_low_cardinality",
            "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder(handle_unknown='ignore')",
            category=HintCategory.ENCODING,
        )
        code = "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder(handle_unknown='ignore')\nX = enc.fit_transform(df)"
        applied = detect_applied_hints(code, [hint])
        assert "onehot_low_cardinality" in applied

    def test_signature_matches_transformer_class(self) -> None:
        hint = _make_hint(
            "numeric_skew_transform",
            "from sklearn.preprocessing import PowerTransformer\npt = PowerTransformer(method='yeo-johnson')",
            category=HintCategory.TRANSFORM,
        )
        code = "pt = PowerTransformer(method='yeo-johnson')\nX = pt.fit_transform(X)"
        applied = detect_applied_hints(code, [hint])
        assert "numeric_skew_transform" in applied

    def test_signature_does_not_match_unrelated_code(self) -> None:
        hint = _make_hint(
            "onehot_low_cardinality",
            "from sklearn.preprocessing import OneHotEncoder\nenc = OneHotEncoder()",
            category=HintCategory.ENCODING,
        )
        code = "from sklearn.preprocessing import LabelEncoder\nenc = LabelEncoder()"
        applied = detect_applied_hints(code, [hint])
        assert "onehot_low_cardinality" not in applied

    def test_signature_respects_word_boundary(self) -> None:
        # Signature detection must use \b (word boundary), not the 2-char literal.
        # Use a hint whose snippet has no fallback substring in _hint_patterns so
        # only the signature path can flip the flag.
        hint = _make_hint("custom_signature_hint", "def MySpecialTransformer(x):\n    return x\n")
        code_hit = "MySpecialTransformer(df)"
        code_miss = "MySpecialTransformerX(df)"
        assert "custom_signature_hint" in detect_applied_hints(code_hit, [hint])
        assert "custom_signature_hint" not in detect_applied_hints(code_miss, [hint])


class TestDetectAppliedHintsPatterns:
    """Per-hint fallback patterns must compile and match realistic code."""

    def test_timeseries_lag_features_pattern_compiles_and_matches_shift(self) -> None:
        # Previously r"shift\\(" raised re.error at compile time — this triggered
        # whenever a datetime-column dataset emitted the timeseries_lag_features hint,
        # crashing every mutation in EvolverAgent for time-series competitions.
        hint = _make_hint(
            "timeseries_lag_features",
            'df["lag_1"] = series.shift(1)\ndf["rolling_mean"] = series.rolling(7).mean()',
            category=HintCategory.TIME_SERIES,
        )
        code = 'df["lag_1"] = df["target"].shift(1)\n'
        applied = detect_applied_hints(code, [hint])
        assert "timeseries_lag_features" in applied

    def test_timeseries_lag_features_pattern_matches_rolling(self) -> None:
        hint = _make_hint(
            "timeseries_lag_features",
            'df["lag_1"] = series.shift(1)\ndf["rolling_mean"] = series.rolling(7).mean()',
            category=HintCategory.TIME_SERIES,
        )
        code = 'df["roll"] = df["y"].rolling(window=7).mean()\n'
        applied = detect_applied_hints(code, [hint])
        assert "timeseries_lag_features" in applied

    def test_lightgbm_custom_rmsle_pattern_matches_objective_kwarg(self) -> None:
        hint = _make_hint(
            "lightgbm_custom_rmsle",
            "def rmsle_objective(y_true, y_pred):\n    return grad, hess",
            category=HintCategory.MODEL_OPTIMIZATION,
        )
        code = "model = lgb.train(objective=rmsle, train_set=ds)"
        applied = detect_applied_hints(code, [hint])
        assert "lightgbm_custom_rmsle" in applied

    def test_remove_collinear_pattern_matches_np_triu(self) -> None:
        hint = _make_hint("remove_collinear", "corr = df.corr().abs()", category=HintCategory.FEATURE_SELECTION)
        code = "upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))"
        applied = detect_applied_hints(code, [hint])
        assert "remove_collinear" in applied

    def test_price_log_transform_pattern_matches_np_log(self) -> None:
        hint = _make_hint("price_log_transform", 'df["price"] = np.log1p(df["price"])', category=HintCategory.TRANSFORM)
        code = 'df["price"] = np.log(df["price"] + 1)'
        applied = detect_applied_hints(code, [hint])
        assert "price_log_transform" in applied


class TestDetectAppliedHintsCommentMarker:
    """The explicit ``# Applied hint:`` comment marker must always be respected."""

    def test_comment_marker_recognized(self) -> None:
        hint = _make_hint("onehot_low_cardinality", "enc = OneHotEncoder()", category=HintCategory.ENCODING)
        code = "# Applied hint: onehot_low_cardinality\nX = pd.DataFrame()"
        applied = detect_applied_hints(code, [hint])
        assert "onehot_low_cardinality" in applied


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.kaggle.com/code/alice/great-notebook", "alice/great-notebook"),
        ("https://www.kaggle.com/alice/great-notebook", "alice/great-notebook"),
        ("https://kaggle.com/code/bob/xyz?scriptVersionId=123", "bob/xyz"),
        ("https://www.kaggle.com/code/carol/nb#comments", "carol/nb"),
    ],
)
def test_extract_kernel_ref_parses_real_urls(url: str, expected: str) -> None:
    # Previously r"kaggle\\.com" required a literal backslash in the URL, so this
    # helper always returned None — silently degrading ScientistAgent kernel analysis.
    from agent_k.agents.scientist import ScientistAgent

    agent = ScientistAgent.__new__(ScientistAgent)
    assert agent._extract_kernel_ref(url) == expected


def test_extract_kernel_ref_returns_none_for_non_kaggle_url() -> None:
    from agent_k.agents.scientist import ScientistAgent

    agent = ScientistAgent.__new__(ScientistAgent)
    assert agent._extract_kernel_ref("https://example.com/alice/nb") is None
