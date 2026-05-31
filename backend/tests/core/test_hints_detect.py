"""Tests for hint detection regex correctness.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from agent_k.core.hints import HintCategory, PreprocessingHint, _hint_patterns, detect_applied_hints


def _make_hint(
    hint_id: str, snippet: str = "", *, category: HintCategory = HintCategory.FEATURE_ENGINEERING
) -> PreprocessingHint:
    return PreprocessingHint(
        id=hint_id,
        category=category,
        priority=0.5,
        applicable_columns=[],
        description="",
        code_snippet=snippet,
        success_rate=0.0,
        last_attempted=None,
        last_result=None,
    )


def test_detect_applied_hints_matches_function_signature() -> None:
    """Signature extracted from a hint snippet should be detected via word boundary."""
    hint = _make_hint(
        "custom_haversine",
        snippet="def haversine(lat1, lon1, lat2, lon2):\n    return 0",
        category=HintCategory.GEOGRAPHIC,
    )
    code = "def main():\n    d = haversine(0.0, 0.0, 1.0, 1.0)\n    return d\n"

    assert detect_applied_hints(code, [hint]) == {"custom_haversine"}


def test_detect_applied_hints_matches_class_signature() -> None:
    """Class names ending in Encoder/Regressor/etc. should match in attribute access too."""
    hint = _make_hint(
        "use_encoder",
        snippet="from sklearn.preprocessing import OneHotEncoder\nencoder = OneHotEncoder()",
        category=HintCategory.ENCODING,
    )
    code = "import sklearn.preprocessing as p\nenc = p.OneHotEncoder(handle_unknown='ignore')\n"

    assert detect_applied_hints(code, [hint]) == {"use_encoder"}


def test_detect_applied_hints_signature_word_boundary_avoids_substring() -> None:
    """A hint signature must match as a whole word, not as a substring."""
    hint = _make_hint("narrow_match", snippet="def encode(value):\n    return value")
    code = "def main():\n    return reencode(x)\n"

    assert detect_applied_hints(code, [hint]) == set()


def test_hint_patterns_compile_for_timeseries_lag_features() -> None:
    r"""`timeseries_lag_features` previously had a malformed `shift\\(` regex that raised on compile."""
    hint = _make_hint("timeseries_lag_features")
    patterns = _hint_patterns(hint)

    assert len(patterns) == 2
    assert any(p.search("df.shift(1)") for p in patterns)
    assert any(p.search("series.rolling(7).mean()") for p in patterns)


def test_hint_patterns_lightgbm_custom_rmsle_matches_assignment() -> None:
    r"""Assignment-form pattern for objective= now uses real `\s` whitespace classes."""
    hint = _make_hint("lightgbm_custom_rmsle")
    patterns = _hint_patterns(hint)

    assert any(p.search("def rmsle_objective(y_pred, train_data): ...") for p in patterns)
    assert any(p.search("objective = rmsle_loss") for p in patterns)


def test_hint_patterns_remove_collinear_matches_np_triu() -> None:
    """`np.triu` reference is now matched with a real escaped dot."""
    hint = _make_hint("remove_collinear")
    patterns = _hint_patterns(hint)

    assert any(p.search("upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))") for p in patterns)
