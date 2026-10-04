"""Tests for dataset profiling and preprocessing hint generation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from agent_k.core.hints import (
    MISSING_PATTERN_CORR_THRESHOLD,
    MISSING_RATE_THRESHOLD,
    MissingPattern,
    _infer_missing_pattern as infer_missing_pattern,
    _pairwise_complete_correlations as pairwise_complete_correlations,
    _sample_values as sample_values,
    build_dataset_profile,
    build_dataset_profile_async,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

__all__ = ()

_ROW_COUNT = 2000


def _pairwise_reference(df: pd.DataFrame, target_columns: Iterable[str]) -> MissingPattern:
    """Correlate every column pair individually, the way the scan did originally.

    Kept as an independent oracle for the vectorised implementation: it walks the
    same decision tree but coerces and correlates one feature at a time.
    """
    if df.empty:
        return MissingPattern.MCAR

    target_name = next(iter(target_columns), None)
    target_series = df[target_name] if target_name and target_name in df.columns else None
    numeric_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    has_mar = False
    has_mnar = False
    for column in df.columns:
        missing_rate = float(df[column].isna().mean())
        if missing_rate < MISSING_RATE_THRESHOLD:
            continue
        indicator = df[column].isna().astype(int)
        if indicator.nunique() < 2:
            continue
        if target_series is not None and pd.api.types.is_numeric_dtype(target_series):
            corr = float(indicator.corr(pd.to_numeric(target_series, errors="coerce")))
            if abs(corr) >= MISSING_PATTERN_CORR_THRESHOLD:
                has_mnar = True
                continue
        for feature in numeric_features:
            if feature == column:
                continue
            corr = float(indicator.corr(pd.to_numeric(df[feature], errors="coerce")))
            if abs(corr) >= MISSING_PATTERN_CORR_THRESHOLD:
                has_mar = True
                break

    if has_mnar:
        return MissingPattern.MNAR
    if has_mar:
        return MissingPattern.MAR
    return MissingPattern.MCAR


def _base_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261004)
    driver = rng.normal(size=_ROW_COUNT)
    target = rng.normal(size=_ROW_COUNT)
    noise = rng.normal(size=_ROW_COUNT)
    return driver, target, noise


def _mcar_frame() -> pd.DataFrame:
    driver, target, noise = _base_columns()
    rng = np.random.default_rng(11)
    values = noise.copy()
    values[rng.random(_ROW_COUNT) < 0.2] = np.nan
    return pd.DataFrame({"driver": driver, "random_gap": values, "target": target})


def _mar_frame() -> pd.DataFrame:
    """Missingness depends on an observed feature, not on the target."""
    driver, target, noise = _base_columns()
    values = noise.copy()
    values[driver > 0.8] = np.nan
    return pd.DataFrame({"driver": driver, "driven_gap": values, "target": target})


def _mnar_frame() -> pd.DataFrame:
    """Missingness depends on the target itself."""
    driver, target, noise = _base_columns()
    values = noise.copy()
    values[target > 0.5] = np.nan
    return pd.DataFrame({"driver": driver, "target_gap": values, "target": target})


def _object_column_frame() -> pd.DataFrame:
    """A non-numeric column carries the missingness, so it is absent from the feature set."""
    driver, target, noise = _base_columns()
    labels = np.array(["present"] * len(driver), dtype=object)
    labels[driver > 0.8] = None
    return pd.DataFrame({"driver": driver, "label": labels, "noise": noise, "target": target})


@pytest.mark.parametrize(
    ("frame_factory", "expected"),
    [
        (_mcar_frame, MissingPattern.MCAR),
        (_mar_frame, MissingPattern.MAR),
        (_mnar_frame, MissingPattern.MNAR),
        (_object_column_frame, MissingPattern.MAR),
    ],
    ids=["mcar", "mar", "mnar", "object_column"],
)
def test_infer_missing_pattern_classifies_frames(
    frame_factory: Callable[[], pd.DataFrame], expected: MissingPattern
) -> None:
    assert infer_missing_pattern(frame_factory(), ["target"]) is expected


@pytest.mark.parametrize(
    "frame_factory",
    [_mcar_frame, _mar_frame, _mnar_frame, _object_column_frame],
    ids=["mcar", "mar", "mnar", "object_column"],
)
def test_infer_missing_pattern_matches_pairwise_reference(frame_factory: Callable[[], pd.DataFrame]) -> None:
    frame = frame_factory()
    assert infer_missing_pattern(frame, ["target"]) is _pairwise_reference(frame, ["target"])


def test_infer_missing_pattern_without_target_column() -> None:
    """With no usable target the scan may still find MAR via the feature set."""
    frame = _mar_frame().drop(columns=["target"])
    assert infer_missing_pattern(frame, []) is _pairwise_reference(frame, [])
    assert infer_missing_pattern(frame, ["absent"]) is MissingPattern.MAR


def test_infer_missing_pattern_ignores_below_threshold_missingness() -> None:
    driver, target, noise = _base_columns()
    values = noise.copy()
    # Mirror the MAR driver but leave the missing rate under the scan threshold.
    values[driver > 3.5] = np.nan
    frame = pd.DataFrame({"driver": driver, "driven_gap": values, "target": target})
    assert float(frame["driven_gap"].isna().mean()) < MISSING_RATE_THRESHOLD
    assert infer_missing_pattern(frame, ["target"]) is MissingPattern.MCAR


def test_infer_missing_pattern_empty_frame() -> None:
    assert infer_missing_pattern(pd.DataFrame(), ["target"]) is MissingPattern.MCAR


@pytest.mark.parametrize("scale", [1.0, 1e6], ids=["unit_scale", "large_scale"])
def test_pairwise_complete_correlations_matches_pandas(scale: float) -> None:
    """The closed form must agree with Series.corr pair by pair, including NaN cases."""
    rng = np.random.default_rng(4242)
    rows, n_indicators, n_features = 600, 7, 9

    indicators = (rng.random((rows, n_indicators)) < 0.35).astype(np.float64)
    features = rng.normal(loc=scale, scale=scale, size=(rows, n_features))
    features[rng.random((rows, n_features)) < 0.25] = np.nan
    features[:, 0] = scale  # Zero variance: pandas scores NaN.
    features[:, 1] = np.nan  # No complete observations at all.
    indicators[:, 0] = 1.0  # Constant indicator: zero variance on the other side.

    actual = pairwise_complete_correlations(indicators, features)

    assert actual.shape == (n_indicators, n_features)
    for i in range(n_indicators):
        for f in range(n_features):
            expected = pd.Series(indicators[:, i]).corr(pd.Series(features[:, f]))
            assert np.isclose(actual[i, f], expected, rtol=1e-9, atol=1e-12, equal_nan=True), (
                f"indicator {i} vs feature {f}: {actual[i, f]} != {expected}"
            )


def test_infer_missing_pattern_matches_reference_on_wide_frames() -> None:
    """Fuzz wide frames so the batched scan is checked against the pairwise oracle."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        rows, cols = 300, 9
        data: dict[str, np.ndarray] = {}
        driver = rng.normal(size=rows)
        for index in range(cols):
            # Mix independent missingness with missingness driven by `driver`, so
            # the frames land on MCAR, MAR and MNAR verdicts across seeds.
            values = rng.normal(loc=rng.choice([0.0, 1e5]), size=rows)
            if rng.random() < 0.5:
                values[driver > rng.uniform(-0.5, 1.0)] = np.nan
            else:
                values[rng.random(rows) < rng.uniform(0.0, 0.4)] = np.nan
            data[f"col{index}"] = values
        data["driver"] = driver
        # Alternate a driver-loaded target (drives MNAR verdicts) with an
        # independent one (leaves MAR/MCAR, which exercise the batched grid).
        loading = rng.uniform(0.3, 1.0) if seed % 2 else 0.0
        data["target"] = driver * loading + rng.normal(size=rows)
        frame = pd.DataFrame(data)

        assert infer_missing_pattern(frame, ["target"]) is _pairwise_reference(frame, ["target"]), f"seed={seed}"


@pytest.mark.parametrize(
    "values",
    [
        [],
        [None, None],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [3, 1, 2],
        list(range(50)),
        [None, 7, None, 7, 8, 9, 10, 11, 12],
        ["$1.50", "$2.50", "", "none", "x"],
        [0.1, 1e-20, 1e20, -0.0, 1234567.891, None],
        [1] * 40 + [2] * 40 + [3, 4, 5, 6],
    ],
    ids=["empty", "all_null", "single_value", "short", "many", "nulls", "strings", "floats", "late_variety"],
)
def test_sample_values_matches_full_column_scan(values: list[Any]) -> None:
    """The head-slice shortcut must return exactly what scanning the column returns."""
    series = pd.Series(values, dtype="object" if not values else None)
    expected = tuple(series.dropna().astype(str).unique().tolist()[:5]) if not series.empty else ()
    assert sample_values(series) == expected


def _write_competition_csvs(tmp_path: Path) -> tuple[Path, Path]:
    frame = _mar_frame()
    frame.insert(0, "id", np.arange(len(frame)))
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    frame.to_csv(train_path, index=False)
    frame.drop(columns=["target"]).to_csv(test_path, index=False)
    return train_path, test_path


async def test_build_dataset_profile_async_matches_sync(tmp_path: Path) -> None:
    train_path, test_path = _write_competition_csvs(tmp_path)

    expected = build_dataset_profile(train_path, test_path, None)
    actual = await build_dataset_profile_async(train_path, test_path, None)

    assert actual.to_dict() == expected.to_dict()


async def test_build_dataset_profile_async_keeps_event_loop_responsive(tmp_path: Path) -> None:
    """The profiling work must not hold the loop, or mission event streaming stalls."""
    train_path, test_path = _write_competition_csvs(tmp_path)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    try:
        profile = await build_dataset_profile_async(train_path, test_path, None)
    finally:
        task.cancel()

    assert profile.row_count == _ROW_COUNT
    # A synchronous call would return before the ticker was ever scheduled.
    assert ticks > 0
