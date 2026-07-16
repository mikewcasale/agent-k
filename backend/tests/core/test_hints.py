"""Tests for column-type detection heuristics in ``agent_k.core.hints``.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pandas as pd
import pytest

from agent_k.core.hints import (
    ColumnType,
    _detect_column_type,
    _is_geo_lat,
    _is_geo_lon,
    _is_ordinal_name,
    _is_price_column,
    _match_geo_pairs,
    _profile_column,
    _strip_geo_token,
)

__all__ = ()


class TestGeoLatDetection:
    """``_is_geo_lat`` must only match column names that carry ``lat``/``latitude`` as a whole word."""

    @pytest.mark.parametrize("name", ["lat", "latitude", "pickup_lat", "start_latitude", "lat_deg", "geo.lat", "LAT"])
    def test_matches_true_latitude_columns(self, name: str) -> None:
        assert _is_geo_lat(name.lower())

    @pytest.mark.parametrize(
        "name",
        [
            "latency_ms",
            "latency",
            "platform",
            "platinum",
            "escalation",
            "translate",
            "related_products",
            "flat_area",
            "collateral",
        ],
    )
    def test_ignores_names_that_only_contain_the_substring(self, name: str) -> None:
        assert not _is_geo_lat(name)


class TestGeoLonDetection:
    """``_is_geo_lon`` must only match column names that carry ``lon``/``lng``/``longitude`` as a whole word."""

    @pytest.mark.parametrize("name", ["lon", "lng", "longitude", "pickup_lon", "dropoff_lng", "end_longitude", "LON"])
    def test_matches_true_longitude_columns(self, name: str) -> None:
        assert _is_geo_lon(name.lower())

    @pytest.mark.parametrize(
        "name",
        ["belongs_to", "belongings", "along_path", "salon_id", "balloon_count", "colony", "colonial", "clone_source"],
    )
    def test_ignores_names_that_only_contain_the_substring(self, name: str) -> None:
        assert not _is_geo_lon(name)


class TestPriceDetection:
    """``_is_price_column`` must only fire on real price/currency-shaped column names."""

    @pytest.mark.parametrize(
        "name", ["price", "unit_cost", "fare_total", "salary_usd", "income_bracket", "amount_paid", "sale_value"]
    )
    def test_matches_true_price_columns(self, name: str) -> None:
        assert _is_price_column(name, sample_values=())

    @pytest.mark.parametrize(
        "name", ["valuation_method", "revalue", "diagnostic_code", "used_by", "revaluated_at", "salvageable"]
    )
    def test_ignores_names_that_only_contain_the_substring(self, name: str) -> None:
        assert not _is_price_column(name, sample_values=())

    def test_currency_symbol_in_sample_values_still_wins(self) -> None:
        assert _is_price_column("mystery_col", sample_values=("$4.20", "$0.99"))


class TestOrdinalDetection:
    """``_is_ordinal_name`` must only fire on real ordinal-shaped column names."""

    @pytest.mark.parametrize(
        "name", ["rank", "grade", "level", "sort_order", "user_rating", "stage_num", "ordinal_position"]
    )
    def test_matches_true_ordinal_columns(self, name: str) -> None:
        assert _is_ordinal_name(name)

    @pytest.mark.parametrize(
        "name", ["frankfurt", "borders_shared", "operating_hours", "berating_score", "levelled_terrain", "graderoot"]
    )
    def test_ignores_names_that_only_contain_the_substring(self, name: str) -> None:
        assert not _is_ordinal_name(name)


class TestDetectColumnTypeRegression:
    """End-to-end regression: substring collisions must not force GEO/PRICE classifications."""

    def test_latency_ms_stays_numeric(self) -> None:
        column_type = _detect_column_type(
            name="latency_ms",
            is_numeric=True,
            is_bool=False,
            is_datetime=False,
            unique_count=500,
            unique_ratio=0.4,
            avg_length=None,
            sample_values=("120", "230", "440"),
            is_target=False,
        )
        assert column_type == ColumnType.NUMERIC_CONTINUOUS

    def test_belongs_to_stays_categorical(self) -> None:
        column_type = _detect_column_type(
            name="belongs_to",
            is_numeric=False,
            is_bool=False,
            is_datetime=False,
            unique_count=5,
            unique_ratio=0.005,
            avg_length=6.0,
            sample_values=("group_a", "group_b"),
            is_target=False,
        )
        assert column_type == ColumnType.CATEGORICAL_LOW_CARDINALITY

    def test_latitude_column_still_detected(self) -> None:
        column_type = _detect_column_type(
            name="pickup_latitude",
            is_numeric=True,
            is_bool=False,
            is_datetime=False,
            unique_count=1000,
            unique_ratio=0.5,
            avg_length=None,
            sample_values=("40.712", "40.750"),
            is_target=False,
        )
        assert column_type == ColumnType.GEOGRAPHIC_LAT

    def test_longitude_column_still_detected(self) -> None:
        column_type = _detect_column_type(
            name="pickup_longitude",
            is_numeric=True,
            is_bool=False,
            is_datetime=False,
            unique_count=1000,
            unique_ratio=0.5,
            avg_length=None,
            sample_values=("-74.006", "-74.010"),
            is_target=False,
        )
        assert column_type == ColumnType.GEOGRAPHIC_LON


class TestProfileColumnRegression:
    """``_profile_column`` glues detection to a DataFrame column; make sure the fix flows through."""

    def test_latency_column_classified_as_numeric_continuous(self) -> None:
        series = pd.Series([12, 34, 56, 78, 90, 120, 200, 340, 12, 88] * 10, name="latency_ms")
        profile = _profile_column("latency_ms", series, target_columns=())
        assert profile.column_type in {ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE}

    def test_true_latitude_column_classified_as_geo_lat(self) -> None:
        series = pd.Series([40.7 + (i % 10) * 0.001 for i in range(200)], name="pickup_latitude")
        profile = _profile_column("pickup_latitude", series, target_columns=())
        assert profile.column_type == ColumnType.GEOGRAPHIC_LAT


class TestStripGeoToken:
    """``_strip_geo_token`` must strip only whole-word tokens, not any substring collision."""

    def test_strips_underscore_token_at_end(self) -> None:
        assert _strip_geo_token("pickup_lat") == "pickup"
        assert _strip_geo_token("pickup_lon") == "pickup"

    def test_strips_longitude_and_latitude_prefixes(self) -> None:
        assert _strip_geo_token("start_latitude") == "start"
        assert _strip_geo_token("start_longitude") == "start"

    def test_does_not_mangle_names_containing_substrings(self) -> None:
        assert _strip_geo_token("latency_ms") == "latencyms"
        assert _strip_geo_token("belongs_to") == "belongsto"


class TestMatchGeoPairs:
    """``_match_geo_pairs`` should only surface pairs when detection is correct."""

    def test_no_pairs_when_only_false_positives(self) -> None:
        latency_profile = _profile_column(
            "latency_ms", pd.Series([10, 20, 30] * 40, name="latency_ms"), target_columns=()
        )
        belongs_profile = _profile_column(
            "belongs_to", pd.Series(["a", "b", "c"] * 40, name="belongs_to"), target_columns=()
        )
        pairs = _match_geo_pairs({"latency_ms": latency_profile, "belongs_to": belongs_profile})
        assert pairs == []

    def test_matched_pair_on_real_geo_columns(self) -> None:
        lat = pd.Series([40.7 + (i % 10) * 0.001 for i in range(200)], name="pickup_latitude")
        lon = pd.Series([-74.0 + (i % 10) * 0.001 for i in range(200)], name="pickup_longitude")
        columns = {
            "pickup_latitude": _profile_column("pickup_latitude", lat, target_columns=()),
            "pickup_longitude": _profile_column("pickup_longitude", lon, target_columns=()),
        }
        pairs = _match_geo_pairs(columns)
        assert pairs == [("pickup_latitude", "pickup_longitude")]
