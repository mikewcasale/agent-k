"""Tests for adaptive preprocessing hint utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import pandas as pd
import pytest

from agent_k.core.hints import (
    ColumnType,
    _is_geo_lat,
    _is_geo_lon,
    _is_ordinal_name,
    _is_price_column,
    _name_tokens,
    _profile_column,
    _strip_geo_token,
)

__all__ = ()


class TestNameTokens:
    """Verify column-name tokenization handles snake_case, kebab-case, and camelCase."""

    def test_snake_case_splits_on_underscore(self) -> None:
        assert _name_tokens("pickup_latitude") == frozenset({"pickup", "latitude"})

    def test_camel_case_splits_on_boundary(self) -> None:
        assert _name_tokens("pickupLatitude") == frozenset({"pickup", "latitude"})

    def test_pascal_case_splits_on_boundary(self) -> None:
        assert _name_tokens("PickupLatitude") == frozenset({"pickup", "latitude"})

    def test_acronym_boundary_is_preserved(self) -> None:
        # "HTTPStatus" should split at the boundary between the acronym and the following word.
        assert _name_tokens("HTTPStatus") == frozenset({"http", "status"})

    def test_kebab_and_dot_are_boundaries(self) -> None:
        assert _name_tokens("pickup-latitude.deg") == frozenset({"pickup", "latitude", "deg"})

    def test_bare_token_yields_single_entry(self) -> None:
        assert _name_tokens("lat") == frozenset({"lat"})


class TestGeoLatDetection:
    """Ensure only genuine latitude column names are flagged, not substrings."""

    @pytest.mark.parametrize(
        "name",
        [
            "lat",
            "Lat",
            "LAT",
            "latitude",
            "pickup_lat",
            "pickup_latitude",
            "pickupLatitude",
            "PickupLatitude",
            "lat_deg",
        ],
    )
    def test_positive(self, name: str) -> None:
        assert _is_geo_lat(name) is True

    @pytest.mark.parametrize(
        "name", ["template", "plate", "flat", "collate", "platform", "salary", "class_label", "elat_flag_column"]
    )
    def test_negative(self, name: str) -> None:
        assert _is_geo_lat(name) is False


class TestGeoLonDetection:
    """Ensure only genuine longitude column names are flagged, not substrings."""

    @pytest.mark.parametrize(
        "name",
        ["lon", "Lon", "lng", "longitude", "pickup_lon", "pickup_longitude", "pickupLongitude", "PickupLongitude"],
    )
    def test_positive(self, name: str) -> None:
        assert _is_geo_lon(name) is True

    @pytest.mark.parametrize("name", ["alone", "melon", "clone", "belong", "salon", "long_description", "colony_id"])
    def test_negative(self, name: str) -> None:
        # Note: bare "long" is intentionally not in the token set (matches historical behavior).
        assert _is_geo_lon(name) is False


class TestPriceDetection:
    """Ensure price-name detection matches tokens, not substrings."""

    @pytest.mark.parametrize(
        "name", ["price", "cost", "amount", "salary", "income", "unit_price", "totalCost", "USD_paid", "value"]
    )
    def test_positive_by_name(self, name: str) -> None:
        assert _is_price_column(name, sample_values=()) is True

    @pytest.mark.parametrize("name", ["postcode", "customer_id", "product_description", "duration", "count_of_events"])
    def test_negative_by_name(self, name: str) -> None:
        assert _is_price_column(name, sample_values=()) is False

    def test_positive_by_currency_symbol_in_samples(self) -> None:
        assert _is_price_column("charge", sample_values=("$12.00", "$45.99")) is True


class TestOrdinalDetection:
    """Ensure ordinal-name detection is token-based."""

    @pytest.mark.parametrize(
        "name", ["rank", "grade", "level", "order", "ordinal", "rating", "stage", "user_rating", "stageNumber"]
    )
    def test_positive(self, name: str) -> None:
        assert _is_ordinal_name(name) is True

    @pytest.mark.parametrize(
        "name", ["border", "border_style", "postage", "franklin", "postmark", "reorderable", "orderer", "engaged"]
    )
    def test_negative(self, name: str) -> None:
        assert _is_ordinal_name(name) is False


class TestStripGeoToken:
    """Ensure geographic base-name extraction lets lat/lon pairs match."""

    def test_latitude_and_longitude_share_base(self) -> None:
        assert _strip_geo_token("pickup_latitude") == _strip_geo_token("pickup_longitude") == "pickup"

    def test_lat_lng_pair_share_base(self) -> None:
        assert _strip_geo_token("origin_lat") == _strip_geo_token("origin_lng") == "origin"


class TestDetectColumnTypeIntegration:
    """End-to-end: exercise _profile_column so token detection reaches ColumnType routing."""

    def test_template_column_is_not_geographic(self) -> None:
        profile = _profile_column("template_id", pd.Series(["a", "b", "a", "c"], name="template_id"), target_columns=())
        assert profile.column_type != ColumnType.GEOGRAPHIC_LAT

    def test_border_style_column_is_not_ordinal(self) -> None:
        # 10 distinct integer levels: without the fix this would be flagged as CATEGORICAL_ORDINAL.
        profile = _profile_column(
            "border_style", pd.Series(list(range(10)) * 4, name="border_style"), target_columns=()
        )
        assert profile.column_type != ColumnType.CATEGORICAL_ORDINAL

    def test_pickup_latitude_is_geographic(self) -> None:
        # Realistic geo data repeats coordinates, so unique_ratio stays below the ID threshold.
        values = [40.7128, 40.7130, 40.7132, 40.7135] * 30
        profile = _profile_column("pickup_latitude", pd.Series(values, name="pickup_latitude"), target_columns=())
        assert profile.column_type == ColumnType.GEOGRAPHIC_LAT

    def test_camel_case_latitude_is_geographic(self) -> None:
        values = [40.7128, 40.7130, 40.7132, 40.7135] * 30
        profile = _profile_column("pickupLatitude", pd.Series(values, name="pickupLatitude"), target_columns=())
        assert profile.column_type == ColumnType.GEOGRAPHIC_LAT

    def test_id_column_still_detected_case_insensitively(self) -> None:
        profile = _profile_column("Order_ID", pd.Series(["a1", "b2", "c3", "d4"], name="Order_ID"), target_columns=())
        assert profile.column_type == ColumnType.ID_COLUMN
