"""Tests for the EVOLVER candidate validation split.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pandas as pd

from agent_k.agents.evolver import _prepare_validation_split, _temporal_order

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ()


class TestValidationSplit:
    """Tests for the candidate scoring holdout."""

    def _write_train(self, path: Path) -> None:
        rows = [{"id": index, "date": f"2021-01-{index + 1:02d}", "target": float(index)} for index in range(10)]
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_temporal_split_holds_out_latest_rows(self, tmp_path: Path) -> None:
        """A temporal split must train on the earliest rows and score the latest ones."""
        train_path = tmp_path / "train.csv"
        self._write_train(train_path)

        train_df, val_features, y_val, id_column = _prepare_validation_split(
            train_path=train_path,
            id_column="id",
            target_columns=["target"],
            validation_split=0.3,
            temporal=True,
            order_column="date",
        )

        assert id_column == "id"
        assert sorted(train_df["id"]) == [0, 1, 2, 3, 4, 5, 6]
        assert sorted(val_features["id"]) == [7, 8, 9]
        assert max(train_df["date"]) < min(val_features["date"])
        assert sorted(y_val["target"]) == [7.0, 8.0, 9.0]

    def test_temporal_split_sorts_unordered_rows(self, tmp_path: Path) -> None:
        """Rows stored out of order are sorted before the holdout is taken."""
        train_path = tmp_path / "train.csv"
        frame = pd.DataFrame(
            [
                {"id": 0, "date": "2021-03-01", "target": 3.0},
                {"id": 1, "date": "2021-01-01", "target": 1.0},
                {"id": 2, "date": "2021-02-01", "target": 2.0},
                {"id": 3, "date": "2021-04-01", "target": 4.0},
            ]
        )
        frame.to_csv(train_path, index=False)

        train_df, val_features, _y_val, _id_column = _prepare_validation_split(
            train_path=train_path,
            id_column="id",
            target_columns=["target"],
            validation_split=0.25,
            temporal=True,
            order_column="date",
        )

        assert list(val_features["id"]) == [3]
        assert sorted(train_df["id"]) == [0, 1, 2]

    def test_temporal_split_without_order_column_uses_row_order(self, tmp_path: Path) -> None:
        """Without an ordering column the file order defines the holdout."""
        train_path = tmp_path / "train.csv"
        self._write_train(train_path)

        train_df, val_features, _y_val, _id_column = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.2, temporal=True
        )

        assert list(train_df["id"]) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert list(val_features["id"]) == [8, 9]

    def test_random_split_is_shuffled_and_deterministic(self, tmp_path: Path) -> None:
        """Non-temporal splits stay shuffled and reproducible."""
        train_path = tmp_path / "train.csv"
        self._write_train(train_path)

        first = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.3
        )
        second = _prepare_validation_split(
            train_path=train_path, id_column="id", target_columns=["target"], validation_split=0.3
        )

        assert list(first[1]["id"]) == list(second[1]["id"])
        assert list(first[0]["id"]) != sorted(first[0]["id"])


class TestTemporalOrder:
    """Tests for temporal row ordering."""

    def test_unparseable_timestamps_stay_in_training(self) -> None:
        """Rows with missing timestamps sort first so they never form the holdout."""
        frame = pd.DataFrame({"date": ["2021-01-03", "not-a-date", "2021-01-01", "2021-01-02"]})

        positions = _temporal_order(frame, "date")

        assert list(positions) == [1, 2, 3, 0]

    def test_missing_column_keeps_row_order(self) -> None:
        """An absent ordering column leaves the original row order untouched."""
        frame = pd.DataFrame({"date": ["2021-01-03", "2021-01-01"]})

        assert list(_temporal_order(frame, "other")) == [0, 1]
        assert list(_temporal_order(frame, None)) == [0, 1]

    def test_non_temporal_column_keeps_row_order(self) -> None:
        """A column with no parseable timestamps leaves the row order untouched."""
        frame = pd.DataFrame({"label": ["b", "a", "c"]})

        assert list(_temporal_order(frame, "label")) == [0, 1, 2]
