"""Training-only dataset summaries and actionable preprocessing guidance.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import pandas as pd

__all__ = ("ColumnProfile", "DatasetProfile", "profile_training_data")


class ColumnProfile(BaseModel):
    """Observable characteristics of one training column."""

    model_config = ConfigDict(frozen=True)
    name: str
    dtype: str
    missing_fraction: float = Field(ge=0, le=1)
    unique_count: int = Field(ge=0)


class DatasetProfile(BaseModel):
    """A compact profile containing no row values or hidden labels."""

    model_config = ConfigDict(frozen=True)
    rows: int = Field(ge=1)
    columns: tuple[ColumnProfile, ...]
    recommendations: tuple[str, ...]


def profile_training_data(frame: pd.DataFrame) -> DatasetProfile:
    """Summarize training data without examining validation or acceptance rows."""
    if frame.empty:
        raise ValueError("Cannot profile an empty training table")
    columns = tuple(
        ColumnProfile(
            name=str(name),
            dtype=str(frame[name].dtype),
            missing_fraction=float(frame[name].isna().mean()),
            unique_count=int(frame[name].nunique()),
        )
        for name in frame.columns
    )
    hints: list[str] = []
    if any(column.missing_fraction for column in columns):
        hints.append("Fit imputation using training rows; retain missingness indicators.")
    if any(column.unique_count == 1 for column in columns):
        hints.append("Remove constant columns using a training-only variance check.")
    hints.append("Fit categorical vocabularies on training rows and handle unknown categories.")
    hints.append("Evaluate a LightGBM custom gradient and Hessian against the fixed baseline.")
    return DatasetProfile(rows=len(frame), columns=columns, recommendations=tuple(hints))
