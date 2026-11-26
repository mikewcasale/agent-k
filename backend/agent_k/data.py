"""Secure loading and validation for discovered AGENT-K datasets.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .datasets import DatasetManifest

__all__ = (
    "DatasetBundle",
    "load_dataset",
    "read_table",
    "resolve_input_path",
)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Validated training and unlabeled prediction frames."""

    manifest: DatasetManifest
    train: pd.DataFrame
    predict: pd.DataFrame

    @property
    def train_features(self) -> pd.DataFrame:
        """Return training features in manifest order."""
        return self.train.loc[:, list(self.manifest.feature_columns)]

    @property
    def train_targets(self) -> pd.DataFrame:
        """Return training targets in manifest order."""
        return self.train.loc[:, list(self.manifest.target_columns)]

    @property
    def predict_features(self) -> pd.DataFrame:
        """Return prediction features in manifest order."""
        return self.predict.loc[:, list(self.manifest.feature_columns)]

    @property
    def predict_ids(self) -> pd.DataFrame:
        """Return prediction IDs in manifest order."""
        return self.predict.loc[:, list(self.manifest.id_columns)]


def load_dataset(
    manifest: DatasetManifest,
    input_root: Path,
    *,
    verify_hashes: bool = True,
) -> DatasetBundle:
    """Load a manifest and enforce the train/predict trust boundary.

    Args:
        manifest: Discovered dataset contract.
        input_root: Root containing manifest-relative assets.
        verify_hashes: Whether to verify immutable asset digests.

    Returns:
        Validated frames ready for candidate training and prediction.
    """
    train_path = resolve_input_path(input_root, manifest.train.path)
    predict_path = resolve_input_path(input_root, manifest.predict.path)
    if verify_hashes:
        _verify_hash(train_path, manifest.train.sha256)
        _verify_hash(predict_path, manifest.predict.sha256)
    train = read_table(train_path)
    predict = read_table(predict_path)
    _validate_columns(manifest, train, predict)
    _validate_ids(manifest, train, predict)
    _validate_targets(manifest, train, predict)
    return DatasetBundle(manifest=manifest, train=train, predict=predict)


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported tabular asset without guessing its format."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {suffix}")


def resolve_input_path(root: Path, relative_path: str) -> Path:
    """Resolve a relative asset path while preventing directory traversal."""
    candidate = (root.resolve() / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"asset escapes input root: {relative_path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"input asset does not exist: {relative_path}")
    return candidate


def _verify_hash(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"asset digest mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _validate_columns(
    manifest: DatasetManifest,
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> None:
    required_train = {
        *manifest.id_columns,
        *manifest.feature_columns,
        *manifest.target_columns,
    }
    required_predict = {*manifest.id_columns, *manifest.feature_columns}
    missing_train = required_train - set(train.columns)
    missing_predict = required_predict - set(predict.columns)
    if missing_train:
        raise ValueError(f"training data is missing columns: {sorted(missing_train)}")
    if missing_predict:
        raise ValueError(f"prediction data is missing columns: {sorted(missing_predict)}")


def _validate_ids(
    manifest: DatasetManifest,
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> None:
    for name, frame in (("training", train), ("prediction", predict)):
        ids = frame.loc[:, list(manifest.id_columns)]
        if ids.isna().any(axis=None):
            raise ValueError(f"{name} IDs contain missing values")
        if ids.duplicated().any():
            raise ValueError(f"{name} IDs are not unique")


def _validate_targets(
    manifest: DatasetManifest,
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> None:
    targets = train.loc[:, list(manifest.target_columns)]
    if targets.isna().any(axis=None):
        raise ValueError("training targets contain missing values")
    exposed = [column for column in manifest.target_columns if column in predict.columns]
    if exposed:
        raise ValueError(f"prediction input exposes target columns: {exposed}")
