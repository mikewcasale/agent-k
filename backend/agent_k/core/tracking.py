"""Experiment tracking utilities for AGENT-K.

@notice: |
    Experiment tracking utilities for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.core.tracking
    provides:
        - agent_k.core.tracking:ExperimentTracker
        - agent_k.core.tracking:create_experiment_tracker
        - agent_k.core.tracking:ExperimentRecord
        - agent_k.core.tracking:HintEffectivenessTracker
    pattern: tracking-store

@agent-guidance:
    do:
        - "Use agent_k.core.tracking as the canonical home for this capability."
    do_not:
        - "Create parallel modules without updating @similar or @graph."

@human-review:
    last-verified: 2026-01-26
    owners:
        - agent-k-core

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

import logfire
from pydantic import BaseModel, ConfigDict, Field

from agent_k.core.sage import Doc

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = (
    "ExperimentMetadata",
    "ExperimentRecord",
    "ExperimentSummary",
    "KaggleSubmissionRecord",
    "HintAttemptRecord",
    "HintEffectivenessTracker",
    "ExperimentTracker",
    "create_experiment_tracker",
    "extract_solution_metadata",
)

SCHEMA_VERSION: Final[str] = "1.0.0"
_DEFAULT_EXPERIMENT_DB: Final[Path] = Path("~/.agent_k/experiments/experiments.sqlite").expanduser()
_MODEL_SIGNATURES: Final[tuple[tuple[str, str, re.Pattern[str]], ...]] = (
    ("LGBMRegressor", "lightgbm", re.compile(r"\bLGBMRegressor\b")),
    ("LGBMClassifier", "lightgbm", re.compile(r"\bLGBMClassifier\b")),
    ("XGBRegressor", "xgboost", re.compile(r"\bXGBRegressor\b")),
    ("CatBoostRegressor", "catboost", re.compile(r"\bCatBoostRegressor\b")),
    ("RandomForestRegressor", "random_forest", re.compile(r"\bRandomForestRegressor\b")),
    ("ExtraTreesRegressor", "extra_trees", re.compile(r"\bExtraTreesRegressor\b")),
    ("GradientBoostingRegressor", "gradient_boosting", re.compile(r"\bGradientBoostingRegressor\b")),
    ("HistGradientBoostingRegressor", "hist_gradient_boosting", re.compile(r"\bHistGradientBoostingRegressor\b")),
    ("KNeighborsRegressor", "knn", re.compile(r"\bKNeighborsRegressor\b")),
    ("KNeighborsClassifier", "knn", re.compile(r"\bKNeighborsClassifier\b")),
    ("Ridge", "linear", re.compile(r"\bRidge\b")),
    ("Lasso", "linear", re.compile(r"\bLasso\b")),
    ("ElasticNet", "linear", re.compile(r"\bElasticNet\b")),
    ("SVR", "svm", re.compile(r"\bSVR\b")),
    ("LinearSVR", "svm", re.compile(r"\bLinearSVR\b")),
)
_HYPERPARAM_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "n_estimators": re.compile(r"(n_estimators\s*=\s*)(\d+)", re.IGNORECASE),
    "learning_rate": re.compile(r"(learning_rate\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_depth": re.compile(r"(max_depth\s*=\s*)(\d+)", re.IGNORECASE),
    "min_child_samples": re.compile(r"(min_child_samples\s*=\s*)(\d+)", re.IGNORECASE),
    "num_leaves": re.compile(r"(num_leaves\s*=\s*)(\d+)", re.IGNORECASE),
    "subsample": re.compile(r"(subsample\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "colsample_bytree": re.compile(r"(colsample_bytree\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "feature_fraction": re.compile(r"(feature_fraction\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "bagging_fraction": re.compile(r"(bagging_fraction\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "bagging_freq": re.compile(r"(bagging_freq\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_leaf": re.compile(r"(min_samples_leaf\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_split": re.compile(r"(min_samples_split\s*=\s*)(\d+)", re.IGNORECASE),
    "max_features": re.compile(r"(max_features\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "n_neighbors": re.compile(r"(n_neighbors\s*=\s*)(\d+)", re.IGNORECASE),
    "leaf_size": re.compile(r"(leaf_size\s*=\s*)(\d+)", re.IGNORECASE),
    "p": re.compile(r"(?<!\w)(p\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "weights": re.compile(r"(weights\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "metric": re.compile(r"(metric\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "algorithm": re.compile(r"(algorithm\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "objective": re.compile(r"(objective\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "huber_delta": re.compile(r"(huber_delta\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "quantile_alpha": re.compile(r"(quantile_alpha\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "lambda_l1": re.compile(r"(lambda_l1\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "lambda_l2": re.compile(r"(lambda_l2\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "min_split_gain": re.compile(r"(min_split_gain\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "min_child_weight": re.compile(r"(min_child_weight\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_bin": re.compile(r"(max_bin\s*=\s*)(\d+)", re.IGNORECASE),
    "alpha": re.compile(r"(alpha\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "l1_ratio": re.compile(r"(l1_ratio\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_iter": re.compile(r"(max_iter\s*=\s*)(\d+)", re.IGNORECASE),
}
_FEATURE_ENGINEERING_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "polynomial_interactions": re.compile(r"\bPolynomialFeatures\b"),
    "binning": re.compile(r"\bKBinsDiscretizer\b|\bpd\.cut\b|\bpd\.qcut\b"),
    "power_transform": re.compile(r"\bPowerTransformer\b"),
    "quantile_transform": re.compile(r"\bQuantileTransformer\b"),
    "scaling": re.compile(r"\b(StandardScaler|MinMaxScaler|RobustScaler)\b"),
    "one_hot": re.compile(r"\bOneHotEncoder\b"),
    "target_encoding": re.compile(r"\bTargetEncoder\b"),
    "ratio_features": re.compile(r"\bratio\b", re.IGNORECASE),
}
_TARGET_TRANSFORM_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("yeo-johnson", re.compile(r"yeo[-_ ]?johnson", re.IGNORECASE)),
    ("box-cox", re.compile(r"box[-_ ]?cox", re.IGNORECASE)),
    ("log1p", re.compile(r"\blog1p\b")),
    ("log", re.compile(r"\bnp\.log\b|\bmath\.log\b")),
)


class ExperimentMetadata(BaseModel):
    """Structured metadata extracted from a solution.

    @notice: |
        Structured metadata extracted from a solution.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: metadata-model
            rationale: "Frozen Pydantic model for solution metadata extraction."
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)
    model_name: str | None = Field(default=None, description="Primary model class name")
    model_family: str | None = Field(default=None, description="Model family label")
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Parsed hyperparameters")
    feature_set: list[str] = Field(default_factory=list, description="Explicit feature list if detected")
    feature_engineering: list[str] = Field(default_factory=list, description="Detected feature engineering tags")
    target_transform: str | None = Field(default=None, description="Target transform keyword")


class ExperimentRecord(BaseModel):
    """Persistent record of an experiment or submission.

    @notice: |
        Persistent record of an experiment or submission.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: record-model
            rationale: "Pydantic model for experiment persistence."
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique record identifier")
    competition_id: str = Field(..., description="Competition identifier")
    phase: str = Field(..., description="Lifecycle phase for the record")
    model_name: str | None = Field(default=None, description="Model class name")
    model_family: str | None = Field(default=None, description="Model family label")
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    feature_set: list[str] = Field(default_factory=list, description="Selected features")
    feature_engineering: list[str] = Field(default_factory=list, description="Feature engineering steps")
    target_transform: str | None = Field(default=None, description="Target transformation")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Recorded metrics")
    cv_score: float | None = Field(default=None, description="Cross-validation score")
    public_score: float | None = Field(default=None, description="Public leaderboard score")
    private_score: float | None = Field(default=None, description="Private leaderboard score")
    submission_id: str | None = Field(default=None, description="Submission identifier")
    rank: int | None = Field(default=None, description="Leaderboard rank")
    code_signature: str | None = Field(default=None, description="Hash of solution code")
    config_signature: str | None = Field(default=None, description="Hash of model/config data")
    dataset_fingerprint: str | None = Field(default=None, description="Dataset fingerprint or version")
    notes: str | None = Field(default=None, description="Optional notes")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Creation timestamp")


class ExperimentSummary(BaseModel):
    """Lightweight summary of a stored experiment.

    @notice: |
        Lightweight summary of a stored experiment.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: summary-model
            rationale: "Frozen Pydantic model for experiment listing."
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)
    record_id: str = Field(..., description="Record identifier")
    competition_id: str = Field(..., description="Competition identifier")
    phase: str = Field(..., description="Lifecycle phase")
    model_name: str | None = Field(default=None, description="Model class name")
    model_family: str | None = Field(default=None, description="Model family label")
    cv_score: float | None = Field(default=None, description="Cross-validation score")
    public_score: float | None = Field(default=None, description="Public leaderboard score")
    config_signature: str | None = Field(default=None, description="Config signature")
    code_signature: str | None = Field(default=None, description="Code signature")
    created_at: datetime = Field(..., description="Creation timestamp")


class KaggleSubmissionRecord(BaseModel):
    """Persistent record of a Kaggle submission and leaderboard scores.

    @notice: |
        Persistent record of a Kaggle submission and leaderboard scores.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: record-model
            rationale: "Pydantic model for Kaggle submission persistence."
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique record identifier")
    competition_id: str = Field(..., description="Competition identifier")
    submission_id: str = Field(..., description="Submission identifier")
    public_score: float | None = Field(default=None, description="Public leaderboard score")
    private_score: float | None = Field(default=None, description="Private leaderboard score")
    rank: int | None = Field(default=None, description="Leaderboard rank")
    model_config_hash: str = Field(..., description="Hash of model configuration signature")
    feature_set_hash: str | None = Field(default=None, description="Hash of feature set")
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Creation timestamp")


class HintAttemptRecord(BaseModel):
    """Track preprocessing hint usage and outcomes.

    @notice: |
        Track preprocessing hint usage and outcomes.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: record-model
            rationale: "Pydantic model for hint effectiveness tracking."
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique record identifier")
    hint_id: str = Field(..., description="Hint identifier")
    competition_id: str = Field(..., description="Competition identifier")
    generation: int = Field(..., ge=0, description="Evolution generation index")
    applied: bool = Field(default=True, description="Whether hint was applied in solution")
    cv_score_before: float | None = Field(default=None, description="CV score before applying hint")
    cv_score_after: float | None = Field(default=None, description="CV score after applying hint")
    delta: float = Field(default=0.0, description="Score delta after applying hint (positive = improvement)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Attempt timestamp")


class ExperimentTracker:
    """SQLite-backed tracker for experiments and submissions.

    @notice: |
        SQLite-backed tracker for experiments and submissions.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: tracking-store
            rationale: "Centralized persistence for experiment metadata."
            violations: "Distributed tracking logic causes inconsistent history."

        @concurrency:
            model: asyncio
            safe: false
            reason: "Uses SQLite connections without cross-process locking."

        @invariants:
            - "Database schema is initialized before writes."
    """

    _table_name: Final[str] = "experiments"
    _submission_table_name: Final[str] = "kaggle_submissions"
    _hint_table_name: Final[str] = "hint_attempts"

    def __init__(self, db_path: Annotated[Path | None, Doc("SQLite database path override.")] = None) -> None:
        self._db_path = (db_path or _resolve_db_path()).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

    def record_experiment(self, record: ExperimentRecord) -> ExperimentRecord:
        """Persist a record to the tracker."""
        if record.config_signature is None:
            signature = _compute_config_signature(record)
            record = record.model_copy(update={"config_signature": signature})

        payload = record.model_dump()
        with logfire.span("experiment.record", competition_id=record.competition_id, phase=record.phase):
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self._table_name} (
                        record_id,
                        competition_id,
                        phase,
                        model_name,
                        model_family,
                        hyperparameters_json,
                        feature_set_json,
                        feature_engineering_json,
                        target_transform,
                        metrics_json,
                        cv_score,
                        public_score,
                        private_score,
                        submission_id,
                        rank,
                        code_signature,
                        config_signature,
                        dataset_fingerprint,
                        notes,
                        created_at
                    ) VALUES (
                        :record_id,
                        :competition_id,
                        :phase,
                        :model_name,
                        :model_family,
                        :hyperparameters_json,
                        :feature_set_json,
                        :feature_engineering_json,
                        :target_transform,
                        :metrics_json,
                        :cv_score,
                        :public_score,
                        :private_score,
                        :submission_id,
                        :rank,
                        :code_signature,
                        :config_signature,
                        :dataset_fingerprint,
                        :notes,
                        :created_at
                    )
                    """,
                    _record_to_row(payload),
                )
        return record

    def list_experiments(self, competition_id: str, *, limit: int = 50) -> list[ExperimentRecord]:
        """Return most recent experiment records for a competition."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM {self._table_name}
                WHERE competition_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (competition_id, limit),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def record_submission(self, record: KaggleSubmissionRecord) -> KaggleSubmissionRecord:
        """Persist a Kaggle submission record."""
        if not record.model_config_hash:
            record = record.model_copy(update={"model_config_hash": _compute_submission_config_hash(record)})

        payload = record.model_dump()
        with logfire.span("submission.record", competition_id=record.competition_id):
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self._submission_table_name} (
                        record_id,
                        competition_id,
                        submission_id,
                        public_score,
                        private_score,
                        rank,
                        model_config_hash,
                        feature_set_hash,
                        hyperparameters_json,
                        created_at
                    ) VALUES (
                        :record_id,
                        :competition_id,
                        :submission_id,
                        :public_score,
                        :private_score,
                        :rank,
                        :model_config_hash,
                        :feature_set_hash,
                        :hyperparameters_json,
                        :created_at
                    )
                    """,
                    _submission_to_row(payload),
                )
        return record

    def record_hint_attempt(self, record: HintAttemptRecord) -> HintAttemptRecord:
        """Persist a preprocessing hint attempt record."""
        with logfire.span("hint_attempt.record", competition_id=record.competition_id, hint_id=record.hint_id):
            with self._connect() as conn:
                conn.execute(
                    f"""
                    INSERT INTO {self._hint_table_name} (
                        id,
                        competition_id,
                        hint_id,
                        applied,
                        delta,
                        generation,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.competition_id,
                        record.hint_id,
                        1 if record.applied else 0,
                        record.delta,
                        record.generation,
                        record.created_at.isoformat(),
                    ),
                )
        return record

    def list_submissions(self, competition_id: str, *, limit: int = 50) -> list[KaggleSubmissionRecord]:
        """Return most recent submission records for a competition."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM {self._submission_table_name}
                WHERE competition_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (competition_id, limit),
            ).fetchall()
        return [_row_to_submission(row) for row in rows]

    def find_duplicate_config(
        self, competition_id: str, *, model_config_hash: str, feature_set_hash: str | None = None
    ) -> KaggleSubmissionRecord | None:
        """Detect previously submitted configurations to avoid duplicate submissions."""
        query = f"""
            SELECT * FROM {self._submission_table_name}
            WHERE competition_id = ? AND model_config_hash = ?
        """
        params: list[Any] = [competition_id, model_config_hash]
        if feature_set_hash:
            query += " AND feature_set_hash = ?"
            params.append(feature_set_hash)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return _row_to_submission(row) if row else None

    def get_improvement_history(
        self, competition_id: str, *, direction: str = "maximize", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Analyze leaderboard score progression over time."""
        submissions = list(reversed(self.list_submissions(competition_id, limit=limit)))
        history: list[dict[str, Any]] = []
        previous_public: float | None = None
        previous_private: float | None = None
        previous_rank: int | None = None
        for submission in submissions:
            delta_public = _delta_score(submission.public_score, previous_public)
            delta_private = _delta_score(submission.private_score, previous_private)
            delta_rank = _delta_rank(submission.rank, previous_rank)
            improved = _is_improvement(public_delta=delta_public, private_delta=delta_private, direction=direction)
            history.append(
                {
                    "submission_id": submission.submission_id,
                    "public_score": submission.public_score,
                    "private_score": submission.private_score,
                    "rank": submission.rank,
                    "model_config_hash": submission.model_config_hash,
                    "feature_set_hash": submission.feature_set_hash,
                    "delta_public": delta_public,
                    "delta_private": delta_private,
                    "delta_rank": delta_rank,
                    "improved": improved,
                    "created_at": submission.created_at,
                }
            )
            if submission.public_score is not None:
                previous_public = submission.public_score
            if submission.private_score is not None:
                previous_private = submission.private_score
            if submission.rank is not None:
                previous_rank = submission.rank
        return history

    def recommend_next_experiment(
        self, competition_id: str, *, direction: str = "maximize", limit: int = 50
    ) -> dict[str, Any]:
        """Recommend the next experiment based on historical performance."""
        submissions = self.list_submissions(competition_id, limit=limit)
        history = self.get_improvement_history(competition_id, direction=direction, limit=limit)
        if not submissions:
            return {
                "status": "no_history",
                "recommendations": [
                    "Run a baseline model to establish a leaderboard anchor.",
                    "Capture feature engineering choices and target transforms for traceability.",
                ],
            }

        best_submission = _select_best_submission(submissions, direction=direction)
        best_experiment = (
            self.find_latest_by_config_signature(competition_id, best_submission.model_config_hash)
            if best_submission
            else None
        )
        recent = history[-3:] if history else []
        stagnating = bool(recent) and not any(item.get("improved") for item in recent)
        recommendations: list[str] = []

        if stagnating:
            recommendations.append("Prioritize feature engineering or target transforms to break stagnation.")
            recommendations.append("Try a different model family to diversify the search.")
        else:
            recommendations.append("Continue hyperparameter tuning around the best-performing configuration.")

        if best_experiment and best_experiment.feature_engineering:
            recommendations.append("Expand feature engineering with interaction or ratio features.")
        if best_experiment and not best_experiment.target_transform:
            recommendations.append("Evaluate target transforms (log1p, Box-Cox, Yeo-Johnson) if not tried yet.")

        return {
            "status": "ok",
            "best_submission_id": best_submission.submission_id if best_submission else None,
            "best_public_score": best_submission.public_score if best_submission else None,
            "best_private_score": best_submission.private_score if best_submission else None,
            "stagnating": stagnating,
            "recommendations": recommendations,
        }

    def find_latest_by_code_signature(self, competition_id: str, code_signature: str) -> ExperimentRecord | None:
        """Return the most recent record for a code signature."""
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM {self._table_name}
                WHERE competition_id = ? AND code_signature = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (competition_id, code_signature),
            ).fetchone()
        return _row_to_record(row) if row else None

    def find_latest_by_config_signature(self, competition_id: str, config_signature: str) -> ExperimentRecord | None:
        """Return the most recent record for a config signature."""
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM {self._table_name}
                WHERE competition_id = ? AND config_signature = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (competition_id, config_signature),
            ).fetchone()
        return _row_to_record(row) if row else None

    def best_experiment(
        self, competition_id: str, *, metric: str = "public_score", direction: str = "maximize"
    ) -> ExperimentRecord | None:
        """Return the best experiment based on a metric."""
        if metric not in {"public_score", "cv_score"}:
            raise ValueError("metric must be public_score or cv_score")
        order = "DESC" if direction == "maximize" else "ASC"
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM {self._table_name}
                WHERE competition_id = ? AND {metric} IS NOT NULL
                ORDER BY {metric} {order}
                LIMIT 1
                """,
                (competition_id,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def summarize(self, record: ExperimentRecord) -> ExperimentSummary:
        """Return a summary payload for an experiment record."""
        return ExperimentSummary(
            record_id=record.record_id,
            competition_id=record.competition_id,
            phase=record.phase,
            model_name=record.model_name,
            model_family=record.model_family,
            cv_score=record.cv_score,
            public_score=record.public_score,
            config_signature=record.config_signature,
            code_signature=record.code_signature,
            created_at=record.created_at,
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # sqlite3 connections used as context managers commit/rollback on exit
        # but do NOT close. Wrap explicit close in a finally to prevent FD leaks.
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    record_id TEXT PRIMARY KEY,
                    competition_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    model_name TEXT,
                    model_family TEXT,
                    hyperparameters_json TEXT,
                    feature_set_json TEXT,
                    feature_engineering_json TEXT,
                    target_transform TEXT,
                    metrics_json TEXT,
                    cv_score REAL,
                    public_score REAL,
                    private_score REAL,
                    submission_id TEXT,
                    rank INTEGER,
                    code_signature TEXT,
                    config_signature TEXT,
                    dataset_fingerprint TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_competition
                ON {self._table_name} (competition_id, created_at)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_code_signature
                ON {self._table_name} (competition_id, code_signature)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_config_signature
                ON {self._table_name} (competition_id, config_signature)
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._submission_table_name} (
                    record_id TEXT PRIMARY KEY,
                    competition_id TEXT NOT NULL,
                    submission_id TEXT NOT NULL,
                    public_score REAL,
                    private_score REAL,
                    rank INTEGER,
                    model_config_hash TEXT NOT NULL,
                    feature_set_hash TEXT,
                    hyperparameters_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._submission_table_name}_competition
                ON {self._submission_table_name} (competition_id, created_at)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._submission_table_name}_config
                ON {self._submission_table_name} (competition_id, model_config_hash)
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._hint_table_name} (
                    id TEXT PRIMARY KEY,
                    competition_id TEXT NOT NULL,
                    hint_id TEXT NOT NULL,
                    applied INTEGER NOT NULL,
                    delta REAL NOT NULL,
                    generation INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self._hint_table_name}_competition
                ON {self._hint_table_name} (competition_id, created_at)
                """
            )


class HintEffectivenessTracker:
    """Track and aggregate hint performance across generations.

    @notice: |
        Track and aggregate hint performance across generations.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: tracking-service
            rationale: "Centralizes hint success tracking and suppression."
            violations: "Distributed tracking makes suppression inconsistent."
    """

    _HINT_TRACKER_PATH: Final[Path] = Path("~/.agent_k/hint_tracker.json").expanduser()
    _suppression_threshold: Final[int] = 3

    def __init__(self, experiment_tracker: ExperimentTracker | None = None) -> None:
        self._records: list[HintAttemptRecord] = []
        self._success_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._failure_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._suppressed: dict[str, set[str]] = defaultdict(set)
        self._amplified: dict[str, set[str]] = defaultdict(set)
        self._experiment_tracker = experiment_tracker
        self.load()

    @property
    def suppressed_hints(self) -> set[str]:
        """Return all suppressed hint ids across competitions."""
        return {hint_id for hints in self._suppressed.values() for hint_id in hints}

    def record_attempt(self, record: HintAttemptRecord) -> None:
        """Record a hint attempt and update suppression state."""
        self._records.append(record)
        if self._experiment_tracker is not None:
            try:
                self._experiment_tracker.record_hint_attempt(record)
            except sqlite3.Error as exc:
                logfire.warning("hint_attempt_persist_failed", error=str(exc))
        if not record.applied:
            return
        key = (record.competition_id, record.hint_id)
        if record.delta > 0:
            self._success_counts[key] += 1
        elif record.delta < 0:
            self._failure_counts[key] += 1
        if self._failure_counts[key] >= self._suppression_threshold:
            self.suppress_hint(record.hint_id, record.competition_id)
        self.save()

    def get_success_rate(self, hint_id: str, competition_id: str) -> float:
        """Return success rate for a hint within a competition."""
        attempts = [
            record
            for record in self._records
            if record.hint_id == hint_id
            and record.competition_id == competition_id
            and record.applied
            and record.delta != 0
        ]
        if not attempts:
            return 0.0
        successes = sum(1 for record in attempts if record.delta > 0)
        return successes / len(attempts)

    def get_last_attempt(self, hint_id: str, competition_id: str) -> HintAttemptRecord | None:
        """Return the most recent attempt for a hint in a competition."""
        attempts = [
            record for record in self._records if record.hint_id == hint_id and record.competition_id == competition_id
        ]
        if not attempts:
            return None
        return max(attempts, key=lambda record: record.created_at)

    def suppress_hint(self, hint_id: str, competition_id: str) -> None:
        """Suppress a hint for a competition."""
        if hint_id in self._suppressed[competition_id]:
            return
        self._suppressed[competition_id].add(hint_id)
        logfire.info("hint_suppressed", hint_id=hint_id, competition_id=competition_id)

    def amplify_hint(self, hint_id: str, competition_id: str) -> None:
        """Amplify a hint to boost its prompt priority."""
        self._amplified[competition_id].add(hint_id)

    def is_suppressed(self, hint_id: str, competition_id: str) -> bool:
        """Check if a hint is suppressed for a competition."""
        return hint_id in self._suppressed[competition_id]

    def is_amplified(self, hint_id: str, competition_id: str) -> bool:
        """Check if a hint is amplified for a competition."""
        return hint_id in self._amplified[competition_id]

    def save(self) -> None:
        """Persist hint statistics to disk."""
        self._HINT_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, dict[str, dict[str, Any]]] = {}
        for (comp_id, hint_id), count in self._success_counts.items():
            data.setdefault(comp_id, {})[hint_id] = {
                "success": count,
                "failure": self._failure_counts.get((comp_id, hint_id), 0),
                "suppressed": hint_id in self._suppressed.get(comp_id, set()),
            }
        try:
            self._HINT_TRACKER_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logfire.warning("hint_tracker_save_failed", error=str(exc))

    def load(self) -> None:
        """Load hint statistics from disk."""
        if not self._HINT_TRACKER_PATH.exists():
            return
        try:
            data = json.loads(self._HINT_TRACKER_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logfire.warning("hint_tracker_load_failed", error=str(exc))
            return
        for comp_id, hints in data.items():
            if not isinstance(hints, dict):
                continue
            for hint_id, stats in hints.items():
                if not isinstance(stats, dict):
                    continue
                self._success_counts[(comp_id, hint_id)] = int(stats.get("success", 0))
                self._failure_counts[(comp_id, hint_id)] = int(stats.get("failure", 0))
                if stats.get("suppressed"):
                    self._suppressed[comp_id].add(hint_id)


def create_experiment_tracker(
    db_path: Annotated[Path | None, Doc("SQLite database path override.")] = None,
) -> ExperimentTracker:
    """Factory for ExperimentTracker instances.

    @dev: |
        See module for behavior details and invariants.

        @notice: |
            Returns a tracker configured with the default database path.

        @factory-for:
            id: agent_k.core.tracking:ExperimentTracker
            rationale: "Centralizes tracker defaults and storage path."
            singleton: false
            cache-key: db_path

        @canonical-home:
            for:
                - "experiment tracker construction"
            notes: "Use create_experiment_tracker to ensure defaults."
    """
    return ExperimentTracker(db_path=db_path)


def extract_solution_metadata(solution_code: str) -> ExperimentMetadata:
    """Extract model and feature metadata from solution code.

    @notice: |
        Extract model and feature metadata from solution code.

    @dev: |
        See module for behavior details and invariants.
    """
    model_name, model_family = _detect_model(solution_code)
    hyperparameters = _extract_hyperparameters(solution_code)
    feature_set = _extract_feature_set(solution_code)
    if not feature_set and "select_dtypes" in solution_code:
        feature_set = ["__auto__"]
    feature_engineering = _extract_feature_engineering(solution_code)
    target_transform = _detect_target_transform(solution_code)
    return ExperimentMetadata(
        model_name=model_name,
        model_family=model_family,
        hyperparameters=hyperparameters,
        feature_set=feature_set,
        feature_engineering=feature_engineering,
        target_transform=target_transform,
    )


def _resolve_db_path() -> Path:
    env_override = os.getenv("AGENT_K_EXPERIMENT_DB")
    return Path(env_override).expanduser() if env_override else _DEFAULT_EXPERIMENT_DB


def _detect_model(solution_code: str) -> tuple[str | None, str | None]:
    for model_name, model_family, pattern in _MODEL_SIGNATURES:
        if pattern.search(solution_code):
            return model_name, model_family
    return None, None


def _extract_hyperparameters(solution_code: str) -> dict[str, Any]:
    hyperparameters: dict[str, Any] = {}
    for key, pattern in _HYPERPARAM_PATTERNS.items():
        match = pattern.search(solution_code)
        if not match:
            continue
        value = _parse_hyperparam_value(match.group(2))
        if value is not None:
            hyperparameters[key] = value
    return hyperparameters


def _parse_hyperparam_value(raw: str) -> Any:
    cleaned = raw.strip()
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
        return cleaned[1:-1]
    if cleaned.lower() in {"none", "null"}:
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _extract_feature_set(solution_code: str) -> list[str]:
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return []

    features: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [target for target in node.targets if isinstance(target, ast.Name)]
        if not targets:
            continue
        target_name = targets[0].id.lower()
        if "feature" not in target_name:
            continue
        extracted = _extract_string_sequence(node.value)
        if extracted:
            features.update(extracted)
    return sorted(features)


def _extract_string_sequence(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
            else:
                return []
        return values
    return []


def _extract_feature_engineering(solution_code: str) -> list[str]:
    return [name for name, pattern in _FEATURE_ENGINEERING_PATTERNS.items() if pattern.search(solution_code)]


def _detect_target_transform(solution_code: str) -> str | None:
    for label, pattern in _TARGET_TRANSFORM_PATTERNS:
        if pattern.search(solution_code):
            return label
    return None


def _compute_config_signature(record: ExperimentRecord) -> str:
    payload = {
        "model_name": record.model_name,
        "model_family": record.model_family,
        "hyperparameters": record.hyperparameters,
        "feature_set": record.feature_set,
        "feature_engineering": record.feature_engineering,
        "target_transform": record.target_transform,
        "dataset_fingerprint": record.dataset_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _record_to_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": payload["record_id"],
        "competition_id": payload["competition_id"],
        "phase": payload["phase"],
        "model_name": payload.get("model_name"),
        "model_family": payload.get("model_family"),
        "hyperparameters_json": json.dumps(payload.get("hyperparameters", {}), sort_keys=True, default=str),
        "feature_set_json": json.dumps(payload.get("feature_set", []), sort_keys=True),
        "feature_engineering_json": json.dumps(payload.get("feature_engineering", []), sort_keys=True),
        "target_transform": payload.get("target_transform"),
        "metrics_json": json.dumps(payload.get("metrics", {}), sort_keys=True, default=str),
        "cv_score": payload.get("cv_score"),
        "public_score": payload.get("public_score"),
        "private_score": payload.get("private_score"),
        "submission_id": payload.get("submission_id"),
        "rank": payload.get("rank"),
        "code_signature": payload.get("code_signature"),
        "config_signature": payload.get("config_signature"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
        "notes": payload.get("notes"),
        "created_at": payload["created_at"].isoformat(),
    }


def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
    payload = dict(row)
    return ExperimentRecord(
        record_id=payload["record_id"],
        competition_id=payload["competition_id"],
        phase=payload["phase"],
        model_name=payload.get("model_name"),
        model_family=payload.get("model_family"),
        hyperparameters=_load_json(payload.get("hyperparameters_json"), default={}),
        feature_set=_load_json(payload.get("feature_set_json"), default=[]),
        feature_engineering=_load_json(payload.get("feature_engineering_json"), default=[]),
        target_transform=payload.get("target_transform"),
        metrics=_load_json(payload.get("metrics_json"), default={}),
        cv_score=payload.get("cv_score"),
        public_score=payload.get("public_score"),
        private_score=payload.get("private_score"),
        submission_id=payload.get("submission_id"),
        rank=payload.get("rank"),
        code_signature=payload.get("code_signature"),
        config_signature=payload.get("config_signature"),
        dataset_fingerprint=payload.get("dataset_fingerprint"),
        notes=payload.get("notes"),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


def _submission_to_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": payload["record_id"],
        "competition_id": payload["competition_id"],
        "submission_id": payload["submission_id"],
        "public_score": payload.get("public_score"),
        "private_score": payload.get("private_score"),
        "rank": payload.get("rank"),
        "model_config_hash": payload.get("model_config_hash"),
        "feature_set_hash": payload.get("feature_set_hash"),
        "hyperparameters_json": json.dumps(payload.get("hyperparameters", {}), sort_keys=True, default=str),
        "created_at": payload["created_at"].isoformat(),
    }


def _row_to_submission(row: sqlite3.Row) -> KaggleSubmissionRecord:
    payload = dict(row)
    return KaggleSubmissionRecord(
        record_id=payload["record_id"],
        competition_id=payload["competition_id"],
        submission_id=payload["submission_id"],
        public_score=payload.get("public_score"),
        private_score=payload.get("private_score"),
        rank=payload.get("rank"),
        model_config_hash=payload.get("model_config_hash"),
        feature_set_hash=payload.get("feature_set_hash"),
        hyperparameters=_load_json(payload.get("hyperparameters_json"), default={}),
        created_at=datetime.fromisoformat(payload["created_at"]),
    )


def _compute_submission_config_hash(record: KaggleSubmissionRecord) -> str:
    payload = {"hyperparameters": record.hyperparameters, "feature_set_hash": record.feature_set_hash}
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _compute_feature_set_hash(feature_set: list[str] | dict[str, Any]) -> str:
    if isinstance(feature_set, dict):
        payload = sorted(feature_set.keys())
    else:
        payload = sorted(feature_set)
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _delta_score(value: float | None, previous: float | None) -> float | None:
    if value is None or previous is None:
        return None
    return value - previous


def _delta_rank(value: int | None, previous: int | None) -> int | None:
    if value is None or previous is None:
        return None
    return previous - value


def _is_improvement(*, public_delta: float | None, private_delta: float | None, direction: str) -> bool | None:
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    delta = private_delta if private_delta is not None else public_delta
    if delta is None:
        return None
    return delta > 0 if direction == "maximize" else delta < 0


def _select_best_submission(
    submissions: list[KaggleSubmissionRecord], *, direction: str
) -> KaggleSubmissionRecord | None:
    if not submissions:
        return None
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    scored: list[tuple[float, KaggleSubmissionRecord]] = []
    for submission in submissions:
        score = submission.private_score if submission.private_score is not None else submission.public_score
        if score is None:
            continue
        scored.append((score, submission))
    if not scored:
        return submissions[0]
    return (
        max(scored, key=lambda item: item[0])[1]
        if direction == "maximize"
        else min(scored, key=lambda item: item[0])[1]
    )


def _load_json(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
