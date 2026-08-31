"""Evolver agent - evolutionary optimization for AGENT-K.

@notice: |
    Evolver agent - evolutionary optimization for AGENT-K.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.agents.evolver
    provides:
        - agent_k.agents.evolver:EvolverAgent
        - agent_k.agents.evolver:EvolverDeps
        - agent_k.agents.evolver:EvolverSettings
        - agent_k.agents.evolver:EvolutionResult
        - agent_k.agents.evolver:EvolutionFailure
        - agent_k.agents.evolver:EVOLUTION_OUTPUT_TYPE
        - agent_k.agents.evolver:evolver_agent
    consumes:
        - agent_k.core.protocols:PlatformAdapter
        - agent_k.ui.agui:EventEmitter
        - agent_k.adapters.openevolve:OpenEvolveRunner
        - agent_k.toolsets.code:code_toolset
    pattern: agent-singleton

@similar:
    - id: agent_k.agents.scientist
        when: "Use for research synthesis rather than optimization."
    - id: agent_k.evolution.framework
        when: "Framework utilities for evolution outside the agent context."

@agent-guidance:
    do:
        - "Use agent_k.agents.evolver as the canonical home for this capability."
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
import csv
import hashlib
import json
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Annotated, Any, Final, Self, cast

import logfire
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, ModelSettings, RunContext, ToolOutput, ToolReturn
from pydantic_ai.builtin_tools import MCPServerTool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, roc_auc_score

from agent_k.adapters.openevolve import OpenEvolveRunner
from agent_k.agents import register_agent
from agent_k.agents.base import MemoryMixin, universal_tool_preparation
from agent_k.agents.prompts import EVOLVER_SYSTEM_PROMPT
from agent_k.core.constants import (
    DEFAULT_KAGGLE_MCP_URL,
    DEFAULT_MODEL,
    EVOLUTION_POPULATION_SIZE,
    MAX_EVOLUTION_GENERATIONS,
    SOLUTION_EXECUTION_TIMEOUT_SECONDS,
)
from agent_k.core.data import CompetitionSchema, stage_competition_data
from agent_k.core.hints import DatasetProfile, PreprocessingHint, compute_hint_priority, detect_applied_hints
from agent_k.core.sage import Doc, Range
from agent_k.core.solution import execute_solution
from agent_k.core.strategy import (
    FitnessInput,
    FitnessPolicy,
    ProblemProfile,
    TechniquePolicy,
    apply_solution_policy,
    build_fitness_function,
    build_fitness_policy,
    build_problem_profile,
    build_technique_policy,
)
from agent_k.core.tracking import (
    ExperimentRecord,
    ExperimentTracker,
    HintAttemptRecord,
    HintEffectivenessTracker,
    extract_solution_metadata,
)
from agent_k.evolution.loss import alternative_objectives
from agent_k.infra.providers import get_model
from agent_k.toolsets import code_toolset, create_production_toolset, prepare_code_execution_tool, prepare_memory_tool

if TYPE_CHECKING:
    from agent_k.core.models import Competition
    from agent_k.core.protocols import PlatformAdapter
    from agent_k.ui.agui import EventEmitter

__all__ = (
    "EVOLUTION_OUTPUT_TYPE",
    "EvolutionFailure",
    "EvolutionResult",
    "EvolverAgent",
    "EvolverDeps",
    "EvolverSettings",
    "EVOLVER_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "evolver_agent",
    "settings",
)

SCHEMA_VERSION: Final[str] = "1.0.0"
_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])(-?\d+\.?\d*)(?![\w.])")
_FILLNA_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>\s*)(?P<var>\w+)\s*=\s*pd\.read_csv\(.*\)$", re.MULTILINE
)
_TARGET_INDEX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<name>TARGET_COLUMNS|TRAIN_TARGET_COLUMNS)\[(?P<index>\d+)\]"
)
_DEF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(def (?!__)\w+).*?(?=^(?:def |class )|\Z)", re.MULTILINE | re.DOTALL
)
_NUMERIC_PIPELINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(numeric_transformer\s*=\s*Pipeline\(steps=\[\s*)(?P<steps>.*?)(\s*\]\))", re.DOTALL
)
_NUMERIC_COLS_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<indent>\s*)numeric_cols\s*=.*$", re.MULTILINE)
_CATEGORICAL_COLS_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<indent>\s*)categorical_cols\s*=.*$", re.MULTILINE)
_KNN_MODEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bKNeighbors(?:Classifier|Regressor)\b")
_ERROR_HINT_PATTERNS: Final[tuple[tuple[re.Pattern[str], str, str], ...]] = (
    (
        re.compile(r"(ModuleNotFoundError|ImportError)", re.IGNORECASE),
        "import_error",
        "Library not available. Use try/except fallback patterns.",
    ),
    (
        re.compile(r"(FileNotFoundError|No such file or directory)", re.IGNORECASE),
        "missing_file",
        "Ensure train.csv, test.csv, and sample_submission.csv exist in the working directory.",
    ),
    (
        re.compile(r"(columns are missing|feature_names mismatch)", re.IGNORECASE),
        "column_mismatch",
        "Align train/test columns: X_test = test_df[X.columns].",
    ),
    (
        re.compile(
            r"(Found array with .* feature|number of features|shape mismatch|incompatible shapes)", re.IGNORECASE
        ),
        "shape_mismatch",
        "Ensure preprocessing produces identical feature shapes for train/test.",
    ),
    (re.compile(r"KeyError", re.IGNORECASE), "key_error", "Check column names in both train and test."),
    (
        re.compile(r"LightGBMError", re.IGNORECASE),
        "lightgbm_error",
        "Check categorical handling and feature name alignment; ensure X_test uses X.columns.",
    ),
    (re.compile(r"(MemoryError|out of memory|OOM)", re.IGNORECASE), "memory_error", "Reduce model size or data."),
    (re.compile(r"TimeoutError", re.IGNORECASE), "timeout", "Reduce complexity or iterations."),
    (re.compile(r"SyntaxError", re.IGNORECASE), "syntax_error", "Fix Python syntax (colons/brackets)."),
    (re.compile(r"IndentationError", re.IGNORECASE), "indentation_error", "Use consistent 4-space indentation."),
    (re.compile(r"TypeError", re.IGNORECASE), "type_error", "Check function arguments and data types."),
    (re.compile(r"AttributeError", re.IGNORECASE), "attribute_error", "Check object methods/attributes exist."),
    (re.compile(r"ValueError", re.IGNORECASE), "value_error", "Check data types and preprocessing consistency."),
)
_ERROR_FEEDBACK_MAX_CHARS: Final[int] = 800
_FAILURE_SUMMARY_LIMIT: Final[int] = 3
_HYPERPARAM_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "n_estimators": re.compile(r"(n_estimators\s*=\s*)(\d+)", re.IGNORECASE),
    "learning_rate": re.compile(r"(learning_rate\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_depth": re.compile(r"(max_depth\s*=\s*)(\d+)", re.IGNORECASE),
    "num_leaves": re.compile(r"(num_leaves\s*=\s*)(\d+)", re.IGNORECASE),
    "min_child_samples": re.compile(r"(min_child_samples\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_leaf": re.compile(r"(min_samples_leaf\s*=\s*)(\d+)", re.IGNORECASE),
    "min_samples_split": re.compile(r"(min_samples_split\s*=\s*)(\d+)", re.IGNORECASE),
    "max_features": re.compile(r"(max_features\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "subsample": re.compile(r"(subsample\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "feature_fraction": re.compile(r"(feature_fraction\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "bagging_fraction": re.compile(r"(bagging_fraction\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "bagging_freq": re.compile(r"(bagging_freq\s*=\s*)(\d+)", re.IGNORECASE),
    "colsample_bytree": re.compile(r"(colsample_bytree\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "n_neighbors": re.compile(r"(n_neighbors\s*=\s*)(\d+)", re.IGNORECASE),
    "leaf_size": re.compile(r"(leaf_size\s*=\s*)(\d+)", re.IGNORECASE),
    "p": re.compile(r"(?<!\w)(p\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "weights": re.compile(r"(weights\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "metric": re.compile(r"(metric\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "algorithm": re.compile(r"(algorithm\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "objective": re.compile(r"(objective\s*=\s*)(\"[^\"]+\"|'[^']+'|\w+)", re.IGNORECASE),
    "huber_delta": re.compile(r"(huber_delta\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "quantile_alpha": re.compile(r"(quantile_alpha\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "loss_weight": re.compile(r"(loss_weight\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "lambda_l1": re.compile(r"(lambda_l1\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "lambda_l2": re.compile(r"(lambda_l2\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "min_split_gain": re.compile(r"(min_split_gain\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "min_child_weight": re.compile(r"(min_child_weight\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_bin": re.compile(r"(max_bin\s*=\s*)(\d+)", re.IGNORECASE),
    "variance_threshold": re.compile(r"(VarianceThreshold\(threshold\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "C": re.compile(r"(\bC\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "alpha": re.compile(r"(alpha\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "l1_ratio": re.compile(r"(l1_ratio\s*=\s*)([\d\.]+)", re.IGNORECASE),
    "max_iter": re.compile(r"(max_iter\s*=\s*)(\d+)", re.IGNORECASE),
}
_HYPERPARAM_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "learning_rate": (0.001, 1.0),
    "subsample": (0.1, 1.0),
    "max_features": (0.1, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "feature_fraction": (0.3, 1.0),
    "bagging_fraction": (0.3, 1.0),
    "lambda_l1": (0.0, 5.0),
    "lambda_l2": (0.0, 5.0),
    "min_split_gain": (0.0, 1.0),
    "min_child_weight": (0.0, 10.0),
    "max_bin": (63.0, 511.0),
    "C": (0.01, 100.0),
    "alpha": (0.0001, 10.0),
    "l1_ratio": (0.0, 1.0),
    "num_leaves": (16.0, 256.0),
    "min_child_samples": (5.0, 100.0),
    "leaf_size": (10.0, 80.0),
    "p": (1.0, 5.0),
    "huber_delta": (0.5, 5.0),
    "quantile_alpha": (0.05, 0.95),
    "loss_weight": (0.05, 0.95),
    "variance_threshold": (0.0, 0.1),
}
_HYPERPARAM_INTEGER_KEYS: Final[set[str]] = {
    "n_estimators",
    "max_depth",
    "num_leaves",
    "min_child_samples",
    "min_samples_leaf",
    "min_samples_split",
    "bagging_freq",
    "n_neighbors",
    "leaf_size",
    "max_bin",
    "max_iter",
}
_CATEGORICAL_HYPERPARAMS: Final[dict[str, tuple[str, ...]]] = {
    "weights": ("uniform", "distance"),
    "metric": ("minkowski", "euclidean", "manhattan", "chebyshev"),
    "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
}
_KNN_PARAM_KEYS: Final[frozenset[str]] = frozenset({"n_neighbors", "weights", "metric", "p", "leaf_size", "algorithm"})
_MODEL_SWAPS: Final[dict[str, str]] = {
    "RandomForestClassifier": "GradientBoostingClassifier",
    "RandomForestRegressor": "GradientBoostingRegressor",
    "GradientBoostingClassifier": "RandomForestClassifier",
    "GradientBoostingRegressor": "RandomForestRegressor",
    "ExtraTreesClassifier": "RandomForestClassifier",
    "ExtraTreesRegressor": "RandomForestRegressor",
    "HistGradientBoostingClassifier": "GradientBoostingClassifier",
    "HistGradientBoostingRegressor": "GradientBoostingRegressor",
    "LogisticRegression": "LinearSVC",
    "LinearRegression": "Ridge",
    "KNeighborsRegressor": "GradientBoostingRegressor",
}
_MODEL_IMPORTS: Final[dict[str, str]] = {
    "RandomForestClassifier": "sklearn.ensemble",
    "RandomForestRegressor": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "GradientBoostingRegressor": "sklearn.ensemble",
    "ExtraTreesClassifier": "sklearn.ensemble",
    "ExtraTreesRegressor": "sklearn.ensemble",
    "HistGradientBoostingClassifier": "sklearn.ensemble",
    "HistGradientBoostingRegressor": "sklearn.ensemble",
    "KNeighborsRegressor": "sklearn.neighbors",
    "LogisticRegression": "sklearn.linear_model",
    "LinearRegression": "sklearn.linear_model",
    "LinearSVC": "sklearn.svm",
    "Ridge": "sklearn.linear_model",
}
_MODEL_FAMILY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("random_forest", re.compile(r"\b(RandomForestClassifier|RandomForestRegressor)\b")),
    ("extra_trees", re.compile(r"\b(ExtraTreesClassifier|ExtraTreesRegressor)\b")),
    ("gradient_boosting", re.compile(r"\b(GradientBoostingClassifier|GradientBoostingRegressor)\b")),
    ("hist_gradient_boosting", re.compile(r"\b(HistGradientBoostingClassifier|HistGradientBoostingRegressor)\b")),
    ("lightgbm", re.compile(r"\b(LGBMClassifier|LGBMRegressor)\b")),
    (
        "linear",
        re.compile(r"\b(LogisticRegression|LinearRegression|Ridge|Lasso|ElasticNet|SGDClassifier|SGDRegressor)\b"),
    ),
    ("svm", re.compile(r"\b(LinearSVC|LinearSVR|SVC|SVR)\b")),
    ("knn", re.compile(r"\b(KNeighborsClassifier|KNeighborsRegressor)\b")),
    ("naive_bayes", re.compile(r"\b(GaussianNB|MultinomialNB|BernoulliNB)\b")),
    ("tree", re.compile(r"\b(DecisionTreeClassifier|DecisionTreeRegressor)\b")),
    ("ensemble", re.compile(r"\b(BaggingClassifier|BaggingRegressor|AdaBoostClassifier|AdaBoostRegressor)\b")),
    ("stacking", re.compile(r"\b(StackingClassifier|StackingRegressor)\b")),
)
_COMPLEXITY_BINS: Final[tuple[int, ...]] = (120, 200, 320, 480, 640)
_MAX_HINTS_ACTIVE: Final[int] = 8
_MAX_HINTS_SUPPRESSED: Final[int] = 3
_HINT_SNIPPET_MAX_CHARS: Final[int] = 220
_HINT_COMMENT_PREFIX: Final[str] = "# Applied hint: "
_ENCODING_HINT_IDS: Final[frozenset[str]] = frozenset(
    {"onehot_low_cardinality", "target_encode_high_cardinality", "frequency_encode_high_cardinality", "ordinal_encode"}
)


@dataclass(frozen=True, slots=True)
class EvolutionArchiveEntry:
    """Tracked candidate for MAP-Elites-style sampling.

    @pattern:
        name: archive-entry
        rationale: "Encapsulates elite metadata for sampling."
        violations: "Unstructured elites make selection brittle."
    """

    code: str
    fitness: float
    cv_score: float
    complexity: int
    complexity_bin: int
    model_family: str
    signature: str

    def to_payload(
        self, *, max_chars: Annotated[int, Doc("Maximum code characters to include."), Range(0, 100_000)]
    ) -> dict[str, Any]:
        """Serialize entry for tool outputs.

        @notice: |
            Converts the archive entry into a JSON-serializable payload.

        @effects:
            state:
                - none
        """
        truncated = False
        code = self.code
        if max_chars > 0 and len(code) > max_chars:
            code = code[:max_chars].rstrip() + "\n# ... truncated"
            truncated = True
        return {
            "fitness": self.fitness,
            "cv_score": self.cv_score,
            "complexity": self.complexity,
            "complexity_bin": self.complexity_bin,
            "model_family": self.model_family,
            "signature": self.signature,
            "code": code,
            "truncated": truncated,
        }


class EvolverSettings(BaseSettings):
    """Configuration for the Evolver agent.

    @notice: |
        Configuration for the Evolver agent.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: settings
            rationale: "Centralizes evolutionary optimization configuration."
            violations: "Ad-hoc overrides lead to unstable evolution runs."
    """

    model_config = SettingsConfigDict(env_prefix="EVOLVER_", env_file=".env", extra="ignore", validate_default=True)
    model: str = Field(default=DEFAULT_MODEL, description="Model identifier for evolution tasks")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature for evolution prompts")
    max_tokens: int = Field(default=4096, ge=1, description="Maximum tokens for responses")
    solution_timeout: int = Field(
        default=SOLUTION_EXECUTION_TIMEOUT_SECONDS,
        ge=1,
        description="Timeout for executing a candidate solution (seconds)",
    )
    tool_retries: int = Field(default=3, ge=0, description="Tool retry attempts")
    output_retries: int = Field(default=4, ge=0, description="Output validation retry attempts")
    population_size: int = Field(default=EVOLUTION_POPULATION_SIZE, ge=1, description="Population size for evolution")
    max_generations: int = Field(default=MAX_EVOLUTION_GENERATIONS, ge=1, description="Maximum evolution generations")
    min_generations: int = Field(default=0, ge=0, description="Minimum generations before convergence checks")
    convergence_threshold: int = Field(default=5, ge=1, description="Generations without improvement before stopping")
    enable_thinking: bool = Field(default=True, description="Enable extended reasoning mode for supported models")
    thinking_budget_tokens: int = Field(default=4096, ge=0, description="Token budget for model thinking mode")
    enable_kaggle_mcp: bool = Field(default=False, description="Enable Kaggle MCP tool access")
    kaggle_mcp_url: str = Field(default=DEFAULT_KAGGLE_MCP_URL, description="Kaggle MCP endpoint")
    enable_submission_tool: bool = Field(default=False, description="Allow submissions during evolution")
    cascade_evaluation: bool = Field(default=True, description="Enable cascade evaluation for candidate filtering")
    cascade_stage1_rows: int = Field(default=300, ge=0, description="Max rows for quick evaluation stage")
    cascade_stage1_timeout: int = Field(default=45, ge=1, description="Timeout for quick evaluation stage")
    cascade_relative_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Fraction of best fitness required to run full evaluation"
    )
    cascade_floor_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Minimum quick fitness required to run full evaluation"
    )
    elite_sample_top: int = Field(default=3, ge=0, description="Default top elites to sample")
    elite_sample_diverse: int = Field(default=2, ge=0, description="Default diverse elites to sample")
    elite_code_max_chars: int = Field(default=8000, ge=256, description="Max chars per elite code snippet")
    use_openevolve: bool = Field(default=False, description="Use OpenEvolve for mutation and selection")

    @model_validator(mode="after")
    def validate_evolution_params(self) -> Self:
        """Validate cross-field evolution configuration."""
        if self.min_generations > self.max_generations:
            raise ValueError("min_generations cannot exceed max_generations")
        if self.convergence_threshold > self.max_generations:
            raise ValueError("convergence_threshold cannot exceed max_generations")
        if self.population_size < 2 and self.max_generations > 1:
            raise ValueError("population_size must be >= 2 when running multiple generations")
        return self

    @property
    def model_settings(self) -> ModelSettings:
        """Build ModelSettings from configuration."""
        settings: ModelSettings = {"temperature": self.temperature, "max_tokens": self.max_tokens}

        if self.enable_thinking and "anthropic" in self.model:
            logfire.info("evolver_thinking_disabled", model=self.model, reason="anthropic_output_tools_incompatible")
            return settings

        return settings


@dataclass
class EvolverDeps:
    """Dependencies for the Evolver agent.

    @notice: |
        Dependencies for the Evolver agent.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: dependency-container
            rationale: "Groups runtime services and evolution state."
            violations: "Scattered state makes evolution hard to resume."

        @collaborators:
            required:
                - agent_k.core.protocols:PlatformAdapter
                - agent_k.ui.agui:EventEmitter
                - agent_k.core.models:Competition
            optional:
                - agent_k.core.tracking:ExperimentTracker
                - agent_k.core.hints:HintEffectivenessTracker
            injection: constructor
            lifecycle: "Allocated per evolution run."

        @invariants:
            - "population_size >= 1"
            - "max_generations >= min_generations"
    """

    competition: Competition
    event_emitter: EventEmitter
    platform_adapter: PlatformAdapter
    data_dir: Path
    train_path: Path
    test_path: Path
    sample_path: Path
    target_columns: list[str]
    train_target_columns: list[str]
    id_column: str
    problem_profile: ProblemProfile | None = None
    technique_policy: TechniquePolicy | None = None
    fitness_policy: FitnessPolicy | None = None
    initial_solution: str = ""
    population_size: int = EVOLUTION_POPULATION_SIZE
    max_generations: int = MAX_EVOLUTION_GENERATIONS
    min_generations: int = 0
    solution_timeout: int = SOLUTION_EXECUTION_TIMEOUT_SECONDS
    target_score: float = 0.0
    generation_offset: int = 0
    best_solution: str | None = None
    best_fitness: float | None = None
    improvement_count: int = 0
    min_improvements_required: int = 0
    generation_history: list[dict[str, Any]] = field(default_factory=list)
    elite_archive: dict[tuple[int, str], EvolutionArchiveEntry] = field(default_factory=dict)
    experiment_tracker: ExperimentTracker | None = None
    dataset_profile: DatasetProfile | None = None
    preprocessing_hints: list[PreprocessingHint] = field(default_factory=list)
    hint_tracker: HintEffectivenessTracker | None = None
    suppressed_hints: set[str] = field(default_factory=set)
    failure_counts: dict[str, int] = field(default_factory=dict)
    last_error_feedback: str | None = None


class EvolutionResult(BaseModel):
    """Result of evolution process.

    @notice: |
        Result of evolution process.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: output-model
            rationale: "Stable schema for successful evolution outputs."
            violations: "Free-form outputs hinder submission automation."
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    best_solution: str = Field(description="Best solution code")
    best_fitness: float = Field(description="Fitness score of best solution")
    generations_completed: int = Field(default=0, ge=0, description="Number of generations completed")
    convergence_achieved: bool = Field(default=False, description="Whether convergence criteria were met")
    convergence_reason: str | None = Field(default=None, description="Reason for convergence if achieved")
    submission_ready: bool = Field(default=False, description="Whether output is ready for submission")


class EvolutionFailure(BaseModel):
    """Failure result for evolution process.

    @notice: |
        Failure result for evolution process.

    @dev: |
        See module for implementation details and extension points.

        @pattern:
            name: output-model
            rationale: "Stable schema for failed evolution outputs."
            violations: "Opaque errors make recovery harder."
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, validate_default=True)
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version")
    error_type: str = Field(description="Classification of failure")
    error_message: str = Field(description="Human-readable error")
    partial_solution: str | None = Field(default=None, description="Best available solution snippet, if any")
    recoverable: bool = Field(default=True, description="Whether the failure is likely recoverable")


EVOLUTION_OUTPUT_TYPE: Final[list[Any]] = [
    ToolOutput[EvolutionResult](EvolutionResult, name="return_success"),
    ToolOutput[EvolutionFailure](EvolutionFailure, name="return_failure"),
    DeferredToolRequests,
]


class EvolverAgent(MemoryMixin):
    """Evolver agent encapsulating evolutionary optimization functionality.

    This class wraps the pydantic-ai Agent and provides all evolution tools
    as instance methods for cleaner organization and testing.

    @notice: |
        Optimizes prototype solutions via evolutionary search.
        Use the module-level evolver_agent or agent registry.

    @dev: |
        Registers evolution tools and coordinates mutation/evaluation cycles.

    @pattern:
        name: agent-singleton
        rationale: "Single instance keeps memory/tool registration consistent."
        violations: "Multiple instances duplicate tool registrations."

    @collaborators:
        required:
            - agent_k.core.protocols:PlatformAdapter
            - agent_k.ui.agui:EventEmitter
        optional:
            - agent_k.adapters.openevolve:OpenEvolveRunner
        injection: deps via RunContext
        lifecycle: "Module-level singleton at import time."

    @concurrency:
        model: asyncio
        safe: false
        reason: "Mutates evolution state and caches."

    @invariants:
        - "self._agent is initialized after __init__ completes."
        - "self._toolset registers evolution tools exactly once."
    """

    def __init__(
        self,
        settings: Annotated[EvolverSettings | None, Doc("Optional settings override.")] = None,
        *,
        register: Annotated[bool, Doc("Register agent in global registry.")] = True,
    ) -> None:
        """Initialize the Evolver agent.

        @notice: |
            Builds the agent singleton and registers tools.

        @dev: |
            Initializes memory backend, toolset, and pydantic-ai Agent.

        @state-changes:
            - self._settings
            - self._toolset
            - self._agent
        """
        self._settings = settings or EvolverSettings()
        self._toolset: FunctionToolset[EvolverDeps] = FunctionToolset(id="evolver")
        self._memory_backend = self._init_memory_backend()
        self._register_tools()
        self._agent = self._create_agent()
        if register:
            register_agent("evolver", self._agent)
        self._setup_memory()

    @property
    def agent(self) -> Agent[EvolverDeps, EvolutionResult | EvolutionFailure]:
        """Return the underlying pydantic-ai Agent."""
        return self._agent

    @property
    def settings(self) -> EvolverSettings:
        """Return current settings."""
        return self._settings

    async def run_openevolve(
        self,
        deps: Annotated[EvolverDeps, Doc("Evolution dependencies and state.")],
        *,
        base_prompt: Annotated[str | None, Doc("Optional base prompt override.")] = None,
        model_specs: Annotated[list[str] | None, Doc("Optional model specs for OpenEvolve.")] = None,
    ) -> EvolutionResult | EvolutionFailure:
        """Run OpenEvolve-backed evolution for a prototype solution.

        @notice: |
            Delegates mutation and evaluation to OpenEvolve when enabled.

        @effects:
            io:
                - OpenEvolve API requests
            state:
                - deps.best_solution
                - deps.best_fitness
        """
        with logfire.span("evolver.openevolve"):
            initial_program = deps.initial_solution or deps.best_solution or ""
            baseline_score = self._score_from_fitness(deps.best_fitness, deps.competition.metric_direction)
            specs = [spec.strip() for spec in (model_specs or []) if isinstance(spec, str) and spec.strip()]
            if not specs:
                specs = [self._settings.model]

            runner = OpenEvolveRunner(
                work_dir=deps.data_dir,
                hints=deps.preprocessing_hints,
                model_specs=specs,
                metric_direction=deps.competition.metric_direction,
                validation_split=0.2,
                timeout_seconds=deps.solution_timeout,
                base_prompt=base_prompt,
            )
            target_fitness = None
            if deps.target_score:
                target_fitness = self._fitness_from_score(deps.target_score, deps.competition.metric_direction)

            try:
                result = await runner.run_evolution(
                    initial_program=initial_program, max_iterations=deps.max_generations, target_score=target_fitness
                )
            except Exception as exc:
                logfire.error("openevolve_failed", error=str(exc))
                return EvolutionFailure(
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    partial_solution=initial_program or deps.best_solution,
                    recoverable=True,
                )

            best_solution = result.get("best_solution") or initial_program
            best_fitness = float(result.get("best_fitness") or 0.0)
            programs = result.get("programs") or []

            deps.best_solution = best_solution
            deps.best_fitness = best_fitness
            deps.generation_history, deps.improvement_count = self._summarize_openevolve_history(
                programs, deps.population_size, deps.competition.metric_direction
            )
            self._record_openevolve_hint_attempts(deps, programs, baseline_score)

            best_score = self._score_from_fitness(best_fitness, deps.competition.metric_direction)
            convergence_achieved = False
            convergence_reason = None
            if best_score is not None:
                if deps.competition.metric_direction == "minimize" and best_score <= deps.target_score:
                    convergence_achieved = True
                    convergence_reason = "target_score"
                elif deps.competition.metric_direction == "maximize" and best_score >= deps.target_score:
                    convergence_achieved = True
                    convergence_reason = "target_score"

            return EvolutionResult(
                best_solution=best_solution,
                best_fitness=best_fitness,
                generations_completed=len(deps.generation_history),
                convergence_achieved=convergence_achieved,
                convergence_reason=convergence_reason,
                submission_ready=False,
            )

    async def mutate_solution(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: Annotated[str, Doc("Solution code to mutate.")],
        mutation_type: Annotated[
            str, Doc("Mutation type (point, structural, hyperparameter, crossover, hint_injection).")
        ],
        mutation_params: Annotated[dict[str, Any] | None, Doc("Optional parameters for the mutation.")] = None,
    ) -> str:
        """Apply mutation to a solution.

        @notice: |
            Applies a mutation strategy and returns the mutated code.

        @effects:
            state:
                - none
        """
        with logfire.span("evolver.mutate", mutation_type=mutation_type):
            await ctx.deps.event_emitter.emit(
                "tool-start",
                {
                    "taskId": "evolution_mutate",
                    "toolCallId": f"mutate_{mutation_type}",
                    "toolType": "code_executor",
                    "operation": f"mutate_{mutation_type}",
                },
            )

            params = dict(mutation_params or {})
            if mutation_type == "hyperparameter" and "magnitude" not in params:
                params["magnitude"] = self._adaptive_magnitude(ctx.deps)
            mutations = {
                "crossover": lambda: self._apply_crossover(solution_code, params.get("other_solution", ""), params),
                "hyperparameter": lambda: self._apply_hyperparameter_mutation(solution_code, params),
                "hint_injection": lambda: self._apply_hint_injection(ctx, solution_code, params),
                "point": lambda: self._apply_point_mutation(solution_code, params),
                "structural": lambda: self._apply_structural_mutation(solution_code, params),
            }
            mutated = mutations.get(mutation_type, lambda: solution_code)()
            mutated = self._apply_solution_policy(ctx, mutated)
            mutated = self._ensure_hint_applied(ctx, mutated, params)
            if not self._is_valid_python(mutated):
                logfire.warning("evolver_mutation_invalid", mutation_type=mutation_type)
                fallback = self._apply_solution_policy(ctx, solution_code)
                return self._ensure_hint_applied(ctx, fallback, params)
            if self._has_invalid_knn_params(mutated):
                logfire.warning("evolver_mutation_invalid_params", mutation_type=mutation_type)
                fallback = self._apply_solution_policy(ctx, solution_code)
                return self._ensure_hint_applied(ctx, fallback, params)
            return mutated

    async def evaluate_fitness(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: Annotated[str, Doc("Solution code to evaluate.")],
        validation_split: Annotated[float, Doc("Fraction of data for validation."), Range(0.0, 0.9)] = 0.2,
    ) -> ToolReturn:
        """Evaluate solution fitness.

        @notice: |
            Runs evaluation and emits fitness telemetry.

        @effects:
            io:
                - local execution
            state:
                - ctx.deps.best_fitness
                - ctx.deps.best_solution
        """
        with logfire.span("evolver.evaluate_fitness"):
            tool_call_id = f"fitness_{id(solution_code):x}"
            await ctx.deps.event_emitter.emit_tool_start(
                task_id="evolution_evaluate",
                tool_call_id=tool_call_id,
                tool_type="code_executor",
                operation="evaluate_fitness",
            )

            solution_code = self._apply_solution_policy(ctx, solution_code)
            original_has_hints = _HINT_COMMENT_PREFIX in solution_code
            solution_code = self._ensure_hint_applied(ctx, solution_code, {})
            if ctx.deps.preprocessing_hints and _HINT_COMMENT_PREFIX not in solution_code:
                hint = self._select_hint_for_injection(ctx, {}, applied=set())
                if hint is not None:
                    solution_code = self._append_hint_comment(solution_code, hint)
            modified_has_hints = _HINT_COMMENT_PREFIX in solution_code
            logfire.info(
                "evaluating_with_hints",
                original_has_hints=original_has_hints,
                modified_has_hints=modified_has_hints,
                hints_available=len(ctx.deps.preprocessing_hints),
            )
            previous_best_fitness = ctx.deps.best_fitness
            result = await self._run_evaluation(ctx, solution_code, validation_split=validation_split)
            eligible_for_archive = result["valid"] and result.get("stage") != "stage1"
            improvement = False
            improvement_delta: float | None = None

            self._update_hint_tracking(ctx, solution_code, result, previous_best_fitness)

            if eligible_for_archive:
                if ctx.deps.best_fitness is None or result["fitness"] > ctx.deps.best_fitness:
                    previous_best = ctx.deps.best_fitness
                    ctx.deps.best_fitness = result["fitness"]
                    ctx.deps.best_solution = solution_code
                    if previous_best is not None:
                        ctx.deps.improvement_count += 1
                        improvement = True
                        improvement_delta = result["fitness"] - previous_best

            if result["valid"]:
                archive_entry = self._build_archive_entry(solution_code, result["fitness"], result["cv_score"])
                if eligible_for_archive:
                    self._update_elite_archive(ctx.deps, archive_entry)
                result.update(
                    {
                        "complexity": archive_entry.complexity,
                        "complexity_bin": archive_entry.complexity_bin,
                        "model_family": archive_entry.model_family,
                        "archive_size": len(ctx.deps.elite_archive),
                        "improvement_count": ctx.deps.improvement_count,
                        "improved": improvement,
                        "improvement_delta": improvement_delta,
                    }
                )

                await ctx.deps.event_emitter.emit(
                    "fitness-update",
                    {
                        "fitness": result["fitness"],
                        "cv_score": result["cv_score"],
                        "validation_split": validation_split,
                        "stage": result.get("stage", "full"),
                        "improvement_count": ctx.deps.improvement_count,
                        "improved": improvement,
                    },
                )
            else:
                await ctx.deps.event_emitter.emit_tool_error(
                    task_id="evolution_evaluate",
                    tool_call_id=tool_call_id,
                    error=result.get("error") or "Invalid solution",
                )

            await ctx.deps.event_emitter.emit_tool_result(
                task_id="evolution_evaluate", tool_call_id=tool_call_id, result=result, duration_ms=result["runtime_ms"]
            )

            summary = f"Fitness {result['fitness']:.4f}, CV {result['cv_score']:.4f}, valid={result['valid']}"
            if not result["valid"] and result.get("error_category"):
                summary = f"{summary}, error={result['error_category']}"
            return ToolReturn(
                return_value=result,
                content=summary,
                metadata={"tool_call_id": tool_call_id, "runtime_ms": result["runtime_ms"]},
            )

    async def record_generation(
        self,
        ctx: RunContext[EvolverDeps],
        generation: Annotated[int, Doc("Generation index (0-based)."), Range(0, 10_000)],
        best_fitness: Annotated[float, Doc("Best fitness in generation.")],
        mean_fitness: Annotated[float, Doc("Mean fitness in generation.")],
        worst_fitness: Annotated[float, Doc("Worst fitness in generation.")],
        mutations: Annotated[dict[str, int], Doc("Mutation counts for the generation.")],
    ) -> None:
        """Record generation metrics.

        @notice: |
            Appends generation metrics and emits telemetry.

        @effects:
            state:
                - ctx.deps.generation_history
        """
        global_generation = generation + ctx.deps.generation_offset
        metrics = {
            "generation": global_generation,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "worst_fitness": worst_fitness,
            "mutations": mutations,
        }

        ctx.deps.generation_history.append(metrics)
        await ctx.deps.event_emitter.emit_generation_complete(
            generation=global_generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            worst_fitness=worst_fitness,
            population_size=ctx.deps.population_size,
            mutations=mutations,
        )

        logfire.info(
            "evolution_generation", generation=global_generation, best_fitness=best_fitness, mean_fitness=mean_fitness
        )

    async def check_convergence(
        self,
        ctx: RunContext[EvolverDeps],
        threshold_generations: Annotated[int, Doc("Generations to check for improvement."), Range(1, 1000)] = 5,
        improvement_threshold: Annotated[float, Doc("Minimum improvement required."), Range(0.0, 10.0)] = 0.001,
    ) -> ToolReturn:
        """Check if evolution has converged.

        @notice: |
            Determines whether fitness has plateaued or target score reached.

        @effects:
            state:
                - none
        """
        history = ctx.deps.generation_history
        policy = self._resolve_technique_policy(ctx.deps)
        if policy is not None:
            if len(ctx.deps.elite_archive) < policy.min_elite_archive_size:
                result = {
                    "converged": False,
                    "reason": (
                        f"Elite archive too small ({len(ctx.deps.elite_archive)}/{policy.min_elite_archive_size})"
                    ),
                }
                return ToolReturn(return_value=result, content=json.dumps(result))
            improvement_threshold = max(improvement_threshold, policy.fitness_improvement_threshold)
        if ctx.deps.min_generations and len(history) < ctx.deps.min_generations:
            result = {
                "converged": False,
                "reason": f"Minimum generations not reached ({len(history)}/{ctx.deps.min_generations})",
            }
            return ToolReturn(return_value=result, content=json.dumps(result))

        if len(history) < threshold_generations:
            result = {"converged": False, "reason": "Not enough generations"}
            return ToolReturn(return_value=result, content=json.dumps(result))

        if ctx.deps.min_improvements_required and ctx.deps.improvement_count < ctx.deps.min_improvements_required:
            result = {
                "converged": False,
                "reason": (
                    "Minimum improvements not reached "
                    f"({ctx.deps.improvement_count}/{ctx.deps.min_improvements_required})"
                ),
                "improvement_count": ctx.deps.improvement_count,
            }
            return ToolReturn(return_value=result, content=json.dumps(result))

        recent_fitness = [g["best_fitness"] for g in history[-threshold_generations:]]
        best = max(recent_fitness)
        improvement = best - min(recent_fitness)
        if improvement < improvement_threshold:
            result = {
                "converged": True,
                "reason": f"No improvement for {threshold_generations} generations",
                "best_fitness": best,
            }
            return ToolReturn(return_value=result, content=json.dumps(result))

        if ctx.deps.target_score > 0:
            target_fitness = self._fitness_from_score(ctx.deps.target_score, ctx.deps.competition.metric_direction)
            if best >= target_fitness:
                result = {"converged": True, "reason": "Target score achieved", "best_fitness": best}
                return ToolReturn(return_value=result, content=json.dumps(result))

        result = {"converged": False, "reason": "Evolution in progress", "recent_improvement": improvement}
        return ToolReturn(return_value=result, content=json.dumps(result))

    async def sample_elites(
        self,
        ctx: RunContext[EvolverDeps],
        num_top: Annotated[int | None, Doc("Number of top elites to sample.")] = None,
        num_diverse: Annotated[int | None, Doc("Number of diverse elites to sample.")] = None,
    ) -> ToolReturn:
        """Sample elite solutions for prompt construction.

        @notice: |
            Selects top and diverse elites from the archive.

        @effects:
            state:
                - none
        """
        top = self._settings.elite_sample_top if num_top is None else max(0, num_top)
        diverse = self._settings.elite_sample_diverse if num_diverse is None else max(0, num_diverse)
        entries = self._select_elite_samples(ctx.deps, top=top, diverse=diverse)
        payload = [entry.to_payload(max_chars=self._settings.elite_code_max_chars) for entry in entries]
        summary = f"Sampled {len(payload)} elites from {len(ctx.deps.elite_archive)} archive cells."
        return ToolReturn(return_value=payload, content=summary)

    async def submit_to_kaggle(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: Annotated[str, Doc("Solution code to submit.")],
        message: Annotated[str, Doc("Submission message.")] = "AGENT-K submission",
    ) -> ToolReturn:
        """Submit solution to Kaggle via the platform adapter.

        @notice: |
            Writes a submission file and triggers adapter submission.

        @effects:
            io:
                - local filesystem access
                - Kaggle API request
        """
        with logfire.span("evolver.submit", competition_id=ctx.deps.competition.id):
            tool_call_id = f"submit_{len(ctx.deps.generation_history)}"
            await ctx.deps.event_emitter.emit(
                "tool-start",
                {
                    "taskId": "evolution_submit",
                    "toolCallId": tool_call_id,
                    "toolType": "kaggle_mcp",
                    "operation": "competitions.submit",
                },
            )

            result = await self._submit_solution(ctx, solution_code, message=message)
            if result.get("status") == "failed":
                await ctx.deps.event_emitter.emit_tool_error(
                    task_id="evolution_submit",
                    tool_call_id=tool_call_id,
                    error=result.get("error", "Submission failed"),
                )
                summary = f"Submission failed: {result.get('error', 'Unknown error')}"
                return ToolReturn(return_value=result, content=summary)

            await ctx.deps.event_emitter.emit_tool_result(
                task_id="evolution_submit",
                tool_call_id=tool_call_id,
                result=result,
                duration_ms=result.get("runtime_ms", 0),
            )

            summary = f"Submission status: {result.get('status', 'unknown')}"
            return ToolReturn(return_value=result, content=summary)

    def _create_agent(self) -> Agent[EvolverDeps, EvolutionResult | EvolutionFailure]:
        """Create the underlying pydantic-ai agent.

        @factory-for:
            id: agent_k.agents.evolver:EvolverAgent
            rationale: "Centralizes agent wiring and toolset preparation."
            singleton: true
            cache-key: "module"

        @canonical-home:
            for:
                - "evolver agent construction"
            notes: "Use EvolverAgent() or module singleton."
        """
        builtin_tools: list[Any] = [prepare_code_execution_tool]
        if self._settings.enable_kaggle_mcp:
            builtin_tools.insert(0, MCPServerTool(id="kaggle", url=self._settings.kaggle_mcp_url))
        if self._memory_backend is not None:
            builtin_tools.append(prepare_memory_tool)

        require_approval = ["submit_to_kaggle"] if self._settings.enable_submission_tool else None
        agent: Agent[EvolverDeps, EvolutionResult | EvolutionFailure] = Agent(
            model=get_model(self._settings.model),
            deps_type=EvolverDeps,
            output_type=EVOLUTION_OUTPUT_TYPE,
            instructions=EVOLVER_SYSTEM_PROMPT,
            name="evolver",
            model_settings=self._settings.model_settings,
            retries=self._settings.tool_retries,
            output_retries=self._settings.output_retries,
            builtin_tools=builtin_tools,
            toolsets=[
                create_production_toolset(
                    [self._toolset, cast("FunctionToolset[EvolverDeps]", code_toolset)],
                    require_approval_for=require_approval,
                )
            ],
            prepare_tools=universal_tool_preparation,
            instrument=True,
        )

        agent.output_validator(self._validate_evolution_result)
        agent.instructions(self._add_evolution_context)
        return agent

    def _register_tools(self) -> None:
        """Register all evolution tools with the toolset."""
        self._toolset.tool(self.mutate_solution)
        self._toolset.tool(self.evaluate_fitness)
        self._toolset.tool(self.record_generation)
        self._toolset.tool(self.check_convergence)
        self._toolset.tool(self.sample_elites)
        if self._settings.enable_submission_tool:
            self._toolset.tool(requires_approval=True)(self.submit_to_kaggle)

    async def _validate_evolution_result(
        self, ctx: RunContext[EvolverDeps], result: EvolutionResult | EvolutionFailure
    ) -> EvolutionResult | EvolutionFailure:
        """Validate evolution results."""
        if ctx.partial_output:
            return result
        match result:
            case EvolutionFailure(error_type=error_type, error_message=error_message) if (
                not error_type or not error_message
            ):
                raise ModelRetry("EvolutionFailure must include error_type and error_message.")
            case EvolutionResult(best_fitness=best_fitness) if best_fitness < 0:
                raise ModelRetry("best_fitness must be >= 0. Provide a valid fitness score.")
            case EvolutionResult(best_solution=best_solution) if not best_solution:
                raise ModelRetry("best_solution is required. Provide the best solution code.")
        return result

    async def _add_evolution_context(self, ctx: RunContext[EvolverDeps]) -> str:
        """Add evolution-specific context to instructions."""
        comp = ctx.deps.competition
        deps = ctx.deps
        sections = [
            (
                "COMPETITION CONTEXT:\n"
                f"- ID: {comp.id}\n"
                f"- Title: {comp.title}\n"
                f"- Metric: {comp.metric.value} ({comp.metric_direction})\n"
                f"- Target Score: {deps.target_score}"
            ),
            (
                "EVOLUTION STATE:\n"
                f"- Generations Completed: {len(deps.generation_history)}\n"
                f"- Population Size: {deps.population_size}\n"
                f"- Max Generations: {deps.max_generations}\n"
                f"- Minimum Generations: {deps.min_generations}\n"
                f"- Improvements: {deps.improvement_count}\n"
                f"- Min Improvements Required: {deps.min_improvements_required}\n"
                f"- Generation Offset: {deps.generation_offset}"
            ),
        ]
        if deps.generation_history:
            last_gen = deps.generation_history[-1]
            sections.append(
                "LAST GENERATION:\n"
                f"- Best Fitness: {last_gen.get('best_fitness', 'N/A')}\n"
                f"- Mean Fitness: {last_gen.get('mean_fitness', 'N/A')}"
            )
        if deps.best_solution:
            sections.append(f"BEST SOLUTION AVAILABLE: {len(deps.best_solution)} chars")
        if deps.elite_archive:
            families = sorted({entry.model_family for entry in deps.elite_archive.values()})
            family_preview = ", ".join(families[:6])
            suffix = "..." if len(families) > 6 else ""
            sections.append(
                "ELITE ARCHIVE:\n"
                f"- Cells: {len(deps.elite_archive)}\n"
                f"- Families: {family_preview}{suffix}\n"
                "- Use sample_elites to retrieve top + diverse candidates."
            )
        policy = self._resolve_technique_policy(deps)
        if policy is not None:
            sections.append(
                "TECHNIQUE POLICY:\n"
                f"- Min Generations: {policy.min_generations}\n"
                f"- Min Population: {policy.min_population_size}\n"
                f"- Elite Archive Min: {policy.min_elite_archive_size}\n"
                f"- Outlier Clipping: {policy.enable_outlier_clipping}\n"
                f"- Target Transform: {policy.enable_target_transform}"
            )

        if deps.failure_counts:
            summary = _summarize_failure_counts(deps.failure_counts, limit=_FAILURE_SUMMARY_LIMIT)
            if summary:
                sections.append(
                    "EXECUTION FAILURES:\n"
                    f"- Top causes: {summary}\n"
                    "- Fix execution errors before applying further mutations."
                )
        if deps.last_error_feedback:
            sections.append(f"LAST EXECUTION ERROR:\n{deps.last_error_feedback}")

        if deps.preprocessing_hints:
            prioritized = self._get_prioritized_hints(deps)
            if prioritized:
                hint_text = self._build_hint_context(prioritized)
                if hint_text:
                    sections.append(hint_text)

        return "\n\n".join(sections)

    def _get_prioritized_hints(self, deps: EvolverDeps) -> list[dict[str, Any]]:
        tracker = deps.hint_tracker
        if not deps.preprocessing_hints:
            return []
        generation = len(deps.generation_history) + deps.generation_offset
        suppressed = set(deps.suppressed_hints)
        prioritized: list[dict[str, Any]] = []

        for hint in deps.preprocessing_hints:
            priority = hint.priority
            success_rate = hint.success_rate
            last_result = None
            is_suppressed = hint.id in suppressed
            if tracker is not None:
                success_rate = tracker.get_success_rate(hint.id, deps.competition.id)
                last_attempt = tracker.get_last_attempt(hint.id, deps.competition.id)
                if last_attempt is not None:
                    last_result = self._hint_result_from_delta(last_attempt.delta)
                priority = compute_hint_priority(hint, tracker, deps.competition.id, generation)
                is_suppressed = is_suppressed or tracker.is_suppressed(hint.id, deps.competition.id)
            prioritized.append(
                {
                    "hint": hint,
                    "priority": 0.0 if is_suppressed else priority,
                    "success_rate": success_rate,
                    "last_result": last_result,
                    "suppressed": is_suppressed,
                }
            )

        prioritized.sort(key=lambda item: (item["suppressed"], -item["priority"], item["hint"].id))
        return prioritized

    def _build_hint_context(self, prioritized: list[dict[str, Any]]) -> str:
        active = [item for item in prioritized if not item["suppressed"]]
        if not active:
            return ""
        active.sort(key=lambda item: item["priority"], reverse=True)
        lines = ["## PREPROCESSING GUIDANCE (Apply at least ONE per mutation)", ""]
        for index, item in enumerate(active[:5], start=1):
            hint = item["hint"]
            title = self._hint_title(hint)
            priority = item["priority"]
            lines.extend(
                [
                    f"### Hint {index}: {title} (priority={priority:.2f})",
                    f"ID: `{hint.id}` - Add comment `# Applied hint: {hint.id}`",
                    f"Description: {hint.description}",
                    "```python",
                    hint.code_snippet.strip(),
                    "```",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _hint_title(self, hint: PreprocessingHint) -> str:
        return hint.id.replace("_", " ").title()

    def _hint_priority_label(self, priority: float) -> str:
        if priority >= 0.75:
            return "HIGH"
        if priority >= 0.5:
            return "MEDIUM"
        if priority > 0.0:
            return "LOW"
        return "SUPPRESSED"

    def _inline_hint_snippet(self, snippet: str) -> str:
        cleaned = " ".join(snippet.strip().split())
        if not cleaned:
            return ""
        if len(cleaned) > _HINT_SNIPPET_MAX_CHARS:
            cleaned = cleaned[:_HINT_SNIPPET_MAX_CHARS].rstrip() + "..."
        return cleaned

    def _hint_result_from_delta(self, delta: float) -> str:
        if delta > 0:
            return "success"
        if delta < 0:
            return "failure"
        return "neutral"

    def _apply_hint_injection(self, ctx: RunContext[EvolverDeps], code: str, params: dict[str, Any]) -> str:
        hint = self._select_hint_for_injection(ctx, params, applied=set())
        if hint is None:
            return code
        return self._inject_hint_snippet(code, hint)

    def _ensure_hint_applied(self, ctx: RunContext[EvolverDeps], code: str, params: dict[str, Any]) -> str:
        logfire.info(
            "hint_injection_attempt",
            code_length=len(code),
            hints_available=len(ctx.deps.preprocessing_hints),
            existing_markers=code.count(_HINT_COMMENT_PREFIX),
        )
        if not ctx.deps.preprocessing_hints:
            return code
        applied = detect_applied_hints(code, ctx.deps.preprocessing_hints)
        generation = len(ctx.deps.generation_history) + ctx.deps.generation_offset
        force_new = bool(ctx.deps.preprocessing_hints) and generation % 3 == 0
        if applied and not force_new:
            return code
        hint = self._select_hint_for_injection(ctx, params, applied=applied)
        if hint is None:
            return code
        injected = self._inject_hint_snippet(code, hint)
        if injected != code:
            return injected
        return self._append_hint_comment(code, hint)

    def _select_hint_for_injection(
        self, ctx: RunContext[EvolverDeps], params: dict[str, Any], *, applied: set[str]
    ) -> PreprocessingHint | None:
        hints = ctx.deps.preprocessing_hints
        if not hints:
            return None
        requested_id = str(params.get("hint_id", "")).strip()
        if requested_id:
            for hint in hints:
                if hint.id == requested_id:
                    return hint

        prioritized = self._get_prioritized_hints(ctx.deps)
        candidates: list[PreprocessingHint] = [item["hint"] for item in prioritized if not item["suppressed"]]
        if applied:
            fresh = [hint for hint in candidates if hint.id not in applied]
            if fresh:
                candidates = fresh

        tracker = ctx.deps.hint_tracker
        if tracker is not None:
            unused = [hint for hint in candidates if tracker.get_last_attempt(hint.id, ctx.deps.competition.id) is None]
            if unused:
                candidates = unused

        if not candidates:
            return None
        generation = len(ctx.deps.generation_history) + ctx.deps.generation_offset
        return candidates[generation % len(candidates)]

    def _inject_hint_snippet(self, code: str, hint: PreprocessingHint) -> str:
        marker = f"{_HINT_COMMENT_PREFIX}{hint.id}"
        if marker in code:
            return code
        block = self._build_hint_block(code, hint)
        imports, body = self._split_imports(code)
        insert_at = self._find_hint_insertion_index(body)
        if insert_at is None:
            new_body = [block, "", *body]
        else:
            new_body = [*body[:insert_at], block, "", *body[insert_at:]]
        parts = [*imports]
        if imports:
            parts.append("")
        parts.extend(new_body)
        return "\n".join(parts).strip() + "\n"

    def _build_hint_block(self, code: str, hint: PreprocessingHint) -> str:
        marker = f"{_HINT_COMMENT_PREFIX}{hint.id}"
        snippet = hint.code_snippet.strip()
        snippet_lines = snippet.splitlines() if snippet else []
        if "np." in hint.code_snippet and "import numpy as np" not in code and "import numpy as np" not in snippet:
            snippet_lines.insert(0, "import numpy as np")
        if "pd." in hint.code_snippet and "import pandas as pd" not in code and "import pandas as pd" not in snippet:
            snippet_lines.insert(0, "import pandas as pd")
        alias_lines = self._hint_alias_lines(code, hint, hint.code_snippet)
        block_lines = [marker, "try:"]
        for line in alias_lines + snippet_lines:
            block_lines.append(f"    {line}" if line.strip() else "    ")
        block_lines.extend(["except Exception:", "    pass"])
        return "\n".join(block_lines)

    def _hint_alias_lines(self, code: str, hint: PreprocessingHint, snippet: str) -> list[str]:
        lines: list[str] = []
        if re.search(r"\bdf\b", snippet) and not re.search(r"(?m)^\s*df\s*=", code):
            df_var = self._find_dataframe_var(code)
            if df_var:
                lines.append(f"df = {df_var}")
        if re.search(r"\bcols\b", snippet) and not re.search(r"(?m)^\s*cols\s*=", code):
            cols_var = self._select_cols_alias(code, hint.id)
            if cols_var:
                lines.append(f"cols = {cols_var}")
        return lines

    def _find_dataframe_var(self, code: str) -> str | None:
        matches: list[str] = re.findall(r"(?m)^(\w+)\s*=\s*pd\.read_(?:csv|parquet|feather)\(", code)
        if not matches:
            return None
        for name in matches:
            if "train" in name.lower():
                return name
        return matches[0]

    def _select_cols_alias(self, code: str, hint_id: str) -> str | None:
        if hint_id in _ENCODING_HINT_IDS and "categorical_cols" in code:
            return "categorical_cols"
        if "numeric_cols" in code:
            return "numeric_cols"
        if "categorical_cols" in code:
            return "categorical_cols"
        return None

    def _find_hint_insertion_index(self, body: list[str]) -> int | None:
        insert_at = None
        for idx, line in enumerate(body):
            if not line or line.startswith((" ", "\t")):
                continue
            if re.search(r"\bpd\.read_(csv|parquet|feather)\b", line):
                insert_at = idx + 1
        return insert_at

    def _append_hint_comment(self, code: str, hint: PreprocessingHint) -> str:
        marker = f"{_HINT_COMMENT_PREFIX}{hint.id}"
        if marker in code:
            return code
        imports, body = self._split_imports(code)
        parts = [*imports]
        if imports:
            parts.append("")
        parts.extend([marker, "", *body])
        return "\n".join(parts).strip() + "\n"

    def _update_hint_tracking(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        result: dict[str, Any],
        previous_best_fitness: float | None,
    ) -> None:
        tracker = ctx.deps.hint_tracker
        if tracker is None or not ctx.deps.preprocessing_hints:
            return
        if not result.get("valid"):
            return
        stage = result.get("stage")
        applied = detect_applied_hints(solution_code, ctx.deps.preprocessing_hints)
        logfire.info("hint_tracking_attempt", stage=stage, applied_count=len(applied))
        if not applied:
            logfire.warning("no_hints_detected_in_solution", stage=stage)
            return
        cv_score_after = result.get("cv_score")
        cv_score_before = self._score_from_fitness(previous_best_fitness, ctx.deps.competition.metric_direction)
        if cv_score_after is None or cv_score_before is None:
            return
        delta = self._score_delta(cv_score_before, cv_score_after, ctx.deps.competition.metric_direction)
        generation = len(ctx.deps.generation_history) + ctx.deps.generation_offset
        saved = False
        for hint in ctx.deps.preprocessing_hints:
            if hint.id not in applied:
                continue
            record = HintAttemptRecord(
                hint_id=hint.id,
                competition_id=ctx.deps.competition.id,
                generation=generation,
                applied=True,
                cv_score_before=cv_score_before,
                cv_score_after=cv_score_after,
                delta=delta,
            )
            tracker.record_attempt(record)
            saved = True
        if saved:
            tracker.save()

    def _summarize_openevolve_history(
        self, programs: list[dict[str, Any]], population_size: int, metric_direction: str
    ) -> tuple[list[dict[str, Any]], int]:
        history: list[dict[str, Any]] = []
        improvement_count = 0
        best_fitness: float | None = None

        for idx, summary in enumerate(sorted(programs, key=lambda item: item.get("iteration", 0)), start=1):
            fitness = summary.get("fitness")
            if not isinstance(fitness, (int, float)):
                cv_score = summary.get("cv_score")
                if isinstance(cv_score, (int, float)):
                    fitness = self._fitness_from_score(cv_score, metric_direction)
                else:
                    fitness = 0.0
            fitness = float(fitness)
            if best_fitness is None or fitness > best_fitness:
                if best_fitness is not None:
                    improvement_count += 1
                best_fitness = fitness
            history.append(
                {
                    "generation": idx,
                    "best_fitness": best_fitness or fitness,
                    "mean_fitness": fitness,
                    "worst_fitness": fitness,
                    "population_size": population_size,
                    "mutations": {"openevolve": 1},
                }
            )

        return history, improvement_count

    def _record_openevolve_hint_attempts(
        self, deps: EvolverDeps, programs: list[dict[str, Any]], baseline_score: float | None
    ) -> None:
        tracker = deps.hint_tracker
        if tracker is None or not deps.preprocessing_hints:
            return

        metric_direction = deps.competition.metric_direction
        best_score = baseline_score
        for summary in sorted(programs, key=lambda item: item.get("iteration", 0)):
            cv_score = summary.get("cv_score")
            if not isinstance(cv_score, (int, float)):
                continue
            applied = summary.get("applied_hints") or []
            if not applied:
                if best_score is None:
                    best_score = float(cv_score)
                else:
                    if metric_direction == "minimize" and cv_score < best_score:
                        best_score = float(cv_score)
                    elif metric_direction == "maximize" and cv_score > best_score:
                        best_score = float(cv_score)
                continue

            if best_score is None:
                best_score = float(cv_score)
            delta = self._score_delta(best_score, float(cv_score), metric_direction)
            generation = int(summary.get("iteration") or 0)
            for hint_id in applied:
                if not isinstance(hint_id, str) or not hint_id.strip():
                    continue
                record = HintAttemptRecord(
                    hint_id=hint_id,
                    competition_id=deps.competition.id,
                    generation=generation,
                    applied=True,
                    cv_score_before=best_score,
                    cv_score_after=float(cv_score),
                    delta=delta,
                )
                tracker.record_attempt(record)

            if metric_direction == "minimize" and cv_score < best_score:
                best_score = float(cv_score)
            elif metric_direction == "maximize" and cv_score > best_score:
                best_score = float(cv_score)

    def _build_execution_env(self, validation_split: float) -> dict[str, str]:
        return {"AGENT_K_VALIDATION_SPLIT": f"{validation_split:.6f}"}

    def _fitness_from_score(self, score: float, direction: str) -> float:
        return 1.0 / (1.0 + max(score, 0.0)) if direction == "minimize" else max(score, 0.0)

    def _score_from_fitness(self, fitness: float | None, direction: str) -> float | None:
        if fitness is None:
            return None
        if direction == "minimize":
            if fitness <= 0:
                return None
            return (1.0 / fitness) - 1.0
        return fitness

    def _score_delta(self, before: float, after: float, direction: str) -> float:
        if direction == "minimize":
            return before - after
        return after - before

    def _solution_complexity(self, code: str) -> int:
        return sum(1 for line in code.splitlines() if line.strip())

    def _complexity_bin(self, complexity: int) -> int:
        for idx, threshold in enumerate(_COMPLEXITY_BINS):
            if complexity <= threshold:
                return idx
        return len(_COMPLEXITY_BINS)

    def _model_family(self, code: str) -> str:
        for family, pattern in _MODEL_FAMILY_PATTERNS:
            if pattern.search(code):
                return family
        return "unknown"

    def _solution_signature(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:12]

    def _is_valid_python(self, code: str) -> bool:
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True

    def _call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _has_invalid_knn_params(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._call_name(node.func)
            if not name or name in {"KNeighborsClassifier", "KNeighborsRegressor"}:
                continue
            if name not in _MODEL_IMPORTS:
                continue
            for keyword in node.keywords:
                if keyword.arg in _KNN_PARAM_KEYS:
                    return True
        return False

    def _apply_solution_policy(self, ctx: RunContext[EvolverDeps], code: str) -> str:
        policy = self._resolve_technique_policy(ctx.deps)
        if policy is None:
            return code
        updated, notes = apply_solution_policy(code, policy)
        if notes:
            logfire.warning("solution_policy_injection_failed", notes=notes)
        updated = self._apply_target_column_guard(ctx.deps, updated)
        updated = self._apply_feature_column_guard(updated)
        return self._normalize_model_imports(updated)

    def _apply_target_column_guard(self, deps: EvolverDeps, code: str) -> str:
        if len(deps.target_columns) != 1 and len(deps.train_target_columns) != 1:
            return code

        def normalize(match: re.Match[str]) -> str:
            index = int(match.group("index"))
            if index <= 0:
                return match.group(0)
            name = match.group("name")
            return f"{name}[0]"

        return _TARGET_INDEX_PATTERN.sub(normalize, code)

    def _apply_feature_column_guard(self, code: str) -> str:
        """Ensure feature column lists don't reference missing columns.

        Some mutated solutions build ``numeric_cols``/``categorical_cols`` lists and then use
        them in a ``ColumnTransformer``. If those lists drift out of sync with the actual
        columns (or train/test alignment), pandas-based column selection will raise KeyError.
        We inject a lightweight runtime guard that intersects these lists with the columns
        present in both frames, preventing hard failures during evolution.
        """
        marker = "# AGENT_K_FEATURE_GUARD"
        if marker in code:
            return code

        numeric_match = _NUMERIC_COLS_PATTERN.search(code)
        cat_match = _CATEGORICAL_COLS_PATTERN.search(code)
        if numeric_match is None and cat_match is None:
            return code

        match = cat_match or numeric_match
        if match is None:
            return code

        line_text = match.group(0)
        if line_text.rstrip().endswith("\\"):
            return code
        openers = line_text.count("[") + line_text.count("(") + line_text.count("{")
        closers = line_text.count("]") + line_text.count(")") + line_text.count("}")
        if openers > closers:
            return code

        indent_prefix = match.group("indent")
        insert_at = match.end()
        guard = dedent(
            f"""

            {indent_prefix}{marker}
            {indent_prefix}_train_cols = set(X.columns) if "X" in locals() else set(
            {indent_prefix}    train_df.columns if "train_df" in locals() else []
            {indent_prefix})
            {indent_prefix}_test_cols = set(test_df.columns) if "test_df" in locals() else _train_cols
            {indent_prefix}_common_cols = _train_cols & _test_cols if _test_cols else _train_cols
            {indent_prefix}if "numeric_cols" in locals():
            {indent_prefix}    numeric_cols = [c for c in list(numeric_cols) if c in _common_cols]
            {indent_prefix}if "categorical_cols" in locals():
            {indent_prefix}    categorical_cols = [c for c in list(categorical_cols) if c in _common_cols]
            """
        ).rstrip()
        updated = f"{code[:insert_at]}{guard}{code[insert_at:]}"
        if not self._is_valid_python(updated):
            return code
        return updated

    def _resolve_technique_policy(self, deps: EvolverDeps) -> TechniquePolicy | None:
        if deps.technique_policy is not None:
            return deps.technique_policy
        profile = self._resolve_problem_profile(deps)
        return build_technique_policy(profile)

    def _resolve_problem_profile(self, deps: EvolverDeps) -> ProblemProfile:
        if deps.problem_profile is not None:
            return deps.problem_profile
        return build_problem_profile(
            deps.competition,
            CompetitionSchema(
                id_column=deps.id_column,
                target_columns=deps.target_columns,
                train_target_columns=deps.train_target_columns,
            ),
        )

    def _resolve_fitness_policy(self, deps: EvolverDeps) -> FitnessPolicy:
        if deps.fitness_policy is not None:
            return deps.fitness_policy
        profile = self._resolve_problem_profile(deps)
        return build_fitness_policy(profile, None, max_runtime_ms=int(deps.solution_timeout * 1000))

    def _compute_fitness(
        self,
        ctx: RunContext[EvolverDeps],
        *,
        cv_score: float,
        runtime_ms: int,
        code: str,
        stage: str | None,
        valid: bool,
    ) -> float:
        policy = self._resolve_fitness_policy(ctx.deps)
        fitness_fn = build_fitness_function(policy)
        return fitness_fn(
            FitnessInput(
                cv_score=cv_score,
                runtime_ms=runtime_ms,
                complexity=self._solution_complexity(code),
                valid=valid,
                stage=stage,
                code=code,
            )
        )

    def _adaptive_magnitude(self, deps: EvolverDeps) -> float:
        history = deps.generation_history[-5:]
        bests: list[float] = []
        for entry in history:
            if isinstance(entry, dict):
                bests.append(float(entry.get("best_fitness", 0.0)))
            else:
                bests.append(float(getattr(entry, "best_fitness", 0.0)))

        if len(bests) < 2:
            return 0.25

        improvement = max(bests) - min(bests)
        if improvement < 1e-4:
            return 0.35
        if improvement > 0.05:
            return 0.15
        return 0.22

    def _build_archive_entry(self, code: str, fitness: float, cv_score: float) -> EvolutionArchiveEntry:
        complexity = self._solution_complexity(code)
        return EvolutionArchiveEntry(
            code=code,
            fitness=fitness,
            cv_score=cv_score,
            complexity=complexity,
            complexity_bin=self._complexity_bin(complexity),
            model_family=self._model_family(code),
            signature=self._solution_signature(code),
        )

    def _update_elite_archive(self, deps: EvolverDeps, entry: EvolutionArchiveEntry) -> None:
        key = (entry.complexity_bin, entry.model_family)
        existing = deps.elite_archive.get(key)
        if existing is None or entry.fitness > existing.fitness:
            deps.elite_archive[key] = entry

    def _select_elite_samples(self, deps: EvolverDeps, *, top: int, diverse: int) -> list[EvolutionArchiveEntry]:
        entries = list(deps.elite_archive.values())
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda entry: entry.fitness, reverse=True)
        selected: list[EvolutionArchiveEntry] = []
        for entry in sorted_entries[:top]:
            selected.append(entry)

        used_signatures = {entry.signature for entry in selected}
        used_families = {entry.model_family for entry in selected}
        used_bins = {entry.complexity_bin for entry in selected}

        target_size = top + diverse
        if diverse > 0:
            for entry in sorted_entries:
                if entry.signature in used_signatures:
                    continue
                if entry.model_family not in used_families or entry.complexity_bin not in used_bins:
                    selected.append(entry)
                    used_signatures.add(entry.signature)
                    used_families.add(entry.model_family)
                    used_bins.add(entry.complexity_bin)
                if len(selected) >= target_size:
                    break

        if len(selected) < target_size:
            for entry in sorted_entries:
                if entry.signature in used_signatures:
                    continue
                selected.append(entry)
                if len(selected) >= target_size:
                    break

        return selected

    async def _run_evaluation(
        self, ctx: RunContext[EvolverDeps], solution_code: str, *, validation_split: float
    ) -> dict[str, Any]:
        code_signature = self._solution_signature(solution_code)
        tracker = ctx.deps.experiment_tracker
        if tracker is not None:
            cached = tracker.find_latest_by_code_signature(ctx.deps.competition.id, code_signature)
            if cached is not None:
                cached_stage = cached.metrics.get("stage")
                if cached_stage in {"full", "cached"} and (
                    cached.cv_score is not None or cached.metrics.get("valid") is False
                ):
                    cached_fitness = cached.metrics.get("fitness")
                    cached_valid = cached.metrics.get("valid", cached.cv_score is not None)
                    cached_error = cached.metrics.get("error")
                    if cached_fitness is None and cached.cv_score is not None:
                        cached_fitness = self._fitness_from_score(
                            cached.cv_score, ctx.deps.competition.metric_direction
                        )
                    return {
                        "fitness": round(float(cached_fitness or 0.0), 6),
                        "cv_score": round(float(cached.cv_score or 0.0), 6),
                        "valid": bool(cached_valid),
                        "runtime_ms": 0,
                        "timed_out": False,
                        "returncode": 0 if cached_valid else 1,
                        "error": cached_error,
                        "cached": True,
                        "stage": "cached",
                    }

        if not self._settings.cascade_evaluation or self._settings.cascade_stage1_rows <= 0:
            result = await self._evaluate_solution(ctx, solution_code, validation_split=validation_split, stage="full")
            result["stage"] = "full"
            return self._record_evaluation(ctx, solution_code, code_signature, result)

        quick_timeout = min(self._settings.cascade_stage1_timeout, ctx.deps.solution_timeout)
        quick_result = await self._evaluate_solution(
            ctx,
            solution_code,
            validation_split=validation_split,
            max_rows=self._settings.cascade_stage1_rows,
            timeout_seconds=quick_timeout,
            stage="stage1",
        )
        quick_result["stage"] = "stage1"
        if not quick_result["valid"]:
            return self._record_evaluation(ctx, solution_code, code_signature, quick_result)

        threshold = None
        if ctx.deps.best_fitness is not None:
            threshold = max(
                ctx.deps.best_fitness * self._settings.cascade_relative_threshold,
                self._settings.cascade_floor_threshold,
            )
        if threshold is not None and quick_result["fitness"] < threshold:
            quick_result["stage_threshold"] = threshold
            return self._record_evaluation(ctx, solution_code, code_signature, quick_result)

        full_result = await self._evaluate_solution(ctx, solution_code, validation_split=validation_split, stage="full")
        full_result["stage"] = "full"
        full_result["stage1_fitness"] = quick_result["fitness"]
        full_result["stage1_cv_score"] = quick_result["cv_score"]
        full_result["stage1_runtime_ms"] = quick_result["runtime_ms"]
        full_result["runtime_ms"] += quick_result["runtime_ms"]
        return self._record_evaluation(ctx, solution_code, code_signature, full_result)

    async def _evaluate_solution(
        self,
        ctx: RunContext[EvolverDeps],
        solution_code: str,
        *,
        validation_split: float,
        stage: str | None,
        max_rows: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(dir=str(ctx.deps.data_dir)) as run_dir:
            run_path = Path(run_dir)
            train_df, val_features, y_val, id_column = _prepare_validation_split(
                train_path=ctx.deps.train_path,
                id_column=ctx.deps.id_column,
                target_columns=list(ctx.deps.train_target_columns or ctx.deps.target_columns),
                validation_split=validation_split,
            )
            if max_rows is not None and max_rows > 0:
                train_df = train_df.head(max_rows).copy()
                val_features = val_features.head(max_rows).copy()
                y_val = y_val.head(max_rows).copy()
            elif ctx.deps.max_generations <= 5:
                train_df = train_df.head(800).copy()
                val_features = val_features.head(800).copy()
                y_val = y_val.head(800).copy()

            target_columns = list(ctx.deps.train_target_columns or ctx.deps.target_columns)
            train_df.to_csv(run_path / "train.csv", index=False)
            val_features.to_csv(run_path / "test.csv", index=False)
            sample_submission = pd.DataFrame({id_column: val_features[id_column].values})
            for col in target_columns:
                sample_submission[col] = 0.0
            sample_submission.to_csv(run_path / "sample_submission.csv", index=False)

            execution = await execute_solution(
                solution_code,
                run_path,
                timeout_seconds=timeout_seconds or ctx.deps.solution_timeout,
                env=self._build_execution_env(validation_split),
                use_builtin_code_execution=True,
                model_spec=self._settings.model,
            )

            submission_path = run_path / "submission.csv"
            score: float | None = None
            error: str | None = None
            if execution.timed_out:
                error = "Execution timed out"
            elif execution.returncode != 0:
                error = f"Execution failed (exit {execution.returncode})"
            elif not submission_path.exists():
                error = "submission.csv not found after execution"
            else:
                try:
                    score = _score_submission(
                        submission_path=submission_path,
                        metric=ctx.deps.competition.metric,
                        id_column=id_column,
                        target_columns=target_columns,
                        y_val=y_val,
                    )
                except Exception as exc:
                    error = f"Unable to score submission: {exc}"

        error_category: str | None = None
        error_feedback = ""
        execution_status = "success"
        stderr_text = execution.stderr or ""
        stderr_trimmed = stderr_text.strip()
        if error is not None:
            error_category, error_feedback = _build_error_feedback(
                stderr=stderr_text, error=error, timed_out=execution.timed_out, returncode=execution.returncode
            )
            self._record_failure(ctx.deps, error_category, error_feedback)
            execution_status = "failed"
        else:
            ctx.deps.last_error_feedback = None

        cv_score = score if score is not None else 0.0
        fitness = (
            self._compute_fitness(
                ctx,
                cv_score=cv_score,
                runtime_ms=execution.runtime_ms,
                code=solution_code,
                stage=stage,
                valid=error is None,
            )
            if error is None
            else 0.0
        )
        return {
            "fitness": round(fitness, 6),
            "cv_score": round(cv_score, 6),
            "valid": error is None,
            "runtime_ms": execution.runtime_ms,
            "timed_out": execution.timed_out,
            "returncode": execution.returncode,
            "error": error,
            "stderr": stderr_trimmed if stderr_trimmed else None,
            "error_category": error_category,
            "error_feedback": error_feedback,
            "execution_status": execution_status,
        }

    def _record_evaluation(
        self, ctx: RunContext[EvolverDeps], solution_code: str, code_signature: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        tracker = ctx.deps.experiment_tracker
        if tracker is None:
            return result

        metadata = extract_solution_metadata(solution_code)
        record = ExperimentRecord(
            competition_id=ctx.deps.competition.id,
            phase="evolution",
            model_name=metadata.model_name,
            model_family=metadata.model_family,
            hyperparameters=metadata.hyperparameters,
            feature_set=metadata.feature_set,
            feature_engineering=metadata.feature_engineering,
            target_transform=metadata.target_transform,
            metrics={
                "fitness": result.get("fitness"),
                "cv_score": result.get("cv_score"),
                "stage": result.get("stage"),
                "runtime_ms": result.get("runtime_ms"),
                "valid": result.get("valid"),
                "error": result.get("error"),
                "error_category": result.get("error_category"),
                "error_feedback": result.get("error_feedback"),
                "timed_out": result.get("timed_out"),
                "execution_status": result.get("execution_status"),
            },
            cv_score=result.get("cv_score"),
            code_signature=code_signature,
            dataset_fingerprint=ctx.deps.competition.id,
        )
        tracker.record_experiment(record)
        return result

    def _record_failure(self, deps: EvolverDeps, error_category: str, error_feedback: str) -> None:
        deps.failure_counts[error_category] = deps.failure_counts.get(error_category, 0) + 1
        deps.last_error_feedback = error_feedback

    async def _submit_solution(
        self, ctx: RunContext[EvolverDeps], solution_code: str, *, message: str
    ) -> dict[str, Any]:
        solution_code = self._apply_solution_policy(ctx, solution_code)
        with tempfile.TemporaryDirectory(dir=str(ctx.deps.data_dir)) as run_dir:
            run_path = Path(run_dir)
            stage_competition_data(
                ctx.deps.train_path,
                ctx.deps.test_path,
                ctx.deps.sample_path,
                run_path,
                competition_id=ctx.deps.competition.id,
            )
            execution = await execute_solution(
                solution_code,
                run_path,
                timeout_seconds=ctx.deps.solution_timeout,
                env=self._build_execution_env(0.2),
                use_builtin_code_execution=True,
                model_spec=self._settings.model,
            )
            submission_path = run_path / "submission.csv"
            error = (
                "Execution timed out"
                if execution.timed_out
                else f"Execution failed (exit {execution.returncode})"
                if execution.returncode != 0
                else "submission.csv not found after execution"
                if not submission_path.exists()
                else None
            )

            if error:
                return {
                    "submission_id": None,
                    "status": "failed",
                    "error": error,
                    "generation": len(ctx.deps.generation_history),
                    "runtime_ms": execution.runtime_ms,
                }

            submission = await ctx.deps.platform_adapter.submit(
                ctx.deps.competition.id, str(submission_path), message=message
            )

        tracker = ctx.deps.experiment_tracker
        if tracker is not None:
            metadata = extract_solution_metadata(solution_code)
            tracker.record_experiment(
                ExperimentRecord(
                    competition_id=ctx.deps.competition.id,
                    phase="submission",
                    model_name=metadata.model_name,
                    model_family=metadata.model_family,
                    hyperparameters=metadata.hyperparameters,
                    feature_set=metadata.feature_set,
                    feature_engineering=metadata.feature_engineering,
                    target_transform=metadata.target_transform,
                    metrics={"status": submission.status},
                    submission_id=submission.id,
                    code_signature=self._solution_signature(solution_code),
                    dataset_fingerprint=ctx.deps.competition.id,
                )
            )

        return {
            "submission_id": submission.id,
            "status": submission.status,
            "generation": len(ctx.deps.generation_history),
            "runtime_ms": execution.runtime_ms,
        }

    def _seeded_rng(self, solution_code: str, params: dict[str, Any], salt: str) -> random.Random:
        seed_input = f"{salt}:{solution_code}:{sorted(params.items())}".encode()
        seed = int(hashlib.sha256(seed_input).hexdigest(), 16)
        return random.Random(seed)

    def _mutate_numbers(self, code: str, rng: random.Random, *, max_changes: int, magnitude: float) -> str:
        changes = 0

        def replacer(match: re.Match[str]) -> str:
            nonlocal changes
            if changes >= max_changes:
                return match.group(0)
            raw = match.group(1)
            try:
                value = float(raw)
            except ValueError:
                return raw
            if value == 0:
                value = 0.1
            direction = rng.choice([-1, 1])
            mutated = value * (1 + direction * magnitude)
            changes += 1
            return str(max(1, int(round(mutated)))) if raw.isdigit() else f"{mutated:.6g}"

        return _NUMBER_PATTERN.sub(replacer, code)

    def _ensure_import(self, code: str, module: str, symbol: str) -> str:
        imports, body = self._split_imports(code)
        import_prefix = f"from {module} import"
        for line in imports:
            stripped = line.strip()
            if stripped.startswith(import_prefix) and symbol in stripped:
                return code

        imports.append(f"{import_prefix} {symbol}")
        merged_imports = self._merge_imports(imports, [])
        if merged_imports:
            return "\n".join([*merged_imports, "", *body])
        return "\n".join(body)

    def _normalize_model_imports(self, code: str) -> str:
        """Ensure sklearn model imports use the correct module."""
        imports, body = self._split_imports(code)
        if not imports:
            return code

        extra_imports: dict[str, list[str]] = {}
        normalized_imports: list[str] = []

        for line in imports:
            match = re.match(r"\s*from\s+(\S+)\s+import\s+(.+)", line)
            if not match:
                normalized_imports.append(line)
                continue

            module = match.group(1)
            symbols = [symbol.strip() for symbol in match.group(2).split(",") if symbol.strip()]
            kept: list[str] = []
            for symbol in symbols:
                base_symbol = symbol.split(" as ", 1)[0].strip()
                expected_module = _MODEL_IMPORTS.get(base_symbol)
                if expected_module and expected_module != module:
                    extra_imports.setdefault(expected_module, []).append(symbol)
                    continue
                kept.append(symbol)

            if kept:
                normalized_imports.append(f"from {module} import {', '.join(kept)}")

        for module, symbols in extra_imports.items():
            unique_symbols: list[str] = []
            seen: set[str] = set()
            for symbol in symbols:
                if symbol in seen:
                    continue
                seen.add(symbol)
                unique_symbols.append(symbol)

            inserted = False
            for idx, line in enumerate(normalized_imports):
                match = re.match(r"\s*from\s+(\S+)\s+import\s+(.+)", line)
                if not match or match.group(1) != module:
                    continue
                existing = [symbol.strip() for symbol in match.group(2).split(",") if symbol.strip()]
                existing_set = set(existing)
                for symbol in unique_symbols:
                    if symbol not in existing_set:
                        existing.append(symbol)
                        existing_set.add(symbol)
                normalized_imports[idx] = f"from {module} import {', '.join(existing)}"
                inserted = True
                break

            if not inserted:
                normalized_imports.append(f"from {module} import {', '.join(unique_symbols)}")

        merged_imports = self._merge_imports(normalized_imports, [])
        if merged_imports:
            return "\n".join([*merged_imports, "", *body])
        return "\n".join(body)

    def _swap_model_family(self, code: str) -> str:
        ordered_swaps = sorted(_MODEL_SWAPS.items(), key=lambda item: len(item[0]), reverse=True)
        for source, target in ordered_swaps:
            pattern = re.compile(rf"\b{re.escape(source)}\b")
            if not pattern.search(code):
                continue
            updated = pattern.sub(target, code)
            module = _MODEL_IMPORTS.get(target)
            if module:
                updated = self._ensure_import(updated, module, target)
            return updated
        return code

    def _inject_scaler(self, code: str) -> str:
        if "StandardScaler" in code:
            return code
        match = _NUMERIC_PIPELINE_PATTERN.search(code)
        if not match:
            return code
        steps_block = match.group("steps")
        if "SimpleImputer" not in steps_block or "StandardScaler" in steps_block:
            return code

        lines = steps_block.splitlines()
        inserted = False
        for idx, line in enumerate(lines):
            if "SimpleImputer" in line:
                indent_match = re.match(r"\s*", line)
                if not indent_match:
                    continue
                indent = indent_match.group(0)
                lines.insert(idx + 1, f'{indent}("scaler", StandardScaler()),')
                inserted = True
                break
        if not inserted:
            return code

        updated_steps = "\n".join(lines)
        updated = f"{code[: match.start('steps')]}{updated_steps}{code[match.end('steps') :]}"
        return self._ensure_import(updated, "sklearn.preprocessing", "StandardScaler")

    def _swap_scaler(self, code: str) -> str:
        scaler_swaps = {
            "StandardScaler": "MinMaxScaler",
            "MinMaxScaler": "RobustScaler",
            "RobustScaler": "StandardScaler",
        }
        for source, target in scaler_swaps.items():
            pattern = re.compile(rf"\b{re.escape(source)}\b")
            if not pattern.search(code):
                continue
            updated = pattern.sub(target, code)
            return self._ensure_import(updated, "sklearn.preprocessing", target)
        return code

    def _inject_feature_engineering(self, code: str) -> str:
        if "PolynomialFeatures" not in code:
            updated = self._insert_numeric_step(code, '("poly", PolynomialFeatures(degree=2, include_bias=False)),')
            if updated != code:
                return self._ensure_import(updated, "sklearn.preprocessing", "PolynomialFeatures")
        if "KBinsDiscretizer" not in code:
            updated = self._insert_numeric_step(
                code, '("binning", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")),'
            )
            if updated != code:
                return self._ensure_import(updated, "sklearn.preprocessing", "KBinsDiscretizer")
        return code

    def _inject_feature_selection(self, code: str) -> str:
        if "VarianceThreshold" in code:
            return code
        updated = self._insert_numeric_step(code, '("selector", VarianceThreshold(threshold=0.0)),')
        if updated != code:
            return self._ensure_import(updated, "sklearn.feature_selection", "VarianceThreshold")
        return code

    def _insert_numeric_step(self, code: str, step_line: str) -> str:
        match = _NUMERIC_PIPELINE_PATTERN.search(code)
        if not match:
            return code
        steps_block = match.group("steps")
        lines = steps_block.splitlines()
        insert_index = None
        for idx, line in enumerate(lines):
            if "SimpleImputer" in line or "scaler" in line:
                insert_index = idx + 1
        if insert_index is None:
            return code
        indent_match = re.match(r"\s*", lines[insert_index - 1])
        if not indent_match:
            return code
        indent = indent_match.group(0)
        lines.insert(insert_index, f"{indent}{step_line}")
        updated_steps = "\n".join(lines)
        return f"{code[: match.start('steps')]}{updated_steps}{code[match.end('steps') :]}"

    def _inject_ratio_features(self, code: str) -> str:
        if "ratio_features" in code:
            return code
        match = _NUMERIC_COLS_PATTERN.search(code)
        if not match or "test_df" not in code or "X =" not in code:
            return code

        indent = match.group("indent")
        snippet = dedent(
            f"""
            {indent}ratio_features = list(zip(numeric_cols[:3], numeric_cols[1:4]))
            {indent}for left, right in ratio_features:
            {indent}    if left not in X.columns or right not in X.columns:
            {indent}        continue
            {indent}    denom = X[right].replace(0, np.nan)
            {indent}    X[f"{{left}}_over_{{right}}"] = X[left] / denom
            {indent}    test_df[f"{{left}}_over_{{right}}"] = (
            {indent}        test_df[left] / test_df[right].replace(0, np.nan)
            {indent}    )
            """
        ).strip("\n")
        insert_at = match.end()
        updated = f"{code[:insert_at]}\n{snippet}{code[insert_at:]}"
        return updated if "import numpy as np" in updated else code

    def _inject_fillna(self, code: str) -> str:
        if "fillna(" in code or not (match := _FILLNA_PATTERN.search(code)):
            return code
        insert = f"\n{match['indent']}{match['var']} = {match['var']}.fillna(0)"
        return f"{code[: match.end()]}{insert}{code[match.end() :]}"

    def _merge_imports(self, primary: list[str], secondary: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for line in primary + secondary:
            normalized = line.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(line)
        return result

    def _split_imports(self, code: str) -> tuple[list[str], list[str]]:
        def is_import(line: str) -> bool:
            stripped = line.lstrip()
            return stripped.startswith(("import ", "from ")) and not stripped.startswith("from __future__")

        lines = code.splitlines()
        return [line for line in lines if is_import(line)], [line for line in lines if not is_import(line)]

    def _extract_top_level_defs(self, code: str) -> dict[str, str]:
        return {
            match.group(1).split()[1].split("(")[0]: match.group(0).rstrip() for match in _DEF_PATTERN.finditer(code)
        }

    def _format_param_value(self, value: Any) -> str:
        return json.dumps(value) if isinstance(value, str) else str(value)

    def _insert_knn_param(self, code: str, param: str, value: Any) -> str:
        match = re.search(r"\bKNeighbors(?:Classifier|Regressor)\s*\(", code)
        if not match:
            return code
        literal = self._format_param_value(value)
        insertion = f"{param}={literal}, "
        return f"{code[: match.end()]}{insertion}{code[match.end() :]}"

    def _replace_or_insert_param(self, code: str, param: str, value: Any) -> str:
        pattern = _HYPERPARAM_PATTERNS.get(param)
        if not pattern:
            return code
        if match := pattern.search(code):
            return pattern.sub(f"{match.group(1)}{self._format_param_value(value)}", code, count=1)
        return self._insert_knn_param(code, param, value)

    def _mutate_numeric_param(self, code: str, param: str, rng: random.Random, params: dict[str, Any]) -> str:
        pattern = _HYPERPARAM_PATTERNS.get(param)
        if not pattern:
            return code
        magnitude = float(params.get("magnitude", 0.2))
        if match := pattern.search(code):
            try:
                value = float(match.group(2))
            except ValueError:
                return code
            mutated = value * (1 + magnitude * rng.choice([-1, 1]))
            bounds = _HYPERPARAM_BOUNDS.get(param)
            if bounds is not None:
                mutated = min(max(mutated, bounds[0]), bounds[1])
            if param in _HYPERPARAM_INTEGER_KEYS:
                mutated_text = str(max(1, int(round(mutated))))
            else:
                mutated_text = f"{max(0.0001, mutated):.6g}"
            return pattern.sub(f"{match.group(1)}{mutated_text}", code, count=1)

        bounds = _HYPERPARAM_BOUNDS.get(param, (1.0, 30.0))
        sampled_value: int | float
        if param in _HYPERPARAM_INTEGER_KEYS:
            sampled_value = max(1, int(round(rng.uniform(bounds[0], bounds[1]))))
        else:
            sampled_value = rng.uniform(bounds[0], bounds[1])
        return self._replace_or_insert_param(code, param, sampled_value)

    def _apply_knn_mutation(self, code: str, params: dict[str, Any]) -> str:
        if not _KNN_MODEL_PATTERN.search(code):
            return code
        rng = self._seeded_rng(code, params, "knn")
        mutation = rng.choice(("metric", "weights", "neighbors", "leaf_size", "algorithm", "scaler"))

        if mutation == "metric":
            metric = rng.choice(("euclidean", "manhattan", "minkowski"))
            updated = self._replace_or_insert_param(code, "metric", metric)
            if metric == "minkowski":
                p_value = rng.randint(1, 5)
            else:
                p_value = 1 if metric == "manhattan" else 2
            return self._replace_or_insert_param(updated, "p", p_value)

        if mutation == "weights":
            return self._replace_or_insert_param(code, "weights", rng.choice(_CATEGORICAL_HYPERPARAMS["weights"]))
        if mutation == "algorithm":
            return self._replace_or_insert_param(code, "algorithm", rng.choice(_CATEGORICAL_HYPERPARAMS["algorithm"]))
        if mutation == "neighbors":
            return self._mutate_numeric_param(code, "n_neighbors", rng, params)
        if mutation == "leaf_size":
            return self._mutate_numeric_param(code, "leaf_size", rng, params)
        if mutation == "scaler":
            swapped = self._swap_scaler(code)
            if swapped != code:
                return swapped
            return self._inject_scaler(code)
        return code

    def _apply_point_mutation(self, code: str, params: dict[str, Any]) -> str:
        rng = self._seeded_rng(code, params, "point")
        magnitude = float(params.get("delta", 0.1))
        max_changes = int(params.get("max_changes", 2))
        return self._mutate_numbers(code, rng, max_changes=max_changes, magnitude=magnitude)

    def _apply_structural_mutation(self, code: str, params: dict[str, Any]) -> str:
        for mutate in (
            self._swap_model_family,
            self._swap_scaler,
            self._inject_scaler,
            self._inject_feature_engineering,
            self._inject_feature_selection,
            self._inject_ratio_features,
            self._inject_fillna,
        ):
            if (result := mutate(code)) != code:
                return result
        return self._apply_point_mutation(code, params)

    def _apply_hyperparameter_mutation(self, code: str, params: dict[str, Any]) -> str:
        rng = self._seeded_rng(code, params, "hyperparameter")
        magnitude = float(params.get("magnitude", 0.2))
        requested = str(params.get("param", "")).strip()

        if _KNN_MODEL_PATTERN.search(code):
            if (knn_mutated := self._apply_knn_mutation(code, params)) != code:
                return knn_mutated

        candidates: list[tuple[str, re.Pattern[str], re.Match[str]]] = []
        for name, pattern in _HYPERPARAM_PATTERNS.items():
            if requested and name != requested:
                continue
            if match := pattern.search(code):
                if name == "objective" and not alternative_objectives(match.group(2)):
                    continue
                candidates.append((name, pattern, match))

        if not candidates:
            return self._apply_point_mutation(code, params)

        name, pattern, match = rng.choice(candidates)
        if name == "objective":
            swapped = rng.choice(alternative_objectives(match.group(2)))
            return pattern.sub(f"{match.group(1)}{json.dumps(swapped)}", code, count=1)
        if name in _CATEGORICAL_HYPERPARAMS:
            current = match.group(2).strip().strip("'\"")
            options = list(_CATEGORICAL_HYPERPARAMS[name])
            choices = [option for option in options if option != current]
            mutated_cat = rng.choice(choices or options)
            return pattern.sub(f"{match.group(1)}{json.dumps(mutated_cat)}", code, count=1)

        try:
            value = float(match.group(2))
        except ValueError:
            return self._apply_point_mutation(code, params)

        mutated_val: float = value * (1 + magnitude * rng.choice([-1, 1]))
        bounds = _HYPERPARAM_BOUNDS.get(name)
        if bounds is not None:
            mutated_val = min(max(mutated_val, bounds[0]), bounds[1])
        if name in _HYPERPARAM_INTEGER_KEYS:
            mutated_text = str(max(1, int(round(mutated_val))))
        else:
            mutated_text = f"{max(0.0001, mutated_val):.6g}"
        return pattern.sub(f"{match.group(1)}{mutated_text}", code, count=1)

    def _apply_crossover(self, code: str, other: str, params: dict[str, Any]) -> str:
        if not other.strip():
            return code
        primary_imports, primary_body = self._split_imports(code)
        other_imports, _ = self._split_imports(other)
        primary_defs = self._extract_top_level_defs(code)
        extra_defs = [block for name, block in self._extract_top_level_defs(other).items() if name not in primary_defs]
        body_parts = ["\n".join(primary_body).strip(), *extra_defs]
        imports = "\n".join(self._merge_imports(primary_imports, other_imports))
        body = "\n\n".join(part for part in body_parts if part)
        return f"{imports}\n\n{body}" if imports else body


def _summarize_failure_counts(counts: dict[str, int], *, limit: int) -> str:
    if not counts or limit <= 0:
        return ""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name}={count}" for name, count in ranked[:limit])


def _extract_traceback_location(stderr: str) -> str | None:
    for line in stderr.splitlines():
        if line.strip().startswith("File "):
            return line.strip()
    return None


def _classify_execution_failure(*, stderr: str, error: str | None, timed_out: bool, returncode: int) -> tuple[str, str]:
    if timed_out:
        return "timeout", "Execution timed out. Reduce model complexity or iterations."

    error_text = (error or "").lower()
    if "submission.csv not found" in error_text:
        return "missing_submission", "Ensure submission.csv is written at the end of the script."
    if "unable to score submission" in error_text:
        return (
            "submission_schema",
            "Ensure submission.csv matches sample_submission columns and contains numeric predictions.",
        )

    combined = "\n".join(value for value in [error or "", stderr] if value)
    for pattern, category, hint in _ERROR_HINT_PATTERNS:
        if pattern.search(combined):
            return category, hint

    if returncode != 0:
        return "execution_failed", "Check stderr traceback and ensure the script runs end-to-end."
    return "unknown_error", "Review stderr/logs and resolve the runtime error."


def _build_error_feedback(*, stderr: str, error: str | None, timed_out: bool, returncode: int) -> tuple[str, str]:
    category, hint = _classify_execution_failure(stderr=stderr, error=error, timed_out=timed_out, returncode=returncode)
    parts = [f"ERROR_CATEGORY: {category}"]
    if error:
        parts.append(f"ERROR: {error}")
    if returncode != 0:
        parts.append(f"RETURN_CODE: {returncode}")

    stderr_clean = stderr.strip()
    if stderr_clean:
        if location := _extract_traceback_location(stderr_clean):
            parts.append(f"LOCATION: {location}")
        last_line = stderr_clean.splitlines()[-1]
        if last_line:
            parts.append(f"STDERR: {last_line}")

    if hint:
        parts.append(f"HINT: {hint}")

    feedback = "\n".join(parts)
    return category, _truncate_text(feedback, max_chars=_ERROR_FEEDBACK_MAX_CHARS)


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n... (truncated)"


def _truncate_csv(path: Path, *, max_rows: int) -> None:
    if max_rows <= 0:
        return
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None:
            return
        with temp_path.open("w", encoding="utf-8", errors="ignore", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(header)
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                writer.writerow(row)
    temp_path.replace(path)


def _prepare_validation_split(
    *, train_path: Path, id_column: str | None, target_columns: list[str], validation_split: float, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Create a deterministic train/validation split and return train, val-features, y_val, id column."""
    if not target_columns:
        raise ValueError("target_columns is empty")
    if not (0.0 < validation_split < 1.0):
        raise ValueError("validation_split must be between 0 and 1")

    frame = pd.read_csv(train_path)
    resolved_id = id_column if id_column and id_column in frame.columns else None
    if resolved_id is None:
        resolved_id = "__row_id__"
        frame[resolved_id] = np.arange(len(frame), dtype=np.int64)

    for col in target_columns:
        if col not in frame.columns:
            raise ValueError(f"Missing target column: {col}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    split = int(round(len(indices) * (1.0 - validation_split)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise ValueError("validation_split produced empty split")

    train_df = frame.iloc[train_idx].copy()
    val_df = frame.iloc[val_idx].copy()
    y_val = val_df[[resolved_id, *target_columns]].copy()
    val_features = val_df.drop(columns=target_columns, errors="ignore").copy()
    return train_df, val_features, y_val, resolved_id


def _score_submission(
    *, submission_path: Path, metric: str, id_column: str, target_columns: list[str], y_val: pd.DataFrame
) -> float:
    """Score a submission.csv against y_val using the competition metric."""
    submission = pd.read_csv(submission_path)
    if id_column not in submission.columns:
        raise ValueError(f"submission.csv missing id column '{id_column}'")
    missing_targets = [col for col in target_columns if col not in submission.columns]
    if missing_targets:
        raise ValueError(f"submission.csv missing target columns: {missing_targets}")

    merged = submission[[id_column, *target_columns]].merge(
        y_val[[id_column, *target_columns]], on=id_column, suffixes=("_pred", "_true"), how="inner"
    )
    if merged.empty:
        raise ValueError("No rows to score after merge")

    y_true = merged[[f"{col}_true" for col in target_columns]].to_numpy(dtype=float)
    y_pred = merged[[f"{col}_pred" for col in target_columns]].to_numpy(dtype=float)
    if not np.isfinite(y_pred).all():
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    key = metric.lower().replace("_", "").replace(" ", "")
    if key in {"rmsle"}:
        y_pred = np.clip(y_pred, 0.0, None)
        return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
    if key in {"rmse"}:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if key in {"mae"}:
        return float(mean_absolute_error(y_true, y_pred))
    if key in {"logloss"}:
        if y_true.shape[1] != 1:
            raise ValueError("logloss only supported for single-target problems")
        y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
        return float(log_loss(y_true[:, 0], y_pred[:, 0]))
    if key in {"auc"}:
        if y_true.shape[1] != 1:
            raise ValueError("auc only supported for single-target problems")
        return float(roc_auc_score(y_true[:, 0], y_pred[:, 0]))

    raise ValueError(f"Unsupported metric: {metric}")


# Module-level singleton for backward compatibility
evolver_agent_instance = EvolverAgent()
evolver_agent = evolver_agent_instance.agent
settings = evolver_agent_instance.settings
