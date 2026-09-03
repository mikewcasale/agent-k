"""OpenEvolve evaluator for AGENT-K solutions.

@notice: |
    OpenEvolve evaluator for AGENT-K solutions.

@dev: |
    See module for implementation details and extension points.

@graph:
    id: agent_k.evolution.evaluator
    provides:
        - agent_k.evolution.evaluator
    pattern: evolution-evaluator

@agent-guidance:
    do:
        - "Use agent_k.evolution.evaluator as the canonical home for this capability."
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
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import logfire
from openevolve.evaluation_result import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agent_k.core.hints import PreprocessingHint

__all__ = ("evaluate", "evaluate_stage1", "evaluate_stage2")

# Ensure repo root is importable when OpenEvolve loads this file directly.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CONTEXT_ENV: Final[str] = "AGENT_K_OPENEVOLVE_CONTEXT"
_DEFAULT_TIMEOUT_SECONDS: Final[int] = 120
_DEFAULT_VALIDATION_SPLIT: Final[float] = 0.2
_STAGE1_TIMEOUT: Final[int] = 5
_STAGE2_TIMEOUT: Final[int] = 30
_STAGE2_DATA_ROWS: Final[int] = 1000
_SUBMISSION_FILENAME: Final[str] = "submission.csv"
_PRISTINE_DIR_NAME: Final[str] = ".agent_k_pristine"
_MANIFEST_FILENAME: Final[str] = "inputs.json"
_MAX_PRISTINE_BYTES: Final[int] = 512 * 1024 * 1024
_RUN_ARTIFACT_NAMES: Final[frozenset[str]] = frozenset({_SUBMISSION_FILENAME, "solution.py", "initial_program.py"})
_EXCLUDED_DIR_NAMES: Final[frozenset[str]] = frozenset({_PRISTINE_DIR_NAME, "openevolve_output", "__pycache__"})
_MUTATED_INPUT_HINT: Final[str] = (
    "MUTATION HINT [MutatedInput]: The solution overwrote a staged competition data file "
    "({files}). Input files are shared by every candidate and must stay read-only:\n"
    "- Read with `pd.read_csv('train.csv')` and keep derived frames in memory\n"
    "- Write derived data under a new name (e.g. `features.csv`), never back over train/test/sample files"
)
_MODEL_FAMILY_PATTERNS: Final[tuple[tuple[float, re.Pattern[str]], ...]] = (
    (3.0, re.compile(r"\b(Stacking|Voting|Bagging|AdaBoost)(?:Regressor|Classifier)?\b")),
    (2.0, re.compile(r"\bKNeighbors(?:Regressor|Classifier)\b")),
    (1.0, re.compile(r"\b(LinearRegression|Ridge|Lasso|ElasticNet|LogisticRegression|SGD(?:Regressor|Classifier))\b")),
    (
        0.0,
        re.compile(
            r"\b(RandomForest|GradientBoosting|HistGradientBoosting|ExtraTrees|DecisionTree|"
            r"XGB(?:Regressor|Classifier)|LGBM(?:Regressor|Classifier)|CatBoost(?:Regressor|Classifier))\b"
        ),
    ),
)


def _load_context() -> dict[str, Any]:
    """Load evaluation context from environment variable."""
    context_json = os.environ.get(_CONTEXT_ENV, "{}")
    result: dict[str, Any] = json.loads(context_json)
    return result


def _resolve_work_dir(context: dict[str, Any]) -> Path:
    """Resolve working directory from context."""
    work_dir_str = context.get("work_dir", ".")
    return Path(work_dir_str).resolve()


def _coerce_timeout(context: dict[str, Any]) -> int:
    """Coerce timeout to integer seconds."""
    timeout = context.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
    return int(timeout)


def _coerce_validation_split(context: dict[str, Any]) -> float:
    """Coerce validation split to float."""
    split = context.get("validation_split", _DEFAULT_VALIDATION_SPLIT)
    return float(split)


def _load_hints(hints_data: list[dict[str, Any]]) -> list[Any]:
    """Load preprocessing hints from context data."""
    from agent_k.core.hints import PreprocessingHint

    hints: list[PreprocessingHint] = []
    for hint_dict in hints_data:
        if isinstance(hint_dict, dict):
            hints.append(PreprocessingHint(**hint_dict))
    return hints


def _iter_input_files(work_dir: Path) -> Iterator[Path]:
    """Yield the staged competition input files under ``work_dir``.

    @notice: |
        Walks the shared evolution working directory and yields every staged
        data file, skipping per-candidate run artifacts and OpenEvolve's own
        bookkeeping directories.

    @dev: |
        Hidden directories (including the pristine cache) and the temporary
        directories created by other evaluators are excluded so only real
        competition inputs are tracked.
    """
    for dir_path, dir_names, file_names in os.walk(work_dir):
        dir_names[:] = sorted(
            name for name in dir_names if name not in _EXCLUDED_DIR_NAMES and not name.startswith(".")
        )
        current = Path(dir_path)
        is_root = current.relative_to(work_dir) == Path()
        for file_name in sorted(file_names):
            if is_root and file_name in _RUN_ARTIFACT_NAMES:
                continue
            path = current / file_name
            if path.is_file():
                yield path


def _fingerprint(path: Path) -> tuple[int, int]:
    """Return a cheap (size, mtime) fingerprint for ``path``."""
    stat_result = path.stat()
    return stat_result.st_size, stat_result.st_mtime_ns


def _snapshot_inputs(work_dir: Path) -> dict[str, tuple[int, int]]:
    """Copy the staged inputs into the pristine cache and fingerprint them."""
    pristine_dir = work_dir / _PRISTINE_DIR_NAME
    fingerprints: dict[str, tuple[int, int]] = {}
    backed_up_bytes = 0

    for path in _iter_input_files(work_dir):
        relative = path.relative_to(work_dir)
        fingerprints[relative.as_posix()] = _fingerprint(path)
        size = path.stat().st_size
        if backed_up_bytes + size > _MAX_PRISTINE_BYTES:
            logfire.warning("openevolve_pristine_snapshot_skipped", file=relative.as_posix(), size=size)
            continue
        backup = pristine_dir / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        backed_up_bytes += size

    return fingerprints


def _read_manifest(manifest_path: Path) -> dict[str, tuple[int, int]] | None:
    """Load a previously written input manifest, or ``None`` when unusable."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return {str(name): (int(size), int(mtime)) for name, (size, mtime) in raw.items()}
    except (TypeError, ValueError):
        return None


def _prepare_workspace(work_dir: Path) -> dict[str, tuple[int, int]]:
    """Reset per-candidate artifacts and return the staged-input fingerprints.

    @notice: |
        Every candidate program runs in the same shared working directory, so
        a stale ``submission.csv`` from an earlier candidate would make a
        candidate that never writes one look valid. This removes that file and
        returns the fingerprints of the staged inputs so a candidate that
        overwrites them can be rolled back before the next candidate runs.

    @dev: |
        The input set is pinned by the first candidate of a run and persisted
        to a manifest inside the pristine cache, so files a candidate creates
        are never mistaken for competition inputs and the pinned set survives
        however OpenEvolve chooses to process candidates. The snapshot is
        capped at ``_MAX_PRISTINE_BYTES``; inputs beyond the cap are still
        fingerprinted (so mutation is reported) but cannot be restored.
    """
    stale_submission = work_dir / _SUBMISSION_FILENAME
    stale_submission.unlink(missing_ok=True)

    manifest_path = work_dir / _PRISTINE_DIR_NAME / _MANIFEST_FILENAME
    existing = _read_manifest(manifest_path)
    if existing is not None:
        return existing

    fingerprints = _snapshot_inputs(work_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(fingerprints, sort_keys=True), encoding="utf-8")
    return fingerprints


def _restore_mutated_inputs(work_dir: Path, fingerprints: dict[str, tuple[int, int]]) -> list[str]:
    """Restore staged inputs a candidate modified and return their names.

    @notice: |
        Returns the relative paths of the input files whose content changed
        during the candidate run, after restoring each one from the pristine
        snapshot so later candidates evaluate against the original data.

    @dev: |
        A file with no snapshot (deleted or skipped by the size cap) is still
        reported, so callers can invalidate the candidate even when rollback
        is impossible.
    """
    pristine_dir = work_dir / _PRISTINE_DIR_NAME
    mutated: list[str] = []

    for relative, expected in fingerprints.items():
        path = work_dir / relative
        if path.exists() and _fingerprint(path) == expected:
            continue
        mutated.append(relative)
        backup = pristine_dir / relative
        if not backup.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)

    return mutated


def _failure_metrics() -> dict[str, float]:
    """Return metrics for a failed evaluation."""
    return {
        "combined_score": 0.0,
        "fitness": 0.0,
        "cv_score": 0.0,
        "cv_variance": 0.0,
        "valid": 0.0,
        "returncode": 1.0,
        "runtime_ms": 0.0,
        "timeout": 0.0,
        "model_family": 0.0,
    }


def _error_artifacts(exc: Exception) -> dict[str, str]:
    """Return artifacts for an error."""
    return {"error": str(exc), "traceback": _truncate(traceback.format_exc(), 1000), "execution_status": "error"}


def _extract_warnings(stderr: str) -> list[str]:
    """Extract warning messages from stderr."""
    warnings: list[str] = []
    for line in stderr.splitlines():
        if "warning" in line.lower() or "deprecated" in line.lower():
            warnings.append(line.strip())
    return warnings[:10]


def _extract_error_feedback(stderr: str, stdout: str, code: str = "") -> str:
    """Extract structured error feedback with actionable mutation hints.

    Analyzes execution output and provides specific guidance for the LLM
    to fix common errors before attempting optimizations.

    Args:
        stderr: Standard error output from execution.
        stdout: Standard output from execution.
        code: Optional solution code for context.

    Returns:
        Structured feedback string with mutation hints.
    """
    feedback_parts: list[str] = []

    # === ERROR-SPECIFIC HINTS ===

    # Import errors - suggest fallback patterns
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        feedback_parts.append(
            "MUTATION HINT [ImportError]: Use try/except fallback pattern:\n"
            "```python\n"
            "try:\n"
            "    from lightgbm import LGBMRegressor\n"
            "except ImportError:\n"
            "    from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor\n"
            "```"
        )

    # Column mismatch errors - common Kaggle issue
    if "columns are missing" in stderr.lower() or "KeyError" in stderr:
        feedback_parts.append(
            "MUTATION HINT [ColumnError]: Ensure test features match train:\n"
            "`X_test = test_df[X.columns]` or `X_test = test_df.reindex(columns=X.columns, fill_value=0)`"
        )

    # Shape mismatch errors
    if "shape" in stderr.lower() and ("mismatch" in stderr.lower() or "ValueError" in stderr):
        feedback_parts.append(
            "MUTATION HINT [ShapeError]: Check data alignment before fit:\n"
            "`assert X_train.shape[1] == X_test.shape[1], 'Feature mismatch'`"
        )

    # Memory errors
    if "MemoryError" in stderr or "out of memory" in stderr.lower():
        feedback_parts.append(
            "MUTATION HINT [MemoryError]: Reduce memory usage:\n"
            "- Use `dtype='float32'` instead of float64\n"
            "- Add `gc.collect()` after large operations\n"
            "- Reduce batch size or n_estimators"
        )

    # Timeout issues
    if "timeout" in stderr.lower() or "timed out" in stderr.lower():
        feedback_parts.append(
            "MUTATION HINT [Timeout]: Speed up execution:\n"
            "- Reduce n_estimators, max_depth, or early_stopping_rounds\n"
            "- Use fewer CV folds (3 instead of 5)\n"
            "- Subsample training data"
        )

    # === OUTPUT QUALITY HINTS ===

    # Missing baseline score
    if "Baseline" not in stdout and "baseline" not in stdout.lower():
        feedback_parts.append(
            "MUTATION HINT [MissingBaseline]: Add baseline logging:\n"
            "`print(f'Baseline {{METRIC}} score: {{score:.6f}}')`"
        )

    # Missing CV fold output
    if "Fold" not in stdout and "fold" not in stdout.lower():
        feedback_parts.append(
            "MUTATION HINT [MissingFolds]: Add per-fold CV logging:\n`print(f'Fold {{i}}: {{fold_score:.6f}}')`"
        )

    # No submission file created
    if "submission" not in stdout.lower() and code:
        feedback_parts.append(
            "MUTATION HINT [MissingSubmission]: Ensure submission.csv is created:\n"
            "`submission_df.to_csv('submission.csv', index=False)`"
        )

    # === SYNTAX/STRUCTURE HINTS ===

    if "SyntaxError" in stderr:
        feedback_parts.append(
            "MUTATION HINT [SyntaxError]: Check for:\n"
            "- Missing colons after if/for/def\n"
            "- Unmatched parentheses/brackets\n"
            "- Invalid indentation"
        )

    if "NameError" in stderr:
        feedback_parts.append(
            "MUTATION HINT [NameError]: Variable used before definition.\n"
            "Ensure all imports are at top and variables are initialized."
        )

    # === INCLUDE TRACEBACK FOR CONTEXT ===

    if "Traceback" in stderr:
        traceback_lines: list[str] = []
        in_traceback = False
        for line in stderr.split("\n"):
            if "Traceback" in line:
                in_traceback = True
            if in_traceback:
                traceback_lines.append(line)
                # Stop after the actual error message
                if line.strip() and not line.startswith(" ") and "Error:" in line:
                    in_traceback = False

        if traceback_lines:
            feedback_parts.append("ERROR TRACEBACK:\n" + "\n".join(traceback_lines[:15]))

    # Return combined feedback or indicate success
    if feedback_parts:
        return "=== ARTIFACT FEEDBACK ===\n\n" + "\n\n".join(feedback_parts)

    return "=== ARTIFACT FEEDBACK ===\nNo errors detected. Focus on optimization."


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def _compute_cv_variance(stdout: str) -> float:
    """Extract CV variance from output for stability tracking.

    Parses stdout for CV fold scores and computes their variance.
    Lower variance indicates more stable/robust solutions.

    Args:
        stdout: Standard output from solution execution.

    Returns:
        Variance of CV fold scores, or 0.0 if not found.
    """
    # Look for patterns like "Fold 1: 0.85", "Fold 2: 0.83", etc.
    fold_pattern = re.compile(r"Fold\s+\d+[:\s]+([0-9.]+)")
    fold_scores = [float(match) for match in fold_pattern.findall(stdout)]

    if len(fold_scores) < 2:
        # Not enough fold scores to compute variance
        return 0.0

    # Compute variance
    mean_score = sum(fold_scores) / len(fold_scores)
    variance = sum((score - mean_score) ** 2 for score in fold_scores) / len(fold_scores)
    return float(variance)


def _model_family_score(code: str) -> float:
    """Score based on model family (prefer tree-based models)."""
    for penalty, pattern in _MODEL_FAMILY_PATTERNS:
        if pattern.search(code):
            return penalty
    return 3.0


def _fitness_from_score(cv_score: float | None, metric_direction: str) -> float:
    """Convert CV score to fitness (higher is better)."""
    if cv_score is None:
        return 0.0

    if metric_direction == "maximize":
        return float(cv_score)
    else:
        return -float(cv_score)


def evaluate(program_path: str) -> EvaluationResult:
    """Evaluate a program and return metrics and artifacts for OpenEvolve.

    @notice: |
        Evaluate a program and return metrics and artifacts for OpenEvolve.

    @dev: |
        See module for behavior details and invariants.
    """
    from agent_k.core.hints import detect_applied_hints
    from agent_k.core.solution import execute_solution, parse_baseline_score

    context = _load_context()
    work_dir = _resolve_work_dir(context)
    timeout = _coerce_timeout(context)
    validation_split = _coerce_validation_split(context)
    metric_direction = str(context.get("metric_direction", "minimize")).lower()
    hints = _load_hints(context.get("hints", []))

    try:
        code = Path(program_path).read_text(encoding="utf-8")
    except OSError as exc:
        logfire.warning("openevolve_read_failed", error=str(exc))
        return EvaluationResult(metrics=_failure_metrics(), artifacts={"error": str(exc)})

    env = {"AGENT_K_VALIDATION_SPLIT": f"{validation_split:.6f}"}
    input_fingerprints = _prepare_workspace(work_dir)

    try:
        result = asyncio.run(execute_solution(code, work_dir, timeout_seconds=timeout, env=env))
    except Exception as exc:
        _restore_mutated_inputs(work_dir, input_fingerprints)
        logfire.error("openevolve_execution_failed", error=str(exc))
        return EvaluationResult(metrics=_failure_metrics(), artifacts=_error_artifacts(exc))

    mutated_inputs = _restore_mutated_inputs(work_dir, input_fingerprints)
    if mutated_inputs:
        logfire.warning("openevolve_inputs_restored", files=mutated_inputs)

    cv_score = parse_baseline_score(result.stdout)
    applied_hints = detect_applied_hints(code, hints) if hints else []
    submission_path = work_dir / _SUBMISSION_FILENAME
    valid = result.returncode == 0 and cv_score is not None and submission_path.exists() and not mutated_inputs

    fitness = _fitness_from_score(cv_score, metric_direction)
    cv_variance = _compute_cv_variance(result.stdout)
    metrics = {
        "combined_score": fitness,
        "fitness": fitness,
        "cv_score": float(cv_score) if cv_score is not None else 0.0,
        "cv_variance": cv_variance,
        "valid": 1.0 if valid else 0.0,
        "returncode": float(result.returncode),
        "runtime_ms": float(result.runtime_ms),
        "timeout": 1.0 if result.timed_out else 0.0,
        "model_family": _model_family_score(code),
    }

    # Build structured error feedback for failed evaluations
    feedback_sections: list[str] = []
    if mutated_inputs:
        feedback_sections.append(_MUTATED_INPUT_HINT.format(files=", ".join(mutated_inputs)))
    if result.returncode != 0:
        feedback_sections.append(_extract_error_feedback(result.stderr, result.stdout))
    error_feedback = "\n\n".join(feedback_sections)

    artifacts = {
        "stdout": _truncate(result.stdout, 2000),
        "stderr": _truncate(result.stderr, 1000),
        "warnings": "\n".join(_extract_warnings(result.stderr)),
        "applied_hints": json.dumps(sorted(applied_hints)),
        "hint_count": str(len(applied_hints)),
        "mutated_inputs": json.dumps(mutated_inputs),
        "submission_exists": str(submission_path.exists()),
        "error_feedback": error_feedback,
        "execution_status": "success" if valid else "failed",
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Stage 1: Quick syntax and import validation.

        This stage runs with a 5-second timeout and validates:
        - Python syntax is correct (AST parsing)
        - Required imports are available
        - Basic structure is present (has main function or training logic)

    Args:
            program_path: Path to the Python solution file.

    Returns:
            EvaluationResult with score 0.0 if validation fails, 0.3 if passes.

    @notice: |
        Stage 1: Quick syntax and import validation.

    @dev: |
        See module for behavior details and invariants.
    """
    try:
        code = Path(program_path).read_text(encoding="utf-8")
    except OSError as exc:
        logfire.warning("stage1_read_failed", error=str(exc))
        return EvaluationResult(
            metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
            artifacts={"error": str(exc), "stage": "stage1"},
        )

    # Step 1: Check syntax by parsing AST
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logfire.info("stage1_syntax_error", error=str(exc))
        return EvaluationResult(
            metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
            artifacts={
                "error": f"Syntax error: {exc}",
                "stage": "stage1",
                "feedback": "Fix Python syntax errors before proceeding.",
            },
        )

    # Step 2: Check for required ML imports
    required_imports = {"pandas", "numpy"}
    found_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found_imports.add(node.module.split(".")[0])

    if not (required_imports & found_imports):
        logfire.info("stage1_missing_imports", missing=list(required_imports))
        return EvaluationResult(
            metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
            artifacts={
                "error": "Missing required imports (pandas/numpy)",
                "stage": "stage1",
                "feedback": "Add required ML library imports.",
            },
        )

    # Step 3: Check for basic ML structure
    has_model_training = False
    for node in ast.walk(tree):
        # Look for common ML patterns: .fit(), .predict(), train_test_split()
        if isinstance(node, ast.Attribute) and node.attr in {"fit", "predict", "train_test_split"}:
            has_model_training = True
            break
        # Look for LightGBM/XGBoost/sklearn model classes
        if isinstance(node, ast.Name) and any(
            pattern in node.id
            for pattern in {"LGBM", "LightGBM", "XGB", "Regressor", "Classifier", "RandomForest", "GradientBoosting"}
        ):
            has_model_training = True
            break

    if not has_model_training:
        logfire.info("stage1_no_training_logic")
        return EvaluationResult(
            metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
            artifacts={
                "error": "No model training logic found",
                "stage": "stage1",
                "feedback": "Add model training code (fit/predict).",
            },
        )

    # All checks passed - return threshold score
    logfire.info("stage1_passed")
    return EvaluationResult(
        metrics={"combined_score": 0.3, "fitness": 0.3, "valid": 1.0}, artifacts={"stage": "stage1", "status": "passed"}
    )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Stage 2: Medium evaluation on subset of data.

        This stage runs with a 30-second timeout and:
        - Executes the solution on a small data subset (first 1000 rows)
        - Validates output format is correct
        - Gets a preliminary score estimate

    Args:
            program_path: Path to the Python solution file.

    Returns:
            EvaluationResult with preliminary fitness score.

    @notice: |
        Stage 2: Medium evaluation on subset of data.

    @dev: |
        See module for behavior details and invariants.
    """
    from agent_k.core.solution import execute_solution, parse_baseline_score

    context = _load_context()
    work_dir = _resolve_work_dir(context)
    metric_direction = str(context.get("metric_direction", "minimize")).lower()

    try:
        code = Path(program_path).read_text(encoding="utf-8")
    except OSError as exc:
        logfire.warning("stage2_read_failed", error=str(exc))
        return EvaluationResult(
            metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
            artifacts={"error": str(exc), "stage": "stage2"},
        )

    # Create a temporary directory with subset data
    subset_dir = Path(tempfile.mkdtemp())
    try:
        # Copy subset of data files
        _create_subset_data(work_dir, subset_dir, max_rows=_STAGE2_DATA_ROWS)

        # Execute with reduced timeout
        env = {"AGENT_K_VALIDATION_SPLIT": "0.2"}
        result = asyncio.run(execute_solution(code, subset_dir, timeout_seconds=_STAGE2_TIMEOUT, env=env))

        # Parse score from output
        cv_score = parse_baseline_score(result.stdout)
        submission_path = subset_dir / "submission.csv"
        valid = result.returncode == 0 and cv_score is not None and submission_path.exists()

        if not valid:
            error_feedback = _extract_error_feedback(result.stderr, result.stdout)
            logfire.info("stage2_execution_failed", returncode=result.returncode)
            return EvaluationResult(
                metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
                artifacts={
                    "error": "Execution failed on subset",
                    "stage": "stage2",
                    "stderr": _truncate(result.stderr, 500),
                    "feedback": error_feedback,
                },
            )

        # Calculate preliminary fitness
        fitness = _fitness_from_score(cv_score, metric_direction)

        # Scale fitness to be conservative (stage2 threshold is 0.6)
        # If it looks promising on subset, give it 0.65 to pass to full eval
        scaled_fitness = min(0.65, fitness) if fitness > 0.4 else 0.0

        logfire.info("stage2_passed", fitness=scaled_fitness, cv_score=cv_score)
        return EvaluationResult(
            metrics={
                "combined_score": scaled_fitness,
                "fitness": scaled_fitness,
                "cv_score": float(cv_score) if cv_score is not None else 0.0,
                "valid": 1.0,
            },
            artifacts={"stage": "stage2", "status": "passed", "subset_rows": str(_STAGE2_DATA_ROWS)},
        )

    except Exception as exc:
        logfire.error("stage2_exception", error=str(exc))
        return EvaluationResult(
            metrics={"combined_score": 0.0, "fitness": 0.0, "valid": 0.0},
            artifacts={"error": str(exc), "stage": "stage2", "traceback": _truncate(traceback.format_exc(), 800)},
        )
    finally:
        # Clean up subset directory
        shutil.rmtree(subset_dir, ignore_errors=True)


def _create_subset_data(source_dir: Path, target_dir: Path, max_rows: int) -> None:
    """Create subset data files in target directory for faster evaluation."""
    import pandas as pd

    # Copy train.csv subset
    train_path = source_dir / "train.csv"
    if train_path.exists():
        try:
            df = pd.read_csv(train_path, nrows=max_rows)
            df.to_csv(target_dir / "train.csv", index=False)
        except Exception as exc:
            logfire.warning("subset_train_failed", error=str(exc))
            # Fallback: copy full file
            shutil.copy2(train_path, target_dir / "train.csv")

    # Copy test.csv subset
    test_path = source_dir / "test.csv"
    if test_path.exists():
        try:
            df = pd.read_csv(test_path, nrows=max_rows)
            df.to_csv(target_dir / "test.csv", index=False)
        except Exception as exc:
            logfire.warning("subset_test_failed", error=str(exc))
            shutil.copy2(test_path, target_dir / "test.csv")

    # Copy sample_submission.csv (always full file - it's small)
    sample_path = source_dir / "sample_submission.csv"
    if sample_path.exists():
        shutil.copy2(sample_path, target_dir / "sample_submission.csv")
