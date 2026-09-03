"""Tests for OpenEvolve evaluator working-directory isolation.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import json
from pathlib import Path

import pytest
from openevolve.evaluation_result import EvaluationResult

from agent_k.evolution.evaluator import _iter_input_files, _prepare_workspace, _restore_mutated_inputs, evaluate

__all__ = ()

_CONTEXT_ENV = "AGENT_K_OPENEVOLVE_CONTEXT"

_TRAIN_CSV = "Id,feature,target\n1,0.5,10\n2,1.5,20\n3,2.5,30\n"
_TEST_CSV = "Id,feature\n4,3.5\n5,4.5\n"
_SAMPLE_CSV = "Id,target\n4,0\n5,0\n"

_WRITES_SUBMISSION = """
import csv

with open("sample_submission.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))

with open("submission.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["Id", "target"])
    writer.writeheader()
    for row in rows:
        writer.writerow({"Id": row["Id"], "target": 20.0})

print("Baseline RMSE score: 0.250000")
"""

_SCORES_WITHOUT_SUBMISSION = """
print("Baseline RMSE score: 0.100000")
"""

_WRITES_DERIVED_FILE = """
with open("features.csv", "w", newline="") as handle:
    handle.write("Id,feature_squared\\n1,0.25\\n")

with open("submission.csv", "w", newline="") as handle:
    handle.write("Id,target\\n4,20.0\\n5,20.0\\n")

print("Baseline RMSE score: 0.200000")
"""

_OVERWRITES_TRAIN = """
with open("train.csv", "w", newline="") as handle:
    handle.write("Id,feature,target\\n1,0.5,10\\n")

with open("submission.csv", "w", newline="") as handle:
    handle.write("Id,target\\n4,20.0\\n5,20.0\\n")

print("Baseline RMSE score: 0.010000")
"""


def _stage_work_dir(work_dir: Path, *, competition_id: str | None = None) -> None:
    """Write the canonical competition files a mission stages before evolving."""
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "train.csv").write_text(_TRAIN_CSV, encoding="utf-8")
    (work_dir / "test.csv").write_text(_TEST_CSV, encoding="utf-8")
    (work_dir / "sample_submission.csv").write_text(_SAMPLE_CSV, encoding="utf-8")
    if competition_id:
        nested = work_dir / competition_id
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "train.csv").write_text(_TRAIN_CSV, encoding="utf-8")


def _set_context(monkeypatch: pytest.MonkeyPatch, work_dir: Path) -> None:
    payload = {"work_dir": str(work_dir), "timeout": 60, "validation_split": 0.2, "metric_direction": "minimize"}
    monkeypatch.setenv(_CONTEXT_ENV, json.dumps(payload))


def _run(program: str, *, tmp_path: Path, name: str) -> EvaluationResult:
    program_path = tmp_path / name
    program_path.write_text(program, encoding="utf-8")
    return evaluate(str(program_path))


class TestIterInputFiles:
    """Tests for ``_iter_input_files``."""

    def test_yields_staged_data_files(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path, competition_id="demo-comp")

        names = {path.relative_to(tmp_path).as_posix() for path in _iter_input_files(tmp_path)}

        assert names == {"train.csv", "test.csv", "sample_submission.csv", "demo-comp/train.csv"}

    def test_skips_run_artifacts_and_bookkeeping_dirs(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path)
        (tmp_path / "submission.csv").write_text("Id,target\n", encoding="utf-8")
        (tmp_path / "solution.py").write_text("print(1)\n", encoding="utf-8")
        (tmp_path / "initial_program.py").write_text("print(1)\n", encoding="utf-8")
        checkpoints = tmp_path / "openevolve_output" / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "program.json").write_text("{}", encoding="utf-8")
        pristine = tmp_path / ".agent_k_pristine"
        pristine.mkdir()
        (pristine / "train.csv").write_text(_TRAIN_CSV, encoding="utf-8")

        names = {path.relative_to(tmp_path).as_posix() for path in _iter_input_files(tmp_path)}

        assert names == {"train.csv", "test.csv", "sample_submission.csv"}


class TestPrepareWorkspace:
    """Tests for ``_prepare_workspace``."""

    def test_removes_stale_submission(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path)
        (tmp_path / "submission.csv").write_text("Id,target\n4,1.0\n", encoding="utf-8")

        _prepare_workspace(tmp_path)

        assert not (tmp_path / "submission.csv").exists()

    def test_snapshots_inputs_once(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path, competition_id="demo-comp")

        fingerprints = _prepare_workspace(tmp_path)

        assert set(fingerprints) == {"train.csv", "test.csv", "sample_submission.csv", "demo-comp/train.csv"}
        assert (tmp_path / ".agent_k_pristine" / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV
        assert (tmp_path / ".agent_k_pristine" / "demo-comp" / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV

    def test_input_set_is_pinned_by_the_first_candidate(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path)
        first = _prepare_workspace(tmp_path)
        (tmp_path / "features.csv").write_text("Id,feature_squared\n1,0.25\n", encoding="utf-8")

        second = _prepare_workspace(tmp_path)

        assert second == first
        assert "features.csv" not in second


class TestRestoreMutatedInputs:
    """Tests for ``_restore_mutated_inputs``."""

    def test_reports_nothing_when_inputs_untouched(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path)
        fingerprints = _prepare_workspace(tmp_path)

        assert _restore_mutated_inputs(tmp_path, fingerprints) == []

    def test_restores_overwritten_input(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path)
        fingerprints = _prepare_workspace(tmp_path)
        (tmp_path / "train.csv").write_text("Id,feature,target\n", encoding="utf-8")

        mutated = _restore_mutated_inputs(tmp_path, fingerprints)

        assert mutated == ["train.csv"]
        assert (tmp_path / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV

    def test_restores_deleted_input(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path)
        fingerprints = _prepare_workspace(tmp_path)
        (tmp_path / "test.csv").unlink()

        mutated = _restore_mutated_inputs(tmp_path, fingerprints)

        assert mutated == ["test.csv"]
        assert (tmp_path / "test.csv").read_text(encoding="utf-8") == _TEST_CSV

    def test_restores_nested_input(self, tmp_path: Path) -> None:
        _stage_work_dir(tmp_path, competition_id="demo-comp")
        fingerprints = _prepare_workspace(tmp_path)
        (tmp_path / "demo-comp" / "train.csv").write_text("corrupted\n", encoding="utf-8")

        mutated = _restore_mutated_inputs(tmp_path, fingerprints)

        assert mutated == ["demo-comp/train.csv"]
        assert (tmp_path / "demo-comp" / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV


class TestEvaluateIsolation:
    """End-to-end tests for ``evaluate`` against a shared working directory."""

    def test_scoring_candidate_is_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work_dir = tmp_path / "work"
        _stage_work_dir(work_dir)
        _set_context(monkeypatch, work_dir)

        result = _run(_WRITES_SUBMISSION, tmp_path=tmp_path, name="candidate_a.py")

        assert result.metrics["valid"] == 1.0
        assert result.artifacts["submission_exists"] == "True"

    def test_stale_submission_does_not_validate_next_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = tmp_path / "work"
        _stage_work_dir(work_dir)
        _set_context(monkeypatch, work_dir)

        first = _run(_WRITES_SUBMISSION, tmp_path=tmp_path, name="candidate_a.py")
        assert first.metrics["valid"] == 1.0

        second = _run(_SCORES_WITHOUT_SUBMISSION, tmp_path=tmp_path, name="candidate_b.py")

        assert second.metrics["returncode"] == 0.0
        assert second.metrics["cv_score"] == pytest.approx(0.1)
        assert second.metrics["valid"] == 0.0
        assert second.artifacts["submission_exists"] == "False"

    def test_mutated_input_is_restored_and_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work_dir = tmp_path / "work"
        _stage_work_dir(work_dir)
        _set_context(monkeypatch, work_dir)

        result = _run(_OVERWRITES_TRAIN, tmp_path=tmp_path, name="candidate_c.py")

        assert (work_dir / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV
        assert json.loads(result.artifacts["mutated_inputs"]) == ["train.csv"]
        assert result.metrics["valid"] == 0.0
        assert "MUTATION HINT [MutatedInput]" in result.artifacts["error_feedback"]
        assert result.artifacts["execution_status"] == "failed"

    def test_derived_file_is_not_treated_as_an_input(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work_dir = tmp_path / "work"
        _stage_work_dir(work_dir)
        _set_context(monkeypatch, work_dir)

        _run(_WRITES_DERIVED_FILE, tmp_path=tmp_path, name="candidate_d.py")
        result = _run(_WRITES_DERIVED_FILE, tmp_path=tmp_path, name="candidate_d2.py")

        assert json.loads(result.artifacts["mutated_inputs"]) == []
        assert result.metrics["valid"] == 1.0

    def test_later_candidate_sees_original_inputs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        work_dir = tmp_path / "work"
        _stage_work_dir(work_dir)
        _set_context(monkeypatch, work_dir)

        _run(_OVERWRITES_TRAIN, tmp_path=tmp_path, name="candidate_c.py")
        result = _run(_WRITES_SUBMISSION, tmp_path=tmp_path, name="candidate_a.py")

        assert result.metrics["valid"] == 1.0
        assert json.loads(result.artifacts["mutated_inputs"]) == []
        assert (work_dir / "train.csv").read_text(encoding="utf-8") == _TRAIN_CSV
