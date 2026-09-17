"""Tests for solution execution utilities.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pytest

from agent_k.core.solution import (
    SOLUTION_SIGNATURE_LENGTH,
    _is_sensitive_env_key,
    _sanitize_env,
    execute_solution,
    parse_baseline_score,
    solution_signature,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import TestEnv

__all__ = ()

pytestmark = pytest.mark.anyio


class TestParseBaselineScore:
    """Tests for baseline score parsing."""

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("Baseline RMSE score: 0.123", 0.123),
            ("baseline accuracy score: -1.5", -1.5),
            ("some text\nBaseline logLoss score: 1.2345\n", 1.2345),
            ("no score here", None),
            ("Baseline score: not-a-number", None),
        ],
    )
    def test_parse_baseline_score(self, output: str, expected: float | None) -> None:
        """Baseline score parsing should handle common formats."""
        result = parse_baseline_score(output)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


class TestSolutionSignature:
    """Tests for formatting-insensitive solution signatures."""

    _BASE = (
        "import pandas as pd\n"
        "import lightgbm as lgb\n"
        "\n"
        "train = pd.read_csv('./train.csv')\n"
        "params = {'objective': 'regression', 'learning_rate': 0.05}\n"
        "model = lgb.train(params, lgb.Dataset(train.drop(columns=['y']), train['y']))\n"
    )

    def test_signature_is_short_hex(self) -> None:
        """Signatures keep the stored 12-character hex shape."""
        signature = solution_signature(self._BASE)

        assert len(signature) == SOLUTION_SIGNATURE_LENGTH
        assert all(char in "0123456789abcdef" for char in signature)

    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(_BASE + "\n\n", id="trailing-blank-lines"),
            pytest.param(_BASE.replace("'./train.csv'", '"./train.csv"'), id="quote-style"),
            pytest.param("# tuned learning rate\n" + _BASE, id="added-comment"),
            pytest.param(_BASE.replace("\n\n", "\n\n\n"), id="extra-blank-line"),
            pytest.param(_BASE.replace("0.05", "(0.05)"), id="redundant-parentheses"),
            pytest.param(_BASE.replace("./train.csv", "/kaggle/input/train.csv"), id="kaggle-input-prefix"),
        ],
    )
    def test_cosmetic_edits_share_a_signature(self, variant: str) -> None:
        """Edits that cannot change execution must not miss the evaluation cache."""
        assert solution_signature(variant) == solution_signature(self._BASE)

    @pytest.mark.parametrize(
        "variant",
        [
            pytest.param(_BASE.replace("0.05", "0.01"), id="hyperparameter"),
            pytest.param(_BASE.replace("'regression'", "'binary'"), id="objective"),
            pytest.param(_BASE + "model.save_model('model.txt')\n", id="added-statement"),
            pytest.param('"""Docstring."""\n' + _BASE, id="added-docstring"),
        ],
    )
    def test_semantic_edits_get_distinct_signatures(self, variant: str) -> None:
        """Anything that can change execution must be evaluated on its own."""
        assert solution_signature(variant) != solution_signature(self._BASE)

    def test_unparsable_code_still_gets_a_stable_signature(self) -> None:
        """Malformed candidates fall back to whitespace normalization."""
        broken = "def f(:\n    return 1\n"

        assert solution_signature(broken) == solution_signature(broken + "\n\n")
        assert solution_signature(broken) != solution_signature("def g(:\n    return 1\n")


class TestEnvSanitization:
    """Tests for environment sanitization helpers."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [("OPENAI_API_KEY", True), ("kaggle_key", True), ("my_token", True), ("PATH", False), ("DATA_DIR", False)],
    )
    def test_is_sensitive_env_key(self, key: str, expected: bool) -> None:
        """Sensitive keys should be detected case-insensitively."""
        assert _is_sensitive_env_key(key) is expected

    def test_sanitize_env_filters_sensitive_keys(self, env: TestEnv, tmp_path: Path) -> None:
        """Sanitization should drop sensitive keys and set defaults."""
        env.set("KAGGLE_KEY", "secret")
        env.set("SAFE_VAR", "ok")

        sanitized = _sanitize_env({"EXTRA": "1"}, work_path=tmp_path)

        assert "KAGGLE_KEY" not in sanitized
        assert sanitized["SAFE_VAR"] == "ok"
        assert sanitized["EXTRA"] == "1"
        assert sanitized["HOME"] == str(tmp_path)
        assert sanitized["PYTHONNOUSERSITE"] == "1"
        assert sanitized["PYTHONDONTWRITEBYTECODE"] == "1"


class TestExecuteSolution:
    """Tests for execute_solution behavior."""

    async def test_execute_solution_nonzero_exit(self, tmp_path: Path) -> None:
        """Execution should capture non-zero return codes."""
        code = "import sys\nsys.exit(7)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=1)

        assert result.returncode == 7
        assert result.timed_out is False

    async def test_execute_solution_timeout(self, tmp_path: Path) -> None:
        """Execution should report timeouts."""
        code = "import time\ntime.sleep(1)\n"
        result = await execute_solution(code, tmp_path, timeout_seconds=0.2)

        assert result.timed_out is True
        assert result.returncode != 0
