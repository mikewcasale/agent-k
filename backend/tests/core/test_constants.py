"""Tests for core constants.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import importlib

import pytest

import agent_k.core.constants as constants_module

__all__ = ()


class TestDefaultModel:
    """Regression tests for DEFAULT_MODEL fallback behavior.

    The fallback must not select a paid provider (Anthropic / OpenAI) so the
    platform stays usable when only OpenRouter credentials are configured.
    Paid-model fallback previously caused production missions to fail at the
    Research phase with `ModelHTTPError 400 - credit balance is too low`.
    """

    def test_fallback_is_free_openrouter_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When DEFAULT_MODEL env is unset, fallback must point to a free OpenRouter model."""
        monkeypatch.delenv("DEFAULT_MODEL", raising=False)
        reloaded = importlib.reload(constants_module)
        try:
            assert reloaded.DEFAULT_MODEL.startswith("openrouter:"), (
                f"DEFAULT_MODEL fallback must use OpenRouter, got {reloaded.DEFAULT_MODEL!r}"
            )
            assert reloaded.DEFAULT_MODEL.endswith(":free"), (
                f"DEFAULT_MODEL fallback must use a :free OpenRouter tier, got {reloaded.DEFAULT_MODEL!r}"
            )
        finally:
            importlib.reload(constants_module)

    def test_env_override_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit DEFAULT_MODEL env value must be honored verbatim."""
        monkeypatch.setenv("DEFAULT_MODEL", "anthropic:claude-sonnet-4-5")
        reloaded = importlib.reload(constants_module)
        try:
            assert reloaded.DEFAULT_MODEL == "anthropic:claude-sonnet-4-5"
        finally:
            monkeypatch.delenv("DEFAULT_MODEL", raising=False)
            importlib.reload(constants_module)
