"""Shared lightweight domain types used by API and worker modules.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from typing import Literal

__all__ = ("CompetitionId", "ErrorCategory", "MetricDirection", "RecoveryStrategy")

type CompetitionId = str
type MetricDirection = Literal["maximize", "minimize"]
type ErrorCategory = Literal["transient", "recoverable", "fatal"]
type RecoveryStrategy = Literal["retry", "fallback", "skip", "replan", "abort"]
