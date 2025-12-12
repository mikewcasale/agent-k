"""Submission service for AGENT-K."""
from __future__ import annotations

from ..core.models import Submission

__all__ = ['SubmissionService']


class SubmissionService:
    """Service handling submissions."""
    
    async def record_submission(self, submission: Submission) -> None:
        """Persist submission metadata."""
        return None
