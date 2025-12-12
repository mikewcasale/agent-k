"""Supplementary tools for the Scientist agent."""
from __future__ import annotations

from .agent import ResearchReport

__all__ = ['summarize_findings']


async def summarize_findings(report: ResearchReport) -> str:
    """Summarize research findings into a brief narrative."""
    return f'{report.competition_id}: {len(report.domain_findings)} domain findings'
