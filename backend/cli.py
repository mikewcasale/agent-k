"""CLI entry point for AGENT-K FastAPI server.

This module provides the uvicorn-compatible app instance for Render deployment.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

from agent_k.ui.ag_ui import create_app

__all__ = ('app', 'health_check')

# Create the FastAPI app instance
app = create_app()


# Add health check endpoint for Render
@app.get('/health')
async def health_check() -> dict[str, str]:
    """Health check endpoint for Render."""
    return {'status': 'healthy'}
