"""CLI entry point for AGENT-K FastAPI server.

This module provides the uvicorn-compatible app instance for Render deployment.
"""
from agent_k.ui.ag_ui.app import create_app

# Create the FastAPI app instance
app = create_app()


# Add health check endpoint for Render
@app.get('/health')
async def health_check() -> dict[str, str]:
    """Health check endpoint for Render."""
    return {'status': 'healthy'}

