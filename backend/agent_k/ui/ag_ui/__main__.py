"""Entry point for running the AG-UI server."""
from __future__ import annotations

import os

import uvicorn

from .app import create_app

app = create_app()


def main() -> None:
    """Run the AG-UI server."""
    port = int(os.environ.get('PORT', '9000'))
    host = os.environ.get('HOST', '0.0.0.0')
    
    uvicorn.run(
        'agent_k.ui.ag_ui.__main__:app',
        host=host,
        port=port,
        reload=os.environ.get('RELOAD', 'false').lower() == 'true',
    )


if __name__ == '__main__':
    main()

