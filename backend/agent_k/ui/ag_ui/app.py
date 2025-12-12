"""FastAPI application for AG-UI protocol.

Provides SSE endpoint for frontend event streaming.
"""
from __future__ import annotations

import uuid
from typing import Any

import logfire
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...core.models import MissionCriteria, MissionResult
from ...graph.state import GraphContext, MissionState
from ..event_stream import EventEmitter

__all__ = ['create_app', 'MissionRequest']


class MissionRequest(BaseModel):
    """Request to start a new mission."""
    
    criteria: MissionCriteria
    user_prompt: str | None = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title='AGENT-K',
        description='Multi-agent Kaggle competition system',
        version='0.1.0',
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    
    # Store active missions
    missions: dict[str, dict[str, Any]] = {}
    
    @app.post('/api/mission/start')
    async def start_mission(request: MissionRequest) -> dict[str, str]:
        """Start a new mission and return mission ID."""
        mission_id = str(uuid.uuid4())
        
        # Create event emitter
        emitter = EventEmitter()
        
        # Initialize mission state
        state = MissionState(
            mission_id=mission_id,
            criteria=request.criteria,
        )
        
        # Store mission
        missions[mission_id] = {
            'state': state,
            'emitter': emitter,
            'context': GraphContext(event_emitter=emitter),
        }
        
        logfire.info('mission_started', mission_id=mission_id)
        
        return {'missionId': mission_id}
    
    @app.get('/api/mission/{mission_id}/stream')
    async def stream_mission(mission_id: str, request: Request) -> StreamingResponse:
        """Stream mission events via SSE."""
        if mission_id not in missions:
            return StreamingResponse(
                iter(['data: {"error": "Mission not found"}\n\n']),
                media_type='text/event-stream',
            )
        
        emitter = missions[mission_id]['emitter']
        
        async def event_generator():
            async for event in emitter.stream():
                if await request.is_disconnected():
                    break
                yield event
        
        return StreamingResponse(
            event_generator(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            },
        )
    
    @app.get('/api/mission/{mission_id}/status')
    async def get_mission_status(mission_id: str) -> dict[str, Any]:
        """Get current mission status."""
        if mission_id not in missions:
            return {'error': 'Mission not found'}
        
        state = missions[mission_id]['state']
        return {
            'missionId': mission_id,
            'status': state.status,
            'currentPhase': state.current_phase,
            'progress': state.overall_progress,
            'competitionId': state.competition_id,
        }
    
    @app.post('/api/mission/{mission_id}/abort')
    async def abort_mission(mission_id: str) -> dict[str, str]:
        """Abort an active mission."""
        if mission_id not in missions:
            return {'error': 'Mission not found'}
        
        emitter = missions[mission_id]['emitter']
        emitter.close()
        
        logfire.info('mission_aborted', mission_id=mission_id)
        
        return {'status': 'aborted'}
    
    return app
