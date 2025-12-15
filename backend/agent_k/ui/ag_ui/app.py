"""FastAPI application for AG-UI protocol.

Provides SSE endpoint for frontend event streaming.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import logfire
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...agents.lycurgus import LycurgusOrchestrator
from ...core.models import MissionCriteria, MissionResult
from ...graph.state import GraphContext, MissionState
from ..event_stream import EventEmitter
from .intent_parser import parse_mission_intent
from .stream_transformer import stream_text_response, transform_to_vercel_stream

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
    
    @app.get('/health')
    async def health_check() -> dict[str, str]:
        """Health check endpoint for Render."""
        return {'status': 'healthy'}
    
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

    @app.post('/agentic_generative_ui/')
    async def handle_chat(request: Request) -> StreamingResponse:
        """Handle chat messages from frontend.

        Intelligently routes to either:
        - Mission mode: Detect intent and start orchestrator
        - Chat mode: Simple Q&A about Agent-K
        """
        try:
            body = await request.json()
            messages = body.get('messages', [])
            chat_id = body.get('id', str(uuid.uuid4()))

            logfire.info('chat_request_received', chat_id=chat_id, message_count=len(messages))

            # Parse for mission intent
            mission_criteria = await parse_mission_intent(messages)

            if mission_criteria:
                # Mission mode: start orchestrator
                logfire.info('mission_mode_activated', chat_id=chat_id, criteria=mission_criteria.model_dump())

                # Create event emitter for this mission
                emitter = EventEmitter()

                # Create async task to run mission
                async def run_mission():
                    try:
                        with logfire.span('mission_execution', chat_id=chat_id):
                            # Use orchestrator as context manager for proper initialization/cleanup
                            async with LycurgusOrchestrator() as orchestrator:
                                result = await orchestrator.execute_mission(
                                    competition_id=None,  # Let LOBBYIST discover
                                    criteria=mission_criteria,
                                )
                                # Emit completion event
                                await emitter.emit('mission-complete', {
                                    'success': result.success,
                                    'final_rank': result.final_rank,
                                    'final_score': result.final_score,
                                })
                    except Exception as e:
                        logfire.error('mission_execution_failed', error=str(e), chat_id=chat_id)
                        await emitter.emit('error-occurred', {
                            'category': 'mission_execution',
                            'message': str(e),
                        })
                    finally:
                        emitter.close()

                # Start mission in background
                asyncio.create_task(run_mission())

                # Stream events to frontend
                return StreamingResponse(
                    transform_to_vercel_stream(emitter),
                    media_type='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                        'X-Vercel-AI-Data-Stream': 'v1',
                    },
                )
            else:
                # Chat mode: simple Q&A
                logfire.info('chat_mode_activated', chat_id=chat_id)

                response_text = """I'm Agent-K, an autonomous multi-agent system for Kaggle competitions.

I can help you discover, research, and compete in Kaggle competitions using a team of specialized AI agents:

- **LOBBYIST**: Discovers competitions matching your criteria
- **SCIENTIST**: Researches winning approaches and analyzes data
- **EVOLVER**: Evolves solutions using evolutionary code search
- **LYCURGUS**: Orchestrates the entire mission

To start a mission, try saying something like:
- "Find me a Kaggle competition"
- "Compete in a computer vision challenge with at least $10,000 prize"
- "Discover featured competitions in NLP"

What would you like to do?"""

                return StreamingResponse(
                    stream_text_response(response_text),
                    media_type='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                        'X-Vercel-AI-Data-Stream': 'v1',
                    },
                )

        except Exception as e:
            logfire.error('chat_endpoint_error', error=str(e))
            # Return error as text stream
            error_text = f"An error occurred: {str(e)}"
            return StreamingResponse(
                stream_text_response(error_text),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Vercel-AI-Data-Stream': 'v1',
                },
            )

    return app
