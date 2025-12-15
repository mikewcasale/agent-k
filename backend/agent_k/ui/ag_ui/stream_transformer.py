"""Stream transformer for converting EventEmitter output to Vercel AI Data Stream format.

Transforms standard SSE events to Vercel AI SDK format for frontend consumption.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from .event_stream import AgentKEvent, EventEmitter

__all__ = ['transform_to_vercel_stream', 'stream_text_response']


# Event types that should be sent as data events (type "8")
AGENT_K_EVENT_TYPES = {
    'state-snapshot',
    'state-delta',
    'phase-start',
    'phase-complete',
    'phase-error',
    'task-start',
    'task-progress',
    'task-complete',
    'task-error',
    'tool-start',
    'tool-thinking',
    'tool-result',
    'tool-error',
    'generation-start',
    'generation-complete',
    'fitness-update',
    'submission-result',
    'convergence-detected',
    'memory-store',
    'memory-retrieve',
    'checkpoint-created',
    'error-occurred',
    'recovery-attempt',
    'recovery-complete',
}


async def transform_to_vercel_stream(emitter: EventEmitter) -> AsyncIterator[str]:
    """Transform EventEmitter output to Vercel AI Data Stream format.

    Vercel AI SDK format:
    - Text delta: `0:"text content"\n`
    - Data event: `8:{"type":"event-type","data":{...}}\n`
    - Finish: `d:{"finishReason":"stop"}\n`

    Args:
        emitter: EventEmitter producing AgentKEvent instances.

    Yields:
        Vercel AI formatted event strings.
    """
    try:
        async for event_str in emitter.stream():
            # Parse the SSE formatted event
            # Event comes as "data: {json}\n\n"
            if not event_str.startswith('data: '):
                continue

            json_str = event_str[6:].strip()
            if not json_str:
                continue

            try:
                event_data = json.loads(json_str)
                event_type = event_data.get('type')

                if event_type in AGENT_K_EVENT_TYPES:
                    # Transform to Vercel AI data event format
                    data_part = json.dumps({
                        'type': event_type,
                        'data': event_data.get('data', {}),
                        'timestamp': event_data.get('timestamp'),
                    })
                    yield f'8:{data_part}\n'

            except json.JSONDecodeError:
                # Skip malformed events
                continue

    except Exception:
        # Ensure stream ends cleanly even on error
        pass
    finally:
        # Always send finish event
        yield 'd:{"finishReason":"stop"}\n'


async def stream_text_response(text: str) -> AsyncIterator[str]:
    """Stream a simple text response in Vercel AI format.

    Used for chat mode (non-mission) responses.

    Args:
        text: Text content to stream.

    Yields:
        Vercel AI formatted text deltas.
    """
    # Stream text in chunks for better UX
    chunk_size = 50
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        # Escape quotes and backslashes for JSON
        escaped = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        yield f'0:"{escaped}"\n'

    # Send finish event
    yield 'd:{"finishReason":"stop"}\n'
