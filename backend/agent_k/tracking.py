"""Bridge mission events and artifacts to the fenced worker API.

(c) Mike Casale 2026.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from .events import AgentKEvent, EventData, EventType
from .stores import ArtifactReference, JobLease, WorkerEvent

if TYPE_CHECKING:
    from .runner import WorkerApiClient

__all__ = ("MissionTracker",)


@dataclass
class MissionTracker:
    """Serialize worker writes with deterministic event identities."""

    lease: JobLease
    client: WorkerApiClient
    sequence: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def append_event(
        self,
        mission_id: str,
        event_type: EventType,
        data: EventData,
        *,
        timestamp: datetime,
    ) -> AgentKEvent:
        """Persist an event before returning its browser representation."""
        if mission_id != str(self.lease.mission_id):
            raise ValueError("Event mission does not match the fenced lease")
        async with self._lock:
            sequence = self.sequence + 1
            event_id = uuid5(NAMESPACE_URL, f"agent-k:{self.lease.job_id}:{sequence}")
            event = WorkerEvent(
                event_id=event_id,
                sequence=sequence,
                type=event_type,
                timestamp=timestamp,
                data=data,
            )
            await self.client.append_events(self.lease, (event,))
            self.sequence = sequence
        return AgentKEvent(
            event_id=str(event_id),
            sequence=sequence,
            mission_id=mission_id,
            type=event_type,
            timestamp=timestamp,
            data=data,
        )

    async def upload(self, path: Path, *, role: str) -> ArtifactReference:
        """Upload an immutable artifact through a scoped, expiring URL."""
        signed = await self.client.request_upload(
            self.lease,
            path,
            content_type="application/octet-stream",
            role=role,
        )
        return await self.client.upload_file(self.lease, signed, path, role=role)
