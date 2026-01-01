"use client";

import type { DataUIPart } from "ai";
import { useEffect, useRef } from "react";
import { useDataStream } from "@/components/data-stream-provider";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import type { CustomUIDataTypes } from "@/lib/types";
import type { AgentKEventType } from "@/lib/types/events";

const AGENT_K_EVENT_TYPES: Set<AgentKEventType> = new Set([
  "state-snapshot",
  "state-delta",
  "phase-start",
  "phase-complete",
  "phase-error",
  "task-start",
  "task-progress",
  "task-complete",
  "task-error",
  "tool-start",
  "tool-thinking",
  "tool-result",
  "tool-error",
  "generation-start",
  "generation-complete",
  "fitness-update",
  "submission-result",
  "convergence-detected",
  "memory-store",
  "memory-retrieve",
  "checkpoint-created",
  "error-occurred",
  "recovery-attempt",
  "recovery-complete",
  "mission-complete",
]);

export function MissionStream() {
  const { state } = useAgentKState();
  const { setDataStream } = useDataStream();
  const missionId = state.mission.missionId;
  const streamRef = useRef<EventSource | null>(null);
  const missionRef = useRef<string | null>(null);

  useEffect(() => {
    if (!missionId) {
      if (streamRef.current) {
        streamRef.current.close();
        streamRef.current = null;
        missionRef.current = null;
      }
      return;
    }

    if (missionRef.current === missionId && streamRef.current) {
      return;
    }

    if (streamRef.current) {
      streamRef.current.close();
    }

    const stream = new EventSource(`/api/mission/${missionId}/stream`);
    streamRef.current = stream;
    missionRef.current = missionId;

    stream.onmessage = (event) => {
      if (!event.data) {
        return;
      }

      let payload: { type?: string; data?: unknown; timestamp?: string } | null =
        null;
      try {
        payload = JSON.parse(event.data) as {
          type?: string;
          data?: unknown;
          timestamp?: string;
        };
      } catch {
        return;
      }

      if (!payload?.type) {
        return;
      }

      if (!AGENT_K_EVENT_TYPES.has(payload.type as AgentKEventType)) {
        return;
      }

      setDataStream((current) => [
        ...current,
        {
          type: payload.type,
          data: payload.data,
          timestamp: payload.timestamp,
        } as DataUIPart<CustomUIDataTypes>,
      ]);

      if (payload.type === "mission-complete") {
        stream.close();
        streamRef.current = null;
        missionRef.current = null;
      }
    };

    stream.onerror = () => {
      if (stream.readyState === EventSource.CLOSED) {
        stream.close();
        streamRef.current = null;
        missionRef.current = null;
      }
    };

    return () => {
      stream.close();
      if (streamRef.current === stream) {
        streamRef.current = null;
        missionRef.current = null;
      }
    };
  }, [missionId, setDataStream]);

  return null;
}
