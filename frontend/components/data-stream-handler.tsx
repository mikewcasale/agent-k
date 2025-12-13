"use client";

import { useEffect, useMemo, useRef } from "react";
import { initialArtifactData, useArtifact } from "@/hooks/use-artifact";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import type { AgentKEvent, AgentKEventType } from "@/lib/types/events";
import type { MissionState } from "@/lib/types/agent-k";
import { artifactDefinitions } from "./artifact";
import { useDataStream } from "./data-stream-provider";

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
]);

export function DataStreamHandler() {
  const { dataStream, setDataStream } = useDataStream();
  const { artifact, setArtifact, setMetadata } = useArtifact();
  const { dispatch, appendToolThinking } = useAgentKState();

  const thinkingBuffer = useRef<Map<string, string>>(new Map());

  const isAgentKEvent = useMemo(
    () => (part: { type: string }): part is AgentKEvent =>
      AGENT_K_EVENT_TYPES.has(part.type as AgentKEventType),
    []
  );

  useEffect(() => {
    if (!dataStream?.length) {
      return;
    }

    const newDeltas = dataStream.slice();
    setDataStream([]);

    for (const delta of newDeltas) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if (isAgentKEvent(delta as any)) {
        handleEvent(delta as unknown as AgentKEvent);
        continue;
      }

      const artifactDefinition = artifactDefinitions.find(
        (currentArtifactDefinition) =>
          currentArtifactDefinition.kind === artifact.kind
      );

      if (artifactDefinition?.onStreamPart) {
        artifactDefinition.onStreamPart({
          streamPart: delta,
          setArtifact,
          setMetadata,
        });
      }

      setArtifact((draftArtifact) => {
        if (!draftArtifact) {
          return { ...initialArtifactData, status: "streaming" };
        }

        switch (delta.type) {
          case "data-id":
            return {
              ...draftArtifact,
              documentId: (delta as any).data?.toString?.() ?? "",
              status: "streaming",
            };

          case "data-title":
            return {
              ...draftArtifact,
              title: delta.data as string,
              status: "streaming",
            };

          case "data-kind":
            return {
              ...draftArtifact,
              kind: delta.data,
              status: "streaming",
            };

          case "data-clear":
            return {
              ...draftArtifact,
              content: "",
              status: "streaming",
            };

          case "data-finish":
            return {
              ...draftArtifact,
              status: "idle",
            };

          default:
            return draftArtifact;
        }
      });
    }
  }, [
    appendToolThinking,
    artifact,
    dataStream,
    dispatch,
    isAgentKEvent,
    setArtifact,
    setDataStream,
    setMetadata,
  ]);

  function handleEvent(event: AgentKEvent) {
    const timestamp = (event as any).timestamp ?? new Date().toISOString();
    switch (event.type) {
      case "state-snapshot":
        dispatch({ type: "SET_STATE", payload: event.data as MissionState });
        break;

      case "state-delta":
        dispatch({
          type: "APPLY_PATCH",
          payload: Array.isArray((event as any).data)
            ? (event as any).data
            : ((event as any).data?.patches ?? []),
        });
        break;

      case "phase-start":
        dispatch({ type: "PHASE_START", payload: { ...(event.data as any), timestamp } });
        break;

      case "phase-complete":
        dispatch({ type: "PHASE_COMPLETE", payload: { ...(event.data as any), timestamp } });
        break;

      case "task-start":
        dispatch({ type: "TASK_START", payload: { ...(event.data as any), timestamp } });
        break;

      case "task-progress":
        dispatch({ type: "TASK_PROGRESS", payload: event.data as any });
        break;

      case "task-complete":
        dispatch({ type: "TASK_COMPLETE", payload: { ...(event.data as any), timestamp } });
        break;

      case "tool-start": {
        const data = event.data as any;
        dispatch({
          type: "TOOL_START",
          payload: {
            taskId: data.taskId,
            toolCall: {
              id: data.toolCallId,
              type: data.toolType,
              operation: data.operation,
              params: data.params,
              startedAt: timestamp,
              thinking: "",
            },
          },
        });
        break;
      }

      case "tool-thinking": {
        const data = event.data as any;
        const { taskId, toolCallId, chunk } = data;
        const key = `${taskId}:${toolCallId}`;
        const current = thinkingBuffer.current.get(key) || "";
        thinkingBuffer.current.set(key, current + chunk);
        appendToolThinking(taskId, toolCallId, thinkingBuffer.current.get(key) ?? "");
        break;
      }

      case "tool-result": {
        const data = event.data as any;
        const resultKey = `${data.taskId}:${data.toolCallId}`;
        thinkingBuffer.current.delete(resultKey);
        dispatch({
          type: "TOOL_RESULT",
          payload: { ...data, timestamp },
        });
        break;
      }

      case "tool-error": {
        const data = event.data as any;
        const errorKey = `${data.taskId}:${data.toolCallId}`;
        thinkingBuffer.current.delete(errorKey);
        dispatch({
          type: "TOOL_ERROR",
          payload: { ...data, timestamp },
        });
        break;
      }

      case "generation-complete":
        dispatch({ type: "GENERATION_COMPLETE", payload: event.data as any });
        break;

      case "submission-result":
        dispatch({ type: "SUBMISSION_RESULT", payload: event.data as any });
        break;

      case "convergence-detected":
        dispatch({ type: "CONVERGENCE_DETECTED", payload: event.data as any });
        break;

      case "memory-store":
      case "memory-retrieve":
        dispatch({
          type: "MEMORY_OPERATION",
          payload: { ...(event.data as any), operation: event.type },
        });
        break;

      case "checkpoint-created":
        dispatch({ type: "CHECKPOINT_CREATED", payload: { ...(event.data as any), timestamp } });
        break;

      case "error-occurred":
        dispatch({ type: "ERROR_OCCURRED", payload: event.data as any });
        break;

      case "recovery-complete":
        dispatch({ type: "RECOVERY_COMPLETE", payload: event.data as any });
        break;

      default:
        // eslint-disable-next-line no-console
        console.warn("Unknown Agent K event:", event.type);
    }
  }

  return null;
}
