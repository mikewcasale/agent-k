import type {
  AgentKPatchOp,
  ErrorEvent,
  GenerationMetrics,
  LeaderboardSubmission,
  MissionPhase,
  MissionState,
  RecoveryStrategy,
  ToolType,
} from "./agent-k";

export type AgentKEventType =
  // State management
  | "state-snapshot"
  | "state-delta"

  // Phase lifecycle
  | "phase-start"
  | "phase-complete"
  | "phase-error"

  // Task lifecycle
  | "task-start"
  | "task-progress"
  | "task-complete"
  | "task-error"

  // Tool usage
  | "tool-start"
  | "tool-thinking"
  | "tool-result"
  | "tool-error"

  // Evolution specific
  | "generation-start"
  | "generation-complete"
  | "fitness-update"
  | "submission-result"
  | "convergence-detected"

  // Memory operations
  | "memory-store"
  | "memory-retrieve"
  | "checkpoint-created"

  // Error handling
  | "error-occurred"
  | "recovery-attempt"
  | "recovery-complete";

export type AgentKEvent<T extends AgentKEventType = AgentKEventType> = {
  type: T;
  timestamp: string;
  data: AgentKEventPayload[T];
};

export type AgentKEventPayload = {
  "state-snapshot": MissionState;
  "state-delta": AgentKPatchOp[];

  "phase-start": { phase: MissionPhase; objectives: string[] };
  "phase-complete": {
    phase: MissionPhase;
    success: boolean;
    durationMs: number;
  };
  "phase-error": { phase: MissionPhase; error: string; recoverable: boolean };

  "task-start": { taskId: string; phase: MissionPhase; name: string };
  "task-progress": { taskId: string; progress: number; message?: string };
  "task-complete": { taskId: string; success: boolean; result?: unknown };
  "task-error": { taskId: string; error: string };

  "tool-start": {
    taskId: string;
    toolCallId: string;
    toolType: ToolType;
    operation: string;
    params?: Record<string, unknown>;
  };
  "tool-thinking": { taskId: string; toolCallId: string; chunk: string };
  "tool-result": {
    taskId: string;
    toolCallId: string;
    result: unknown;
    durationMs: number;
  };
  "tool-error": { taskId: string; toolCallId: string; error: string };

  "generation-start": { generation: number; populationSize: number };
  "generation-complete": GenerationMetrics;
  "fitness-update": { generation: number; fitnessValues: number[] };
  "submission-result": LeaderboardSubmission;
  "convergence-detected": { generation: number; reason: string };

  "memory-store": { key: string; scope: string; category: string };
  "memory-retrieve": { key: string; found: boolean };
  "checkpoint-created": { name: string; phase: MissionPhase };

  "error-occurred": ErrorEvent;
  "recovery-attempt": {
    errorId: string;
    strategy: RecoveryStrategy;
    attempt: number;
  };
  "recovery-complete": {
    errorId: string;
    success: boolean;
    resolution?: string;
  };
};
