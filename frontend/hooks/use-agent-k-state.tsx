"use client";

import type React from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react";
import type {
  AgentKPatchOp,
  AgentKState,
  AgentKUIState,
  ErrorEvent,
  GenerationMetrics,
  LeaderboardSubmission,
  MissionPhase,
  MissionState,
  PhasePlan,
  PlannedTask,
  ToolCall,
} from "@/lib/types/agent-k";
import { applyPatches } from "@/lib/utils/json-patch";

const UI_STATE_STORAGE_KEY = "agentk-ui-state";

type AgentKAction =
  // State management
  | { type: "SET_STATE"; payload: MissionState }
  | { type: "APPLY_PATCH"; payload: AgentKPatchOp[] }
  | { type: "RESET_STATE" }

  // Phase lifecycle
  | { type: "PHASE_START"; payload: { phase: MissionPhase; objectives: string[]; timestamp?: string } }
  | { type: "PHASE_COMPLETE"; payload: { phase: MissionPhase; success: boolean; durationMs?: number; timestamp?: string } }

  // Task lifecycle
  | { type: "TASK_START"; payload: { taskId: string; phase: MissionPhase; name: string; timestamp?: string } }
  | { type: "TASK_PROGRESS"; payload: { taskId: string; progress: number; message?: string } }
  | { type: "TASK_COMPLETE"; payload: { taskId: string; success: boolean; result?: unknown; timestamp?: string } }

  // Tool usage
  | { type: "TOOL_START"; payload: { taskId: string; toolCall: ToolCall } }
  | { type: "TOOL_THINKING"; payload: { taskId: string; toolCallId: string; thinking: string } }
  | { type: "TOOL_RESULT"; payload: { taskId: string; toolCallId: string; result: unknown; durationMs: number; timestamp?: string } }
  | { type: "TOOL_ERROR"; payload: { taskId: string; toolCallId: string; error: string; timestamp?: string } }

  // Evolution
  | { type: "GENERATION_COMPLETE"; payload: GenerationMetrics }
  | { type: "SUBMISSION_RESULT"; payload: LeaderboardSubmission }
  | { type: "CONVERGENCE_DETECTED"; payload: { generation: number; reason: string } }

  // Memory
  | { type: "MEMORY_OPERATION"; payload: { key: string; operation: string; scope?: string; category?: string } }
  | { type: "CHECKPOINT_CREATED"; payload: { name: string; phase: MissionPhase; timestamp?: string } }

  // Errors
  | { type: "ERROR_OCCURRED"; payload: ErrorEvent }
  | { type: "RECOVERY_COMPLETE"; payload: { errorId: string; success: boolean; resolution?: string } }

  // UI actions
  | { type: "TOGGLE_PHASE"; payload: MissionPhase }
  | { type: "TOGGLE_TASK"; payload: string }
  | { type: "TOGGLE_TOOL_CALL"; payload: string }
  | { type: "SET_ACTIVE_TAB"; payload: AgentKUIState["activeTab"] }
  | { type: "TOGGLE_THINKING_BLOCKS" }
  | { type: "SET_EVOLUTION_CHART_TYPE"; payload: AgentKUIState["evolutionChartType"] }
  | { type: "SET_EVOLUTION_CHART_RANGE"; payload: AgentKUIState["evolutionChartRange"] }
  | { type: "HYDRATE_UI"; payload: Partial<AgentKUIState> };

const AgentKStateContext = createContext<AgentKContextValue | null>(null);

function cloneMissionState(state: MissionState): MissionState {
  if (typeof structuredClone === "function") {
    return structuredClone(state);
  }
  return JSON.parse(JSON.stringify(state)) as MissionState;
}

function createInitialMissionState(): MissionState {
  return {
    missionId: "",
    status: "idle",
    overallProgress: 0,
    phases: [],
    memory: { entries: [], checkpoints: [], totalSizeBytes: 0 },
    errors: [],
  };
}

function createInitialUIState(): AgentKUIState {
  return {
    expandedPhases: new Set(),
    expandedTasks: new Set(),
    expandedToolCalls: new Set(),
    activeTab: "plan",
    showThinkingBlocks: true,
    showToolDetails: true,
    logFilter: [],
    toolFilter: [],
    evolutionChartType: "fitness",
    evolutionChartRange: "all",
  };
}

function findTaskById(
  mission: MissionState,
  taskId: string
): { phase: PhasePlan; task: PlannedTask } | null {
  for (const phase of mission.phases) {
    const task = phase.tasks.find((currentTask) => currentTask.id === taskId);
    if (task) return { phase, task };
  }
  return null;
}

function agentKReducer(state: AgentKState, action: AgentKAction): AgentKState {
  switch (action.type) {
    case "SET_STATE":
      return { ...state, mission: action.payload };

    case "APPLY_PATCH":
      return { ...state, mission: applyPatches(state.mission, action.payload) };

    case "RESET_STATE":
      return { mission: createInitialMissionState(), ui: createInitialUIState() };

    case "PHASE_START": {
      const mission = cloneMissionState(state.mission);
      const phase = mission.phases.find((p) => p.phase === action.payload.phase);
      if (phase) {
        phase.status = "in_progress";
        phase.startedAt = phase.startedAt ?? action.payload.timestamp ?? new Date().toISOString();
        mission.currentPhase = phase.phase;
      }
      return { ...state, mission };
    }

    case "PHASE_COMPLETE": {
      const mission = cloneMissionState(state.mission);
      const phase = mission.phases.find((p) => p.phase === action.payload.phase);
      if (phase) {
        phase.status = action.payload.success ? "completed" : "failed";
        phase.progress = 100;
        phase.completedAt = action.payload.timestamp ?? new Date().toISOString();
      }
      mission.currentPhase = undefined;
      return { ...state, mission };
    }

    case "TASK_START": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      if (lookup) {
        lookup.task.status = "in_progress";
        lookup.task.startedAt = action.payload.timestamp ?? new Date().toISOString();
        mission.currentTaskId = lookup.task.id;
        mission.currentPhase = lookup.phase.phase;
      }
      return { ...state, mission };
    }

    case "TASK_PROGRESS": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      if (lookup) {
        lookup.task.progress = action.payload.progress;
        if (lookup.task.status === "pending") {
          lookup.task.status = "in_progress";
        }
      }
      return { ...state, mission };
    }

    case "TASK_COMPLETE": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      if (lookup) {
        lookup.task.status = action.payload.success ? "completed" : "failed";
        lookup.task.completedAt = action.payload.timestamp ?? new Date().toISOString();
        if (action.payload.result !== undefined) {
          lookup.task.result = action.payload.result;
        }
      }
      mission.currentTaskId = undefined;
      return { ...state, mission };
    }

    case "TOOL_START": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      if (lookup) {
        lookup.task.toolCalls = lookup.task.toolCalls ?? [];
        const existing = lookup.task.toolCalls.find((t) => t.id === action.payload.toolCall.id);
        if (existing) {
          Object.assign(existing, action.payload.toolCall);
        } else {
          lookup.task.toolCalls.push(action.payload.toolCall);
        }
      }
      return { ...state, mission };
    }

    case "TOOL_THINKING": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      const toolCall = lookup?.task.toolCalls.find(
        (call) => call.id === action.payload.toolCallId
      );
      if (toolCall) {
        toolCall.thinking = action.payload.thinking;
      }
      return { ...state, mission };
    }

    case "TOOL_RESULT": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      const toolCall = lookup?.task.toolCalls.find(
        (call) => call.id === action.payload.toolCallId
      );
      if (toolCall) {
        toolCall.result = action.payload.result;
        toolCall.durationMs = action.payload.durationMs;
        toolCall.completedAt = action.payload.timestamp ?? new Date().toISOString();
        toolCall.error = undefined;
      }
      return { ...state, mission };
    }

    case "TOOL_ERROR": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      const toolCall = lookup?.task.toolCalls.find(
        (call) => call.id === action.payload.toolCallId
      );
      if (toolCall) {
        toolCall.error = action.payload.error;
        toolCall.completedAt = action.payload.timestamp ?? new Date().toISOString();
      }
      return { ...state, mission };
    }

    case "GENERATION_COMPLETE": {
      const mission = cloneMissionState(state.mission);
      mission.evolution = mission.evolution ?? {
        currentGeneration: 0,
        maxGenerations: 0,
        populationSize: 0,
        generationHistory: [],
        convergenceDetected: false,
        leaderboardSubmissions: [],
      };
      mission.evolution.currentGeneration = action.payload.generation;
      mission.evolution.generationHistory = [
        ...(mission.evolution.generationHistory ?? []),
        action.payload,
      ];
      mission.evolution.populationSize = action.payload.populationSize;
      mission.evolution.bestSolution = mission.evolution.bestSolution ?? undefined;
      return { ...state, mission };
    }

    case "SUBMISSION_RESULT": {
      const mission = cloneMissionState(state.mission);
      mission.evolution = mission.evolution ?? {
        currentGeneration: 0,
        maxGenerations: 0,
        populationSize: 0,
        generationHistory: [],
        convergenceDetected: false,
        leaderboardSubmissions: [],
      };
      mission.evolution.leaderboardSubmissions = [
        ...(mission.evolution.leaderboardSubmissions ?? []),
        action.payload,
      ];
      return { ...state, mission };
    }

    case "CONVERGENCE_DETECTED": {
      const mission = cloneMissionState(state.mission);
      mission.evolution = mission.evolution ?? {
        currentGeneration: 0,
        maxGenerations: 0,
        populationSize: 0,
        generationHistory: [],
        convergenceDetected: false,
        leaderboardSubmissions: [],
      };
      mission.evolution.convergenceDetected = true;
      mission.evolution.convergenceReason = action.payload.reason;
      return { ...state, mission };
    }

    case "MEMORY_OPERATION": {
      const mission = cloneMissionState(state.mission);
      const now = new Date().toISOString();
      const entry = mission.memory.entries.find((item) => item.key === action.payload.key);

      if (entry) {
        entry.accessedAt = now;
        entry.accessCount = (entry.accessCount ?? 0) + 1;
      } else {
        mission.memory.entries.push({
          key: action.payload.key,
          scope: (action.payload.scope as any) ?? "session",
          category: action.payload.category ?? "general",
          valuePreview: "",
          createdAt: now,
          accessedAt: now,
          accessCount: 1,
          sizeBytes: 0,
        });
      }
      return { ...state, mission };
    }

    case "CHECKPOINT_CREATED": {
      const mission = cloneMissionState(state.mission);
      mission.memory.checkpoints = [
        ...mission.memory.checkpoints,
        {
          name: action.payload.name,
          phase: action.payload.phase,
          timestamp: action.payload.timestamp ?? new Date().toISOString(),
          stateSnapshot: "",
        },
      ];
      return { ...state, mission };
    }

    case "ERROR_OCCURRED": {
      const mission = cloneMissionState(state.mission);
      mission.errors = [...mission.errors, action.payload];
      return { ...state, mission };
    }

    case "RECOVERY_COMPLETE": {
      const mission = cloneMissionState(state.mission);
      const error = mission.errors.find((err) => err.id === action.payload.errorId);
      if (error) {
        error.resolved = action.payload.success;
        error.resolution = action.payload.resolution;
      }
      return { ...state, mission };
    }

    case "TOGGLE_PHASE": {
      const expandedPhases = new Set(state.ui.expandedPhases);
      if (expandedPhases.has(action.payload)) {
        expandedPhases.delete(action.payload);
      } else {
        expandedPhases.add(action.payload);
      }
      return { ...state, ui: { ...state.ui, expandedPhases } };
    }

    case "TOGGLE_TASK": {
      const expandedTasks = new Set(state.ui.expandedTasks);
      if (expandedTasks.has(action.payload)) {
        expandedTasks.delete(action.payload);
      } else {
        expandedTasks.add(action.payload);
      }
      return { ...state, ui: { ...state.ui, expandedTasks } };
    }

    case "TOGGLE_TOOL_CALL": {
      const expandedToolCalls = new Set(state.ui.expandedToolCalls);
      if (expandedToolCalls.has(action.payload)) {
        expandedToolCalls.delete(action.payload);
      } else {
        expandedToolCalls.add(action.payload);
      }
      return { ...state, ui: { ...state.ui, expandedToolCalls } };
    }

    case "SET_ACTIVE_TAB":
      return { ...state, ui: { ...state.ui, activeTab: action.payload } };

    case "TOGGLE_THINKING_BLOCKS":
      return {
        ...state,
        ui: { ...state.ui, showThinkingBlocks: !state.ui.showThinkingBlocks },
      };

    case "SET_EVOLUTION_CHART_TYPE":
      return {
        ...state,
        ui: { ...state.ui, evolutionChartType: action.payload },
      };

    case "SET_EVOLUTION_CHART_RANGE":
      return {
        ...state,
        ui: { ...state.ui, evolutionChartRange: action.payload },
      };

    case "HYDRATE_UI": {
      const expandedPhases = new Set(
        (action.payload.expandedPhases as MissionPhase[] | undefined) ?? []
      );
      const expandedTasks = new Set((action.payload.expandedTasks as string[] | undefined) ?? []);
      const expandedToolCalls = new Set(
        (action.payload.expandedToolCalls as string[] | undefined) ?? []
      );
      return {
        ...state,
        ui: {
          ...state.ui,
          ...action.payload,
          expandedPhases,
          expandedTasks,
          expandedToolCalls,
        },
      };
    }

    default:
      return state;
  }
}

interface AgentKContextValue {
  state: AgentKState;
  dispatch: React.Dispatch<AgentKAction>;
  currentPhase?: PhasePlan;
  currentTask?: PlannedTask;
  isEvolutionActive: boolean;
  hasErrors: boolean;
  togglePhase: (phase: MissionPhase) => void;
  toggleTask: (taskId: string) => void;
  toggleToolCall: (toolCallId: string) => void;
  setActiveTab: (tab: AgentKUIState["activeTab"]) => void;
  appendToolThinking: (taskId: string, toolCallId: string, thinking: string) => void;
}

export function AgentKStateProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(agentKReducer, {
    mission: createInitialMissionState(),
    ui: createInitialUIState(),
  });

  // Hydrate UI state from localStorage
  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem(UI_STATE_STORAGE_KEY);
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw) as Partial<AgentKUIState>;
      dispatch({ type: "HYDRATE_UI", payload: parsed });
    } catch {
      // ignore malformed UI state
    }
  }, [dispatch]);

  // Persist UI state
  useEffect(() => {
    if (typeof window === "undefined") return;
    const serializable = {
      ...state.ui,
      expandedPhases: Array.from(state.ui.expandedPhases),
      expandedTasks: Array.from(state.ui.expandedTasks),
      expandedToolCalls: Array.from(state.ui.expandedToolCalls),
    };
    window.localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(serializable));
  }, [state.ui]);

  const currentPhase = useMemo(
    () => state.mission.phases.find((phase) => phase.status === "in_progress"),
    [state.mission.phases]
  );

  const currentTask = useMemo(() => {
    if (state.mission.currentTaskId) {
      return findTaskById(state.mission, state.mission.currentTaskId)?.task;
    }
    return currentPhase?.tasks.find((task) => task.status === "in_progress");
  }, [currentPhase, state.mission.currentTaskId, state.mission.phases]);

  const isEvolutionActive = useMemo(
    () =>
      state.mission.currentPhase === "evolution" &&
      state.mission.status === "executing",
    [state.mission.currentPhase, state.mission.status]
  );

  const hasErrors = useMemo(
    () => state.mission.errors.some((error) => !error.resolved),
    [state.mission.errors]
  );

  const togglePhase = useCallback(
    (phase: MissionPhase) => dispatch({ type: "TOGGLE_PHASE", payload: phase }),
    []
  );

  const toggleTask = useCallback(
    (taskId: string) => dispatch({ type: "TOGGLE_TASK", payload: taskId }),
    []
  );

  const toggleToolCall = useCallback(
    (toolCallId: string) => dispatch({ type: "TOGGLE_TOOL_CALL", payload: toolCallId }),
    []
  );

  const setActiveTab = useCallback(
    (tab: AgentKUIState["activeTab"]) => dispatch({ type: "SET_ACTIVE_TAB", payload: tab }),
    []
  );

  const appendToolThinking = useCallback(
    (taskId: string, toolCallId: string, thinking: string) =>
      dispatch({ type: "TOOL_THINKING", payload: { taskId, toolCallId, thinking } }),
    []
  );

  const value = useMemo(
    () => ({
      state,
      dispatch,
      currentPhase,
      currentTask,
      isEvolutionActive,
      hasErrors,
      togglePhase,
      toggleTask,
      toggleToolCall,
      setActiveTab,
      appendToolThinking,
    }),
    [
      state,
      currentPhase,
      currentTask,
      isEvolutionActive,
      hasErrors,
      togglePhase,
      toggleTask,
      toggleToolCall,
      setActiveTab,
      appendToolThinking,
    ]
  );

  return (
    <AgentKStateContext.Provider value={value}>
      {children}
    </AgentKStateContext.Provider>
  );
}

export function useAgentKState() {
  const context = useContext(AgentKStateContext);
  if (!context) {
    throw new Error("useAgentKState must be used within AgentKStateProvider");
  }
  return context;
}
