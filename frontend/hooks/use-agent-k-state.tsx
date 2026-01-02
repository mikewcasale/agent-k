"use client";

import type React from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from "react";
import type {
  AgentKPatchOp,
  AgentKState,
  AgentKUIState,
  CompetitionInfo,
  ErrorEvent,
  GenerationMetrics,
  LeaderboardSubmission,
  MissionPhase,
  MissionState,
  PhasePlan,
  PlannedTask,
  ToolCall,
} from "@/lib/types/agent-k";
import {
  bestResultFromMission,
  upsertBestResult,
} from "@/lib/utils/best-results";
import { applyPatches } from "@/lib/utils/json-patch";

const UI_STATE_STORAGE_KEY = "agentk-ui-state";

const MISSION_PHASE_ORDER: MissionPhase[] = [
  "discovery",
  "research",
  "prototype",
  "evolution",
  "submission",
];

const PHASE_DISPLAY_NAMES: Record<MissionPhase, string> = {
  discovery: "Discovery",
  research: "Research",
  prototype: "Prototype",
  evolution: "Evolution",
  submission: "Submission",
};

const PHASE_TIMEOUTS_MS: Record<MissionPhase, number> = {
  discovery: 5 * 60 * 1000,
  research: 10 * 60 * 1000,
  prototype: 15 * 60 * 1000,
  evolution: 120 * 60 * 1000,
  submission: 2 * 60 * 1000,
};

const PHASE_OBJECTIVES: Record<MissionPhase, string[]> = {
  discovery: [
    "Find competitions matching criteria",
    "Validate competition accessibility",
    "Rank by fit score",
  ],
  research: [
    "Analyze leaderboard and score distribution",
    "Review relevant papers and techniques",
    "Perform exploratory data analysis",
    "Synthesize strategy recommendations",
  ],
  prototype: [
    "Generate baseline solution code",
    "Validate solution structure",
    "Establish baseline score",
  ],
  evolution: [
    "Generate improved solutions",
    "Track fitness improvements",
    "Detect convergence",
  ],
  submission: [
    "Prepare final submission",
    "Submit to leaderboard",
    "Record final score",
  ],
};

type TaskTemplate = {
  id: string;
  name: string;
  description: string;
  agent: PlannedTask["agent"];
  toolsRequired: PlannedTask["toolsRequired"];
  estimatedDurationMs?: number;
  priority?: PlannedTask["priority"];
};

const PHASE_TASKS: Record<MissionPhase, TaskTemplate[]> = {
  discovery: [
    {
      id: "discovery-search",
      name: "Search competitions",
      description: "Find competitions that match the mission criteria.",
      agent: "lobbyist",
      toolsRequired: ["kaggle_mcp", "web_search"],
    },
    {
      id: "discovery-score",
      name: "Score fit",
      description: "Rank candidates by fit score and accessibility.",
      agent: "lobbyist",
      toolsRequired: [],
    },
  ],
  research: [
    {
      id: "research-leaderboard",
      name: "Analyze leaderboard",
      description: "Review leaderboard trends and metric behavior.",
      agent: "scientist",
      toolsRequired: ["kaggle_mcp"],
    },
    {
      id: "research-techniques",
      name: "Survey techniques",
      description: "Identify high-impact approaches and baselines.",
      agent: "scientist",
      toolsRequired: ["web_search", "browser"],
    },
  ],
  prototype: [
    {
      id: "prototype-baseline",
      name: "Build baseline",
      description: "Generate and validate a baseline solution.",
      agent: "evolver",
      toolsRequired: ["code_executor"],
    },
  ],
  evolution: [
    {
      id: "evolution-optimize",
      name: "Optimize solutions",
      description: "Iterate on candidate solutions and track fitness.",
      agent: "evolver",
      toolsRequired: ["code_executor", "memory"],
    },
  ],
  submission: [
    {
      id: "submission-submit",
      name: "Submit results",
      description: "Package and submit the best solution.",
      agent: "lycurgus",
      toolsRequired: ["kaggle_mcp"],
    },
  ],
};

function buildTask(template: TaskTemplate): PlannedTask {
  return {
    id: template.id,
    name: template.name,
    description: template.description,
    agent: template.agent,
    toolsRequired: template.toolsRequired,
    estimatedDurationMs: template.estimatedDurationMs ?? 30_000,
    priority: template.priority ?? "medium",
    dependencies: [],
    status: "pending",
    progress: 0,
    toolCalls: [],
  };
}

function buildPhasePlan(phase: MissionPhase, objectives?: string[]): PhasePlan {
  return {
    phase,
    displayName: PHASE_DISPLAY_NAMES[phase],
    objectives: objectives ?? PHASE_OBJECTIVES[phase],
    successCriteria: [],
    tasks: (PHASE_TASKS[phase] ?? []).map(buildTask),
    timeoutMs: PHASE_TIMEOUTS_MS[phase],
    status: "pending",
    progress: 0,
  };
}

function ensurePhasePlan(
  phases: PhasePlan[],
  phase: MissionPhase,
  objectives?: string[]
): PhasePlan {
  let plan = phases.find((item) => item.phase === phase);
  if (!plan) {
    plan = buildPhasePlan(phase, objectives);
    phases.push(plan);
    phases.sort(
      (a, b) =>
        MISSION_PHASE_ORDER.indexOf(a.phase) -
        MISSION_PHASE_ORDER.indexOf(b.phase)
    );
  } else if (objectives?.length) {
    plan.objectives = objectives;
  }
  return plan;
}

function createDefaultPhasePlans(): PhasePlan[] {
  return MISSION_PHASE_ORDER.map((phase) => buildPhasePlan(phase));
}

function calculateOverallProgress(phases: PhasePlan[]): number {
  if (!phases.length) {
    return 0;
  }
  const completed = phases.filter(
    (phase) => phase.status === "completed"
  ).length;
  const inProgress = phases.some((phase) => phase.status === "in_progress");
  const progress = ((completed + (inProgress ? 0.5 : 0)) / phases.length) * 100;
  return Math.min(100, Math.max(0, Math.round(progress)));
}

type AgentKAction =
  // State management
  | { type: "SET_STATE"; payload: MissionState }
  | {
      type: "SET_COMPETITION";
      payload: { competition: CompetitionInfo; missionId?: string };
    }
  | { type: "APPLY_PATCH"; payload: AgentKPatchOp[] }
  | { type: "RESET_STATE" }

  // Phase lifecycle
  | {
      type: "PHASE_START";
      payload: {
        phase: MissionPhase;
        objectives: string[];
        timestamp?: string;
      };
    }
  | {
      type: "PHASE_COMPLETE";
      payload: {
        phase: MissionPhase;
        success: boolean;
        durationMs?: number;
        timestamp?: string;
      };
    }

  // Task lifecycle
  | {
      type: "TASK_START";
      payload: {
        taskId: string;
        phase: MissionPhase;
        name: string;
        timestamp?: string;
      };
    }
  | {
      type: "TASK_PROGRESS";
      payload: { taskId: string; progress: number; message?: string };
    }
  | {
      type: "TASK_COMPLETE";
      payload: {
        taskId: string;
        success: boolean;
        result?: unknown;
        timestamp?: string;
      };
    }

  // Tool usage
  | { type: "TOOL_START"; payload: { taskId: string; toolCall: ToolCall } }
  | {
      type: "TOOL_THINKING";
      payload: { taskId: string; toolCallId: string; thinking: string };
    }
  | {
      type: "TOOL_RESULT";
      payload: {
        taskId: string;
        toolCallId: string;
        result: unknown;
        durationMs: number;
        timestamp?: string;
      };
    }
  | {
      type: "TOOL_ERROR";
      payload: {
        taskId: string;
        toolCallId: string;
        error: string;
        timestamp?: string;
      };
    }

  // Evolution
  | { type: "GENERATION_COMPLETE"; payload: GenerationMetrics }
  | { type: "SUBMISSION_RESULT"; payload: LeaderboardSubmission }
  | {
      type: "CONVERGENCE_DETECTED";
      payload: { generation: number; reason: string };
    }

  // Memory
  | {
      type: "MEMORY_OPERATION";
      payload: {
        key: string;
        operation: string;
        scope?: string;
        category?: string;
      };
    }
  | {
      type: "CHECKPOINT_CREATED";
      payload: { name: string; phase: MissionPhase; timestamp?: string };
    }

  // Errors
  | { type: "ERROR_OCCURRED"; payload: ErrorEvent }
  | {
      type: "RECOVERY_COMPLETE";
      payload: { errorId: string; success: boolean; resolution?: string };
    }

  // Mission lifecycle
  | {
      type: "MISSION_COMPLETE";
      payload: {
        success: boolean;
        finalRank?: number;
        finalScore?: number;
        errorMessage?: string;
        totalSubmissions?: number;
        evolutionGenerations?: number;
        durationMs?: number;
        phasesCompleted?: MissionPhase[];
      };
    }

  // UI actions
  | { type: "TOGGLE_PHASE"; payload: MissionPhase }
  | { type: "TOGGLE_TASK"; payload: string }
  | { type: "TOGGLE_TOOL_CALL"; payload: string }
  | { type: "SET_ACTIVE_TAB"; payload: AgentKUIState["activeTab"] }
  | { type: "TOGGLE_THINKING_BLOCKS" }
  | {
      type: "SET_EVOLUTION_CHART_TYPE";
      payload: AgentKUIState["evolutionChartType"];
    }
  | {
      type: "SET_EVOLUTION_CHART_RANGE";
      payload: AgentKUIState["evolutionChartRange"];
    }
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
    if (task) {
      return { phase, task };
    }
  }
  return null;
}

function agentKReducer(state: AgentKState, action: AgentKAction): AgentKState {
  switch (action.type) {
    case "SET_STATE":
      return { ...state, mission: action.payload };

    case "SET_COMPETITION": {
      const mission = cloneMissionState(state.mission);
      mission.competition = action.payload.competition;
      mission.competitionId = action.payload.competition.id;
      if (action.payload.missionId) {
        mission.missionId = action.payload.missionId;
      }
      if (mission.status === "idle") {
        mission.status = "planning";
      }
      if (!mission.phases.length) {
        mission.phases = createDefaultPhasePlans();
      }
      mission.overallProgress = calculateOverallProgress(mission.phases);
      return { ...state, mission };
    }

    case "APPLY_PATCH":
      return { ...state, mission: applyPatches(state.mission, action.payload) };

    case "RESET_STATE":
      return {
        mission: createInitialMissionState(),
        ui: createInitialUIState(),
      };

    case "PHASE_START": {
      const mission = cloneMissionState(state.mission);
      const phase = ensurePhasePlan(
        mission.phases,
        action.payload.phase,
        action.payload.objectives
      );
      phase.status = "in_progress";
      phase.progress = Math.max(phase.progress, 5);
      phase.startedAt =
        phase.startedAt ?? action.payload.timestamp ?? new Date().toISOString();
      for (const task of phase.tasks) {
        if (task.status === "pending") {
          task.status = "in_progress";
          task.startedAt = task.startedAt ?? phase.startedAt;
          task.progress = Math.max(task.progress, 5);
        }
      }
      mission.currentPhase = phase.phase;
      mission.status = "executing";
      mission.overallProgress = calculateOverallProgress(mission.phases);
      return { ...state, mission };
    }

    case "PHASE_COMPLETE": {
      const mission = cloneMissionState(state.mission);
      const phase = ensurePhasePlan(mission.phases, action.payload.phase);
      phase.status = action.payload.success ? "completed" : "failed";
      phase.progress = 100;
      phase.completedAt = action.payload.timestamp ?? new Date().toISOString();
      for (const task of phase.tasks) {
        task.status = action.payload.success ? "completed" : "failed";
        task.progress = action.payload.success ? 100 : task.progress;
        task.completedAt = phase.completedAt;
      }
      if (!action.payload.success) {
        mission.status = "failed";
      }
      mission.currentPhase = undefined;
      mission.overallProgress = calculateOverallProgress(mission.phases);
      return { ...state, mission };
    }

    case "TASK_START": {
      const mission = cloneMissionState(state.mission);
      const lookup = findTaskById(mission, action.payload.taskId);
      if (lookup) {
        lookup.task.status = "in_progress";
        lookup.task.startedAt =
          action.payload.timestamp ?? new Date().toISOString();
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
        lookup.task.completedAt =
          action.payload.timestamp ?? new Date().toISOString();
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
        const existing = lookup.task.toolCalls.find(
          (t) => t.id === action.payload.toolCall.id
        );
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
        toolCall.completedAt =
          action.payload.timestamp ?? new Date().toISOString();
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
        toolCall.completedAt =
          action.payload.timestamp ?? new Date().toISOString();
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
      mission.evolution.bestSolution =
        mission.evolution.bestSolution ?? undefined;
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
      const entry = mission.memory.entries.find(
        (item) => item.key === action.payload.key
      );

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
      const error = mission.errors.find(
        (err) => err.id === action.payload.errorId
      );
      if (error) {
        error.resolved = action.payload.success;
        error.resolution = action.payload.resolution;
      }
      return { ...state, mission };
    }

    case "MISSION_COMPLETE": {
      const mission = cloneMissionState(state.mission);
      mission.status = action.payload.success ? "completed" : "failed";
      mission.overallProgress = 100;
      mission.currentPhase = undefined;
      mission.currentTaskId = undefined;
      mission.result = {
        success: action.payload.success,
        finalRank: action.payload.finalRank,
        finalScore: action.payload.finalScore,
        totalSubmissions: action.payload.totalSubmissions ?? 0,
        evolutionGenerations: action.payload.evolutionGenerations ?? 0,
        durationMs: action.payload.durationMs ?? 0,
        phasesCompleted: action.payload.phasesCompleted ?? [],
        errorMessage: action.payload.errorMessage,
      };
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
      const expandedTasks = new Set(
        (action.payload.expandedTasks as string[] | undefined) ?? []
      );
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

type AgentKContextValue = {
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
  appendToolThinking: (
    taskId: string,
    toolCallId: string,
    thinking: string
  ) => void;
};

export function AgentKStateProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [state, dispatch] = useReducer(agentKReducer, {
    mission: createInitialMissionState(),
    ui: createInitialUIState(),
  });
  const lastBestResultKey = useRef<string | null>(null);

  // Hydrate UI state from localStorage
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const raw = window.localStorage.getItem(UI_STATE_STORAGE_KEY);
    if (!raw) {
      return;
    }
    try {
      const parsed = JSON.parse(raw) as Partial<AgentKUIState>;
      dispatch({ type: "HYDRATE_UI", payload: parsed });
    } catch {
      // ignore malformed UI state
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist UI state
  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const serializable = {
      ...state.ui,
      expandedPhases: Array.from(state.ui.expandedPhases),
      expandedTasks: Array.from(state.ui.expandedTasks),
      expandedToolCalls: Array.from(state.ui.expandedToolCalls),
    };
    window.localStorage.setItem(
      UI_STATE_STORAGE_KEY,
      JSON.stringify(serializable)
    );
  }, [state.ui]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const bestResult = bestResultFromMission(state.mission);
    if (!bestResult) {
      return;
    }
    const nextKey = [
      bestResult.competitionId,
      bestResult.submissionId ?? "",
      bestResult.rank ?? "",
      bestResult.recordedAt,
    ].join("|");
    if (nextKey === lastBestResultKey.current) {
      return;
    }
    lastBestResultKey.current = nextKey;
    upsertBestResult(bestResult);
  }, [state.mission]);

  const currentPhase = useMemo(
    () => state.mission.phases.find((phase) => phase.status === "in_progress"),
    [state.mission.phases]
  );

  const currentTask = useMemo(() => {
    if (state.mission.currentTaskId) {
      return findTaskById(state.mission, state.mission.currentTaskId)?.task;
    }
    return currentPhase?.tasks.find((task) => task.status === "in_progress");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPhase, state.mission.currentTaskId, state.mission]);

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
    (toolCallId: string) =>
      dispatch({ type: "TOGGLE_TOOL_CALL", payload: toolCallId }),
    []
  );

  const setActiveTab = useCallback(
    (tab: AgentKUIState["activeTab"]) =>
      dispatch({ type: "SET_ACTIVE_TAB", payload: tab }),
    []
  );

  const appendToolThinking = useCallback(
    (taskId: string, toolCallId: string, thinking: string) =>
      dispatch({
        type: "TOOL_THINKING",
        payload: { taskId, toolCallId, thinking },
      }),
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
