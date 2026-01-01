// =============================================================================
// Core Enums
// =============================================================================

export type MissionPhase =
  | "discovery"
  | "research"
  | "prototype"
  | "evolution"
  | "submission";

export type TaskStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "blocked"
  | "skipped";

export type TaskPriority = "critical" | "high" | "medium" | "low";

export type ToolType =
  | "web_search"
  | "kaggle_mcp"
  | "code_executor"
  | "memory"
  | "browser";

export type MemoryOperation = "store" | "retrieve" | "checkpoint" | "restore";

export type ErrorCategory = "transient" | "recoverable" | "fatal";

export type RecoveryStrategy = "retry" | "fallback" | "skip" | "replan" | "abort";

// =============================================================================
// Tool Events
// =============================================================================

export interface ToolCall {
  id: string;
  type: ToolType;
  operation: string;
  params?: Record<string, unknown>;
  thinking?: string;
  result?: unknown;
  error?: string;
  startedAt: string;
  completedAt?: string;
  durationMs?: number;
}

export interface WebSearchCall extends ToolCall {
  type: "web_search";
  query: string;
  resultCount?: number;
  results?: Array<{
    title: string;
    url: string;
    snippet: string;
  }>;
}

export interface KaggleMCPCall extends ToolCall {
  type: "kaggle_mcp";
  operation:
    | "competitions.list"
    | "competitions.leaderboard"
    | "competitions.submit"
    | "competitions.data"
    | "kernels.list";
  competitionId?: string;
}

export interface CodeExecutorCall extends ToolCall {
  type: "code_executor";
  code: string;
  language: "python";
  stdout?: string;
  stderr?: string;
  executionTimeMs?: number;
  memoryUsageMb?: number;
}

export interface MemoryCall extends ToolCall {
  type: "memory";
  operation: MemoryOperation;
  key: string;
  scope?: "session" | "persistent" | "global";
  valuePreview?: string;
}

// =============================================================================
// Task and Phase Models
// =============================================================================

export interface PlannedTask {
  id: string;
  name: string;
  description: string;
  agent: "lobbyist" | "scientist" | "evolver" | "lycurgus";
  toolsRequired: ToolType[];
  estimatedDurationMs: number;
  actualDurationMs?: number;
  priority: TaskPriority;
  dependencies: string[];
  status: TaskStatus;
  progress: number; // 0-100
  result?: unknown;
  error?: string;
  toolCalls: ToolCall[];
  startedAt?: string;
  completedAt?: string;
}

export interface PhasePlan {
  phase: MissionPhase;
  displayName: string;
  objectives: string[];
  successCriteria: string[];
  tasks: PlannedTask[];
  timeoutMs: number;
  fallbackStrategy?: string;
  status: TaskStatus;
  progress: number;
  startedAt?: string;
  completedAt?: string;
}

// =============================================================================
// Evolution-Specific State
// =============================================================================

export interface GenerationMetrics {
  generation: number;
  bestFitness: number;
  meanFitness: number;
  worstFitness: number;
  populationSize: number;
  mutations: {
    point: number;
    structural: number;
    hyperparameter: number;
    crossover: number;
  };
  timestamp: string;
}

export interface LeaderboardSubmission {
  submissionId: string;
  generation: number;
  cvScore: number;
  publicScore?: number;
  rank?: number;
  totalTeams?: number;
  percentile?: number;
  submittedAt: string;
}

export interface EvolutionState {
  currentGeneration: number;
  maxGenerations: number;
  populationSize: number;
  bestSolution?: {
    code: string;
    fitness: number;
    generation: number;
  };
  generationHistory: GenerationMetrics[];
  convergenceDetected: boolean;
  convergenceReason?: string;
  leaderboardSubmissions: LeaderboardSubmission[];
}

// =============================================================================
// Competition and Research State
// =============================================================================

export type CompetitionSelectionMode = "search" | "direct";

export type CompetitionPaidStatus = "any" | "paid" | "free";

export interface CompetitionSearchCriteria {
  paidStatus: CompetitionPaidStatus;
  domains: string[];
  competitionTypes: string[];
  minPrize: number | null;
  minDaysRemaining: number;
}

export interface CompetitionInfo {
  id: string;
  title: string;
  description?: string;
  competitionType: string;
  metric: string;
  metricDirection: "maximize" | "minimize";
  deadline: string;
  prizePool?: number;
  maxTeamSize: number;
  maxDailySubmissions: number;
  tags: string[];
  url?: string;
}

export interface LeaderboardAnalysis {
  topScore: number;
  medianScore: number;
  targetScore: number;
  targetPercentile: number;
  totalTeams: number;
  scoreDistribution: Array<{ score: number; count: number }>;
  commonApproaches: string[];
  improvementOpportunities: string[];
}

export interface ResearchFindings {
  leaderboardAnalysis?: LeaderboardAnalysis;
  papers: Array<{
    title: string;
    url: string;
    relevance: number;
    keyInsights: string[];
  }>;
  approaches: Array<{
    name: string;
    description: string;
    estimatedScore: number;
    complexity: "low" | "medium" | "high";
  }>;
  edaResults?: {
    classDistribution?: Record<string, number>;
    featureCount: number;
    sampleCount: number;
    dataQualityIssues: string[];
  };
  strategyRecommendations: string[];
}

// =============================================================================
// Error and Recovery State
// =============================================================================

export interface ErrorEvent {
  id: string;
  timestamp: string;
  category: ErrorCategory;
  errorType: string;
  message: string;
  context: string;
  taskId?: string;
  phase?: MissionPhase;
  recoveryStrategy: RecoveryStrategy;
  recoveryAttempts: number;
  resolved: boolean;
  resolution?: string;
}

// =============================================================================
// Memory State
// =============================================================================

export interface MemoryEntry {
  key: string;
  scope: "session" | "persistent" | "global";
  category: string;
  valuePreview: string;
  createdAt: string;
  accessedAt: string;
  accessCount: number;
  sizeBytes: number;
}

export interface MemoryState {
  entries: MemoryEntry[];
  checkpoints: Array<{
    name: string;
    phase: MissionPhase;
    timestamp: string;
    stateSnapshot: string;
  }>;
  totalSizeBytes: number;
}

// =============================================================================
// Mission State (Root)
// =============================================================================

export interface MissionState {
  // Identity
  missionId: string;
  competitionId?: string;

  // Status
  status: "idle" | "planning" | "executing" | "paused" | "completed" | "failed";
  currentPhase?: MissionPhase;
  currentTaskId?: string;

  // Progress
  overallProgress: number; // 0-100
  startedAt?: string;
  estimatedCompletionAt?: string;

  // Planning
  phases: PhasePlan[];

  // Competition context
  competition?: CompetitionInfo;
  research?: ResearchFindings;

  // Evolution state (only during evolution phase)
  evolution?: EvolutionState;

  // Memory
  memory: MemoryState;

  // Errors
  errors: ErrorEvent[];

  // Final results
  result?: {
    success: boolean;
    finalRank?: number;
    finalScore?: number;
    totalSubmissions: number;
    evolutionGenerations: number;
    durationMs: number;
    phasesCompleted: MissionPhase[];
    errorMessage?: string;
  };
}

// =============================================================================
// UI View State
// =============================================================================

export interface AgentKUIState {
  expandedPhases: Set<MissionPhase>;
  expandedTasks: Set<string>;
  expandedToolCalls: Set<string>;

  activeTab: "plan" | "evolution" | "research" | "memory" | "logs";
  showThinkingBlocks: boolean;
  showToolDetails: boolean;

  logFilter: TaskStatus[];
  toolFilter: ToolType[];

  evolutionChartType: "fitness" | "mutations" | "submissions";
  evolutionChartRange: "all" | "last50" | "last10";
}

export interface AgentKState {
  mission: MissionState;
  ui: AgentKUIState;
}

// =============================================================================
// Extended Patch Operations
// =============================================================================

export type AgentKPatchOp =
  // Phase updates
  | { op: "replace"; path: `/phases/${number}/status`; value: TaskStatus }
  | { op: "replace"; path: `/phases/${number}/progress`; value: number }
  | {
      op: "replace";
      path: `/phases/${number}/tasks/${number}/status`;
      value: TaskStatus;
    }

  // Tool call streaming
  | {
      op: "add";
      path: `/phases/${number}/tasks/${number}/toolCalls/-`;
      value: ToolCall;
    }
  | {
      op: "replace";
      path: `/phases/${number}/tasks/${number}/toolCalls/${number}/result`;
      value: unknown;
    }
  | {
      op: "replace";
      path: `/phases/${number}/tasks/${number}/toolCalls/${number}/thinking`;
      value: string;
    }

  // Evolution updates
  | { op: "add"; path: "/evolution/generationHistory/-"; value: GenerationMetrics }
  | { op: "replace"; path: "/evolution/currentGeneration"; value: number }
  | {
      op: "replace";
      path: "/evolution/bestSolution";
      value: EvolutionState["bestSolution"];
    }
  | {
      op: "add";
      path: "/evolution/leaderboardSubmissions/-";
      value: LeaderboardSubmission;
    }

  // Memory operations
  | { op: "add"; path: "/memory/entries/-"; value: MemoryEntry }
  | {
      op: "add";
      path: "/memory/checkpoints/-";
      value: MemoryState["checkpoints"][0];
    }

  // Error events
  | { op: "add"; path: "/errors/-"; value: ErrorEvent }
  | { op: "replace"; path: `/errors/${number}/resolved`; value: boolean };
