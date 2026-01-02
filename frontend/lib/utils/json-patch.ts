import type {
  AgentKPatchOp,
  MissionState,
  PhasePlan,
  PlannedTask,
  ToolCall,
} from "@/lib/types/agent-k";

function cloneMissionState(state: MissionState): MissionState {
  if (typeof structuredClone === "function") {
    return structuredClone(state);
  }
  return JSON.parse(JSON.stringify(state)) as MissionState;
}

function getPhase(phases: PhasePlan[], index: number): PhasePlan | undefined {
  if (Number.isNaN(index) || index < 0) {
    return;
  }
  return phases[index];
}

function getTask(tasks: PlannedTask[], index: number): PlannedTask | undefined {
  if (Number.isNaN(index) || index < 0) {
    return;
  }
  return tasks[index];
}

export function applyPatches(
  state: MissionState,
  operations: AgentKPatchOp[]
): MissionState {
  const next = cloneMissionState(state);

  for (const op of operations) {
    const pathParts = op.path.split("/").filter(Boolean);

    if (!pathParts.length) {
      continue;
    }

    // Phase-level operations
    if (pathParts[0] === "phases") {
      const phaseIndex = Number.parseInt(pathParts[1], 10);
      const phase = getPhase(next.phases, phaseIndex);
      if (!phase) {
        continue;
      }

      // Replace operations on phase
      if (op.op === "replace" && pathParts.length === 3) {
        if (pathParts[2] === "status" && typeof op.value === "string") {
          phase.status = op.value as PhasePlan["status"];
          continue;
        }
        if (pathParts[2] === "progress" && typeof op.value === "number") {
          phase.progress = op.value;
          continue;
        }
      }

      // Task-level operations
      if (pathParts[2] === "tasks") {
        const taskIndex = Number.parseInt(pathParts[3], 10);
        const task = getTask(phase.tasks, taskIndex);
        if (!task) {
          continue;
        }

        if (
          op.op === "replace" &&
          pathParts.length === 5 &&
          pathParts[4] === "status" &&
          typeof op.value === "string"
        ) {
          task.status = op.value as PlannedTask["status"];
          continue;
        }

        // Tool call paths
        if (pathParts[4] === "toolCalls") {
          if (op.op === "add" && pathParts[5] === "-" && op.value) {
            task.toolCalls = [...task.toolCalls, op.value as ToolCall];
            continue;
          }

          const callIndex = Number.parseInt(pathParts[5], 10);
          const toolCall = getTask(phase.tasks, taskIndex)?.toolCalls?.[
            callIndex
          ];
          if (!toolCall) {
            continue;
          }

          if (op.op === "replace" && pathParts.length === 7) {
            if (pathParts[6] === "result") {
              toolCall.result = op.value as ToolCall["result"];
              continue;
            }
            if (pathParts[6] === "thinking" && typeof op.value === "string") {
              toolCall.thinking = op.value;
              continue;
            }
          }
        }
      }
    }

    // Evolution updates
    if (pathParts[0] === "evolution") {
      next.evolution = next.evolution ?? {
        currentGeneration: 0,
        maxGenerations: 0,
        populationSize: 0,
        improvementCount: 0,
        minImprovementsRequired: 0,
        generationHistory: [],
        convergenceDetected: false,
        leaderboardSubmissions: [],
      };

      if (op.op === "add" && op.path === "/evolution/generationHistory/-") {
        next.evolution.generationHistory = [
          ...next.evolution.generationHistory,
          op.value as any,
        ];
        continue;
      }

      if (
        op.op === "replace" &&
        op.path === "/evolution/currentGeneration" &&
        typeof op.value === "number"
      ) {
        next.evolution.currentGeneration = op.value;
        continue;
      }

      if (op.op === "replace" && op.path === "/evolution/bestSolution") {
        next.evolution.bestSolution = op.value as any;
        continue;
      }

      if (
        op.op === "add" &&
        op.path === "/evolution/leaderboardSubmissions/-"
      ) {
        next.evolution.leaderboardSubmissions = [
          ...(next.evolution.leaderboardSubmissions ?? []),
          op.value as any,
        ];
        continue;
      }
    }

    // Memory updates
    if (pathParts[0] === "memory") {
      if (op.op === "add" && op.path === "/memory/entries/-") {
        next.memory.entries = [...next.memory.entries, op.value as any];
        continue;
      }
      if (op.op === "add" && op.path === "/memory/checkpoints/-") {
        next.memory.checkpoints = [...next.memory.checkpoints, op.value as any];
        continue;
      }
    }

    // Error events
    if (pathParts[0] === "errors") {
      if (op.op === "add" && op.path === "/errors/-") {
        next.errors = [...next.errors, op.value as any];
        continue;
      }

      const errorIndex = Number.parseInt(pathParts[1], 10);
      if (
        op.op === "replace" &&
        pathParts[2] === "resolved" &&
        typeof op.value === "boolean" &&
        next.errors[errorIndex]
      ) {
        next.errors[errorIndex].resolved = op.value;
      }
    }
  }

  return next;
}
