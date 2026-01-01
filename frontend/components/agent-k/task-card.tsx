"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  CheckCircle2,
  Circle,
  Clock,
  Loader2,
  Sparkles,
} from "lucide-react";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import type { PlannedTask } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils/time";
import { ToolCallCard } from "./tool-call-card";

const agentColors: Record<PlannedTask["agent"], string> = {
  lobbyist:
    "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-200",
  scientist: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-200",
  evolver:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200",
  lycurgus:
    "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200",
};

type TaskCardProps = {
  task: PlannedTask;
};

export function TaskCard({ task }: TaskCardProps) {
  const { state, toggleTask } = useAgentKState();
  const isExpanded = state.ui.expandedTasks.has(task.id);

  const statusIcon = {
    pending: Circle,
    in_progress: Loader2,
    completed: CheckCircle2,
    failed: AlertCircle,
    blocked: AlertCircle,
    skipped: Circle,
  }[task.status];

  const StatusIcon = statusIcon;

  return (
    <motion.div
      className="rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
      layout
    >
      <button
        className="flex w-full items-start gap-3 p-4 text-left hover:bg-zinc-50 dark:hover:bg-zinc-800/60"
        onClick={() => toggleTask(task.id)}
        type="button"
      >
        <div className="mt-0.5">
          <StatusIcon
            className={cn(
              "size-4",
              task.status === "completed"
                ? "text-emerald-500"
                : task.status === "in_progress"
                  ? "animate-spin text-blue-500"
                  : task.status === "failed"
                    ? "text-red-500"
                    : "text-zinc-400"
            )}
          />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-sm text-zinc-800 dark:text-zinc-100">
              {task.name}
            </span>
            <span
              className={cn(
                "rounded-full px-2 py-0.5 font-medium text-[11px] uppercase tracking-wide",
                agentColors[task.agent]
              )}
            >
              {task.agent}
            </span>
            <span className="text-[11px] text-zinc-500 uppercase">
              {task.priority} priority
            </span>
          </div>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">
            {task.description}
          </p>

          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <Clock className="size-3" />
              est {formatDuration(task.estimatedDurationMs)}
            </span>
            {task.actualDurationMs !== undefined && (
              <span className="flex items-center gap-1">
                <Sparkles className="size-3" />
                actual {formatDuration(task.actualDurationMs)}
              </span>
            )}
          </div>

          <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
            <motion.div
              animate={{ width: `${Math.max(5, task.progress)}%` }}
              className={cn(
                "h-full rounded-full",
                task.status === "completed"
                  ? "bg-emerald-500"
                  : task.status === "failed"
                    ? "bg-red-500"
                    : "bg-blue-500"
              )}
              initial={{ width: 0 }}
              transition={{ duration: 0.4 }}
            />
          </div>
        </div>

        <Chevron />
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            animate={{ height: "auto" }}
            className="overflow-hidden border-zinc-200 border-t dark:border-zinc-800"
            exit={{ height: 0 }}
            initial={{ height: 0 }}
          >
            <div className="space-y-3 p-4">
              {task.toolCalls?.length ? (
                task.toolCalls.map((toolCall) => (
                  <ToolCallCard
                    key={toolCall.id}
                    taskId={task.id}
                    toolCall={toolCall}
                  />
                ))
              ) : (
                <p className="text-xs text-zinc-500">No tool calls yet.</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function Chevron() {
  return (
    <svg
      className="mt-1 size-4 text-zinc-400"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        d="M9 5l7 7-7 7"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
      />
    </svg>
  );
}
