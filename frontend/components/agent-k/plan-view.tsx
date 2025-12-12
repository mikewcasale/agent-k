"use client";

import { motion } from "framer-motion";
import { Compass, Loader2 } from "lucide-react";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { cn } from "@/lib/utils";
import { formatRelativeTime } from "@/lib/utils/time";
import { PhaseCard } from "./phase-card";

export function PlanView() {
  const { state } = useAgentKState();
  const phases = state.mission.phases;

  if (!phases.length) {
    return (
      <div className="flex items-center gap-3 rounded-lg border border-dashed border-zinc-300 bg-white px-4 py-6 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
        <Compass className="size-4 text-blue-500" />
        Waiting for mission plan...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="relative space-y-6">
        <div className="absolute left-6 top-0 h-full w-0.5 bg-gradient-to-b from-blue-500/60 via-zinc-200 to-transparent dark:via-zinc-800" />
        {phases.map((phase) => (
          <div className="relative pl-14" key={phase.phase}>
            <div className="absolute left-5 top-6 size-3 rounded-full border-2 border-white bg-blue-500 shadow ring-2 ring-white dark:border-zinc-900 dark:ring-zinc-900" />
            <PhaseCard phase={phase} />
          </div>
        ))}
      </div>

      <CurrentTaskFocus />
    </div>
  );
}

function CurrentTaskFocus() {
  const { currentPhase, currentTask } = useAgentKState();

  if (!currentPhase || !currentTask) return null;

  return (
    <motion.div
      layout
      className="rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 to-white shadow-sm dark:border-indigo-900/60 dark:from-indigo-950/30 dark:to-zinc-900"
    >
      <div className="flex items-start gap-3 p-4">
        <div className="mt-0.5 flex size-9 items-center justify-center rounded-xl bg-indigo-500 text-white shadow">
          <Loader2 className="size-4 animate-spin" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-xs uppercase tracking-wide text-indigo-600 dark:text-indigo-300">
            In progress • {currentPhase.displayName}
          </p>
          <p className="text-base font-semibold text-zinc-900 dark:text-white">
            {currentTask.name}
          </p>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">
            {currentTask.description}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-zinc-500 dark:text-zinc-400">
            <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-indigo-700 dark:bg-indigo-900/60 dark:text-indigo-200">
              {currentTask.agent}
            </span>
            <span>
              Progress{" "}
              <strong className="text-indigo-700 dark:text-indigo-200">
                {Math.round(currentTask.progress)}%
              </strong>
            </span>
            {currentTask.startedAt && (
              <span>Started {formatRelativeTime(currentTask.startedAt)}</span>
            )}
          </div>
          <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-indigo-100 dark:bg-indigo-900/50">
            <div
              className="h-full bg-indigo-500"
              style={{ width: `${Math.max(5, currentTask.progress)}%` }}
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
