"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Circle,
  Dna,
  FlaskConical,
  Hammer,
  Loader2,
  Search,
  Send,
} from "lucide-react";
import type React from "react";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import type { MissionPhase, PhasePlan } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils/time";
import { TaskCard } from "./task-card";

const phaseIcons: Record<MissionPhase, React.ElementType> = {
  discovery: Search,
  research: FlaskConical,
  prototype: Hammer,
  evolution: Dna,
  submission: Send,
};

const phaseColors: Record<MissionPhase, string> = {
  discovery: "from-violet-500 to-purple-600",
  research: "from-blue-500 to-cyan-600",
  prototype: "from-amber-500 to-orange-600",
  evolution: "from-emerald-500 to-green-600",
  submission: "from-pink-500 to-rose-600",
};

type PhaseCardProps = {
  phase: PhasePlan;
};

export function PhaseCard({ phase }: PhaseCardProps) {
  const { state, togglePhase } = useAgentKState();
  const isExpanded = state.ui.expandedPhases.has(phase.phase);

  const Icon = phaseIcons[phase.phase];
  const completedTasks = phase.tasks.filter(
    (t) => t.status === "completed"
  ).length;
  const totalTasks = phase.tasks.length;

  const statusIcon = {
    pending: Circle,
    in_progress: Loader2,
    completed: CheckCircle2,
    failed: AlertCircle,
    blocked: AlertCircle,
    skipped: Circle,
  }[phase.status];

  const StatusIcon = statusIcon;

  return (
    <motion.div
      className={cn(
        "overflow-hidden rounded-2xl border shadow-sm transition-colors",
        phase.status === "in_progress"
          ? "border-blue-300 dark:border-blue-700"
          : "border-zinc-200 dark:border-zinc-800"
      )}
      layout
    >
      <button
        className="flex w-full items-center gap-4 p-4 text-left hover:bg-zinc-50 dark:hover:bg-zinc-800/60"
        onClick={() => togglePhase(phase.phase)}
        type="button"
      >
        <div
          className={cn(
            "flex size-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br text-white",
            phaseColors[phase.phase]
          )}
        >
          <Icon className="size-6" />
        </div>

        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold text-lg text-zinc-900 dark:text-white">
              {phase.displayName}
            </span>
            <span className="text-xs text-zinc-500 uppercase tracking-wide">
              {phase.tasks.length} tasks
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <StatusIcon
                className={cn(
                  "size-4",
                  phase.status === "completed"
                    ? "text-emerald-500"
                    : phase.status === "in_progress"
                      ? "animate-spin text-blue-500"
                      : phase.status === "failed"
                        ? "text-red-500"
                        : "text-zinc-400"
                )}
              />
              {phase.status.replaceAll("_", " ")}
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle2 className="size-3" />
              {completedTasks}/{totalTasks} done
            </span>
            <span className="flex items-center gap-1">
              <AlertCircle className="size-3" />
              {formatDuration(phase.timeoutMs)} timeout
            </span>
          </div>

          <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
            <motion.div
              animate={{ width: `${Math.max(5, phase.progress)}%` }}
              className="h-full bg-gradient-to-r from-blue-500 to-violet-500"
              initial={{ width: 0 }}
              transition={{ duration: 0.4 }}
            />
          </div>
        </div>

        <ChevronDown
          className={cn(
            "size-5 text-zinc-400 transition-transform",
            isExpanded && "rotate-180"
          )}
        />
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            animate={{ height: "auto" }}
            className="overflow-hidden border-zinc-200 border-t bg-white dark:border-zinc-800 dark:bg-zinc-900"
            exit={{ height: 0 }}
            initial={{ height: 0 }}
          >
            <div className="flex flex-col gap-4 p-4">
              {phase.objectives?.length ? (
                <div className="rounded-lg bg-zinc-50 p-3 text-sm text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
                  <p className="mb-2 text-xs text-zinc-500 uppercase tracking-wide">
                    Objectives
                  </p>
                  <ul className="space-y-1">
                    {phase.objectives.map((objective) => (
                      <li className="flex items-start gap-2" key={objective}>
                        <CheckCircle2 className="mt-0.5 size-4 text-emerald-500" />
                        <span>{objective}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div className="space-y-3">
                {phase.tasks.map((task) => (
                  <TaskCard key={task.id} task={task} />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
