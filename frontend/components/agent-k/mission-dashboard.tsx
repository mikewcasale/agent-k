"use client";

import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  BrainCircuit,
  LineChart,
  ListChecks,
  Pause,
  Play,
  Square,
} from "lucide-react";
import type React from "react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { buttonVariants } from "@/components/ui/button";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { cn } from "@/lib/utils";
import {
  buildCompetitionRulesUrl,
  isRulesAcceptanceError,
} from "@/lib/utils/kaggle";
import { formatRelativeTime } from "@/lib/utils/time";
import { EvolutionView } from "./evolution-view";
import { LogsView } from "./logs-view";
import { MemoryView } from "./memory-view";
import { PlanView } from "./plan-view";
import { ResearchView } from "./research-view";

const tabs: Array<{
  key: "plan" | "evolution" | "research" | "memory" | "logs";
  label: string;
  icon: React.ElementType;
}> = [
  { key: "plan", label: "Plan", icon: ListChecks },
  { key: "evolution", label: "Evolution", icon: LineChart },
  { key: "research", label: "Research", icon: BookOpen },
  { key: "memory", label: "Memory", icon: BrainCircuit },
  { key: "logs", label: "Logs", icon: Activity },
];

export function MissionDashboard() {
  const { state, setActiveTab } = useAgentKState();
  const mission = state.mission;
  const [showRulesDialog, setShowRulesDialog] = useState(false);
  const activeTab = state.ui.activeTab;

  const rulesUrl = useMemo(
    () => buildCompetitionRulesUrl(mission.competition, mission.competitionId),
    [mission.competition, mission.competitionId]
  );
  const rulesError = useMemo(() => {
    const messages: string[] = [];
    for (const error of mission.errors) {
      messages.push(error.message);
    }
    for (const phase of mission.phases) {
      for (const task of phase.tasks) {
        for (const toolCall of task.toolCalls ?? []) {
          if (toolCall.error) {
            messages.push(toolCall.error);
          }
        }
      }
    }
    return messages.find((message) => isRulesAcceptanceError(message));
  }, [mission.errors, mission.phases]);

  // Avoid rendering until mission initialized
  if (!mission.phases.length && !mission.evolution && !mission.competition) {
    return null;
  }

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="my-6 overflow-hidden rounded-3xl border border-zinc-200 bg-gradient-to-br from-white via-zinc-50 to-blue-50 shadow-lg dark:border-zinc-800 dark:from-zinc-950 dark:via-zinc-900 dark:to-blue-950/30"
      initial={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
    >
      <div className="border-zinc-200 border-b px-4 py-4 md:px-6 dark:border-zinc-800">
        <Header />
        {rulesError && (
          <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-amber-900 text-sm">
            <div className="flex items-center gap-2 font-semibold">
              <AlertTriangle className="size-4 text-amber-600" />
              Kaggle rules acceptance required
            </div>
            <p className="mt-1 text-amber-800">
              Kaggle blocks API downloads until you accept the competition rules
              in your active browser session.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                className="rounded-full bg-amber-600 px-4 py-1.5 font-semibold text-white text-xs hover:bg-amber-700"
                onClick={() => setShowRulesDialog(true)}
                type="button"
              >
                Review Kaggle Rules
              </button>
              <a
                className={cn(buttonVariants({ variant: "outline" }))}
                href={rulesUrl}
                rel="noreferrer"
                target="_blank"
              >
                Open rules page
              </a>
            </div>
          </div>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {tabs.map((tab) => (
            <button
              className={cn(
                "inline-flex items-center gap-2 rounded-full px-3 py-1.5 font-medium text-sm transition-colors",
                activeTab === tab.key
                  ? "bg-blue-600 text-white shadow-sm"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
              )}
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              type="button"
            >
              <tab.icon className="size-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="p-4 md:p-6">
        {activeTab === "plan" && <PlanView />}
        {activeTab === "evolution" && <EvolutionView />}
        {activeTab === "research" && <ResearchView />}
        {activeTab === "memory" && <MemoryView />}
        {activeTab === "logs" && <LogsView />}
      </div>

      <AlertDialog onOpenChange={setShowRulesDialog} open={showRulesDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Accept Kaggle competition rules</AlertDialogTitle>
            <AlertDialogDescription>
              Kaggle requires a one-time manual acceptance in your browser
              session before API access works. Open the rules page, confirm you
              are signed in to the correct Kaggle account, accept the rules,
              then return here to retry the mission.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-amber-900 text-xs">
            Rules page:{" "}
            <a
              className="font-semibold text-blue-700 hover:text-blue-800"
              href={rulesUrl}
              rel="noreferrer"
              target="_blank"
            >
              {rulesUrl}
            </a>
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>Not now</AlertDialogCancel>
            <a
              className={cn(buttonVariants({ variant: "outline" }))}
              href={rulesUrl}
              rel="noreferrer"
              target="_blank"
            >
              Open rules page
            </a>
            <AlertDialogAction onClick={() => setShowRulesDialog(false)}>
              I've accepted
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </motion.div>
  );
}

function Header() {
  const { state, dispatch, isEvolutionActive, hasErrors } = useAgentKState();
  const mission = state.mission;
  const [pendingAction, setPendingAction] = useState<
    "pause" | "resume" | "abort" | null
  >(null);
  const [showAbortDialog, setShowAbortDialog] = useState(false);
  const title = mission.competition?.title || "Agent K Mission";
  const subtitle =
    mission.competition?.id || mission.missionId || "Tracking multi-agent flow";
  const overallProgress = mission.overallProgress ?? 0;
  const isTerminal =
    mission.status === "completed" || mission.status === "failed";
  const isPaused = mission.status === "paused" || mission.status === "pausing";

  async function requestControl(action: "pause" | "resume" | "abort") {
    if (!mission.missionId || pendingAction) {
      return;
    }
    setPendingAction(action);
    try {
      const response = await fetch(
        `/api/mission/${encodeURIComponent(mission.missionId)}/${action}`,
        { method: "POST" }
      );
      const payload = (await response.json()) as {
        error?: string | { message?: string };
      };
      if (!response.ok) {
        const message =
          typeof payload.error === "string"
            ? payload.error
            : (payload.error?.message ?? "Mission control request failed.");
        throw new Error(message);
      }

      const status =
        action === "pause"
          ? "pausing"
          : action === "abort"
            ? "cancelling"
            : "executing";
      dispatch({ type: "SET_MISSION_STATUS", payload: status });
      toast.success(
        action === "pause"
          ? "Pause requested."
          : action === "resume"
            ? "Resume requested."
            : "Mission cancellation requested."
      );
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Mission control request failed."
      );
    } finally {
      setPendingAction(null);
      setShowAbortDialog(false);
    }
  }

  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <h2 className="font-semibold text-xl text-zinc-900 dark:text-white">
            {title}
          </h2>
          {isEvolutionActive && (
            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-emerald-700 text-xs dark:bg-emerald-900/40 dark:text-emerald-200">
              Evolution live
            </span>
          )}
          {hasErrors && (
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-amber-700 text-xs dark:bg-amber-900/40 dark:text-amber-200">
              <AlertTriangle className="size-3" />
              Issues
            </span>
          )}
        </div>
        <p className="text-sm text-zinc-600 dark:text-zinc-300">{subtitle}</p>
        <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500">
          <span>Status: {mission.status}</span>
          {mission.estimatedCompletionAt && (
            <span>ETA {formatRelativeTime(mission.estimatedCompletionAt)}</span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        {mission.missionId && !isTerminal && (
          <div className="flex items-center gap-1 rounded-full bg-zinc-100 p-1 dark:bg-zinc-800">
            <button
              aria-label={isPaused ? "Resume mission" : "Pause mission"}
              className="inline-flex items-center gap-1 rounded-full px-3 py-1.5 font-medium text-xs hover:bg-white disabled:opacity-50 dark:hover:bg-zinc-700"
              disabled={
                pendingAction !== null || mission.status === "cancelling"
              }
              onClick={() => requestControl(isPaused ? "resume" : "pause")}
              type="button"
            >
              {isPaused ? (
                <Play className="size-3.5" />
              ) : (
                <Pause className="size-3.5" />
              )}
              {pendingAction === "pause"
                ? "Pausing"
                : pendingAction === "resume"
                  ? "Resuming"
                  : isPaused
                    ? "Resume"
                    : "Pause"}
            </button>
            <button
              aria-label="Cancel mission"
              className="inline-flex items-center gap-1 rounded-full px-3 py-1.5 font-medium text-red-700 text-xs hover:bg-red-50 disabled:opacity-50 dark:text-red-300 dark:hover:bg-red-950/40"
              disabled={
                pendingAction !== null || mission.status === "cancelling"
              }
              onClick={() => setShowAbortDialog(true)}
              type="button"
            >
              <Square className="size-3.5" />
              {pendingAction === "abort" ? "Cancelling" : "Cancel"}
            </button>
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Progress</span>
          <div className="h-2 w-32 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-violet-500"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
          <span className="font-semibold text-sm text-zinc-900 dark:text-white">
            {Math.round(overallProgress)}%
          </span>
        </div>
      </div>

      <AlertDialog onOpenChange={setShowAbortDialog} open={showAbortDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel this mission?</AlertDialogTitle>
            <AlertDialogDescription>
              The worker will stop at the next durable boundary. This cannot be
              resumed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep running</AlertDialogCancel>
            <AlertDialogAction onClick={() => requestControl("abort")}>
              Cancel mission
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
