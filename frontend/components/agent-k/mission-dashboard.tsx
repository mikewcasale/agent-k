"use client";

import type React from "react";
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
} from "lucide-react";
import { EvolutionView } from "./evolution-view";
import { LogsView } from "./logs-view";
import { MemoryView } from "./memory-view";
import { PlanView } from "./plan-view";
import { ResearchView } from "./research-view";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { cn } from "@/lib/utils";
import { formatRelativeTime } from "@/lib/utils/time";

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
  const { state, setActiveTab, hasErrors } = useAgentKState();
  const mission = state.mission;

  // Avoid rendering until mission initialized
  if (!mission.phases.length && !mission.evolution && !mission.competition) {
    return null;
  }

  const activeTab = state.ui.activeTab;
  const overallProgress = mission.overallProgress ?? 0;

  return (
    <motion.div
      className="my-6 overflow-hidden rounded-3xl border border-zinc-200 bg-gradient-to-br from-white via-zinc-50 to-blue-50 shadow-lg dark:border-zinc-800 dark:from-zinc-950 dark:via-zinc-900 dark:to-blue-950/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="border-b border-zinc-200 px-4 py-4 dark:border-zinc-800 md:px-6">
        <Header />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              className={cn(
                "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                activeTab === tab.key
                  ? "bg-blue-600 text-white shadow-sm"
                  : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
              )}
              onClick={() => setActiveTab(tab.key)}
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
    </motion.div>
  );
}

function Header() {
  const { state, isEvolutionActive, hasErrors } = useAgentKState();
  const mission = state.mission;
  const title = mission.competition?.title || "Agent K Mission";
  const subtitle = mission.competition?.id || mission.missionId || "Tracking multi-agent flow";
  const overallProgress = mission.overallProgress ?? 0;

  return (
    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <h2 className="text-xl font-semibold text-zinc-900 dark:text-white">{title}</h2>
          {isEvolutionActive && (
            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">
              Evolution live
            </span>
          )}
          {hasErrors && (
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-900/40 dark:text-amber-200">
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
        <div className="hidden items-center gap-2 rounded-full bg-zinc-100 px-3 py-1.5 text-sm text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 md:flex">
          <Pause className="size-4" />
          <Play className="size-4" />
          <span className="text-xs text-zinc-500">Controls coming soon</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Progress</span>
          <div className="h-2 w-32 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-700">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-violet-500"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
          <span className="text-sm font-semibold text-zinc-900 dark:text-white">
            {Math.round(overallProgress)}%
          </span>
        </div>
      </div>
    </div>
  );
}
