"use client";

import type React from "react";
import { Flame, Medal, Trophy, Zap } from "lucide-react";
import { CodeBlock } from "@/components/elements/code-block";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { cn } from "@/lib/utils";
import { formatDateTime } from "@/lib/utils/time";
import { FitnessChart } from "./fitness-chart";

export function EvolutionView() {
  const { state } = useAgentKState();
  const evolution = state.mission.evolution;

  if (!evolution) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-300 bg-white px-4 py-6 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
        Evolution not started yet.
      </div>
    );
  }

  const latestGeneration = evolution.generationHistory.at(-1);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <StatCard
          icon={Flame}
          label="Generation"
          value={evolution.currentGeneration}
          hint={`Max ${evolution.maxGenerations || "?"}`}
        />
        <StatCard
          icon={Trophy}
          label="Best Fitness"
          value={
            latestGeneration?.bestFitness !== undefined
              ? latestGeneration.bestFitness.toFixed(4)
              : "—"
          }
          hint={latestGeneration ? `Mean ${latestGeneration.meanFitness.toFixed(4)}` : undefined}
        />
        <StatCard
          icon={Zap}
          label="Population"
          value={evolution.populationSize}
          hint={evolution.convergenceDetected ? evolution.convergenceReason : "Active"}
        />
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-zinc-500">Fitness over time</p>
            <p className="text-sm text-zinc-700 dark:text-zinc-200">
              Tracking best/mean/worst across generations
            </p>
          </div>
          {state.mission.research?.leaderboardAnalysis?.targetScore && (
            <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700 dark:bg-amber-900/40 dark:text-amber-200">
              Target {state.mission.research.leaderboardAnalysis.targetScore.toFixed(4)}
            </span>
          )}
        </div>
        <FitnessChart />
      </div>

      <MutationBreakdown />

      <LeaderboardTracker />

      <BestSolutionPreview />
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  hint,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-blue-500 text-white">
          <Icon className="size-5" />
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
          <p className="text-lg font-semibold text-zinc-900 dark:text-white">{value}</p>
          {hint && <p className="text-xs text-zinc-500">{hint}</p>}
        </div>
      </div>
    </div>
  );
}

function MutationBreakdown() {
  const { state } = useAgentKState();
  const latest = state.mission.evolution?.generationHistory.at(-1);

  if (!latest) return null;

  const mutations = latest.mutations;
  const entries = [
    { label: "Point", value: mutations.point, color: "bg-blue-500" },
    { label: "Structural", value: mutations.structural, color: "bg-violet-500" },
    { label: "Hyperparam", value: mutations.hyperparameter, color: "bg-amber-500" },
    { label: "Crossover", value: mutations.crossover, color: "bg-emerald-500" },
  ];

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-3 flex items-center justify-between">
        <p className="text-sm font-semibold text-zinc-900 dark:text-white">Mutation breakdown</p>
        <span className="text-xs text-zinc-500">Generation {latest.generation}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-4">
        {entries.map((entry) => (
          <div
            key={entry.label}
            className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-sm dark:border-zinc-800 dark:bg-zinc-800/60"
          >
            <div className="flex items-center justify-between">
              <span className="text-zinc-700 dark:text-zinc-200">{entry.label}</span>
              <span className={cn("h-2 w-2 rounded-full", entry.color)} />
            </div>
            <p className="mt-2 text-xl font-semibold text-zinc-900 dark:text-white">
              {entry.value}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function LeaderboardTracker() {
  const { state } = useAgentKState();
  const submissions = state.mission.evolution?.leaderboardSubmissions ?? [];

  if (!submissions.length) return null;

  const sorted = [...submissions].sort(
    (a, b) => new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime()
  );

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-3 flex items-center gap-2">
        <Medal className="size-4 text-amber-500" />
        <p className="text-sm font-semibold text-zinc-900 dark:text-white">Leaderboard tracking</p>
      </div>
      <div className="divide-y divide-zinc-200 text-sm dark:divide-zinc-800">
        {sorted.map((submission) => (
          <div
            className="flex flex-wrap items-center justify-between gap-2 py-3"
            key={submission.submissionId}
          >
            <div>
              <p className="font-medium text-zinc-900 dark:text-white">
                Gen {submission.generation}
              </p>
              <p className="text-xs text-zinc-500">
                Submitted {formatDateTime(submission.submittedAt)}
              </p>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <Badge color="emerald">
                CV {submission.cvScore.toFixed(4)}
              </Badge>
              {submission.publicScore !== undefined && (
                <Badge color="blue">Public {submission.publicScore.toFixed(4)}</Badge>
              )}
              {submission.rank && submission.totalTeams && (
                <Badge color="amber">
                  Rank {submission.rank}/{submission.totalTeams}
                </Badge>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function BestSolutionPreview() {
  const { state } = useAgentKState();
  const best = state.mission.evolution?.bestSolution;

  if (!best) return null;

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-zinc-500">Best solution</p>
          <p className="text-sm text-zinc-700 dark:text-zinc-200">
            Fitness {best.fitness.toFixed(4)} at generation {best.generation}
          </p>
        </div>
      </div>
      <CodeBlock code={best.code} language="python" />
    </div>
  );
}

function Badge({ children, color }: { children: React.ReactNode; color: "emerald" | "blue" | "amber" }) {
  const palette = {
    emerald: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200",
    blue: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-200",
    amber: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200",
  }[color];
  return (
    <span className={cn("rounded-full px-2 py-0.5 font-medium", palette)}>{children}</span>
  );
}
