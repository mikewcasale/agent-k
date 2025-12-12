"use client";

import { Book, ExternalLink, Info, ListChecks } from "lucide-react";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { cn } from "@/lib/utils";
import { formatDateTime } from "@/lib/utils/time";

export function ResearchView() {
  const { state } = useAgentKState();
  const competition = state.mission.competition;
  const research = state.mission.research;

  if (!competition && !research) {
    return (
      <div className="rounded-lg border border-dashed border-zinc-300 bg-white px-4 py-6 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
        Research context will appear here once available.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {competition && <CompetitionInfo />}
      {research?.leaderboardAnalysis && <LeaderboardAnalysis />}
      {research?.approaches?.length ? <ApproachesList /> : null}
      {research?.papers?.length ? <PapersList /> : null}
      {research?.strategyRecommendations?.length ? <Recommendations /> : null}
    </div>
  );
}

function CompetitionInfo() {
  const { state } = useAgentKState();
  const competition = state.mission.competition;
  if (!competition) return null;

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-xs uppercase tracking-wide text-zinc-500">Competition</p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-white">{competition.title}</h3>
        <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700 dark:bg-blue-900/40 dark:text-blue-200">
          {competition.competitionType}
        </span>
        <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">
          Metric {competition.metric}
        </span>
      </div>
      {competition.description && (
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">{competition.description}</p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-zinc-500">
        <span>Deadline {formatDateTime(competition.deadline)}</span>
        <span>Max team size {competition.maxTeamSize}</span>
        <span>Daily submissions {competition.maxDailySubmissions}</span>
        {competition.prizePool && <span>Prize ${competition.prizePool.toLocaleString()}</span>}
      </div>
      {competition.tags?.length ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {competition.tags.map((tag) => (
            <span
              key={tag}
              className="rounded-full bg-zinc-100 px-2 py-0.5 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"
            >
              {tag}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function LeaderboardAnalysis() {
  const { state } = useAgentKState();
  const analysis = state.mission.research?.leaderboardAnalysis;
  if (!analysis) return null;

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-2">
        <Info className="size-4 text-blue-500" />
        <p className="text-sm font-semibold text-zinc-900 dark:text-white">Leaderboard analysis</p>
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-4">
        <Tile label="Top score" value={analysis.topScore.toFixed(4)} color="emerald" />
        <Tile label="Median score" value={analysis.medianScore.toFixed(4)} color="blue" />
        <Tile label="Target score" value={analysis.targetScore.toFixed(4)} color="amber" />
        <Tile label="Teams" value={analysis.totalTeams} color="indigo" />
      </div>
      {analysis.improvementOpportunities?.length ? (
        <div className="mt-3 text-sm text-zinc-600 dark:text-zinc-300">
          <p className="mb-1 text-xs uppercase tracking-wide text-zinc-500">Opportunities</p>
          <ul className="list-disc space-y-1 pl-5">
            {analysis.improvementOpportunities.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function ApproachesList() {
  const { state } = useAgentKState();
  const approaches = state.mission.research?.approaches ?? [];
  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-2">
        <ListChecks className="size-4 text-emerald-500" />
        <p className="text-sm font-semibold text-zinc-900 dark:text-white">Candidate approaches</p>
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        {approaches.map((approach) => (
          <div
            key={approach.name}
            className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-800/60"
          >
            <div className="flex items-center justify-between">
              <p className="font-medium text-zinc-900 dark:text-white">{approach.name}</p>
              <span className="text-xs text-zinc-500">Complexity {approach.complexity}</span>
            </div>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-300">{approach.description}</p>
            <p className="mt-2 text-xs text-zinc-500">
              Estimated score {approach.estimatedScore.toFixed(4)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function PapersList() {
  const { state } = useAgentKState();
  const papers = state.mission.research?.papers ?? [];

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-2">
        <Book className="size-4 text-indigo-500" />
        <p className="text-sm font-semibold text-zinc-900 dark:text-white">Papers & references</p>
      </div>
      <div className="mt-3 space-y-3">
        {papers.map((paper) => (
          <div
            key={paper.url}
            className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-sm dark:border-zinc-800 dark:bg-zinc-800/60"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="font-medium text-zinc-900 dark:text-white">{paper.title}</p>
              <a
                className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-500"
                href={paper.url}
                rel="noreferrer"
                target="_blank"
              >
                Open <ExternalLink className="size-3" />
              </a>
            </div>
            <p className="mt-1 text-xs text-zinc-500">Relevance {paper.relevance}/10</p>
            {paper.keyInsights?.length ? (
              <ul className="mt-1 list-disc space-y-1 pl-4 text-xs text-zinc-500">
                {paper.keyInsights.map((insight) => (
                  <li key={insight}>{insight}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function Recommendations() {
  const { state } = useAgentKState();
  const recs = state.mission.research?.strategyRecommendations ?? [];
  if (!recs.length) return null;

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-sm font-semibold text-zinc-900 dark:text-white">Strategy recommendations</p>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zinc-600 dark:text-zinc-300">
        {recs.map((rec) => (
          <li key={rec}>{rec}</li>
        ))}
      </ul>
    </div>
  );
}

function Tile({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color: "emerald" | "blue" | "amber" | "indigo";
}) {
  const palette = {
    emerald: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200",
    blue: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-200",
    amber: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200",
    indigo: "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200",
  }[color];
  return (
    <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-800/60">
      <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
      <p className={cn("mt-1 text-lg font-semibold", palette)}>{value}</p>
    </div>
  );
}
