"use client";

import { useMemo, useState } from "react";
import { Award, Calendar, Flag, Tags } from "lucide-react";
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
import type { CompetitionInfo } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";

type CompetitionPreviewProps = {
  competition: CompetitionInfo;
  matchCount?: number | null;
  isStarting?: boolean;
  onConfirm: () => void;
  onBack: () => void;
};

function formatCompetitionType(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatMetric(value: string) {
  return value.replace(/_/g, " ").toUpperCase();
}

function buildRulesUrl(competition: CompetitionInfo) {
  if (competition.url) {
    try {
      const parsed = new URL(competition.url);
      const cleanedPath = parsed.pathname.replace(/\/rules\/?$/, "").replace(/\/$/, "");
      return `${parsed.origin}${cleanedPath}/rules`;
    } catch {
      // Fall back to the canonical rules path.
    }
  }
  return `https://www.kaggle.com/competitions/${competition.id}/rules`;
}

export function CompetitionPreview({
  competition,
  matchCount,
  isStarting = false,
  onConfirm,
  onBack,
}: CompetitionPreviewProps) {
  const [showRulesDialog, setShowRulesDialog] = useState(false);
  const rulesUrl = useMemo(() => buildRulesUrl(competition), [competition]);
  const deadline = new Date(competition.deadline);
  const daysRemaining = Number.isNaN(deadline.getTime())
    ? null
    : Math.ceil((deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24));

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-indigo-100 px-2.5 py-1 text-xs font-semibold text-indigo-700">
              {formatCompetitionType(competition.competitionType)}
            </span>
            {matchCount !== null && matchCount !== undefined && (
              <span className="text-xs text-zinc-500">{matchCount} matches</span>
            )}
          </div>
          <h3 className="text-xl font-semibold text-zinc-900">{competition.title}</h3>
          {competition.description && (
            <p className="max-w-2xl text-sm text-zinc-600">
              {competition.description}
            </p>
          )}
        </div>
        <div className="flex flex-col gap-2 rounded-xl border border-zinc-100 bg-zinc-50 px-4 py-3 text-xs text-zinc-600">
          <div className="flex items-center gap-2">
            <Award className="size-3.5 text-emerald-600" />
            {competition.prizePool
              ? `$${competition.prizePool.toLocaleString()} prize`
              : "No prize pool"}
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="size-3.5 text-blue-600" />
            {Number.isNaN(deadline.getTime())
              ? "Deadline unavailable"
              : deadline.toLocaleDateString()}
          </div>
          {daysRemaining !== null && (
            <div className="flex items-center gap-2">
              <Flag className="size-3.5 text-amber-600" />
              {daysRemaining} days remaining
            </div>
          )}
        </div>
      </div>

      <div className="mt-4 grid gap-3 text-sm text-zinc-600 md:grid-cols-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-zinc-700">Metric</span>
          <span>
            {formatMetric(competition.metric)} ({competition.metricDirection})
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-semibold text-zinc-700">Submissions</span>
          <span>{competition.maxDailySubmissions} per day</span>
        </div>
      </div>

      {competition.tags?.length ? (
        <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
          <Tags className="size-3.5" />
          {competition.tags.map((tag) => (
            <span
              className="rounded-full border border-zinc-200 bg-zinc-50 px-2 py-0.5"
              key={tag}
            >
              {tag}
            </span>
          ))}
        </div>
      ) : null}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          className={cn(
            "rounded-xl border border-zinc-200 px-4 py-2 text-sm font-semibold text-zinc-600 transition-colors hover:border-zinc-300 hover:bg-zinc-50",
            isStarting && "cursor-not-allowed opacity-60"
          )}
          disabled={isStarting}
          onClick={onBack}
          type="button"
        >
          Back to Search
        </button>
        <button
          className={cn(
            "rounded-xl bg-gradient-to-r from-emerald-600 to-green-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-all",
            isStarting
              ? "cursor-not-allowed opacity-70"
              : "hover:from-emerald-700 hover:to-green-700"
          )}
          disabled={isStarting}
          onClick={onConfirm}
          type="button"
        >
          {isStarting ? "Starting..." : "Start Mission"}
        </button>
        <button
          className={cn(
            "rounded-xl border border-zinc-200 px-4 py-2 text-sm font-semibold text-zinc-600 transition-colors hover:border-zinc-300 hover:bg-zinc-50",
            isStarting && "cursor-not-allowed opacity-60"
          )}
          disabled={isStarting}
          onClick={() => setShowRulesDialog(true)}
          type="button"
        >
          Review Kaggle Rules
        </button>
      </div>

      <AlertDialog onOpenChange={setShowRulesDialog} open={showRulesDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Accept Kaggle competition rules</AlertDialogTitle>
            <AlertDialogDescription>
              Kaggle requires a one-time manual acceptance in your browser session before
              API access works. Open the rules page, confirm you are signed in to the
              correct Kaggle account, accept the rules, then return here to start the
              mission.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-xs text-zinc-600">
            Rules page:{" "}
            <a
              className="font-semibold text-blue-600 hover:text-blue-700"
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
            <AlertDialogAction onClick={onConfirm}>I've accepted, start mission</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
