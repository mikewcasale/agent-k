"use client";

import {
  ArrowRight,
  ArrowUpDown,
  ExternalLink,
  FileText,
  HeartPulse,
  TrendingUp,
  Trophy,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  BEST_RESULTS_STORAGE_KEY,
  formatResultAge,
  loadBestResults,
  mergeBestResults,
  saveBestResults,
  sortBestResults,
  type BestResultEntry,
} from "@/lib/utils/best-results";
import { buildCompetitionSubmissionsUrl } from "@/lib/utils/kaggle";
import { cn } from "@/lib/utils";

const MAX_RESULTS = 3;

type CategoryMeta = {
  label: string;
  icon: typeof Trophy;
  className: string;
};

const CATEGORY_STYLES: Array<{
  matches: string[];
  meta: CategoryMeta;
}> = [
  {
    matches: ["finance", "trading"],
    meta: {
      label: "Finance",
      icon: TrendingUp,
      className: "bg-blue-500/10 text-blue-500 dark:bg-blue-500/20 dark:text-blue-300",
    },
  },
  {
    matches: ["medical", "health", "biomedical"],
    meta: {
      label: "Medical",
      icon: HeartPulse,
      className:
        "bg-purple-500/10 text-purple-500 dark:bg-purple-500/20 dark:text-purple-300",
    },
  },
  {
    matches: ["nlp", "text", "language"],
    meta: {
      label: "NLP",
      icon: FileText,
      className:
        "bg-emerald-500/10 text-emerald-600 dark:bg-emerald-500/20 dark:text-emerald-300",
    },
  },
];

function formatCategoryLabel(category?: string): string {
  if (!category) {
    return "Kaggle";
  }

  const trimmed = category.trim();
  if (trimmed.length <= 4 && trimmed === trimmed.toLowerCase()) {
    return trimmed.toUpperCase();
  }

  return trimmed
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function resolveCategory(category?: string): CategoryMeta {
  const normalized = (category ?? "").toLowerCase();
  for (const { matches, meta } of CATEGORY_STYLES) {
    if (matches.some((match) => normalized.includes(match))) {
      return { ...meta, label: formatCategoryLabel(category ?? meta.label) };
    }
  }

  return {
    label: formatCategoryLabel(category),
    icon: Trophy,
    className:
      "bg-zinc-100 text-zinc-500 dark:bg-white/10 dark:text-zinc-300",
  };
}

function formatOrdinal(rank: number): string {
  const mod10 = rank % 10;
  const mod100 = rank % 100;
  if (mod10 === 1 && mod100 !== 11) {
    return `${rank}st`;
  }
  if (mod10 === 2 && mod100 !== 12) {
    return `${rank}nd`;
  }
  if (mod10 === 3 && mod100 !== 13) {
    return `${rank}rd`;
  }
  return `${rank}th`;
}

function formatRankLabel(entry: BestResultEntry): string {
  if (entry.rank != null) {
    if (entry.rank <= 3) {
      return formatOrdinal(entry.rank);
    }

    if (entry.percentile != null) {
      const percent = entry.percentile > 1 ? entry.percentile : entry.percentile * 100;
      return `Top ${Math.round(percent)}%`;
    }

    if (entry.totalTeams != null) {
      return `Rank ${entry.rank}/${entry.totalTeams}`;
    }

    return `Rank ${entry.rank}`;
  }

  if (entry.percentile != null) {
    const percent = entry.percentile > 1 ? entry.percentile : entry.percentile * 100;
    return `Top ${Math.round(percent)}%`;
  }

  return "-";
}

function getRankBadgeClass(entry: BestResultEntry): string {
  if (entry.rank != null && entry.rank <= 3) {
    return "border-amber-200 bg-amber-100 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/15 dark:text-amber-200";
  }
  return "border-border bg-muted text-muted-foreground";
}

export function BestResults() {
  const [results, setResults] = useState<BestResultEntry[]>([]);

  useEffect(() => {
    const storedResults = loadBestResults();
    setResults(storedResults);

    let isActive = true;
    const controller = new AbortController();

    const hydrateBestResults = async () => {
      try {
        const response = await fetch("/api/mission/best-results", {
          signal: controller.signal,
        });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as {
          results?: BestResultEntry[];
        };
        if (!payload.results || !payload.results.length) {
          return;
        }
        const merged = mergeBestResults(loadBestResults(), payload.results);
        saveBestResults(merged);
        if (isActive) {
          setResults(merged);
        }
      } catch {
        // Ignore history fetch errors.
      }
    };

    void hydrateBestResults();

    const handleStorage = (event: StorageEvent) => {
      if (event.key === BEST_RESULTS_STORAGE_KEY) {
        setResults(loadBestResults());
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => {
      isActive = false;
      controller.abort();
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  const entries = useMemo(
    () => sortBestResults(results).slice(0, MAX_RESULTS),
    [results]
  );

  return (
    <section className="relative overflow-hidden rounded-3xl border border-border bg-card p-6 shadow-sm md:p-8">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="flex items-center gap-3 font-medium text-foreground text-xl">
          <span className="flex size-10 items-center justify-center rounded-full bg-muted text-foreground">
            <Trophy className="size-5" />
          </span>
          Best Results
        </h2>
        <button
          className="flex items-center gap-1 font-medium text-muted-foreground text-xs transition-colors hover:text-blue-500"
          type="button"
        >
          View All <ArrowRight className="size-3" />
        </button>
      </div>
      <div className="overflow-x-auto rounded-xl border border-border">
        {entries.length === 0 ? (
          <div className="px-4 py-6 text-center text-muted-foreground text-sm">
            No historical results yet. Run a mission to capture your best
            submissions.
          </div>
        ) : (
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-border border-b bg-muted">
                <th className="group cursor-pointer px-4 py-3 font-bold text-muted-foreground text-xs uppercase tracking-wider transition-colors hover:text-blue-500">
                  <div className="flex items-center gap-1">
                    Competition Name
                    <ArrowUpDown className="size-3 opacity-50 transition-opacity group-hover:opacity-100" />
                  </div>
                </th>
                <th className="group w-24 cursor-pointer px-4 py-3 text-center font-bold text-muted-foreground text-xs uppercase tracking-wider transition-colors hover:text-blue-500">
                  <div className="flex items-center justify-center gap-1">
                    Rank
                    <ArrowUpDown className="size-3 opacity-50 transition-opacity group-hover:opacity-100" />
                  </div>
                </th>
                <th className="px-4 py-3 text-right font-bold text-muted-foreground text-xs uppercase tracking-wider">
                  Link
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {entries.map((entry) => {
                const categoryMeta = resolveCategory(entry.category);
                const Icon = categoryMeta.icon;
                const rankLabel = formatRankLabel(entry);
                const resultAge = formatResultAge(
                  entry.submittedAt ?? entry.recordedAt
                );
                const submissionsUrl = buildCompetitionSubmissionsUrl(
                  { id: entry.competitionId, url: entry.competitionUrl },
                  entry.competitionId,
                  entry.submissionId
                );

                return (
                  <tr
                    className="group transition-colors hover:bg-muted/50"
                    key={`${entry.competitionId}-${entry.submissionId ?? "latest"}`}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <div
                          className={cn(
                            "flex size-8 items-center justify-center rounded",
                            categoryMeta.className
                          )}
                        >
                          <Icon className="size-4" />
                        </div>
                        <div>
                          <div className="font-medium text-foreground text-sm transition-colors group-hover:text-blue-500">
                            {entry.competitionTitle}
                          </div>
                          <div className="text-[10px] text-muted-foreground">
                            {categoryMeta.label} - {resultAge}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1 rounded border px-2 py-1 text-xs font-bold",
                          getRankBadgeClass(entry)
                        )}
                      >
                        {rankLabel}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <a
                        className="inline-flex size-8 items-center justify-center rounded-full bg-muted text-muted-foreground transition-all hover:bg-blue-500 hover:text-white"
                        href={submissionsUrl}
                        rel="noreferrer"
                        target="_blank"
                      >
                        <ExternalLink className="size-4" />
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
