"use client";

import { motion } from "framer-motion";
import { Link2, Search, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { CompetitionPreview } from "@/components/agent-k/competition-preview";
import { DirectUrlInput } from "@/components/agent-k/direct-url-input";
import { SearchCriteriaForm } from "@/components/agent-k/search-criteria-form";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import type {
  CompetitionInfo,
  CompetitionSearchCriteria,
  CompetitionSelectionMode,
} from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";

const DEFAULT_CRITERIA: CompetitionSearchCriteria = {
  paidStatus: "any",
  domains: [],
  competitionTypes: [],
  minPrize: null,
  minDaysRemaining: 7,
};

const tabs: Array<{
  id: CompetitionSelectionMode;
  label: string;
  description: string;
  icon: typeof Search;
}> = [
  {
    id: "search",
    label: "Search Competitions",
    description: "Filter by prizes, domains, and timelines",
    icon: Search,
  },
  {
    id: "direct",
    label: "Enter URL",
    description: "Jump straight to a known competition",
    icon: Link2,
  },
];

export function CompetitionSelector() {
  const { dispatch } = useAgentKState();
  const [mode, setMode] = useState<CompetitionSelectionMode>("search");
  const [criteria, setCriteria] = useState<CompetitionSearchCriteria>(DEFAULT_CRITERIA);
  const [url, setUrl] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCompetition, setSelectedCompetition] = useState<CompetitionInfo | null>(null);
  const [matchCount, setMatchCount] = useState<number | null>(null);
  const [lastFetchedUrl, setLastFetchedUrl] = useState("");

  const isLoading = isSearching || isFetching;
  const activeTab = tabs.find((tab) => tab.id === mode) ?? tabs[0];

  const missionCriteria = useMemo(() => {
    const base = mode === "search" ? criteria : DEFAULT_CRITERIA;
    const payload: Record<string, unknown> = {
      min_days_remaining: base.minDaysRemaining,
    };
    if (base.competitionTypes.length) {
      payload.target_competition_types = base.competitionTypes;
    }
    if (base.minPrize) {
      payload.min_prize_pool = base.minPrize;
    }
    if (base.domains.length) {
      payload.target_domains = base.domains;
    }
    if (base.paidStatus === "paid" && !base.minPrize) {
      payload.min_prize_pool = 1;
    }
    return payload;
  }, [criteria, mode]);

  const handleSearch = async (nextCriteria: CompetitionSearchCriteria) => {
    setIsSearching(true);
    setError(null);
    setSelectedCompetition(null);
    setMatchCount(null);

    try {
      const response = await fetch("/api/competitions/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paid_only: nextCriteria.paidStatus === "paid",
          domains: nextCriteria.domains,
          competition_types: nextCriteria.competitionTypes,
          min_prize: nextCriteria.minPrize ?? undefined,
          min_days_remaining: nextCriteria.minDaysRemaining,
        }),
      });
      const data = (await response.json()) as {
        competitions?: CompetitionInfo[];
        count?: number;
        error?: string;
      };

      if (!response.ok || data.error) {
        setError(data.error ?? "Competition search failed.");
        return;
      }

      let competitions = data.competitions ?? [];
      if (nextCriteria.paidStatus === "free") {
        competitions = competitions.filter(
          (competition) => !competition.prizePool || competition.prizePool === 0
        );
      }

      if (!competitions.length) {
        setError("No competitions matched those filters.");
        setMatchCount(0);
        return;
      }

      setSelectedCompetition(competitions[0]);
      setMatchCount(competitions.length);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Competition search failed.");
    } finally {
      setIsSearching(false);
    }
  };

  const handleFetch = async (nextUrl: string) => {
    if (nextUrl === lastFetchedUrl) return;
    setIsFetching(true);
    setError(null);
    setSelectedCompetition(null);
    setMatchCount(null);

    try {
      const response = await fetch("/api/competitions/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: nextUrl }),
      });
      const data = (await response.json()) as {
        competition?: CompetitionInfo;
        error?: string;
      };

      if (!response.ok || data.error || !data.competition) {
        setError(data.error ?? "Unable to fetch competition.");
        setLastFetchedUrl("");
        return;
      }

      setSelectedCompetition(data.competition);
      setLastFetchedUrl(nextUrl);
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Unable to fetch competition.");
      setLastFetchedUrl("");
    } finally {
      setIsFetching(false);
    }
  };

  const handleConfirm = async () => {
    if (!selectedCompetition) return;
    setIsStarting(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        criteria: missionCriteria,
        competition_id: selectedCompetition.id,
      };
      if (selectedCompetition.url) {
        payload.competition_url = selectedCompetition.url;
      }
      const response = await fetch("/api/mission/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await response.json()) as { missionId?: string; error?: string };
      if (!response.ok || data.error) {
        setError(data.error ?? "Unable to start mission.");
        return;
      }

      dispatch({
        type: "SET_COMPETITION",
        payload: { competition: selectedCompetition, missionId: data.missionId },
      });
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : "Unable to start mission.");
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-5xl px-4 py-10">
        <motion.div
          animate={{ opacity: 1, y: 0 }}
          className="rounded-3xl border border-border bg-card/90 p-6 shadow-lg backdrop-blur md:p-8"
          initial={{ opacity: 0, y: 16 }}
          transition={{ duration: 0.3 }}
        >
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-blue-500 dark:text-blue-400">
                <Sparkles className="size-4" />
                Competition Selection
              </div>
              <h1 className="mt-2 text-2xl font-semibold text-foreground md:text-3xl">
                Choose the mission you want Agent-K to run.
              </h1>
              <p className="mt-2 text-sm text-muted-foreground">
                Search by criteria or jump directly to a Kaggle competition URL.
              </p>
            </div>
            <div className="rounded-2xl border border-border bg-muted px-4 py-3 text-xs text-muted-foreground">
              Mission kickoff is instant once you confirm.
            </div>
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={cn(
                  "flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                  mode === tab.id
                    ? "border-blue-500/50 bg-blue-500/10 text-blue-600 dark:text-blue-300"
                    : "border-border bg-background text-muted-foreground hover:border-blue-500/40 hover:bg-blue-500/5"
                )}
                onClick={() => {
                  setMode(tab.id);
                  setError(null);
                  setSelectedCompetition(null);
                  setMatchCount(null);
                }}
                type="button"
              >
                <tab.icon className="size-4" />
                {tab.label}
              </button>
            ))}
          </div>

          <div className="mt-6 rounded-2xl border border-border bg-muted/40 p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground">
              <activeTab.icon className="size-4 text-blue-500" />
              {activeTab.label}
            </div>
            <p className="mb-6 text-sm text-muted-foreground">
              {activeTab.description}
            </p>

            {mode === "search" ? (
              <SearchCriteriaForm
                criteria={criteria}
                isLoading={isSearching}
                onChange={setCriteria}
                onSubmit={handleSearch}
              />
            ) : (
              <DirectUrlInput
                error={error}
                isLoading={isFetching}
                onChange={setUrl}
                onSubmit={handleFetch}
                value={url}
              />
            )}
          </div>

          {error && mode === "search" && (
            <div className="mt-4 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          )}

          {selectedCompetition && (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="mt-6"
              initial={{ opacity: 0, y: 12 }}
              transition={{ duration: 0.2 }}
            >
              <CompetitionPreview
                competition={selectedCompetition}
                isStarting={isStarting}
                matchCount={mode === "search" ? matchCount : null}
                onBack={() => {
                  setSelectedCompetition(null);
                  setError(null);
                  setMatchCount(null);
                }}
                onConfirm={handleConfirm}
              />
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
