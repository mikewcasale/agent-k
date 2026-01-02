"use client";

import { motion } from "framer-motion";
import { Filter, Link2, Search, Sparkles } from "lucide-react";
import { useMemo, useState } from "react";
import { CompetitionPreview } from "@/components/agent-k/competition-preview";
import { DirectUrlInput } from "@/components/agent-k/direct-url-input";
import { EvolutionModelSelector } from "@/components/agent-k/evolution-model-selector";
import { BestResults } from "@/components/agent-k/best-results";
import { SearchCriteriaForm } from "@/components/agent-k/search-criteria-form";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import {
  DEFAULT_EVOLUTION_MODELS,
  evolutionModels,
} from "@/lib/ai/agent-k-models";
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

export function CompetitionSelector() {
  const { dispatch } = useAgentKState();
  const [mode, setMode] = useState<CompetitionSelectionMode>("search");
  const [criteria, setCriteria] =
    useState<CompetitionSearchCriteria>(DEFAULT_CRITERIA);
  const [url, setUrl] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCompetition, setSelectedCompetition] =
    useState<CompetitionInfo | null>(null);
  const [matchCount, setMatchCount] = useState<number | null>(null);
  const [lastFetchedUrl, setLastFetchedUrl] = useState("");
  const [selectedEvolutionModels, setSelectedEvolutionModels] = useState<
    string[]
  >(DEFAULT_EVOLUTION_MODELS);

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
      setError(
        fetchError instanceof Error
          ? fetchError.message
          : "Competition search failed."
      );
    } finally {
      setIsSearching(false);
    }
  };

  const handleFetch = async (nextUrl: string) => {
    if (nextUrl === lastFetchedUrl) {
      return;
    }
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
      setError(
        fetchError instanceof Error
          ? fetchError.message
          : "Unable to fetch competition."
      );
      setLastFetchedUrl("");
    } finally {
      setIsFetching(false);
    }
  };

  const handleConfirm = async () => {
    if (!selectedCompetition) {
      return;
    }
    setIsStarting(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        criteria: missionCriteria,
        competition_id: selectedCompetition.id,
        evolution_models: selectedEvolutionModels,
      };
      if (selectedCompetition.url) {
        payload.competition_url = selectedCompetition.url;
      }
      const response = await fetch("/api/mission/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await response.json()) as {
        missionId?: string;
        error?: string;
      };
      if (!response.ok || data.error) {
        setError(data.error ?? "Unable to start mission.");
        return;
      }

      dispatch({
        type: "SET_COMPETITION",
        payload: {
          competition: selectedCompetition,
          missionId: data.missionId,
        },
      });
    } catch (fetchError) {
      setError(
        fetchError instanceof Error
          ? fetchError.message
          : "Unable to start mission."
      );
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <div className="min-h-dvh bg-background">
      <div className="mx-auto w-full max-w-[85rem] px-4 py-10 sm:px-6 lg:px-8">
        <motion.div
          animate={{ opacity: 1, y: 0 }}
          initial={{ opacity: 0, y: 16 }}
          transition={{ duration: 0.3 }}
        >
          {/* Header */}
          <header className="mb-10 flex flex-col gap-6 border-border border-b pb-8 md:flex-row md:items-end md:justify-between">
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-blue-500">
                <Sparkles className="size-4" />
                <span className="font-bold text-xs uppercase tracking-[0.2em]">
                  Competition Selection
                </span>
              </div>
              <div>
                <h1 className="mb-2 font-semibold text-3xl text-foreground tracking-tight lg:text-4xl">
                  Mission{" "}
                  <span className="font-normal text-muted-foreground">
                    Configuration
                  </span>
                </h1>
                <p className="max-w-2xl font-light text-base text-muted-foreground">
                  Select a target mission and configure evolutionary models for
                  Agent-K.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex rounded-xl border border-border bg-muted p-1">
                <button
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-4 py-2 font-medium text-sm transition-all",
                    mode === "search"
                      ? "border border-border bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-card/50 hover:text-foreground"
                  )}
                  onClick={() => {
                    setMode("search");
                    setError(null);
                    setSelectedCompetition(null);
                    setMatchCount(null);
                  }}
                  type="button"
                >
                  <Search className="size-4" />
                  Search
                </button>
                <button
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-4 py-2 font-medium text-sm transition-all",
                    mode === "direct"
                      ? "border border-border bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-card/50 hover:text-foreground"
                  )}
                  onClick={() => {
                    setMode("direct");
                    setError(null);
                    setSelectedCompetition(null);
                    setMatchCount(null);
                  }}
                  type="button"
                >
                  <Link2 className="size-4" />
                  URL
                </button>
              </div>
            </div>
          </header>

          {/* Main content grid */}
          <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-12">
            {/* Left column - Search Criteria */}
            <div className="flex flex-col gap-6 lg:col-span-7">
              <section className="group relative overflow-hidden rounded-3xl border border-border bg-card p-6 shadow-sm md:p-8">
                {/* Decorative gradient */}
                <div className="-right-32 -top-32 pointer-events-none absolute size-96 rounded-full bg-blue-500/5 blur-3xl" />

                <div className="relative z-10">
                  <div className="mb-8 flex items-center justify-between">
                    <h2 className="flex items-center gap-3 font-medium text-foreground text-xl">
                      <span className="flex size-10 items-center justify-center rounded-full bg-muted text-foreground">
                        <Filter className="size-5" />
                      </span>
                      Search Criteria
                    </h2>
                  </div>

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
              </section>
              <BestResults />
            </div>

            {/* Right column - Evolution Models */}
            <div className="flex flex-col gap-6 lg:col-span-5">
              <EvolutionModelSelector
                disabled={isStarting}
                onChange={setSelectedEvolutionModels}
                options={evolutionModels}
                recommended={DEFAULT_EVOLUTION_MODELS}
                value={selectedEvolutionModels}
              />
            </div>
          </div>

          {/* Error message */}
          {error && mode === "search" && (
            <div className="mt-6 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-destructive text-sm">
              {error}
            </div>
          )}

          {/* Competition Preview */}
          {selectedCompetition && (
            <motion.div
              animate={{ opacity: 1, y: 0 }}
              className="mt-8"
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
