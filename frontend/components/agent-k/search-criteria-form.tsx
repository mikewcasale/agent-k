"use client";

import { Award, Calendar, Filter, Target } from "lucide-react";
import type { CompetitionSearchCriteria } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";

const paidOptions = [
  { id: "any", label: "Any" },
  { id: "paid", label: "Paid only" },
  { id: "free", label: "Free only" },
] as const;

const domainOptions = [
  { id: "finance", label: "Finance" },
  { id: "medical", label: "Medical/Healthcare" },
  { id: "weather", label: "Weather/Climate" },
  { id: "computer_vision", label: "Computer Vision" },
  { id: "nlp", label: "NLP" },
  { id: "tabular", label: "Tabular" },
  { id: "time_series", label: "Time Series" },
  { id: "audio", label: "Audio" },
  { id: "geospatial", label: "Geospatial" },
] as const;

const competitionTypeOptions = [
  { id: "featured", label: "Featured" },
  { id: "research", label: "Research" },
  { id: "playground", label: "Playground" },
  { id: "getting_started", label: "Getting Started" },
  { id: "community", label: "Community" },
] as const;

type SearchCriteriaFormProps = {
  criteria: CompetitionSearchCriteria;
  onChange: (criteria: CompetitionSearchCriteria) => void;
  onSubmit: (criteria: CompetitionSearchCriteria) => void;
  isLoading?: boolean;
};

export function SearchCriteriaForm({
  criteria,
  onChange,
  onSubmit,
  isLoading = false,
}: SearchCriteriaFormProps) {
  const toggleSelection = (value: string, list: string[]) =>
    list.includes(value) ? list.filter((item) => item !== value) : [...list, value];

  return (
    <form
      className="space-y-6"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(criteria);
      }}
    >
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <Filter className="size-3.5" />
          Paid status
        </label>
        <div className="flex flex-wrap gap-2">
          {paidOptions.map((option) => (
            <button
              key={option.id}
              className={cn(
                "rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
                criteria.paidStatus === option.id
                  ? "border-blue-500/60 bg-blue-500/10 text-blue-600 dark:text-blue-300"
                  : "border-border bg-background text-muted-foreground hover:border-blue-500/40 hover:bg-blue-500/5"
              )}
              onClick={() =>
                onChange({ ...criteria, paidStatus: option.id as typeof criteria.paidStatus })
              }
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-3">
        <label className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <Target className="size-3.5" />
          Subject domains
        </label>
        <div className="flex flex-wrap gap-2">
          {domainOptions.map((option) => {
            const isActive = criteria.domains.includes(option.id);
            return (
              <button
                key={option.id}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
                  isActive
                    ? "border-emerald-500/60 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300"
                    : "border-border bg-background text-muted-foreground hover:border-emerald-500/40 hover:bg-emerald-500/5"
                )}
                onClick={() =>
                  onChange({
                    ...criteria,
                    domains: toggleSelection(option.id, criteria.domains),
                  })
                }
                type="button"
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-y-3">
        <label className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          <Award className="size-3.5" />
          Competition types
        </label>
        <div className="flex flex-wrap gap-2">
          {competitionTypeOptions.map((option) => {
            const isActive = criteria.competitionTypes.includes(option.id);
            return (
              <button
                key={option.id}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-xs font-medium transition-colors",
                  isActive
                    ? "border-purple-500/60 bg-purple-500/10 text-purple-600 dark:text-purple-300"
                    : "border-border bg-background text-muted-foreground hover:border-purple-500/40 hover:bg-purple-500/5"
                )}
                onClick={() =>
                  onChange({
                    ...criteria,
                    competitionTypes: toggleSelection(option.id, criteria.competitionTypes),
                  })
                }
                type="button"
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Award className="size-3.5" />
            Min prize ($)
          </label>
          <input
            className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm text-foreground shadow-sm outline-none placeholder:text-muted-foreground focus:border-blue-500/60 focus:ring-2 focus:ring-blue-500/20"
            min={0}
            onChange={(event) => {
              const value = event.target.value;
              onChange({
                ...criteria,
                minPrize: value === "" ? null : Number.parseInt(value, 10),
              });
            }}
            placeholder="0"
            type="number"
            value={criteria.minPrize ?? ""}
          />
        </div>
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <Calendar className="size-3.5" />
            Min days remaining
          </label>
          <input
            className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm text-foreground shadow-sm outline-none placeholder:text-muted-foreground focus:border-blue-500/60 focus:ring-2 focus:ring-blue-500/20"
            min={1}
            onChange={(event) => {
              const value = Number.parseInt(event.target.value, 10);
              onChange({
                ...criteria,
                minDaysRemaining: Number.isNaN(value) ? 7 : Math.max(1, value),
              });
            }}
            placeholder="7"
            type="number"
            value={criteria.minDaysRemaining}
          />
        </div>
      </div>

      <button
        className={cn(
          "flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition-all",
          isLoading ? "cursor-not-allowed opacity-80" : "hover:from-blue-700 hover:to-indigo-700"
        )}
        disabled={isLoading}
        type="submit"
      >
        {isLoading ? "Searching..." : "Search Competitions"}
      </button>
    </form>
  );
}
