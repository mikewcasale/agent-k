"use client";

import { DollarSign, Filter, Target, Trophy } from "lucide-react";
import type { CompetitionSearchCriteria } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";

const paidOptions = [
  { id: "any", label: "Any" },
  { id: "paid", label: "Paid only" },
  { id: "free", label: "Free only" },
] as const;

const domainOptions = [
  { id: "finance", label: "Finance" },
  { id: "medical", label: "Medical" },
  { id: "weather", label: "Weather" },
  { id: "computer_vision", label: "Computer Vision" },
  { id: "nlp", label: "NLP" },
  { id: "tabular", label: "Tabular" },
  { id: "audio", label: "Audio" },
] as const;

const competitionTypeOptions = [
  { id: "featured", label: "Featured" },
  { id: "research", label: "Research" },
  { id: "playground", label: "Playground" },
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
    list.includes(value)
      ? list.filter((item) => item !== value)
      : [...list, value];

  return (
    <form
      className="space-y-8"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit(criteria);
      }}
    >
      <div className="group/filter">
        <p className="mb-4 flex items-center gap-2 font-bold text-muted-foreground text-xs uppercase tracking-widest transition-colors group-hover/filter:text-blue-500">
          <DollarSign className="size-3.5" />
          Paid Status
        </p>
        <div className="flex gap-2">
          {paidOptions.map((option) => (
            <button
              className={cn(
                "rounded-full px-5 py-2 font-medium text-sm transition-all",
                criteria.paidStatus === option.id
                  ? "bg-foreground text-background shadow-lg"
                  : "border border-border bg-transparent text-muted-foreground hover:border-muted-foreground/50 hover:text-foreground"
              )}
              key={option.id}
              onClick={() =>
                onChange({
                  ...criteria,
                  paidStatus: option.id as typeof criteria.paidStatus,
                })
              }
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="group/filter">
        <p className="mb-4 flex items-center gap-2 font-bold text-muted-foreground text-xs uppercase tracking-widest transition-colors group-hover/filter:text-blue-500">
          <Target className="size-3.5" />
          Subject Domains
        </p>
        <div className="flex flex-wrap gap-2">
          {domainOptions.map((option) => {
            const isActive = criteria.domains.includes(option.id);
            return (
              <button
                className={cn(
                  "rounded-lg px-3 py-1.5 text-sm transition-colors",
                  isActive
                    ? "border border-blue-500/30 bg-blue-500/10 text-blue-500 hover:bg-blue-500/20"
                    : "border border-transparent bg-muted text-muted-foreground hover:bg-muted/80"
                )}
                key={option.id}
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

      <div className="group/filter">
        <p className="mb-4 flex items-center gap-2 font-bold text-muted-foreground text-xs uppercase tracking-widest transition-colors group-hover/filter:text-blue-500">
          <Trophy className="size-3.5" />
          Competition Types
        </p>
        <div className="flex flex-wrap gap-2">
          {competitionTypeOptions.map((option) => {
            const isActive = criteria.competitionTypes.includes(option.id);
            return (
              <button
                className={cn(
                  "rounded-lg px-3 py-1.5 text-sm transition-colors",
                  isActive
                    ? "border border-blue-500/30 bg-blue-500/10 text-blue-500 hover:bg-blue-500/20"
                    : "border border-transparent bg-muted text-muted-foreground hover:bg-muted/80"
                )}
                key={option.id}
                onClick={() =>
                  onChange({
                    ...criteria,
                    competitionTypes: toggleSelection(
                      option.id,
                      criteria.competitionTypes
                    ),
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

      <div className="grid grid-cols-2 gap-6 pt-2">
        <div className="space-y-3">
          <label
            className="block font-bold text-muted-foreground text-xs uppercase tracking-widest"
            htmlFor="min-prize-input"
          >
            Min Prize ($)
          </label>
          <div className="group relative">
            <input
              className="block w-full rounded-xl border border-border bg-background/50 px-4 py-3.5 font-mono text-foreground text-sm placeholder-muted-foreground transition-all focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              id="min-prize-input"
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
        </div>
        <div className="space-y-3">
          <label
            className="block font-bold text-muted-foreground text-xs uppercase tracking-widest"
            htmlFor="min-days-input"
          >
            Min Days Remaining
          </label>
          <div className="group relative">
            <input
              className="block w-full rounded-xl border border-border bg-background/50 px-4 py-3.5 font-mono text-foreground text-sm placeholder-muted-foreground transition-all focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              id="min-days-input"
              min={1}
              onChange={(event) => {
                const value = Number.parseInt(event.target.value, 10);
                onChange({
                  ...criteria,
                  minDaysRemaining: Number.isNaN(value)
                    ? 7
                    : Math.max(1, value),
                });
              }}
              placeholder="7"
              type="number"
              value={criteria.minDaysRemaining}
            />
          </div>
        </div>
      </div>

      <div className="pt-6">
        <button
          className={cn(
            "flex w-full items-center justify-center gap-3 rounded-xl bg-blue-500 px-4 py-4 font-semibold text-lg text-white shadow-[0_0_20px_-5px_rgba(59,130,246,0.3)] transition-all",
            isLoading
              ? "cursor-not-allowed opacity-80"
              : "hover:bg-blue-600 active:scale-[0.99]"
          )}
          disabled={isLoading}
          type="submit"
        >
          <Filter className="size-5" />
          <span>{isLoading ? "Searching..." : "Find Competitions"}</span>
        </button>
        <p className="mt-4 text-center text-muted-foreground text-xs">
          Mission kickoff is instant once you confirm selection.
        </p>
      </div>
    </form>
  );
}
