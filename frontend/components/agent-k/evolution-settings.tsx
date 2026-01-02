"use client";

import { Dna } from "lucide-react";
import type { CompetitionSearchCriteria } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";

const DEFAULT_MAX_EVOLUTION_ROUNDS = 100;
const MAX_EVOLUTION_ROUNDS = 200;

type EvolutionSettingsProps = {
  criteria: CompetitionSearchCriteria;
  onChange: (criteria: CompetitionSearchCriteria) => void;
  disabled?: boolean;
};

export function EvolutionSettings({
  criteria,
  onChange,
  disabled = false,
}: EvolutionSettingsProps) {
  return (
    <section
      className={cn(
        "rounded-3xl border border-border bg-card p-6 shadow-sm",
        disabled && "opacity-60"
      )}
    >
      <div className="mb-6 flex items-center justify-between">
        <h2 className="flex items-center gap-3 font-medium text-foreground text-xl">
          <span className="flex size-10 items-center justify-center rounded-full bg-muted text-foreground">
            <Dna className="size-5" />
          </span>
          Evolution Limits
        </h2>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <div className="space-y-3">
          <label
            className="block font-bold text-muted-foreground text-xs uppercase tracking-widest"
            htmlFor="max-evolution-rounds-input"
          >
            Max Evolution Rounds
          </label>
          <div className="group relative">
            <input
              className="block w-full rounded-xl border border-border bg-background/50 px-4 py-3.5 font-mono text-foreground text-sm placeholder-muted-foreground transition-all focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              disabled={disabled}
              id="max-evolution-rounds-input"
              min={1}
              onChange={(event) => {
                const value = Number.parseInt(event.target.value, 10);
                onChange({
                  ...criteria,
                  maxEvolutionRounds: Number.isNaN(value)
                    ? DEFAULT_MAX_EVOLUTION_ROUNDS
                    : Math.min(MAX_EVOLUTION_ROUNDS, Math.max(1, value)),
                });
              }}
              placeholder="100"
              step={1}
              type="number"
              max={MAX_EVOLUTION_ROUNDS}
              value={criteria.maxEvolutionRounds}
            />
          </div>
        </div>
        <div className="space-y-3">
          <label
            className="block font-bold text-muted-foreground text-xs uppercase tracking-widest"
            htmlFor="min-improvements-input"
          >
            Min Improvements Before Submission
          </label>
          <div className="group relative">
            <input
              className="block w-full rounded-xl border border-border bg-background/50 px-4 py-3.5 font-mono text-foreground text-sm placeholder-muted-foreground transition-all focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              disabled={disabled}
              id="min-improvements-input"
              min={0}
              onChange={(event) => {
                const value = Number.parseInt(event.target.value, 10);
                onChange({
                  ...criteria,
                  minImprovementsRequired: Number.isNaN(value)
                    ? 0
                    : Math.max(0, value),
                });
              }}
              placeholder="0"
              step={1}
              type="number"
              value={criteria.minImprovementsRequired}
            />
          </div>
        </div>
      </div>

      <p className="mt-4 text-muted-foreground text-xs">
        Default 100 rounds. Max {MAX_EVOLUTION_ROUNDS}. Set 0 to allow
        submission without improvement gates.
      </p>
    </section>
  );
}
