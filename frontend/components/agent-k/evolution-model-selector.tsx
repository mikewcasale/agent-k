"use client";

import { Brain, Check, Code, Layers, Sparkles, Terminal } from "lucide-react";
import type { EvolutionModelOption } from "@/lib/ai/agent-k-models";
import { cn } from "@/lib/utils";

const iconMap = {
  sparkles: Sparkles,
  brain: Brain,
  code: Code,
  terminal: Terminal,
} as const;

type EvolutionModelSelectorProps = {
  options: EvolutionModelOption[];
  recommended: string[];
  value: string[];
  disabled?: boolean;
  onChange: (next: string[]) => void;
};

export function EvolutionModelSelector({
  options,
  recommended,
  value,
  disabled = false,
  onChange,
}: EvolutionModelSelectorProps) {
  const toggleModel = (modelId: string) => {
    if (disabled) {
      return;
    }
    const isActive = value.includes(modelId);
    const next = isActive
      ? value.filter((item) => item !== modelId)
      : [...value, modelId];
    if (!next.length) {
      return;
    }
    onChange(next);
  };

  const selectAll = () => {
    if (disabled) {
      return;
    }
    onChange(options.map((option) => option.id));
  };

  const resetToRecommended = () => {
    if (disabled || !recommended.length) {
      return;
    }
    onChange(recommended);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between px-1">
        <h2 className="flex items-center gap-3 font-medium text-foreground text-xl">
          <span className="flex size-10 items-center justify-center rounded-full bg-muted text-foreground">
            <Layers className="size-5" />
          </span>
          Evolution Models
        </h2>
        <div className="flex items-center gap-2">
          <button
            className={cn(
              "font-medium text-muted-foreground text-xs transition-colors",
              disabled ? "cursor-not-allowed opacity-60" : "hover:text-blue-500"
            )}
            disabled={disabled}
            onClick={selectAll}
            type="button"
          >
            Select All
          </button>
          <span className="text-border">|</span>
          <button
            className={cn(
              "font-medium text-blue-500 text-xs transition-colors",
              disabled ? "cursor-not-allowed opacity-60" : "hover:text-blue-400"
            )}
            disabled={disabled}
            onClick={resetToRecommended}
            type="button"
          >
            Recommended
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {options.map((option) => {
          const isActive = value.includes(option.id);
          const isRecommended = recommended.includes(option.id);
          const IconComponent = option.icon ? iconMap[option.icon] : Sparkles;

          return (
            <label
              className={cn(
                "relative block cursor-pointer",
                disabled && "cursor-not-allowed opacity-60"
              )}
              key={option.id}
            >
              <input
                checked={isActive}
                className="peer sr-only"
                disabled={disabled}
                onChange={() => toggleModel(option.id)}
                type="checkbox"
              />
              <div
                className={cn(
                  "relative overflow-hidden rounded-2xl border p-5 transition-all",
                  isActive
                    ? "border-blue-500 bg-card shadow-lg"
                    : "border-border bg-card hover:border-muted-foreground/30 hover:bg-muted/50"
                )}
              >
                {isActive && (
                  <div className="pointer-events-none absolute inset-0 bg-blue-500/5" />
                )}
                <div className="relative z-10 mb-3 flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div
                      className={cn(
                        "flex size-10 items-center justify-center rounded-lg border shadow-lg",
                        isRecommended
                          ? "border-transparent bg-gradient-to-br from-indigo-500 to-purple-600 text-white"
                          : "border-border bg-muted text-muted-foreground"
                      )}
                    >
                      {isRecommended ? (
                        <span className="font-bold text-xs tracking-tighter">
                          C4.5
                        </span>
                      ) : (
                        <IconComponent className="size-5" />
                      )}
                    </div>
                    <div>
                      <h3 className="font-semibold text-base text-foreground">
                        {option.label}
                      </h3>
                      <div className="flex items-center gap-1.5">
                        {isRecommended ? (
                          <span className="flex items-center gap-1 font-bold text-[10px] text-emerald-500 uppercase tracking-wider">
                            <Check className="size-2.5" /> Recommended
                          </span>
                        ) : option.freeTier ? (
                          <span className="font-bold text-[10px] text-amber-500 uppercase tracking-wider">
                            Free Tier
                          </span>
                        ) : (
                          <span className="text-muted-foreground text-xs">
                            {option.provider}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div
                    className={cn(
                      "flex size-6 items-center justify-center rounded-full border-2 transition-all",
                      isActive
                        ? "border-blue-500 bg-blue-500 text-white shadow-sm"
                        : "border-border"
                    )}
                  >
                    {isActive && <Check className="size-3.5 font-bold" />}
                  </div>
                </div>
                <p className="relative z-10 mb-4 pl-[3.25rem] text-muted-foreground text-sm leading-relaxed">
                  {option.description}
                </p>
                <div className="relative z-10 pl-[3.25rem]">
                  <code className="rounded border border-border bg-muted px-2 py-1 font-mono text-[10px] text-muted-foreground">
                    {option.id}
                  </code>
                </div>
              </div>
            </label>
          );
        })}
        <div className="mt-4 border-border border-t pt-4 text-center">
          <p className="text-muted-foreground text-xs">
            Queued for evolution upon mission start.
          </p>
        </div>
      </div>
    </div>
  );
}
