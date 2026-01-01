"use client";

import { Link2, Loader2, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

const KAGGLE_URL_PATTERN = /kaggle\.com\/competitions\/([a-zA-Z0-9-]+)/;

type DirectUrlInputProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  isLoading?: boolean;
  error?: string | null;
};

export function DirectUrlInput({
  value,
  onChange,
  onSubmit,
  isLoading = false,
  error,
}: DirectUrlInputProps) {
  const [touched, setTouched] = useState(false);
  const trimmed = value.trim();
  const isValid = useMemo(
    () => trimmed.length > 0 && KAGGLE_URL_PATTERN.test(trimmed),
    [trimmed]
  );
  const showInvalid = touched && trimmed.length > 0 && !isValid;

  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault();
        setTouched(true);
        if (isValid && !isLoading) {
          onSubmit(trimmed);
        }
      }}
    >
      <div className="space-y-2">
        <label
          className="flex items-center gap-2 font-semibold text-muted-foreground text-xs uppercase tracking-wide"
          htmlFor="kaggle-url-input"
        >
          <Link2 className="size-3.5" />
          Kaggle competition URL
        </label>
        <div className="relative">
          <input
            className={cn(
              "w-full rounded-xl border bg-background px-3 py-2 pr-10 text-foreground text-sm shadow-sm outline-none placeholder:text-muted-foreground focus:ring-2",
              showInvalid
                ? "border-destructive/60 focus:border-destructive focus:ring-destructive/30"
                : "border-border focus:border-blue-500/60 focus:ring-blue-500/20"
            )}
            id="kaggle-url-input"
            onBlur={() => {
              setTouched(true);
              if (isValid && !isLoading) {
                onSubmit(trimmed);
              }
            }}
            onChange={(event) => onChange(event.target.value)}
            placeholder="https://www.kaggle.com/competitions/titanic"
            type="url"
            value={value}
          />
          <div className="-translate-y-1/2 pointer-events-none absolute top-1/2 right-3 text-muted-foreground">
            {isLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Link2 className="size-4" />
            )}
          </div>
        </div>
      </div>

      {showInvalid && (
        <div className="flex items-center gap-2 text-destructive text-xs">
          <TriangleAlert className="size-3.5" />
          Enter a valid Kaggle competition URL.
        </div>
      )}

      {error && !showInvalid && (
        <div className="flex items-center gap-2 text-destructive text-xs">
          <TriangleAlert className="size-3.5" />
          {error}
        </div>
      )}

      <button
        className={cn(
          "flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-zinc-800 to-zinc-600 px-4 py-3 font-semibold text-sm text-white shadow-sm transition-all",
          isLoading
            ? "cursor-not-allowed opacity-80"
            : "hover:from-zinc-700 hover:to-zinc-500"
        )}
        disabled={isLoading || !isValid}
        type="submit"
      >
        {isLoading ? "Checking URL..." : "Fetch Competition"}
      </button>
    </form>
  );
}
