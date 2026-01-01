"use client";

import { AlertTriangle, CheckCircle2, Clock, RefreshCcw } from "lucide-react";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { formatDateTime } from "@/lib/utils/time";

export function LogsView() {
  const { state } = useAgentKState();
  const errors = state.mission.errors;

  if (!errors.length) {
    return (
      <div className="rounded-lg border border-zinc-300 border-dashed bg-white px-4 py-6 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
        No errors reported. Live logs will appear here when available.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {errors.map((error) => (
        <div
          className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
          key={error.id}
        >
          <div className="flex items-start gap-3">
            <div className="mt-0.5">
              {error.resolved ? (
                <CheckCircle2 className="size-4 text-emerald-500" />
              ) : (
                <AlertTriangle className="size-4 text-amber-500" />
              )}
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex flex-wrap items-center gap-2 font-semibold text-sm text-zinc-900 dark:text-white">
                <span>{error.errorType}</span>
                <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] text-amber-700 dark:bg-amber-900/40 dark:text-amber-200">
                  {error.category}
                </span>
                <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[11px] text-blue-700 dark:bg-blue-900/40 dark:text-blue-200">
                  Strategy {error.recoveryStrategy}
                </span>
              </div>
              <p className="text-sm text-zinc-600 dark:text-zinc-300">
                {error.message}
              </p>
              {error.context && (
                <p className="text-xs text-zinc-500">
                  Context: {error.context}
                </p>
              )}
              <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500">
                <span className="inline-flex items-center gap-1">
                  <Clock className="size-3" />
                  {formatDateTime(error.timestamp)}
                </span>
                <span className="inline-flex items-center gap-1">
                  <RefreshCcw className="size-3" />
                  Attempts {error.recoveryAttempts}
                </span>
                {error.resolution && (
                  <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-300">
                    Resolved: {error.resolution}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
