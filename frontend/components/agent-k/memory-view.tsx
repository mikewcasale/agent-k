"use client";

import { Database, History, ScanEye } from "lucide-react";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import { cn } from "@/lib/utils";
import { formatDateTime } from "@/lib/utils/time";

export function MemoryView() {
  const { state } = useAgentKState();
  const memory = state.mission.memory;

  if (!memory) {
    return (
      <div className="rounded-lg border border-zinc-300 border-dashed bg-white px-4 py-6 text-sm text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
        Memory state not available.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-3 md:grid-cols-3">
        <MemoryStat
          color="emerald"
          icon={Database}
          label="Entries"
          value={memory.entries.length}
        />
        <MemoryStat
          color="blue"
          icon={History}
          label="Checkpoints"
          value={memory.checkpoints.length}
        />
        <MemoryStat
          color="amber"
          icon={ScanEye}
          label="Total size"
          value={`${Math.max(1, Math.round(memory.totalSizeBytes / 1024))} KB`}
        />
      </div>

      <CheckpointTimeline />
      <MemoryTable />
    </div>
  );
}

function MemoryStat({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  color: "emerald" | "blue" | "amber";
}) {
  const palette = {
    emerald: "from-emerald-500 to-green-500",
    blue: "from-blue-500 to-indigo-500",
    amber: "from-amber-500 to-orange-500",
  }[color];

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "flex size-10 items-center justify-center rounded-lg bg-gradient-to-br text-white",
            palette
          )}
        >
          <Icon className="size-5" />
        </div>
        <div>
          <p className="text-xs text-zinc-500 uppercase tracking-wide">
            {label}
          </p>
          <p className="font-semibold text-lg text-zinc-900 dark:text-white">
            {value}
          </p>
        </div>
      </div>
    </div>
  );
}

function CheckpointTimeline() {
  const { state } = useAgentKState();
  const checkpoints = state.mission.memory.checkpoints;
  if (!checkpoints.length) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <p className="font-semibold text-sm text-zinc-900 dark:text-white">
        Checkpoint timeline
      </p>
      <div className="mt-3 space-y-3">
        {checkpoints.map((checkpoint) => (
          <div
            className="flex items-center justify-between rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-800/60"
            key={`${checkpoint.name}-${checkpoint.timestamp}`}
          >
            <div>
              <p className="font-medium text-zinc-900 dark:text-white">
                {checkpoint.name}
              </p>
              <p className="text-xs text-zinc-500">
                {checkpoint.phase} • {formatDateTime(checkpoint.timestamp)}
              </p>
            </div>
            <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-[11px] text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200">
              Snapshot
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MemoryTable() {
  const { state } = useAgentKState();
  const entries = state.mission.memory.entries;
  if (!entries.length) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="border-zinc-200 border-b px-4 py-3 dark:border-zinc-800">
        <p className="font-semibold text-sm text-zinc-900 dark:text-white">
          Memory entries
        </p>
      </div>
      <div className="divide-y divide-zinc-200 text-sm dark:divide-zinc-800">
        <div className="grid grid-cols-5 px-4 py-2 text-xs text-zinc-500 uppercase tracking-wide">
          <span>Key</span>
          <span>Scope</span>
          <span>Category</span>
          <span>Accessed</span>
          <span>Reads</span>
        </div>
        {entries.map((entry) => (
          <div
            className="grid grid-cols-5 items-center px-4 py-3 text-sm text-zinc-700 dark:text-zinc-200"
            key={`${entry.key}-${entry.scope}`}
          >
            <span className="truncate font-medium">{entry.key}</span>
            <span className="text-xs text-zinc-500 uppercase">
              {entry.scope}
            </span>
            <span className="text-xs text-zinc-500">{entry.category}</span>
            <span className="text-xs text-zinc-500">
              {formatDateTime(entry.accessedAt)}
            </span>
            <span className="text-xs text-zinc-500">{entry.accessCount}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
