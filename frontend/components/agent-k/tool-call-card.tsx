"use client";

import type React from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  ChevronDown,
  Clock,
  Globe2,
  Loader2,
  Search,
  TerminalSquare,
} from "lucide-react";
import { useState } from "react";
import { CodeBlock } from "@/components/elements/code-block";
import { useAgentKState } from "@/hooks/use-agent-k-state";
import type { ToolCall, ToolType } from "@/lib/types/agent-k";
import { cn } from "@/lib/utils";
import { formatDuration } from "@/lib/utils/time";

const toolIcons: Record<ToolType, React.ElementType> = {
  web_search: Search,
  kaggle_mcp: Globe2,
  code_executor: TerminalSquare,
  memory: Brain,
  browser: Globe2,
};

const toolColors: Record<ToolType, string> = {
  web_search: "bg-violet-100 text-violet-600 dark:bg-violet-900/40 dark:text-violet-200",
  kaggle_mcp: "bg-sky-100 text-sky-600 dark:bg-sky-900/40 dark:text-sky-200",
  code_executor: "bg-amber-100 text-amber-600 dark:bg-amber-900/40 dark:text-amber-200",
  memory: "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-200",
  browser: "bg-indigo-100 text-indigo-600 dark:bg-indigo-900/40 dark:text-indigo-200",
};

const toolNames: Record<ToolType, string> = {
  web_search: "Web Search",
  kaggle_mcp: "Kaggle MCP",
  code_executor: "Code Executor",
  memory: "Memory",
  browser: "Browser",
};

interface ToolCallCardProps {
  toolCall: ToolCall;
  taskId: string;
}

export function ToolCallCard({ toolCall, taskId }: ToolCallCardProps) {
  const { state, toggleToolCall } = useAgentKState();
  const [showThinking, setShowThinking] = useState(state.ui.showThinkingBlocks);

  const isExpanded = state.ui.expandedToolCalls.has(toolCall.id);
  const Icon = toolIcons[toolCall.type];
  const hasError = Boolean(toolCall.error);
  const isComplete = Boolean(toolCall.completedAt || toolCall.result);
  const isRunning = !isComplete && !hasError;

  return (
    <motion.div
      layout
      className="overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
    >
      <button
        className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-zinc-50 dark:hover:bg-zinc-800/60"
        onClick={() => toggleToolCall(toolCall.id)}
      >
        <div
          className={cn(
            "flex size-8 items-center justify-center rounded-lg",
            toolColors[toolCall.type]
          )}
        >
          <Icon className="size-4" />
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-100">
              {toolNames[toolCall.type]}
            </span>
            <span className="text-xs text-zinc-500">{toolCall.operation}</span>
          </div>
          {toolCall.params && (
            <p className="truncate text-xs text-zinc-500">
              {JSON.stringify(toolCall.params).slice(0, 60)}
            </p>
          )}
        </div>

        <div className="flex items-center gap-2">
          {toolCall.durationMs !== undefined && (
            <span className="flex items-center gap-1 text-xs text-zinc-500">
              <Clock className="size-3" />
              {formatDuration(toolCall.durationMs)}
            </span>
          )}
          {hasError ? (
            <AlertCircle className="size-4 text-red-500" />
          ) : isComplete ? (
            <CheckCircle2 className="size-4 text-emerald-500" />
          ) : isRunning ? (
            <Loader2 className="size-4 animate-spin text-blue-500" />
          ) : null}
          <ChevronDown
            className={cn(
              "size-4 text-zinc-400 transition-transform",
              isExpanded && "rotate-180"
            )}
          />
        </div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="space-y-3 border-t border-zinc-200 p-3 text-xs dark:border-zinc-800">
              {toolCall.thinking && (
                <div>
                  <button
                    className="mb-1 flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowThinking((prev) => !prev);
                    }}
                  >
                    <ChevronDown
                      className={cn(
                        "size-3 transition-transform",
                        showThinking && "rotate-180"
                      )}
                    />
                    Thinking
                  </button>

                  <AnimatePresence>
                    {showThinking && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className={cn(
                          "rounded border-l-2 pl-3",
                          isRunning
                            ? "border-blue-400 bg-blue-50/60 dark:border-blue-500 dark:bg-blue-950/40"
                            : "border-zinc-300 bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800/60"
                        )}
                      >
                        <p className="whitespace-pre-wrap py-2 font-mono text-[11px] text-zinc-700 dark:text-zinc-300">
                          {toolCall.thinking}
                          {isRunning && (
                            <span className="ml-1 inline-block h-4 w-1 bg-blue-500 animate-pulse" />
                          )}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              {toolCall.type === "code_executor" && "code" in toolCall && (
                <div className="space-y-1">
                  <p className="text-[11px] uppercase tracking-wide text-zinc-500">Code</p>
                  <CodeBlock code={(toolCall as any).code ?? ""} language="python" />
                </div>
              )}

              {toolCall.result && (
                <div className="space-y-1">
                  <p className="text-[11px] uppercase tracking-wide text-zinc-500">Result</p>
                  <div className="max-h-48 overflow-auto rounded-md bg-zinc-100 p-2 font-mono text-[11px] text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200">
                    {typeof toolCall.result === "string"
                      ? toolCall.result
                      : JSON.stringify(toolCall.result, null, 2)}
                  </div>
                </div>
              )}

              {toolCall.error && (
                <div className="rounded-md bg-red-100 px-2 py-1 text-red-700 dark:bg-red-900/30 dark:text-red-200">
                  {toolCall.error}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
