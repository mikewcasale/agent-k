"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAgentKState } from "@/hooks/use-agent-k-state";

export function FitnessChart() {
  const { state } = useAgentKState();
  const evolution = state.mission.evolution;

  if (!evolution || !evolution.generationHistory.length) return null;

  const data = useMemo(() => {
    return evolution.generationHistory.map((gen) => ({
      generation: gen.generation,
      best: gen.bestFitness,
      mean: gen.meanFitness,
      worst: gen.worstFitness,
    }));
  }, [evolution.generationHistory]);

  const targetScore = state.mission.research?.leaderboardAnalysis?.targetScore;

  const allValues = data.flatMap((d) => [d.best, d.mean, d.worst]);
  const minValue = Math.min(...allValues, targetScore ?? Infinity);
  const maxValue = Math.max(...allValues, targetScore ?? -Infinity);
  const padding = (maxValue - minValue) * 0.1 || 1;

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            className="stroke-zinc-200 dark:stroke-zinc-700"
          />

          <XAxis
            dataKey="generation"
            tick={{ fontSize: 12 }}
            className="text-zinc-500"
            label={{ value: "Generation", position: "bottom", offset: 0 }}
          />

          <YAxis
            domain={[minValue - padding, maxValue + padding]}
            tick={{ fontSize: 12 }}
            className="text-zinc-500"
            tickFormatter={(value) => value.toFixed(3)}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: "var(--tooltip-bg, #fff)",
              border: "1px solid var(--tooltip-border, #e5e7eb)",
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(value: number) => value.toFixed(4)}
          />

          <Legend />

          {targetScore && (
            <ReferenceLine
              y={targetScore}
              stroke="#F59E0B"
              strokeDasharray="5 5"
              label={{
                value: `Target: ${targetScore.toFixed(4)}`,
                position: "right",
                fill: "#F59E0B",
                fontSize: 11,
              }}
            />
          )}

          <Line
            type="monotone"
            dataKey="best"
            stroke="#10B981"
            strokeWidth={2}
            dot={false}
            name="Best Fitness"
            animationDuration={300}
          />

          <Line
            type="monotone"
            dataKey="mean"
            stroke="#3B82F6"
            strokeWidth={1.5}
            dot={false}
            name="Mean Fitness"
            animationDuration={300}
          />

          <Line
            type="monotone"
            dataKey="worst"
            stroke="#EF4444"
            strokeWidth={1}
            strokeDasharray="3 3"
            dot={false}
            name="Worst Fitness"
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
