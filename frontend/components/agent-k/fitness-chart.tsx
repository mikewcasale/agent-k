"use client";

import { useMemo } from "react";
import {
  CartesianGrid as RCartesianGrid,
  Legend as RLegend,
  Line as RLine,
  LineChart as RLineChart,
  ReferenceLine as RReferenceLine,
  ResponsiveContainer as RResponsiveContainer,
  Tooltip as RTooltip,
  XAxis as RXAxis,
  YAxis as RYAxis,
} from "recharts";
import { useAgentKState } from "@/hooks/use-agent-k-state";

// Workaround for React 19 type compatibility with recharts
// See: https://github.com/recharts/recharts/issues/3615
/* eslint-disable @typescript-eslint/no-explicit-any */
const CartesianGrid = RCartesianGrid as any;
const Legend = RLegend as any;
const Line = RLine as any;
const LineChart = RLineChart as any;
const ReferenceLine = RReferenceLine as any;
const ResponsiveContainer = RResponsiveContainer as any;
const Tooltip = RTooltip as any;
const XAxis = RXAxis as any;
const YAxis = RYAxis as any;
/* eslint-enable @typescript-eslint/no-explicit-any */

export function FitnessChart() {
  const { state } = useAgentKState();
  const evolution = state.mission.evolution;

  const data = useMemo(() => {
    if (!evolution) {
      return [];
    }
    return evolution.generationHistory.map((gen) => ({
      generation: gen.generation,
      best: gen.bestFitness,
      mean: gen.meanFitness,
      worst: gen.worstFitness,
    }));
  }, [evolution]);

  if (!evolution || !evolution.generationHistory.length) {
    return null;
  }

  const targetScore = state.mission.research?.leaderboardAnalysis?.targetScore;

  const allValues = data.flatMap((d) => [d.best, d.mean, d.worst]);
  const minValue = Math.min(
    ...allValues,
    targetScore ?? Number.POSITIVE_INFINITY
  );
  const maxValue = Math.max(
    ...allValues,
    targetScore ?? Number.NEGATIVE_INFINITY
  );
  const padding = (maxValue - minValue) * 0.1 || 1;

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer>
        <LineChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid
            className="stroke-zinc-200 dark:stroke-zinc-700"
            strokeDasharray="3 3"
          />

          <XAxis
            className="text-zinc-500"
            dataKey="generation"
            label={{ value: "Generation", position: "bottom", offset: 0 }}
            tick={{ fontSize: 12 }}
          />

          <YAxis
            className="text-zinc-500"
            domain={[minValue - padding, maxValue + padding]}
            tick={{ fontSize: 12 }}
            tickFormatter={(value: number) => value.toFixed(3)}
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
              label={{
                value: `Target: ${targetScore.toFixed(4)}`,
                position: "right",
                fill: "#F59E0B",
                fontSize: 11,
              }}
              stroke="#F59E0B"
              strokeDasharray="5 5"
              y={targetScore}
            />
          )}

          <Line
            animationDuration={300}
            dataKey="best"
            dot={false}
            name="Best Fitness"
            stroke="#10B981"
            strokeWidth={2}
            type="monotone"
          />

          <Line
            animationDuration={300}
            dataKey="mean"
            dot={false}
            name="Mean Fitness"
            stroke="#3B82F6"
            strokeWidth={1.5}
            type="monotone"
          />

          <Line
            animationDuration={300}
            dataKey="worst"
            dot={false}
            name="Worst Fitness"
            stroke="#EF4444"
            strokeDasharray="3 3"
            strokeWidth={1}
            type="monotone"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
