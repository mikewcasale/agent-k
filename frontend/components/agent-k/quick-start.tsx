"use client";

import { motion } from "framer-motion";
import {
  Award,
  Calendar,
  Code2,
  Database,
  FlaskConical,
  Layers,
  Search,
  Sparkles,
  Target,
} from "lucide-react";

const domains = [
  { id: "cv", label: "Computer Vision", icon: Layers },
  { id: "nlp", label: "NLP", icon: Code2 },
  { id: "tabular", label: "Tabular", icon: Database },
  { id: "timeseries", label: "Time Series", icon: FlaskConical },
];

const competitionTypes = [
  { id: "featured", label: "Featured", color: "blue" },
  { id: "research", label: "Research", color: "violet" },
  { id: "playground", label: "Playground", color: "emerald" },
  { id: "getting-started", label: "Getting Started", color: "amber" },
];

type QuickStartProps = {
  onStartMission: (criteria: MissionCriteria) => void;
};

type MissionCriteria = {
  domains: string[];
  competitionTypes: string[];
  minPrize: number;
  minDaysRemaining: number;
};

export function QuickStart({ onStartMission }: QuickStartProps) {
  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto mt-8 w-full max-w-2xl rounded-2xl border border-zinc-200 bg-gradient-to-br from-white via-zinc-50 to-blue-50/50 p-6 shadow-sm dark:border-zinc-800 dark:from-zinc-950 dark:via-zinc-900 dark:to-blue-950/20"
      initial={{ opacity: 0, y: 20 }}
      transition={{ delay: 0.8 }}
    >
      <div className="mb-4 flex items-center gap-2">
        <Sparkles className="size-5 text-blue-500" />
        <h3 className="font-semibold text-zinc-900 dark:text-white">
          Quick Mission Setup
        </h3>
      </div>

      <p className="mb-6 text-sm text-zinc-500">
        Configure your mission criteria or use the chat to describe what
        you&apos;re looking for.
      </p>

      {/* Domain Selection */}
      <div className="mb-5">
        <label className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-600 dark:text-zinc-400">
          <Target className="size-3.5" />
          Target Domains
        </label>
        <div className="flex flex-wrap gap-2">
          {domains.map((domain) => (
            <button
              className="flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:border-blue-300 hover:bg-blue-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:border-blue-600 dark:hover:bg-blue-950/30"
              key={domain.id}
              type="button"
            >
              <domain.icon className="size-3.5" />
              {domain.label}
            </button>
          ))}
        </div>
      </div>

      {/* Competition Type */}
      <div className="mb-5">
        <label className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-600 dark:text-zinc-400">
          <Award className="size-3.5" />
          Competition Types
        </label>
        <div className="flex flex-wrap gap-2">
          {competitionTypes.map((type) => (
            <button
              className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors hover:border-blue-300 hover:bg-blue-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:border-blue-600 dark:hover:bg-blue-950/30"
              key={type.id}
              type="button"
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      {/* Prize & Timeline */}
      <div className="mb-6 grid grid-cols-2 gap-4">
        <div>
          <label className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-600 dark:text-zinc-400">
            <Award className="size-3.5" />
            Min Prize
          </label>
          <select className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
            <option value="0">Any</option>
            <option value="1000">$1,000+</option>
            <option value="10000">$10,000+</option>
            <option value="50000">$50,000+</option>
            <option value="100000">$100,000+</option>
          </select>
        </div>
        <div>
          <label className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-600 dark:text-zinc-400">
            <Calendar className="size-3.5" />
            Min Days Left
          </label>
          <select className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
            <option value="0">Any</option>
            <option value="7">7+ days</option>
            <option value="14">14+ days</option>
            <option value="30">30+ days</option>
            <option value="60">60+ days</option>
          </select>
        </div>
      </div>

      {/* Start Button */}
      <button
        className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-4 py-3 font-medium text-white shadow-sm transition-all hover:from-blue-700 hover:to-violet-700 hover:shadow-md"
        type="button"
      >
        <Search className="size-4" />
        Start Discovery Mission
      </button>

      <p className="mt-3 text-center text-xs text-zinc-400">
        Or describe your criteria in the chat below
      </p>
    </motion.div>
  );
}

