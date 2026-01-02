import type { LeaderboardSubmission, MissionState } from "@/lib/types/agent-k";

export type BestResultEntry = {
  competitionId: string;
  competitionTitle: string;
  competitionUrl?: string;
  submissionId?: string;
  rank?: number;
  totalTeams?: number;
  percentile?: number;
  score?: number;
  submittedAt?: string;
  category?: string;
  recordedAt: string;
};

export const BEST_RESULTS_STORAGE_KEY = "agentk-best-results";

const DAY_MS = 24 * 60 * 60 * 1000;

function isBestResultEntry(value: unknown): value is BestResultEntry {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as BestResultEntry;
  return Boolean(
    record.competitionId &&
      record.competitionTitle &&
      record.recordedAt &&
      typeof record.competitionId === "string" &&
      typeof record.competitionTitle === "string" &&
      typeof record.recordedAt === "string"
  );
}

function isBetterResult(candidate: BestResultEntry, current: BestResultEntry) {
  if (candidate.rank != null && current.rank == null) {
    return true;
  }

  if (candidate.rank == null && current.rank != null) {
    return false;
  }

  if (candidate.rank != null && current.rank != null) {
    return candidate.rank < current.rank;
  }

  if (candidate.percentile != null && current.percentile == null) {
    return true;
  }

  if (candidate.percentile == null && current.percentile != null) {
    return false;
  }

  if (candidate.percentile != null && current.percentile != null) {
    return candidate.percentile < current.percentile;
  }

  if (candidate.score != null && current.score == null) {
    return true;
  }

  if (candidate.score == null && current.score != null) {
    return false;
  }

  if (candidate.score != null && current.score != null) {
    return candidate.score > current.score;
  }

  const candidateTime = Date.parse(candidate.recordedAt);
  const currentTime = Date.parse(current.recordedAt);

  if (!Number.isNaN(candidateTime) && !Number.isNaN(currentTime)) {
    return candidateTime > currentTime;
  }

  return false;
}

export function loadBestResults(): BestResultEntry[] {
  if (typeof window === "undefined") {
    return [];
  }

  const raw = window.localStorage.getItem(BEST_RESULTS_STORAGE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter(isBestResultEntry);
  } catch {
    return [];
  }
}

export function saveBestResults(results: BestResultEntry[]): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(
    BEST_RESULTS_STORAGE_KEY,
    JSON.stringify(results)
  );
}

export function upsertBestResult(entry: BestResultEntry): BestResultEntry[] {
  const current = loadBestResults();
  const index = current.findIndex(
    (result) => result.competitionId === entry.competitionId
  );

  let next = current.slice();
  if (index === -1) {
    next = [entry, ...next];
  } else if (isBetterResult(entry, current[index])) {
    next[index] = { ...current[index], ...entry };
  }

  next = sortBestResults(next);
  saveBestResults(next);
  return next;
}

export function mergeBestResults(
  baseResults: BestResultEntry[],
  incomingResults: BestResultEntry[]
): BestResultEntry[] {
  const merged = new Map<string, BestResultEntry>();

  for (const entry of baseResults) {
    if (isBestResultEntry(entry)) {
      merged.set(entry.competitionId, entry);
    }
  }

  for (const entry of incomingResults) {
    if (!isBestResultEntry(entry)) {
      continue;
    }
    const existing = merged.get(entry.competitionId);
    if (!existing || isBetterResult(entry, existing)) {
      merged.set(entry.competitionId, entry);
    }
  }

  return sortBestResults(Array.from(merged.values()));
}

function selectBestSubmission(
  submissions: LeaderboardSubmission[]
): LeaderboardSubmission | null {
  if (!submissions.length) {
    return null;
  }

  return submissions.reduce<LeaderboardSubmission>((best, current) => {
    if (best.rank == null && current.rank != null) {
      return current;
    }

    if (best.rank != null && current.rank == null) {
      return best;
    }

    if (best.rank != null && current.rank != null) {
      return current.rank < best.rank ? current : best;
    }

    if (best.percentile == null && current.percentile != null) {
      return current;
    }

    if (best.percentile != null && current.percentile == null) {
      return best;
    }

    if (best.percentile != null && current.percentile != null) {
      return current.percentile < best.percentile ? current : best;
    }

    if (best.publicScore == null && current.publicScore != null) {
      return current;
    }

    if (best.publicScore != null && current.publicScore == null) {
      return best;
    }

    if (best.publicScore != null && current.publicScore != null) {
      return current.publicScore > best.publicScore ? current : best;
    }

    if (best.cvScore !== current.cvScore) {
      return current.cvScore > best.cvScore ? current : best;
    }

    const bestTime = Date.parse(best.submittedAt);
    const currentTime = Date.parse(current.submittedAt);
    if (!Number.isNaN(bestTime) && !Number.isNaN(currentTime)) {
      return currentTime > bestTime ? current : best;
    }

    return best;
  }, submissions[0]);
}

function deriveCategory(mission: MissionState): string | undefined {
  const tags = mission.competition?.tags ?? [];
  if (tags.length) {
    return tags[0];
  }
  return mission.competition?.competitionType;
}

export function bestResultFromMission(
  mission: MissionState
): BestResultEntry | null {
  if (!mission.competition) {
    return null;
  }

  const submissions = mission.evolution?.leaderboardSubmissions ?? [];
  const bestSubmission = selectBestSubmission(submissions);
  const result = mission.result;

  if (!bestSubmission && !result) {
    return null;
  }

  const recordedAt =
    bestSubmission?.submittedAt ??
    mission.startedAt ??
    new Date().toISOString();

  const rank = bestSubmission?.rank ?? result?.finalRank;
  const score =
    bestSubmission?.publicScore ??
    bestSubmission?.cvScore ??
    result?.finalScore;

  return {
    competitionId: mission.competition.id,
    competitionTitle: mission.competition.title,
    competitionUrl: mission.competition.url,
    submissionId: bestSubmission?.submissionId,
    rank,
    totalTeams: bestSubmission?.totalTeams,
    percentile: bestSubmission?.percentile,
    score,
    submittedAt: bestSubmission?.submittedAt,
    category: deriveCategory(mission),
    recordedAt,
  };
}

export function sortBestResults(results: BestResultEntry[]): BestResultEntry[] {
  return results.slice().sort((a, b) => {
    if (a.rank != null && b.rank != null) {
      return a.rank - b.rank;
    }

    if (a.rank != null) {
      return -1;
    }

    if (b.rank != null) {
      return 1;
    }

    const aTime = Date.parse(a.recordedAt);
    const bTime = Date.parse(b.recordedAt);

    if (!Number.isNaN(aTime) && !Number.isNaN(bTime)) {
      return bTime - aTime;
    }

    const aAge = Math.abs(Date.now() - aTime);
    const bAge = Math.abs(Date.now() - bTime);

    if (!Number.isNaN(aAge) && !Number.isNaN(bAge)) {
      return aAge - bAge;
    }

    return 0;
  });
}

export function formatResultAge(timestamp?: string): string {
  if (!timestamp) {
    return "-";
  }

  const target = Date.parse(timestamp);
  if (Number.isNaN(target)) {
    return "-";
  }

  const diff = Date.now() - target;
  const absDiff = Math.abs(diff);

  if (absDiff < DAY_MS) {
    return "Today";
  }

  const days = Math.round(absDiff / DAY_MS);
  if (days < 7) {
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }

  if (days < 30) {
    const weeks = Math.round(days / 7);
    return `${weeks} week${weeks === 1 ? "" : "s"} ago`;
  }

  const months = Math.round(days / 30);
  return `${months} month${months === 1 ? "" : "s"} ago`;
}
