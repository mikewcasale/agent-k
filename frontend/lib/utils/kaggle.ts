import type { CompetitionInfo } from "@/lib/types/agent-k";

const RULES_SUFFIX_REGEX = /\/rules\/?$/;
const TRAILING_SLASH_REGEX = /\/$/;

export function buildCompetitionRulesUrl(
  competition?: CompetitionInfo | null,
  competitionId?: string | null
) {
  if (competition?.url) {
    try {
      const parsed = new URL(competition.url);
      const cleanedPath = parsed.pathname
        .replace(RULES_SUFFIX_REGEX, "")
        .replace(TRAILING_SLASH_REGEX, "");
      return `${parsed.origin}${cleanedPath}/rules`;
    } catch {
      // Fall back to the canonical rules path.
    }
  }
  const id = competition?.id ?? competitionId ?? "";
  if (id) {
    return `https://www.kaggle.com/competitions/${id}/rules`;
  }
  return "https://www.kaggle.com/competitions";
}

export function isRulesAcceptanceError(message?: string | null) {
  if (!message) {
    return false;
  }
  const normalized = message.toLowerCase();
  return (
    (normalized.includes("accept") && normalized.includes("rules")) ||
    normalized.includes("competitions.userservice/acceptrules") ||
    normalized.includes("competitions.participate")
  );
}
