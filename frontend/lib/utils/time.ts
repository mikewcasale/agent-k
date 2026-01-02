const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;

export function formatDuration(durationMs?: number): string {
  if (durationMs === undefined || Number.isNaN(durationMs)) {
    return "—";
  }
  if (durationMs < SECOND) {
    return `${durationMs} ms`;
  }

  const hours = Math.floor(durationMs / HOUR);
  const minutes = Math.floor((durationMs % HOUR) / MINUTE);
  const seconds = Math.floor((durationMs % MINUTE) / SECOND);

  const parts = [];
  if (hours) parts.push(`${hours}h`);
  if (minutes) parts.push(`${minutes}m`);
  if (seconds || (!hours && !minutes)) parts.push(`${seconds}s`);
  return parts.join(" ");
}

export function formatRelativeTime(timestamp?: string): string {
  if (!timestamp) {
    return "—";
  }
  const target = new Date(timestamp).getTime();
  if (Number.isNaN(target)) {
    return "—";
  }

  const diff = target - Date.now();
  const absDiff = Math.abs(diff);

  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

  if (absDiff >= HOUR) {
    const hours = Math.round(diff / HOUR);
    return formatter.format(hours, "hour");
  }

  if (absDiff >= MINUTE) {
    const minutes = Math.round(diff / MINUTE);
    return formatter.format(minutes, "minute");
  }

  const seconds = Math.round(diff / SECOND);
  return formatter.format(seconds, "second");
}

export function formatDateTime(timestamp?: string): string {
  if (!timestamp) {
    return "—";
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return date.toLocaleString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric",
  });
}
