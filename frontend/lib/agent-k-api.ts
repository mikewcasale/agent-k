import "server-only";

type BackendPayload = Record<string, unknown>;

function errorMessage(value: unknown): string | null {
  if (typeof value === "string") {
    return value.trim() || null;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  for (const key of ["message", "msg"] as const) {
    const message = Reflect.get(value, key);
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  return null;
}

/** Read one backend response without exposing a non-JSON body as structured data. */
export async function readAgentKBackendPayload(
  response: Response
): Promise<BackendPayload> {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    const payload: unknown = JSON.parse(text);
    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      return payload as BackendPayload;
    }
    return { error: "The Agent-K backend returned an invalid response." };
  } catch {
    return { error: "The Agent-K backend returned an invalid response." };
  }
}

/** Convert backend error envelopes to the frontend's stable string error contract. */
export function normalizeAgentKBackendPayload(
  payload: BackendPayload
): BackendPayload {
  if (payload.error !== undefined) {
    return {
      ...payload,
      error:
        errorMessage(payload.error) ?? "The Agent-K backend request failed.",
    };
  }

  const detail = payload.detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const message = errorMessage(detail[0]);
    if (message) {
      return { ...payload, error: message };
    }
  }
  const message = errorMessage(detail);
  return message ? { ...payload, error: message } : payload;
}
