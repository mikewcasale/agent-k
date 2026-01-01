import { auth } from "@/app/(auth)/auth";
import { PYTHON_BACKEND_URL } from "@/lib/constants";
import { ChatSDKError } from "@/lib/errors";

async function readBackendPayload(response: Response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return { error: text };
  }
}

function normalizeBackendPayload(payload: Record<string, unknown>) {
  if (payload.error) {
    return payload;
  }

  const detail = payload.detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string } | undefined;
    if (first?.msg) {
      return { error: first.msg };
    }
  }

  return payload;
}

export async function POST(request: Request) {
  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  try {
    const backendUrl = new URL("/api/competitions/search", PYTHON_BACKEND_URL);
    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const payload = normalizeBackendPayload(
      await readBackendPayload(backendResponse)
    );

    return Response.json(payload, { status: backendResponse.status });
  } catch {
    return Response.json(
      { error: "Unable to reach the Agent-K backend." },
      { status: 503 }
    );
  }
}
