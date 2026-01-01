import type { NextRequest } from "next/server";
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

export async function GET(request: NextRequest) {
  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  try {
    const backendUrl = new URL("/api/mission/best-results", PYTHON_BACKEND_URL);
    const limit = request.nextUrl.searchParams.get("limit");
    if (limit) {
      backendUrl.searchParams.set("limit", limit);
    }

    const backendResponse = await fetch(backendUrl, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
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
