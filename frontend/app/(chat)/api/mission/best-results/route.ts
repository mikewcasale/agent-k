import type { NextRequest } from "next/server";
import { auth } from "@/app/(auth)/auth";
import {
  normalizeAgentKBackendPayload,
  readAgentKBackendPayload,
} from "@/lib/agent-k-api";
import { agentKAuthorization } from "@/lib/agent-k-auth";
import { PYTHON_BACKEND_URL } from "@/lib/constants";
import { ChatSDKError } from "@/lib/errors";

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
      headers: {
        "Content-Type": "application/json",
        Authorization: agentKAuthorization(session.user.id),
      },
    });

    const payload = normalizeAgentKBackendPayload(
      await readAgentKBackendPayload(backendResponse)
    );

    return Response.json(payload, { status: backendResponse.status });
  } catch {
    return Response.json(
      { error: "Unable to reach the Agent-K backend." },
      { status: 503 }
    );
  }
}
