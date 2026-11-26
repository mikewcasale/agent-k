import { auth } from "@/app/(auth)/auth";
import {
  normalizeAgentKBackendPayload,
  readAgentKBackendPayload,
} from "@/lib/agent-k-api";
import { agentKAuthorization } from "@/lib/agent-k-auth";
import { PYTHON_BACKEND_URL } from "@/lib/constants";
import { ChatSDKError } from "@/lib/errors";

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
      headers: {
        "Content-Type": "application/json",
        Authorization: agentKAuthorization(session.user.id),
      },
      body: JSON.stringify(body),
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
