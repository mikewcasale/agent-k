import { auth } from "@/app/(auth)/auth";
import { agentKAuthorization } from "@/lib/agent-k-auth";
import { PYTHON_BACKEND_URL } from "@/lib/constants";
import { ChatSDKError } from "@/lib/errors";

const CONTROL_ACTIONS = new Set(["abort", "pause", "resume"]);

export async function POST(
  _request: Request,
  { params }: { params: Promise<{ missionId: string; action: string }> }
) {
  const { missionId, action } = await params;
  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }
  if (!CONTROL_ACTIONS.has(action)) {
    return Response.json(
      { error: "Unsupported mission control." },
      { status: 404 }
    );
  }

  try {
    const backendUrl = new URL(
      `/api/mission/${encodeURIComponent(missionId)}/${action}`,
      PYTHON_BACKEND_URL
    );
    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      headers: {
        Authorization: agentKAuthorization(session.user.id),
      },
    });
    const payload = await backendResponse.json().catch(() => ({
      error: "The Agent-K backend returned an invalid response.",
    }));
    return Response.json(payload, { status: backendResponse.status });
  } catch {
    return Response.json(
      { error: "Unable to reach the Agent-K backend." },
      { status: 503 }
    );
  }
}
