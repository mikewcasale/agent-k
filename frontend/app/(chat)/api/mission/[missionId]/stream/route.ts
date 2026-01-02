import { auth } from "@/app/(auth)/auth";
import { PYTHON_BACKEND_URL } from "@/lib/constants";
import { ChatSDKError } from "@/lib/errors";

export const maxDuration = 60;

export async function GET(
  _request: Request,
  { params }: { params: { missionId: string } }
) {
  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  try {
    const backendUrl = new URL(
      `/api/mission/${params.missionId}/stream`,
      PYTHON_BACKEND_URL
    );
    const backendResponse = await fetch(backendUrl, {
      headers: { Accept: "text/event-stream" },
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      return Response.json(
        { error: errorText || "Unable to reach the Agent-K backend." },
        { status: backendResponse.status }
      );
    }

    return new Response(backendResponse.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return Response.json(
      { error: "Unable to reach the Agent-K backend." },
      { status: 503 }
    );
  }
}
