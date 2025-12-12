/**
 * Proxy route for Agent K (Python Pydantic AI backend)
 *
 * This route forwards chat requests to the Python backend running on port 9000
 * and streams the response back to the client using the Vercel AI Data Stream Protocol.
 */

import { auth } from "@/app/(auth)/auth";
import { PYTHON_BACKEND_URL } from "@/lib/constants";
import { ChatSDKError } from "@/lib/errors";
import type { ChatMessage } from "@/lib/types";

export const maxDuration = 60;

type AgentKRequestBody = {
  id: string;
  message: ChatMessage;
};

/**
 * Convert frontend message format to Vercel AI protocol format
 */
function convertToVercelUIMessages(messages: ChatMessage[]) {
  return messages.map((msg) => ({
    id: msg.id,
    role: msg.role,
    parts: msg.parts,
  }));
}

export async function POST(request: Request) {
  let requestBody: AgentKRequestBody;

  try {
    const json = await request.json();
    requestBody = json;
  } catch {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  try {
    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError("unauthorized:chat").toResponse();
    }

    const { id, message } = requestBody;

    // Build messages array for the Vercel AI protocol
    // For now, we send just the current message
    // In a full implementation, you'd fetch history from the database
    const messages = [message];

    // Forward request to Python backend
    const backendResponse = await fetch(PYTHON_BACKEND_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        trigger: "submit-message",
        id,
        messages: convertToVercelUIMessages(messages),
      }),
    });

    if (!backendResponse.ok) {
      console.error(
        "Agent K backend error:",
        backendResponse.status,
        await backendResponse.text()
      );
      return new ChatSDKError("offline:chat").toResponse();
    }

    // Stream the response back to the client
    // The Python backend returns SSE format which we pass through directly
    return new Response(backendResponse.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Vercel-AI-Data-Stream": "v1",
        ...Object.fromEntries(
          [...backendResponse.headers.entries()].filter(
            ([key]) =>
              key.toLowerCase().startsWith("x-") ||
              key.toLowerCase() === "content-type"
          )
        ),
      },
    });
  } catch (error) {
    console.error("Agent K proxy error:", error);

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    return new ChatSDKError("offline:chat").toResponse();
  }
}

