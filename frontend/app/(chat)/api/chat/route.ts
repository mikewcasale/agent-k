import { geolocation } from "@vercel/functions";
import {
  convertToModelMessages,
  createUIMessageStream,
  JsonToSseTransformStream,
  smoothStream,
  stepCountIs,
  streamText,
} from "ai";
import { unstable_cache as cache } from "next/cache";
import { after } from "next/server";
import {
  createResumableStreamContext,
  type ResumableStreamContext,
} from "resumable-stream";
import type { ModelCatalog } from "tokenlens/core";
import { fetchModels } from "tokenlens/fetch";
import { getUsage } from "tokenlens/helpers";
import { auth, type UserType } from "@/app/(auth)/auth";
import type { VisibilityType } from "@/components/visibility-selector";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import type { ChatModel } from "@/lib/ai/models";
import { type RequestHints, systemPrompt } from "@/lib/ai/prompts";
import { myProvider } from "@/lib/ai/providers";
import { createDocument } from "@/lib/ai/tools/create-document";
import { getWeather } from "@/lib/ai/tools/get-weather";
import { requestSuggestions } from "@/lib/ai/tools/request-suggestions";
import { updateDocument } from "@/lib/ai/tools/update-document";
import {
  acquireRequestSlot,
  checkConcurrentLimit,
} from "@/lib/concurrent-limit";
import { isProductionEnvironment, PYTHON_BACKEND_URL } from "@/lib/constants";
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
  updateChatLastContextById,
} from "@/lib/db/queries";
import type { DBMessage } from "@/lib/db/schema";
import { checkTokenBudget, incrementTokenUsage } from "@/lib/db/token-usage";
import { ChatSDKError } from "@/lib/errors";
import { checkRateLimit, recordRequest } from "@/lib/rate-limit";
import type { ChatMessage } from "@/lib/types";
import type { AppUsage } from "@/lib/usage";
import { convertToUIMessages, generateUUID } from "@/lib/utils";
import { validateMessage } from "@/lib/validation";
import { generateTitleFromUserMessage } from "../../actions";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

export const maxDuration = 60;

let globalStreamContext: ResumableStreamContext | null = null;

/**
 * Transform Agent K backend SSE stream to Vercel AI SDK format.
 */
function transformAgentKStream(
 inputStream: ReadableStream<Uint8Array>
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  const agentKEventTypes = new Set([
    "state-snapshot",
    "state-delta",
    "phase-start",
    "phase-complete",
    "phase-error",
    "task-start",
    "task-progress",
    "task-complete",
    "task-error",
    "tool-start",
    "tool-thinking",
    "tool-result",
    "tool-error",
    "generation-start",
    "generation-complete",
    "fitness-update",
    "submission-result",
    "convergence-detected",
    "memory-store",
    "memory-retrieve",
    "checkpoint-created",
    "error-occurred",
    "recovery-attempt",
    "recovery-complete",
  ]);

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = inputStream.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.enqueue(encoder.encode(`d:{"finishReason":"stop"}\n`));
            controller.close();
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") continue;

            try {
              const event = JSON.parse(data);

              if (event.type === "text-delta" && event.textDelta) {
                const escaped = event.textDelta.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
                controller.enqueue(encoder.encode(`0:"${escaped}"\n`));
                continue;
              }

              if (agentKEventTypes.has(event.type) && event.data !== undefined) {
                const dp = JSON.stringify({
                  type: event.type,
                  data: event.data,
                  timestamp: event.timestamp,
                });
                controller.enqueue(encoder.encode(`8:${dp}\n`));
                continue;
              }

              if (event.type === "tool-output-available" && event.output) {
                const output = event.output;
                if (output.type === "state-snapshot") {
                  const dp = JSON.stringify({ type: "state-snapshot", data: output.snapshot });
                  controller.enqueue(encoder.encode(`8:${dp}\n`));
                } else if (output.type === "state-delta") {
                  const dp = JSON.stringify({ type: "state-delta", data: output.delta });
                  controller.enqueue(encoder.encode(`8:${dp}\n`));
                }
                continue;
              }

              if (agentKEventTypes.has(event.type)) {
                const dp = JSON.stringify({
                  type: event.type,
                  data: event.snapshot ?? event.delta ?? event,
                  timestamp: event.timestamp,
                });
                controller.enqueue(encoder.encode(`8:${dp}\n`));
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      } catch (error) {
        controller.error(error);
      }
    },
  });
}

/**
 * Handle requests for the Agent K (Pydantic AI) backend
 */
async function handleAgentKRequest({
  id,
  message,
  selectedVisibilityType,
  session,
}: {
  id: string;
  message: ChatMessage;
  selectedVisibilityType: VisibilityType;
  session: { user: { id: string } };
}) {
  // Save the chat if it doesn't exist
  const chat = await getChatById({ id });

  if (chat) {
    if (chat.userId !== session.user.id) {
      return new ChatSDKError("forbidden:chat").toResponse();
    }
  } else {
    // For Agent K, use a simple title from the message text
    // (avoids requiring Vercel AI Gateway for title generation)
    const firstTextPart = message.parts.find((p) => p.type === "text");
    const messageText =
      firstTextPart && "text" in firstTextPart ? firstTextPart.text : "";
    const title =
      messageText.slice(0, 50) + (messageText.length > 50 ? "..." : "") ||
      "Agent K Chat";
    await saveChat({
      id,
      userId: session.user.id,
      title,
      visibility: selectedVisibilityType,
    });
  }

  // Save the user message
  await saveMessages({
    messages: [
      {
        chatId: id,
        id: message.id,
        role: "user",
        parts: message.parts,
        attachments: [],
        createdAt: new Date(),
      },
    ],
  });

  // Forward to Python backend
  const backendResponse = await fetch(PYTHON_BACKEND_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      trigger: "submit-message",
      id,
      messages: [
        {
          id: message.id,
          role: message.role,
          parts: message.parts,
        },
      ],
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

  // Transform and stream the response back
  const transformedStream = transformAgentKStream(backendResponse.body!);
  return new Response(transformedStream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Vercel-AI-Data-Stream": "v1",
    },
  });
}

const getTokenlensCatalog = cache(
  async (): Promise<ModelCatalog | undefined> => {
    try {
      return await fetchModels();
    } catch (err) {
      console.warn(
        "TokenLens: catalog fetch failed, using default catalog",
        err
      );
      return; // tokenlens helpers will fall back to defaultCatalog
    }
  },
  ["tokenlens-catalog"],
  { revalidate: 24 * 60 * 60 } // 24 hours
);

export function getStreamContext() {
  if (!globalStreamContext) {
    try {
      globalStreamContext = createResumableStreamContext({
        waitUntil: after,
      });
    } catch (error: any) {
      if (error.message.includes("REDIS_URL")) {
        console.log(
          " > Resumable streams are disabled due to missing REDIS_URL"
        );
      } else {
        console.error(error);
      }
    }
  }

  return globalStreamContext;
}

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (error) {
    console.error("Schema validation error:", error);
    return new ChatSDKError("bad_request:api").toResponse();
  }

  try {
    const {
      id,
      message,
      selectedChatModel,
      selectedVisibilityType,
    }: {
      id: string;
      message: ChatMessage;
      selectedChatModel: ChatModel["id"];
      selectedVisibilityType: VisibilityType;
    } = requestBody;

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError("unauthorized:chat").toResponse();
    }

    const userType: UserType = session.user.type;
    const userId = session.user.id;
    const requestId = generateUUID();

    // =========================================================================
    // Defense Layer 1: Input Validation
    // =========================================================================
    const validationResult = validateMessage(message, userType);
    if (!validationResult.valid) {
      return new ChatSDKError("bad_request:input_size").toResponse();
    }

    // =========================================================================
    // Defense Layer 2: Rate Limiting (Burst Protection)
    // =========================================================================
    // NOTE: Rate limiting disabled for development
    // const rateLimitResult = await checkRateLimit(userId, userType);
    // if (!rateLimitResult.allowed) {
    //   return new ChatSDKError("rate_limit:burst").toResponse();
    // }

    // =========================================================================
    // Defense Layer 3: Daily Message Limit
    // =========================================================================
    // NOTE: Disabled for development
    // const messageCount = await getMessageCountByUserId({
    //   id: userId,
    //   differenceInHours: 24,
    // });

    // if (messageCount > entitlementsByUserType[userType].maxMessagesPerDay) {
    //   return new ChatSDKError("rate_limit:chat").toResponse();
    // }

    // =========================================================================
    // Defense Layer 4: Token Budget Check
    // =========================================================================
    // NOTE: Disabled for development
    // const tokenBudgetStatus = await checkTokenBudget(userId, userType);
    // if (!tokenBudgetStatus.withinBudget) {
    //   return new ChatSDKError("rate_limit:tokens").toResponse();
    // }

    // =========================================================================
    // Defense Layer 5: Concurrent Request Limit
    // =========================================================================
    // NOTE: Disabled for development
    // const concurrentResult = await checkConcurrentLimit(userId, userType);
    // if (!concurrentResult.allowed) {
    //   return new ChatSDKError("rate_limit:concurrent").toResponse();
    // }

    // Acquire request slot and get release function
    // NOTE: Disabled for development
    // const releaseSlot = await acquireRequestSlot(userId, requestId);
    const releaseSlot = () => Promise.resolve(); // No-op for development

    // Record this request for rate limiting
    // NOTE: Disabled for development
    // await recordRequest(userId);

    // Route to Python backend for Agent K model
    if (selectedChatModel === "agent-k") {
      try {
        return await handleAgentKRequest({ id, message, selectedVisibilityType, session });
      } finally {
        await releaseSlot();
      }
    }

    const chat = await getChatById({ id });
    let messagesFromDb: DBMessage[] = [];

    if (chat) {
      if (chat.userId !== session.user.id) {
        return new ChatSDKError("forbidden:chat").toResponse();
      }
      // Only fetch messages if chat already exists
      messagesFromDb = await getMessagesByChatId({ id });
    } else {
      const title = await generateTitleFromUserMessage({
        message,
      });

      await saveChat({
        id,
        userId: session.user.id,
        title,
        visibility: selectedVisibilityType,
      });
      // New chat - no need to fetch messages, it's empty
    }

    // Filter out empty text parts to prevent Anthropic API errors
    const sanitizedMessage = {
      ...message,
      parts: message.parts.filter(
        (part) => part.type !== "text" || (part.type === "text" && part.text.trim().length > 0)
      ),
    };

    const uiMessages = [...convertToUIMessages(messagesFromDb), sanitizedMessage];

    const { longitude, latitude, city, country } = geolocation(request);

    const requestHints: RequestHints = {
      longitude,
      latitude,
      city,
      country,
    };

    await saveMessages({
      messages: [
        {
          chatId: id,
          id: message.id,
          role: "user",
          parts: message.parts,
          attachments: [],
          createdAt: new Date(),
        },
      ],
    });

    const streamId = generateUUID();
    await createStreamId({ streamId, chatId: id });

    let finalMergedUsage: AppUsage | undefined;

    const stream = createUIMessageStream({
      execute: ({ writer: dataStream }) => {
        const result = streamText({
          model: myProvider.languageModel(selectedChatModel),
          system: systemPrompt({ selectedChatModel, requestHints }),
          messages: convertToModelMessages(uiMessages),
          stopWhen: stepCountIs(5),
          experimental_activeTools:
            selectedChatModel === "chat-model-reasoning"
              ? []
              : [
                  "getWeather",
                  "createDocument",
                  "updateDocument",
                  "requestSuggestions",
                ],
          experimental_transform: smoothStream({ chunking: "word" }),
          tools: {
            getWeather,
            createDocument: createDocument({ session, dataStream }),
            updateDocument: updateDocument({ session, dataStream }),
            requestSuggestions: requestSuggestions({
              session,
              dataStream,
            }),
          },
          experimental_telemetry: {
            isEnabled: isProductionEnvironment,
            functionId: "stream-text",
          },
          onFinish: async ({ usage }) => {
            try {
              // Track token usage for budget enforcement
              if (usage.inputTokens || usage.outputTokens) {
                await incrementTokenUsage(
                  userId,
                  usage.inputTokens ?? 0,
                  usage.outputTokens ?? 0
                );
              }

              const providers = await getTokenlensCatalog();
              const modelId =
                myProvider.languageModel(selectedChatModel).modelId;
              if (!modelId) {
                finalMergedUsage = usage;
                dataStream.write({
                  type: "data-usage",
                  data: finalMergedUsage,
                });
                return;
              }

              if (!providers) {
                finalMergedUsage = usage;
                dataStream.write({
                  type: "data-usage",
                  data: finalMergedUsage,
                });
                return;
              }

              const summary = getUsage({ modelId, usage, providers });
              finalMergedUsage = { ...usage, ...summary, modelId } as AppUsage;
              dataStream.write({ type: "data-usage", data: finalMergedUsage });
            } catch (err) {
              console.warn("TokenLens enrichment failed", err);
              finalMergedUsage = usage;
              dataStream.write({ type: "data-usage", data: finalMergedUsage });
            }
          },
        });

        result.consumeStream();

        dataStream.merge(
          result.toUIMessageStream({
            sendReasoning: true,
          })
        );
      },
      generateId: generateUUID,
      onFinish: async ({ messages }) => {
        // Release the concurrent request slot
        await releaseSlot();

        await saveMessages({
          messages: messages.map((currentMessage) => ({
            id: currentMessage.id,
            role: currentMessage.role,
            parts: currentMessage.parts,
            createdAt: new Date(),
            attachments: [],
            chatId: id,
          })),
        });

        if (finalMergedUsage) {
          try {
            await updateChatLastContextById({
              chatId: id,
              context: finalMergedUsage,
            });
          } catch (err) {
            console.warn("Unable to persist last usage for chat", id, err);
          }
        }
      },
      onError: () => {
        // Release slot on error too
        releaseSlot().catch(() => {});
        return "Oops, an error occurred!";
      },
    });

    // const streamContext = getStreamContext();

    // if (streamContext) {
    //   return new Response(
    //     await streamContext.resumableStream(streamId, () =>
    //       stream.pipeThrough(new JsonToSseTransformStream())
    //     )
    //   );
    // }

    return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
  } catch (error) {
    const vercelId = request.headers.get("x-vercel-id");

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    // Check for Vercel AI Gateway credit card error
    if (
      error instanceof Error &&
      error.message?.includes(
        "AI Gateway requires a valid credit card on file to service requests"
      )
    ) {
      return new ChatSDKError("bad_request:activate_gateway").toResponse();
    }

    console.error("Unhandled error in chat API:", error, { vercelId });
    return new ChatSDKError("offline:chat").toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== session.user.id) {
    return new ChatSDKError("forbidden:chat").toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
