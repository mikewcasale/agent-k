import { z } from "zod";

/**
 * Maximum allowed characters per text part.
 * This is a hard limit enforced at schema level.
 * User-type-specific limits are enforced separately in validation.ts
 */
const MAX_TEXT_CHARS_HARD_LIMIT = 100_000;

/**
 * Maximum parts per message (hard limit).
 */
const MAX_PARTS_HARD_LIMIT = 20;

const textPartSchema = z
  .object({
    type: z.enum(["text"]),
    text: z.string().max(MAX_TEXT_CHARS_HARD_LIMIT),
  })
  .passthrough();

const filePartSchema = z
  .object({
    type: z.enum(["file"]),
    mediaType: z.enum(["image/jpeg", "image/png", "image/gif", "image/webp"]),
    name: z.string().min(1).max(255),
    url: z.string().url().max(10_000_000), // 10MB base64 URL limit
  })
  .passthrough();

const partSchema = z.union([textPartSchema, filePartSchema]);

export const postRequestBodySchema = z.object({
  id: z.string().uuid(),
  message: z
    .object({
      id: z.string().uuid(),
      role: z.enum(["user"]),
      parts: z.array(partSchema).max(MAX_PARTS_HARD_LIMIT),
    })
    .passthrough(),
  selectedChatModel: z.enum(["chat-model", "chat-model-reasoning", "agent-k", "devstral-local"]),
  selectedVisibilityType: z.enum(["public", "private"]),
});

export type PostRequestBody = z.infer<typeof postRequestBodySchema>;
