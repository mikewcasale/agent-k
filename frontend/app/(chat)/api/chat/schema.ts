import { z } from "zod";

const textPartSchema = z
  .object({
    type: z.enum(["text"]),
    text: z.string().max(2000),
  })
  .passthrough();

const filePartSchema = z
  .object({
    type: z.enum(["file"]),
    mediaType: z.enum(["image/jpeg", "image/png"]),
    name: z.string().min(1).max(100),
    url: z.string().url(),
  })
  .passthrough();

const partSchema = z.union([textPartSchema, filePartSchema]);

export const postRequestBodySchema = z.object({
  id: z.string().uuid(),
  message: z
    .object({
      id: z.string().uuid(),
      role: z.enum(["user"]),
      parts: z.array(partSchema),
    })
    .passthrough(),
  selectedChatModel: z.enum(["chat-model", "chat-model-reasoning", "agent-k"]),
  selectedVisibilityType: z.enum(["public", "private"]),
});

export type PostRequestBody = z.infer<typeof postRequestBodySchema>;
