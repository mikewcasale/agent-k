export const DEFAULT_CHAT_MODEL: string = "chat-model";

export type ChatModel = {
  id: string;
  name: string;
  description: string;
};

export const chatModels: ChatModel[] = [
  {
    id: "chat-model",
    name: "Claude Sonnet",
    description: "Anthropic's Claude Sonnet - balanced intelligence and speed",
  },
  {
    id: "chat-model-reasoning",
    name: "Claude Reasoning",
    description:
      "Uses extended thinking for complex reasoning tasks",
  },
  {
    id: "anthropic-haiku",
    name: "Claude Haiku",
    description: "Fast and efficient for quick interactions",
  },
  {
    id: "devstral-local",
    name: "Devstral (Local)",
    description: "Mistral's Devstral coding model running locally via LM Studio",
  },
  {
    id: "openrouter-devstral",
    name: "Devstral (OpenRouter)",
    description: "Mistral's Devstral via OpenRouter - tool-capable coding model",
  },
];
