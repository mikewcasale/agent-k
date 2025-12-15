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
    id: "agent-k",
    name: "Agent K (Pydantic AI)",
    description:
      "Python-powered AI agent using Pydantic AI with Claude Sonnet",
  },
];
