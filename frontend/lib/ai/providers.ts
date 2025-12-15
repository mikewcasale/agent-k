import { anthropic } from "@ai-sdk/anthropic";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  customProvider,
  extractReasoningMiddleware,
  wrapLanguageModel,
} from "ai";
import { isTestEnvironment } from "../constants";

// Use OpenAI-compatible provider for local LM Studio server
const devstralProvider = createOpenAICompatible({
  name: "lmstudio",
  baseURL: process.env.DEVSTRAL_BASE_URL ?? "http://192.168.105.1:1234/v1",
});

// Create the devstral model
const devstralModel = devstralProvider.chatModel("mistralai/devstral-small-2-2512");

export const myProvider = isTestEnvironment
  ? (() => {
      const {
        artifactModel,
        chatModel,
        reasoningModel,
        titleModel,
      } = require("./models.mock");
      return customProvider({
        languageModels: {
          "chat-model": chatModel,
          "chat-model-reasoning": reasoningModel,
          "title-model": titleModel,
          "artifact-model": artifactModel,
          "devstral-local": chatModel,
        },
      });
    })()
  : customProvider({
      languageModels: {
        "chat-model": anthropic("claude-sonnet-4-20250514"),
        "chat-model-reasoning": wrapLanguageModel({
          model: anthropic("claude-sonnet-4-20250514"),
          middleware: extractReasoningMiddleware({ tagName: "thinking" }),
        }),
        "title-model": anthropic("claude-sonnet-4-20250514"),
        "artifact-model": anthropic("claude-sonnet-4-20250514"),
        "devstral-local": devstralModel,
      },
    });
