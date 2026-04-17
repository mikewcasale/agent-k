export type EvolutionModelOption = {
  id: string;
  label: string;
  description: string;
  provider?: string;
  icon?: "sparkles" | "brain" | "code" | "terminal";
  freeTier?: boolean;
};

export const evolutionModels: EvolutionModelOption[] = [
  {
    id: "anthropic:claude-sonnet-4-5",
    label: "Claude Sonnet 4.5",
    description: "High-accuracy generalist for complex strategy.",
    provider: "Anthropic",
    icon: "sparkles",
  },
  {
    id: "openrouter:openai/gpt-5.2",
    label: "GPT-5.2",
    description: "Frontier reasoning model.",
    provider: "OpenRouter",
    icon: "brain",
  },
  {
    id: "openrouter:openai/gpt-oss-120b:free",
    label: "GPT-OSS 120B",
    description: "Free-tier large open model with reliable function calling.",
    provider: "OpenRouter",
    icon: "code",
    freeTier: true,
  },
  {
    id: "openrouter:openai/gpt-oss-20b:free",
    label: "GPT-OSS 20B",
    description: "Smaller free open model for fast mutations.",
    provider: "OpenRouter",
    icon: "code",
    freeTier: true,
  },
  {
    id: "devstral:local",
    label: "Devstral Local",
    description: "Low-latency iterations.",
    provider: "Local Endpoint",
    icon: "terminal",
  },
];

export const DEFAULT_EVOLUTION_MODELS: string[] = [
  "openrouter:openai/gpt-oss-120b:free",
  "openrouter:openai/gpt-oss-20b:free",
];
