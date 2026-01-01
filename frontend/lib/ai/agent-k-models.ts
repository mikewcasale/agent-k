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
    id: "openrouter:mistralai/devstral-2512:free",
    label: "Devstral 2512",
    description: "Fast code mutations.",
    provider: "OpenRouter",
    icon: "code",
    freeTier: true,
  },
  {
    id: "openrouter:kwaipilot/kat-coder-pro:free",
    label: "KAT Coder Pro",
    description: "Free-tier code specialist.",
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
  "anthropic:claude-sonnet-4-5",
];
