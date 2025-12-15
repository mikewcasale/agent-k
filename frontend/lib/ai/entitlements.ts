import type { UserType } from "@/app/(auth)/auth";
import type { ChatModel } from "./models";

// =============================================================================
// Types
// =============================================================================

export type RateLimitConfig = {
  perMinute: number;
  perHour: number;
};

export type InputLimits = {
  maxCharsPerMessage: number;
  maxPartsPerMessage: number;
  maxAttachmentsPerMessage: number;
  maxAttachmentSizeBytes: number;
  maxEstimatedTokensPerMessage: number;
};

export type TokenBudget = {
  dailyTokenLimit: number;
  dailyCostLimitCents: number;
  monthlyTokenLimit: number;
  monthlyCostLimitCents: number;
};

export type Entitlements = {
  // Message limits
  maxMessagesPerDay: number;
  availableChatModelIds: ChatModel["id"][];

  // Rate limiting
  rateLimit: RateLimitConfig;

  // Input validation
  inputLimits: InputLimits;

  // Token/cost budgets
  tokenBudget: TokenBudget;

  // Concurrent requests
  maxConcurrentRequests: number;
};

// =============================================================================
// Entitlements by User Type
// =============================================================================

export const entitlementsByUserType: Record<UserType, Entitlements> = {
  /**
   * For users without an account (guests)
   */
  guest: {
    maxMessagesPerDay: 20,
    availableChatModelIds: ["chat-model", "chat-model-reasoning", "agent-k", "devstral-local"],

    rateLimit: {
      perMinute: 5,
      perHour: 15,
    },

    inputLimits: {
      maxCharsPerMessage: 10_000,
      maxPartsPerMessage: 5,
      maxAttachmentsPerMessage: 2,
      maxAttachmentSizeBytes: 2 * 1024 * 1024, // 2MB
      maxEstimatedTokensPerMessage: 4000,
    },

    tokenBudget: {
      dailyTokenLimit: 100_000,
      dailyCostLimitCents: 50, // $0.50
      monthlyTokenLimit: 1_000_000,
      monthlyCostLimitCents: 500, // $5.00
    },

    maxConcurrentRequests: 1,
  },

  /**
   * For users with an account (registered users)
   */
  regular: {
    maxMessagesPerDay: 100,
    availableChatModelIds: ["chat-model", "chat-model-reasoning", "agent-k", "devstral-local"],

    rateLimit: {
      perMinute: 10,
      perHour: 50,
    },

    inputLimits: {
      maxCharsPerMessage: 50_000,
      maxPartsPerMessage: 10,
      maxAttachmentsPerMessage: 5,
      maxAttachmentSizeBytes: 10 * 1024 * 1024, // 10MB
      maxEstimatedTokensPerMessage: 16_000,
    },

    tokenBudget: {
      dailyTokenLimit: 500_000,
      dailyCostLimitCents: 500, // $5.00
      monthlyTokenLimit: 10_000_000,
      monthlyCostLimitCents: 5000, // $50.00
    },

    maxConcurrentRequests: 2,
  },
};

// =============================================================================
// Premium Tier (Future Use)
// =============================================================================

/**
 * Premium tier entitlements for paid users.
 * Uncomment and add to entitlementsByUserType when implementing subscriptions.
 */
export const premiumEntitlements: Entitlements = {
  maxMessagesPerDay: 1000,
  availableChatModelIds: ["chat-model", "chat-model-reasoning", "agent-k", "devstral-local"],

  rateLimit: {
    perMinute: 30,
    perHour: 200,
  },

  inputLimits: {
    maxCharsPerMessage: 100_000,
    maxPartsPerMessage: 20,
    maxAttachmentsPerMessage: 10,
    maxAttachmentSizeBytes: 50 * 1024 * 1024, // 50MB
    maxEstimatedTokensPerMessage: 32_000,
  },

  tokenBudget: {
    dailyTokenLimit: 5_000_000,
    dailyCostLimitCents: 5000, // $50.00
    monthlyTokenLimit: 100_000_000,
    monthlyCostLimitCents: 50_000, // $500.00
  },

  maxConcurrentRequests: 5,
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get entitlements for a user type.
 */
export function getEntitlements(userType: UserType): Entitlements {
  return entitlementsByUserType[userType];
}

/**
 * Check if a model is available for a user type.
 */
export function isModelAvailable(
  modelId: ChatModel["id"],
  userType: UserType
): boolean {
  const entitlements = getEntitlements(userType);
  return entitlements.availableChatModelIds.includes(modelId);
}
