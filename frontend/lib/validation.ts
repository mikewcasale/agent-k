/**
 * Input validation utilities for API abuse prevention.
 *
 * Provides configurable limits for message sizes, parts, and attachments
 * with fast token estimation for cost-based validation.
 */

import type { UserType } from "@/app/(auth)/auth";

// =============================================================================
// Types
// =============================================================================

export type ValidationLimits = {
  maxCharsPerMessage: number;
  maxPartsPerMessage: number;
  maxAttachmentsPerMessage: number;
  maxAttachmentSizeBytes: number;
  maxEstimatedTokensPerMessage: number;
};

export type ValidationResult = {
  valid: boolean;
  error?: string;
  estimatedTokens?: number;
};

// =============================================================================
// Default Limits by User Type
// =============================================================================

export const validationLimitsByUserType: Record<UserType, ValidationLimits> = {
  guest: {
    maxCharsPerMessage: 10_000,
    maxPartsPerMessage: 5,
    maxAttachmentsPerMessage: 2,
    maxAttachmentSizeBytes: 2 * 1024 * 1024, // 2MB
    maxEstimatedTokensPerMessage: 4000,
  },
  regular: {
    maxCharsPerMessage: 50_000,
    maxPartsPerMessage: 10,
    maxAttachmentsPerMessage: 5,
    maxAttachmentSizeBytes: 10 * 1024 * 1024, // 10MB
    maxEstimatedTokensPerMessage: 16_000,
  },
};

// =============================================================================
// Token Estimation
// =============================================================================

/**
 * Fast token estimation using character count approximation.
 * Claude/GPT models average ~4 characters per token for English text.
 * This is a conservative estimate for rate limiting purposes.
 */
export function estimateTokenCount(text: string): number {
  // Average ~4 chars per token, but be conservative (use 3.5)
  return Math.ceil(text.length / 3.5);
}

/**
 * Estimate tokens for a message with multiple parts.
 */
export function estimateMessageTokens(
  parts: Array<{ type: string; text?: string }>
): number {
  let totalTokens = 0;

  for (const part of parts) {
    if (part.type === "text" && part.text) {
      totalTokens += estimateTokenCount(part.text);
    }
    // File parts add overhead for base64 encoding and metadata
    if (part.type === "file") {
      totalTokens += 100; // Estimated overhead per file
    }
  }

  return totalTokens;
}

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Validate a message against user-specific limits.
 */
export function validateMessage(
  message: {
    parts: Array<{ type: string; text?: string; url?: string }>;
  },
  userType: UserType
): ValidationResult {
  const limits = validationLimitsByUserType[userType];

  // Check number of parts
  if (message.parts.length > limits.maxPartsPerMessage) {
    return {
      valid: false,
      error: `Message exceeds maximum parts limit (${limits.maxPartsPerMessage})`,
    };
  }

  // Count attachments and check text length
  let totalChars = 0;
  let attachmentCount = 0;

  for (const part of message.parts) {
    if (part.type === "text" && part.text) {
      totalChars += part.text.length;
    }
    if (part.type === "file") {
      attachmentCount++;
    }
  }

  // Check total character count
  if (totalChars > limits.maxCharsPerMessage) {
    return {
      valid: false,
      error: `Message exceeds maximum character limit (${limits.maxCharsPerMessage.toLocaleString()} chars)`,
    };
  }

  // Check attachment count
  if (attachmentCount > limits.maxAttachmentsPerMessage) {
    return {
      valid: false,
      error: `Message exceeds maximum attachment limit (${limits.maxAttachmentsPerMessage})`,
    };
  }

  // Estimate tokens
  const estimatedTokens = estimateMessageTokens(message.parts);

  if (estimatedTokens > limits.maxEstimatedTokensPerMessage) {
    return {
      valid: false,
      error: `Message exceeds estimated token limit (${limits.maxEstimatedTokensPerMessage.toLocaleString()} tokens)`,
      estimatedTokens,
    };
  }

  return {
    valid: true,
    estimatedTokens,
  };
}

/**
 * Validate attachment size from a data URL or fetch the size.
 */
export function validateAttachmentSize(
  url: string,
  maxSizeBytes: number
): ValidationResult {
  // For data URLs, we can estimate size from base64 length
  if (url.startsWith("data:")) {
    const base64Part = url.split(",").at(1);
    if (base64Part) {
      // Base64 encoding increases size by ~33%
      const estimatedBytes = Math.ceil((base64Part.length * 3) / 4);
      if (estimatedBytes > maxSizeBytes) {
        return {
          valid: false,
          error: `Attachment exceeds maximum size (${Math.round(maxSizeBytes / 1024 / 1024)}MB)`,
        };
      }
    }
  }

  return { valid: true };
}
