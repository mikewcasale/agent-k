/**
 * Token usage tracking and budget queries.
 *
 * Provides functions to track and query token usage for cost-based rate limiting.
 */

import "server-only";

import { and, eq, gte, sql } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import type { UserType } from "@/app/(auth)/auth";
import { userTokenUsage } from "./schema";

// biome-ignore lint: Forbidden non-null assertion.
const client = postgres(process.env.POSTGRES_URL!);
const db = drizzle(client);

// =============================================================================
// Types
// =============================================================================

export type TokenUsageRecord = {
  inputTokens: number;
  outputTokens: number;
  estimatedCostCents: number;
  requestCount: number;
};

export type TokenBudget = {
  dailyTokenLimit: number;
  dailyCostLimitCents: number;
  monthlyTokenLimit: number;
  monthlyCostLimitCents: number;
};

export type TokenBudgetStatus = {
  dailyUsage: TokenUsageRecord;
  monthlyUsage: TokenUsageRecord;
  dailyBudget: TokenBudget;
  withinBudget: boolean;
  remainingDailyTokens: number;
  remainingDailyCostCents: number;
};

// =============================================================================
// Budget Configuration by User Type
// =============================================================================

export const tokenBudgetsByUserType: Record<UserType, TokenBudget> = {
  guest: {
    dailyTokenLimit: 100_000,
    dailyCostLimitCents: 50, // $0.50
    monthlyTokenLimit: 1_000_000,
    monthlyCostLimitCents: 500, // $5.00
  },
  regular: {
    dailyTokenLimit: 500_000,
    dailyCostLimitCents: 500, // $5.00
    monthlyTokenLimit: 10_000_000,
    monthlyCostLimitCents: 5000, // $50.00
  },
};

// Premium tier (for future use)
export const premiumTokenBudget: TokenBudget = {
  dailyTokenLimit: 5_000_000,
  dailyCostLimitCents: 5000, // $50.00
  monthlyTokenLimit: 100_000_000,
  monthlyCostLimitCents: 50_000, // $500.00
};

// =============================================================================
// Cost Estimation
// =============================================================================

/**
 * Cost per 1000 tokens in cents.
 * These can be overridden via environment variables.
 */
const INPUT_COST_PER_1K_CENTS =
  Number.parseFloat(process.env.CLAUDE_INPUT_COST_PER_1K ?? "0.3") * 100;
const OUTPUT_COST_PER_1K_CENTS =
  Number.parseFloat(process.env.CLAUDE_OUTPUT_COST_PER_1K ?? "1.5") * 100;

/**
 * Estimate cost in cents for token usage.
 */
export function estimateCostCents(
  inputTokens: number,
  outputTokens: number
): number {
  const inputCost = (inputTokens / 1000) * INPUT_COST_PER_1K_CENTS;
  const outputCost = (outputTokens / 1000) * OUTPUT_COST_PER_1K_CENTS;
  return Math.ceil(inputCost + outputCost);
}

// =============================================================================
// Database Queries
// =============================================================================

/**
 * Get today's date as a string (YYYY-MM-DD).
 */
function getTodayString(): string {
  return new Date().toISOString().split("T").at(0) ?? "";
}

/**
 * Get the first day of the current month as a string.
 */
function getMonthStartString(): string {
  const now = new Date();
  return (
    new Date(now.getFullYear(), now.getMonth(), 1)
      .toISOString()
      .split("T")
      .at(0) ?? ""
  );
}

/**
 * Get token usage for a user on a specific date.
 */
export async function getTokenUsageForDate(
  userId: string,
  dateString: string
): Promise<TokenUsageRecord | null> {
  const results = await db
    .select({
      inputTokens: userTokenUsage.inputTokens,
      outputTokens: userTokenUsage.outputTokens,
      estimatedCostCents: userTokenUsage.estimatedCostCents,
      requestCount: userTokenUsage.requestCount,
    })
    .from(userTokenUsage)
    .where(
      and(
        eq(userTokenUsage.userId, userId),
        eq(userTokenUsage.date, dateString)
      )
    )
    .limit(1);

  if (results.length === 0) {
    return null;
  }

  return results.at(0) ?? null;
}

/**
 * Get aggregated token usage for a user in a date range.
 */
export async function getTokenUsageForRange(
  userId: string,
  startDate: string
): Promise<TokenUsageRecord> {
  const results = await db
    .select({
      inputTokens: sql<number>`COALESCE(SUM(${userTokenUsage.inputTokens}), 0)`,
      outputTokens: sql<number>`COALESCE(SUM(${userTokenUsage.outputTokens}), 0)`,
      estimatedCostCents: sql<number>`COALESCE(SUM(${userTokenUsage.estimatedCostCents}), 0)`,
      requestCount: sql<number>`COALESCE(SUM(${userTokenUsage.requestCount}), 0)`,
    })
    .from(userTokenUsage)
    .where(
      and(
        eq(userTokenUsage.userId, userId),
        gte(userTokenUsage.date, startDate)
      )
    );

  const result = results.at(0);

  return {
    inputTokens: Number(result?.inputTokens ?? 0),
    outputTokens: Number(result?.outputTokens ?? 0),
    estimatedCostCents: Number(result?.estimatedCostCents ?? 0),
    requestCount: Number(result?.requestCount ?? 0),
  };
}

/**
 * Increment token usage for a user.
 * Creates a new record if one doesn't exist for today.
 */
export async function incrementTokenUsage(
  userId: string,
  inputTokens: number,
  outputTokens: number
): Promise<void> {
  const today = getTodayString();
  const costCents = estimateCostCents(inputTokens, outputTokens);

  await db
    .insert(userTokenUsage)
    .values({
      userId,
      date: today,
      inputTokens,
      outputTokens,
      estimatedCostCents: costCents,
      requestCount: 1,
    })
    .onConflictDoUpdate({
      target: [userTokenUsage.userId, userTokenUsage.date],
      set: {
        inputTokens: sql`${userTokenUsage.inputTokens} + ${inputTokens}`,
        outputTokens: sql`${userTokenUsage.outputTokens} + ${outputTokens}`,
        estimatedCostCents: sql`${userTokenUsage.estimatedCostCents} + ${costCents}`,
        requestCount: sql`${userTokenUsage.requestCount} + 1`,
        updatedAt: sql`now()`,
      },
    });
}

/**
 * Check if a user is within their token budget.
 */
export async function checkTokenBudget(
  userId: string,
  userType: UserType
): Promise<TokenBudgetStatus> {
  const budget = tokenBudgetsByUserType[userType];
  const today = getTodayString();
  const monthStart = getMonthStartString();

  const [dailyUsage, monthlyUsage] = await Promise.all([
    getTokenUsageForDate(userId, today).then(
      (r) =>
        r ?? {
          inputTokens: 0,
          outputTokens: 0,
          estimatedCostCents: 0,
          requestCount: 0,
        }
    ),
    getTokenUsageForRange(userId, monthStart),
  ]);

  const totalDailyTokens = dailyUsage.inputTokens + dailyUsage.outputTokens;
  const totalMonthlyTokens =
    monthlyUsage.inputTokens + monthlyUsage.outputTokens;

  const withinDailyTokens = totalDailyTokens < budget.dailyTokenLimit;
  const withinDailyCost =
    dailyUsage.estimatedCostCents < budget.dailyCostLimitCents;
  const withinMonthlyTokens = totalMonthlyTokens < budget.monthlyTokenLimit;
  const withinMonthlyCost =
    monthlyUsage.estimatedCostCents < budget.monthlyCostLimitCents;

  const withinBudget =
    withinDailyTokens &&
    withinDailyCost &&
    withinMonthlyTokens &&
    withinMonthlyCost;

  return {
    dailyUsage,
    monthlyUsage,
    dailyBudget: budget,
    withinBudget,
    remainingDailyTokens: Math.max(
      0,
      budget.dailyTokenLimit - totalDailyTokens
    ),
    remainingDailyCostCents: Math.max(
      0,
      budget.dailyCostLimitCents - dailyUsage.estimatedCostCents
    ),
  };
}

/**
 * Get a formatted usage summary for a user.
 */
export async function getUsageSummary(
  userId: string,
  userType: UserType
): Promise<{
  daily: { used: number; limit: number; percentage: number };
  monthly: { used: number; limit: number; percentage: number };
  costDaily: { usedCents: number; limitCents: number; percentage: number };
}> {
  const status = await checkTokenBudget(userId, userType);

  const dailyTokensUsed =
    status.dailyUsage.inputTokens + status.dailyUsage.outputTokens;
  const monthlyTokensUsed =
    status.monthlyUsage.inputTokens + status.monthlyUsage.outputTokens;

  return {
    daily: {
      used: dailyTokensUsed,
      limit: status.dailyBudget.dailyTokenLimit,
      percentage: Math.round(
        (dailyTokensUsed / status.dailyBudget.dailyTokenLimit) * 100
      ),
    },
    monthly: {
      used: monthlyTokensUsed,
      limit: status.dailyBudget.monthlyTokenLimit,
      percentage: Math.round(
        (monthlyTokensUsed / status.dailyBudget.monthlyTokenLimit) * 100
      ),
    },
    costDaily: {
      usedCents: status.dailyUsage.estimatedCostCents,
      limitCents: status.dailyBudget.dailyCostLimitCents,
      percentage: Math.round(
        (status.dailyUsage.estimatedCostCents /
          status.dailyBudget.dailyCostLimitCents) *
          100
      ),
    },
  };
}
