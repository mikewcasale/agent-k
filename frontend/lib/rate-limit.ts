/**
 * Sliding window rate limiter with Redis and in-memory fallback.
 *
 * Provides multi-window rate limiting (per-minute, per-hour, per-day)
 * with configurable limits per user type.
 */

import type { UserType } from "@/app/(auth)/auth";

// =============================================================================
// Types
// =============================================================================

export type RateLimitWindow = "minute" | "hour" | "day";

export type RateLimitConfig = {
  windowMs: number;
  maxRequests: number;
};

export type UserRateLimits = {
  perMinute: number;
  perHour: number;
  perDay: number;
};

export type RateLimitResult = {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
  window: RateLimitWindow;
};

type RateLimitEntry = {
  count: number;
  windowStart: number;
};

// =============================================================================
// Configuration
// =============================================================================

const WINDOW_MS: Record<RateLimitWindow, number> = {
  minute: 60 * 1000,
  hour: 60 * 60 * 1000,
  day: 24 * 60 * 60 * 1000,
};

/**
 * Rate limits by user type.
 * These are checked in order: minute -> hour -> day
 */
export const rateLimitsByUserType: Record<UserType, UserRateLimits> = {
  guest: {
    perMinute: 5,
    perHour: 15,
    perDay: 20,
  },
  regular: {
    perMinute: 10,
    perHour: 50,
    perDay: 100,
  },
};

// Premium tier (for future use)
export const premiumRateLimits: UserRateLimits = {
  perMinute: 30,
  perHour: 200,
  perDay: 1000,
};

// =============================================================================
// In-Memory Store (Fallback)
// =============================================================================

/**
 * Simple in-memory rate limit store.
 * Uses a Map with automatic cleanup of stale entries.
 */
class InMemoryRateLimitStore {
  private readonly store = new Map<string, RateLimitEntry>();
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // Clean up stale entries every 5 minutes
    this.cleanupInterval = setInterval(() => this.cleanup(), 5 * 60 * 1000);
  }

  /**
   * Get or create a rate limit entry for a key.
   */
  get(key: string, windowMs: number): RateLimitEntry {
    const now = Date.now();
    const entry = this.store.get(key);

    if (!entry || now - entry.windowStart >= windowMs) {
      // Window expired, create new entry
      const newEntry: RateLimitEntry = { count: 0, windowStart: now };
      this.store.set(key, newEntry);
      return newEntry;
    }

    return entry;
  }

  /**
   * Increment the count for a key.
   */
  increment(key: string, windowMs: number): RateLimitEntry {
    const entry = this.get(key, windowMs);
    entry.count++;
    this.store.set(key, entry);
    return entry;
  }

  /**
   * Remove stale entries from the store.
   */
  private cleanup(): void {
    const now = Date.now();
    const maxAge = WINDOW_MS.day; // Keep entries for at least a day

    for (const [key, entry] of this.store.entries()) {
      if (now - entry.windowStart > maxAge) {
        this.store.delete(key);
      }
    }
  }

  /**
   * Dispose of the store and cleanup interval.
   */
  dispose(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.store.clear();
  }
}

// =============================================================================
// Redis Store (Production)
// =============================================================================

/**
 * Redis-based rate limit store using Upstash.
 * Falls back to in-memory if Redis is not configured.
 */
class RedisRateLimitStore {
  private readonly redisUrl: string | undefined;
  private readonly redisToken: string | undefined;

  constructor() {
    this.redisUrl = process.env.UPSTASH_REDIS_REST_URL;
    this.redisToken = process.env.UPSTASH_REDIS_REST_TOKEN;
  }

  isConfigured(): boolean {
    return Boolean(this.redisUrl && this.redisToken);
  }

  /**
   * Get the current count for a key.
   */
  async get(key: string): Promise<number> {
    if (!this.isConfigured()) {
      return 0;
    }

    const response = await fetch(`${this.redisUrl}/get/${key}`, {
      headers: { Authorization: `Bearer ${this.redisToken}` },
    });

    const data = await response.json();
    return data.result ? Number.parseInt(data.result, 10) : 0;
  }

  /**
   * Increment the count for a key with expiration.
   */
  async increment(key: string, windowMs: number): Promise<number> {
    if (!this.isConfigured()) {
      return 0;
    }

    const ttlSeconds = Math.ceil(windowMs / 1000);

    // Use INCR and EXPIRE in a pipeline
    const response = await fetch(`${this.redisUrl}/pipeline`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.redisToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify([
        ["INCR", key],
        ["EXPIRE", key, ttlSeconds.toString()],
      ]),
    });

    const data = await response.json();
    // First result is the INCR result
    return data[0]?.result ?? 1;
  }
}

// =============================================================================
// Rate Limiter
// =============================================================================

const memoryStore = new InMemoryRateLimitStore();
const redisStore = new RedisRateLimitStore();

/**
 * Create a rate limit key for a user and window.
 */
function createKey(
  userId: string,
  window: RateLimitWindow,
  prefix = "ratelimit"
): string {
  const now = Date.now();
  const windowStart = Math.floor(now / WINDOW_MS[window]) * WINDOW_MS[window];
  return `${prefix}:${userId}:${window}:${windowStart}`;
}

/**
 * Check rate limit for a single window.
 */
async function checkWindow(
  userId: string,
  window: RateLimitWindow,
  maxRequests: number
): Promise<RateLimitResult> {
  const key = createKey(userId, window);
  const windowMs = WINDOW_MS[window];

  let count: number;

  if (redisStore.isConfigured()) {
    count = await redisStore.get(key);
  } else {
    const entry = await memoryStore.get(key, windowMs);
    count = entry.count;
  }

  const resetAt = new Date(
    Math.floor(Date.now() / windowMs) * windowMs + windowMs
  );

  return {
    allowed: count < maxRequests,
    remaining: Math.max(0, maxRequests - count),
    resetAt,
    window,
  };
}

/**
 * Increment rate limit counters for all windows.
 */
async function incrementAllWindows(userId: string): Promise<void> {
  const windows: RateLimitWindow[] = ["minute", "hour", "day"];

  if (redisStore.isConfigured()) {
    await Promise.all(
      windows.map((window) =>
        redisStore.increment(createKey(userId, window), WINDOW_MS[window])
      )
    );
  } else {
    await Promise.all(
      windows.map((window) =>
        memoryStore.increment(createKey(userId, window), WINDOW_MS[window])
      )
    );
  }
}

/**
 * Check all rate limit windows for a user.
 * Returns the first exceeded limit or success.
 */
export async function checkRateLimit(
  userId: string,
  userType: UserType
): Promise<RateLimitResult> {
  const limits = rateLimitsByUserType[userType];

  // Check windows in order of shortest to longest
  const checks: Array<{ window: RateLimitWindow; limit: number }> = [
    { window: "minute", limit: limits.perMinute },
    { window: "hour", limit: limits.perHour },
    { window: "day", limit: limits.perDay },
  ];

  for (const { window, limit } of checks) {
    const result = await checkWindow(userId, window, limit);
    if (!result.allowed) {
      return result;
    }
  }

  // All windows passed, return success with day remaining
  const dayResult = await checkWindow(userId, "day", limits.perDay);
  return dayResult;
}

/**
 * Record a request for rate limiting.
 * Call this after successfully processing a request.
 */
export async function recordRequest(userId: string): Promise<void> {
  await incrementAllWindows(userId);
}

/**
 * Get rate limit status for a user without incrementing.
 */
export async function getRateLimitStatus(
  userId: string,
  userType: UserType
): Promise<{
  minute: RateLimitResult;
  hour: RateLimitResult;
  day: RateLimitResult;
}> {
  const limits = rateLimitsByUserType[userType];

  const [minute, hour, day] = await Promise.all([
    checkWindow(userId, "minute", limits.perMinute),
    checkWindow(userId, "hour", limits.perHour),
    checkWindow(userId, "day", limits.perDay),
  ]);

  return { minute, hour, day };
}

// =============================================================================
// IP-Based Rate Limiting (for unauthenticated requests)
// =============================================================================

const IP_RATE_LIMITS = {
  perMinute: 20,
  perHour: 100,
};

/**
 * Check rate limit for an IP address.
 * Used for unauthenticated or anonymous requests.
 */
export async function checkIpRateLimit(ip: string): Promise<RateLimitResult> {
  const checks: Array<{ window: RateLimitWindow; limit: number }> = [
    { window: "minute", limit: IP_RATE_LIMITS.perMinute },
    { window: "hour", limit: IP_RATE_LIMITS.perHour },
  ];

  for (const { window, limit } of checks) {
    const key = createKey(ip, window, "ip-ratelimit");
    const windowMs = WINDOW_MS[window];

    let count: number;

    if (redisStore.isConfigured()) {
      count = await redisStore.get(key);
    } else {
      const entry = await memoryStore.get(key, windowMs);
      count = entry.count;
    }

    if (count >= limit) {
      const resetAt = new Date(
        Math.floor(Date.now() / windowMs) * windowMs + windowMs
      );
      return {
        allowed: false,
        remaining: 0,
        resetAt,
        window,
      };
    }
  }

  return {
    allowed: true,
    remaining: IP_RATE_LIMITS.perMinute,
    resetAt: new Date(Date.now() + WINDOW_MS.minute),
    window: "minute",
  };
}

/**
 * Record a request for IP-based rate limiting.
 */
export async function recordIpRequest(ip: string): Promise<void> {
  const windows: RateLimitWindow[] = ["minute", "hour"];

  if (redisStore.isConfigured()) {
    await Promise.all(
      windows.map((window) =>
        redisStore.increment(
          createKey(ip, window, "ip-ratelimit"),
          WINDOW_MS[window]
        )
      )
    );
  } else {
    await Promise.all(
      windows.map((window) =>
        memoryStore.increment(
          createKey(ip, window, "ip-ratelimit"),
          WINDOW_MS[window]
        )
      )
    );
  }
}
