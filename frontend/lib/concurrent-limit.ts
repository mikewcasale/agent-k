/**
 * Concurrent request limiter.
 *
 * Prevents users from opening too many parallel streams/requests
 * to prevent resource exhaustion and cost abuse.
 */

import type { UserType } from "@/app/(auth)/auth";

// =============================================================================
// Types
// =============================================================================

export type ConcurrentLimitResult = {
  allowed: boolean;
  current: number;
  limit: number;
};

type ActiveRequest = {
  startedAt: number;
  streamId?: string;
};

// =============================================================================
// Configuration
// =============================================================================

/**
 * Maximum concurrent requests by user type.
 */
export const concurrentLimitsByUserType: Record<UserType, number> = {
  guest: 1,
  regular: 2,
};

// Premium tier (for future use)
export const premiumConcurrentLimit = 5;

/**
 * Request timeout in milliseconds.
 * Requests older than this are automatically cleaned up.
 */
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

// =============================================================================
// In-Memory Store
// =============================================================================

/**
 * In-memory store for tracking active requests per user.
 * For production with multiple instances, use Redis.
 */
class ConcurrentRequestStore {
  private readonly activeRequests = new Map<
    string,
    Map<string, ActiveRequest>
  >();
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // Clean up stale requests every minute
    this.cleanupInterval = setInterval(() => this.cleanup(), 60 * 1000);
  }

  /**
   * Get the count of active requests for a user.
   */
  getCount(userId: string): number {
    const userRequests = this.activeRequests.get(userId);
    return userRequests?.size ?? 0;
  }

  /**
   * Get all active request IDs for a user.
   */
  getActiveRequestIds(userId: string): string[] {
    const userRequests = this.activeRequests.get(userId);
    return userRequests ? [...userRequests.keys()] : [];
  }

  /**
   * Add a new active request for a user.
   */
  add(userId: string, requestId: string, streamId?: string): void {
    let userRequests = this.activeRequests.get(userId);

    if (!userRequests) {
      userRequests = new Map();
      this.activeRequests.set(userId, userRequests);
    }

    userRequests.set(requestId, {
      startedAt: Date.now(),
      streamId,
    });
  }

  /**
   * Remove an active request for a user.
   */
  remove(userId: string, requestId: string): boolean {
    const userRequests = this.activeRequests.get(userId);

    if (!userRequests) {
      return false;
    }

    const removed = userRequests.delete(requestId);

    // Clean up empty user entry
    if (userRequests.size === 0) {
      this.activeRequests.delete(userId);
    }

    return removed;
  }

  /**
   * Remove all requests for a user.
   */
  removeAll(userId: string): number {
    const userRequests = this.activeRequests.get(userId);
    const count = userRequests?.size ?? 0;
    this.activeRequests.delete(userId);
    return count;
  }

  /**
   * Clean up stale requests (older than timeout).
   */
  private cleanup(): void {
    const now = Date.now();

    for (const [userId, userRequests] of this.activeRequests.entries()) {
      for (const [requestId, request] of userRequests.entries()) {
        if (now - request.startedAt > REQUEST_TIMEOUT_MS) {
          userRequests.delete(requestId);
        }
      }

      // Clean up empty user entry
      if (userRequests.size === 0) {
        this.activeRequests.delete(userId);
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
    this.activeRequests.clear();
  }
}

// =============================================================================
// Redis Store (Production)
// =============================================================================

/**
 * Redis-based concurrent request store.
 * Uses sorted sets with timestamps for automatic expiration.
 */
class RedisConcurrentStore {
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
   * Get the count of active requests for a user.
   */
  async getCount(userId: string): Promise<number> {
    if (!this.isConfigured()) {
      return 0;
    }

    const key = `concurrent:${userId}`;
    const now = Date.now();
    const minScore = now - REQUEST_TIMEOUT_MS;

    // First, clean up expired entries
    await this.cleanup(userId);

    // Count remaining entries
    const response = await fetch(
      `${this.redisUrl}/zcount/${key}/${minScore}/+inf`,
      {
        headers: { Authorization: `Bearer ${this.redisToken}` },
      }
    );

    const data = await response.json();
    return data.result ?? 0;
  }

  /**
   * Add a request to the active set.
   */
  async add(userId: string, requestId: string): Promise<void> {
    if (!this.isConfigured()) {
      return;
    }

    const key = `concurrent:${userId}`;
    const score = Date.now();

    await fetch(`${this.redisUrl}/zadd/${key}/${score}/${requestId}`, {
      headers: { Authorization: `Bearer ${this.redisToken}` },
    });

    // Set expiration on the key
    await fetch(
      `${this.redisUrl}/expire/${key}/${Math.ceil(REQUEST_TIMEOUT_MS / 1000)}`,
      {
        headers: { Authorization: `Bearer ${this.redisToken}` },
      }
    );
  }

  /**
   * Remove a request from the active set.
   */
  async remove(userId: string, requestId: string): Promise<void> {
    if (!this.isConfigured()) {
      return;
    }

    const key = `concurrent:${userId}`;

    await fetch(`${this.redisUrl}/zrem/${key}/${requestId}`, {
      headers: { Authorization: `Bearer ${this.redisToken}` },
    });
  }

  /**
   * Clean up expired entries.
   */
  private async cleanup(userId: string): Promise<void> {
    const key = `concurrent:${userId}`;
    const minScore = Date.now() - REQUEST_TIMEOUT_MS;

    await fetch(`${this.redisUrl}/zremrangebyscore/${key}/-inf/${minScore}`, {
      headers: { Authorization: `Bearer ${this.redisToken}` },
    });
  }
}

// =============================================================================
// Concurrent Limiter
// =============================================================================

const memoryStore = new ConcurrentRequestStore();
const redisStore = new RedisConcurrentStore();

/**
 * Check if a user can start a new request.
 */
export async function checkConcurrentLimit(
  userId: string,
  userType: UserType
): Promise<ConcurrentLimitResult> {
  const limit = concurrentLimitsByUserType[userType];

  let current: number;

  if (redisStore.isConfigured()) {
    current = await redisStore.getCount(userId);
  } else {
    current = memoryStore.getCount(userId);
  }

  return {
    allowed: current < limit,
    current,
    limit,
  };
}

/**
 * Acquire a slot for a new request.
 * Returns a release function to call when the request completes.
 */
export async function acquireRequestSlot(
  userId: string,
  requestId: string,
  streamId?: string
): Promise<() => Promise<void>> {
  if (redisStore.isConfigured()) {
    await redisStore.add(userId, requestId);

    return async () => {
      await redisStore.remove(userId, requestId);
    };
  }

  memoryStore.add(userId, requestId, streamId);

  return () => {
    memoryStore.remove(userId, requestId);
    return Promise.resolve();
  };
}

/**
 * Get the current concurrent request count for a user.
 */
export function getConcurrentCount(userId: string): number | Promise<number> {
  if (redisStore.isConfigured()) {
    return redisStore.getCount(userId);
  }

  return memoryStore.getCount(userId);
}

/**
 * Force release all requests for a user.
 * Use sparingly - mainly for cleanup/testing.
 */
export function forceReleaseAll(userId: string): number {
  return memoryStore.removeAll(userId);
}
