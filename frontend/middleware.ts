/**
 * Next.js Middleware for API Abuse Prevention.
 *
 * Provides:
 * - IP-based rate limiting for unauthenticated requests
 * - Bot detection via User-Agent checks
 * - Request logging for abuse detection
 */

import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

// =============================================================================
// Configuration
// =============================================================================

/**
 * Paths that require rate limiting.
 */
const RATE_LIMITED_PATHS = ["/api/chat", "/api/agent-k"];

/**
 * Suspicious User-Agent patterns (bots, scrapers, etc.)
 */
const SUSPICIOUS_USER_AGENTS = [
  /curl/i,
  /wget/i,
  /python-requests/i,
  /scrapy/i,
  /httpie/i,
  /postman/i,
];

/**
 * Known good bot patterns (allow through)
 */
const ALLOWED_BOTS = [/googlebot/i, /bingbot/i, /slackbot/i, /discordbot/i];

// =============================================================================
// In-Memory Rate Limit Store (Edge Runtime Compatible)
// =============================================================================

/**
 * Simple in-memory rate limit store for Edge Runtime.
 * Note: This does not persist across instances in serverless environments.
 * For production, use Vercel KV or Upstash Redis.
 */
const ipRequestCounts = new Map<
  string,
  { count: number; windowStart: number }
>();

const WINDOW_MS = 60 * 1000; // 1 minute window
const MAX_REQUESTS_PER_MINUTE = 30; // Max requests per IP per minute

/**
 * Check if an IP is rate limited.
 */
function isIpRateLimited(ip: string): { limited: boolean; remaining: number } {
  const now = Date.now();
  const entry = ipRequestCounts.get(ip);

  if (!entry || now - entry.windowStart >= WINDOW_MS) {
    // New window
    ipRequestCounts.set(ip, { count: 1, windowStart: now });
    return { limited: false, remaining: MAX_REQUESTS_PER_MINUTE - 1 };
  }

  if (entry.count >= MAX_REQUESTS_PER_MINUTE) {
    return { limited: true, remaining: 0 };
  }

  entry.count++;
  return { limited: false, remaining: MAX_REQUESTS_PER_MINUTE - entry.count };
}

/**
 * Clean up stale entries periodically.
 */
function cleanupStaleEntries(): void {
  const now = Date.now();
  for (const [ip, entry] of ipRequestCounts.entries()) {
    if (now - entry.windowStart >= WINDOW_MS * 5) {
      ipRequestCounts.delete(ip);
    }
  }
}

// Run cleanup every 100 requests (approximate)
let requestCounter = 0;

// =============================================================================
// Middleware Logic
// =============================================================================

/**
 * Get client IP from request headers.
 */
function getClientIp(request: NextRequest): string {
  return (
    request.headers.get("x-forwarded-for")?.split(",").at(0)?.trim() ??
    request.headers.get("x-real-ip") ??
    "unknown"
  );
}

/**
 * Check if User-Agent is suspicious.
 */
function isSuspiciousUserAgent(userAgent: string | null): boolean {
  if (!userAgent) {
    return true; // No User-Agent is suspicious
  }

  // Allow known good bots
  if (ALLOWED_BOTS.some((pattern) => pattern.test(userAgent))) {
    return false;
  }

  // Check for suspicious patterns
  return SUSPICIOUS_USER_AGENTS.some((pattern) => pattern.test(userAgent));
}

/**
 * Create a rate limit response.
 */
function createRateLimitResponse(retryAfter: number): NextResponse {
  return NextResponse.json(
    {
      code: "rate_limit:burst",
      message: "Too many requests. Please slow down and try again.",
    },
    {
      status: 429,
      headers: {
        "Retry-After": retryAfter.toString(),
        "X-RateLimit-Limit": MAX_REQUESTS_PER_MINUTE.toString(),
        "X-RateLimit-Remaining": "0",
      },
    }
  );
}

/**
 * Create a forbidden response for suspicious requests.
 */
function createForbiddenResponse(): NextResponse {
  return NextResponse.json(
    {
      code: "forbidden:api",
      message: "Request blocked.",
    },
    { status: 403 }
  );
}

// =============================================================================
// Middleware Entry Point
// =============================================================================

export function middleware(request: NextRequest): NextResponse {
  const { pathname } = request.nextUrl;

  // Only apply to rate-limited paths
  if (!RATE_LIMITED_PATHS.some((path) => pathname.startsWith(path))) {
    return NextResponse.next();
  }

  // Get client information
  const ip = getClientIp(request);
  const userAgent = request.headers.get("user-agent");

  // Periodic cleanup
  requestCounter++;
  if (requestCounter % 100 === 0) {
    cleanupStaleEntries();
  }

  // Check for suspicious User-Agent (only block in production)
  if (
    process.env.NODE_ENV === "production" &&
    isSuspiciousUserAgent(userAgent)
  ) {
    // Log suspicious request
    console.warn("Suspicious request blocked", {
      ip,
      userAgent,
      path: pathname,
      timestamp: new Date().toISOString(),
    });
    return createForbiddenResponse();
  }

  // IP-based rate limiting
  const { limited, remaining } = isIpRateLimited(ip);

  if (limited) {
    console.warn("IP rate limited", {
      ip,
      path: pathname,
      timestamp: new Date().toISOString(),
    });
    return createRateLimitResponse(60); // Retry after 60 seconds
  }

  // Add rate limit headers to response
  const response = NextResponse.next();
  response.headers.set("X-RateLimit-Limit", MAX_REQUESTS_PER_MINUTE.toString());
  response.headers.set("X-RateLimit-Remaining", remaining.toString());

  return response;
}

// =============================================================================
// Middleware Config
// =============================================================================

export const config = {
  matcher: [
    /*
     * Match all API routes that need rate limiting:
     * - /api/chat
     * - /api/agent-k
     */
    "/api/chat/:path*",
    "/api/agent-k/:path*",
  ],
};
