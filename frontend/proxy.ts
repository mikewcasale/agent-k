import { type NextRequest, NextResponse } from "next/server";
import { getToken } from "next-auth/jwt";
import {
  guestRegex,
  isDevelopmentEnvironment,
  isProductionEnvironment,
} from "./lib/constants";

// =============================================================================
// IP Rate Limiting (Abuse Prevention)
// =============================================================================

const RATE_LIMITED_PATHS = ["/api/chat", "/api/agent-k"];
const SUSPICIOUS_USER_AGENTS = [
  /curl/i,
  /wget/i,
  /python-requests/i,
  /scrapy/i,
  /httpie/i,
  /postman/i,
];
const ALLOWED_BOTS = [/googlebot/i, /bingbot/i, /slackbot/i, /discordbot/i];

const ipRequestCounts = new Map<
  string,
  { count: number; windowStart: number }
>();
const WINDOW_MS = 60 * 1000;
const MAX_REQUESTS_PER_MINUTE = 30;

function getClientIp(request: NextRequest): string {
  return (
    request.headers.get("x-forwarded-for")?.split(",").at(0)?.trim() ??
    request.headers.get("x-real-ip") ??
    "unknown"
  );
}

function isIpRateLimited(ip: string): { limited: boolean; remaining: number } {
  const now = Date.now();
  const entry = ipRequestCounts.get(ip);

  if (!entry || now - entry.windowStart >= WINDOW_MS) {
    ipRequestCounts.set(ip, { count: 1, windowStart: now });
    return { limited: false, remaining: MAX_REQUESTS_PER_MINUTE - 1 };
  }

  if (entry.count >= MAX_REQUESTS_PER_MINUTE) {
    return { limited: true, remaining: 0 };
  }

  entry.count++;
  return { limited: false, remaining: MAX_REQUESTS_PER_MINUTE - entry.count };
}

function isSuspiciousUserAgent(userAgent: string | null): boolean {
  if (!userAgent) return true;
  if (ALLOWED_BOTS.some((p) => p.test(userAgent))) return false;
  return SUSPICIOUS_USER_AGENTS.some((p) => p.test(userAgent));
}

// =============================================================================
// Proxy Handler
// =============================================================================

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  /*
   * Playwright starts the dev server and requires a 200 status to
   * begin the tests, so this ensures that the tests can start
   */
  if (pathname.startsWith("/ping")) {
    return new Response("pong", { status: 200 });
  }

  // Rate limiting for API routes (abuse prevention)
  if (RATE_LIMITED_PATHS.some((path) => pathname.startsWith(path))) {
    const ip = getClientIp(request);
    const userAgent = request.headers.get("user-agent");

    // Block suspicious user agents in production
    if (isProductionEnvironment && isSuspiciousUserAgent(userAgent)) {
      return NextResponse.json(
        { code: "forbidden:api", message: "Request blocked." },
        { status: 403 }
      );
    }

    // IP-based rate limiting
    const { limited, remaining } = isIpRateLimited(ip);
    if (limited) {
      return NextResponse.json(
        {
          code: "rate_limit:burst",
          message: "Too many requests. Please slow down.",
        },
        {
          status: 429,
          headers: {
            "Retry-After": "60",
            "X-RateLimit-Limit": MAX_REQUESTS_PER_MINUTE.toString(),
            "X-RateLimit-Remaining": "0",
          },
        }
      );
    }
  }

  if (pathname.startsWith("/api/auth")) {
    return NextResponse.next();
  }

  const token = await getToken({
    req: request,
    secret: process.env.AUTH_SECRET,
    secureCookie: !isDevelopmentEnvironment,
  });

  if (!token) {
    const redirectUrl = encodeURIComponent(request.url);

    return NextResponse.redirect(
      new URL(`/api/auth/guest?redirectUrl=${redirectUrl}`, request.url)
    );
  }

  const isGuest = guestRegex.test(token?.email ?? "");

  if (token && !isGuest && ["/login", "/register"].includes(pathname)) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/",
    "/chat/:id",
    "/api/:path*",
    "/login",
    "/register",

    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico, sitemap.xml, robots.txt (metadata files)
     */
    "/((?!_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt).*)",
  ],
};
