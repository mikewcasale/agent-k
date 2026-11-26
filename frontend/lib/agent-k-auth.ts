import "server-only";

import { createHmac, randomUUID } from "node:crypto";

const DEFAULT_ISSUER = "agent-k-frontend";
const DEFAULT_AUDIENCE = "agent-k-api";
const DEFAULT_TTL_SECONDS = 60;

function encode(value: object): string {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

function tokenTtl(): number {
  const configured = Number.parseInt(
    process.env.AGENT_K_GATEWAY_TOKEN_TTL_SECONDS ?? "",
    10
  );
  if (
    !Number.isSafeInteger(configured) ||
    configured < 30 ||
    configured > 300
  ) {
    return DEFAULT_TTL_SECONDS;
  }
  return configured;
}

/** Create a short-lived signed identity token for one authenticated frontend user. */
export function agentKAuthorization(userId: string): string {
  const secret = process.env.AGENT_K_GATEWAY_SECRET;
  if (!secret) {
    throw new Error("AGENT_K_GATEWAY_SECRET is not configured");
  }
  if (!userId) {
    throw new Error("Authenticated Agent-K requests require a user id");
  }

  const issuedAt = Math.floor(Date.now() / 1000);
  const header = encode({ alg: "HS256", typ: "JWT" });
  const claims = encode({
    sub: userId,
    iss: process.env.AGENT_K_GATEWAY_ISSUER ?? DEFAULT_ISSUER,
    aud: process.env.AGENT_K_GATEWAY_AUDIENCE ?? DEFAULT_AUDIENCE,
    iat: issuedAt,
    exp: issuedAt + tokenTtl(),
    jti: randomUUID(),
  });
  const signingInput = `${header}.${claims}`;
  const signature = createHmac("sha256", secret)
    .update(signingInput)
    .digest("base64url");
  return `Bearer ${signingInput}.${signature}`;
}
