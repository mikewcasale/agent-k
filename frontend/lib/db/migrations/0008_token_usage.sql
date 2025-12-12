-- Token usage tracking for API abuse prevention
-- Tracks daily token consumption per user for cost-based rate limiting

CREATE TABLE "UserTokenUsage" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "userId" uuid NOT NULL REFERENCES "User"("id") ON DELETE CASCADE,
  "date" date NOT NULL,
  "inputTokens" integer NOT NULL DEFAULT 0,
  "outputTokens" integer NOT NULL DEFAULT 0,
  "estimatedCostCents" integer NOT NULL DEFAULT 0,
  "requestCount" integer NOT NULL DEFAULT 0,
  "createdAt" timestamp NOT NULL DEFAULT now(),
  "updatedAt" timestamp NOT NULL DEFAULT now(),
  UNIQUE("userId", "date")
);

-- Index for efficient lookups by user and date range
CREATE INDEX "UserTokenUsage_userId_date_idx" ON "UserTokenUsage"("userId", "date" DESC);

-- Index for aggregation queries (monthly reports)
CREATE INDEX "UserTokenUsage_date_idx" ON "UserTokenUsage"("date" DESC);

