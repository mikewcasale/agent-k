ALTER TABLE runtime_migrations
    ADD COLUMN IF NOT EXISTS processing_token UUID,
    ADD COLUMN IF NOT EXISTS processing_expires_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS runtime_migrations_claim_idx
    ON runtime_migrations (status, processing_expires_at, queued_at);

CREATE TABLE IF NOT EXISTS legacy_mission_states (
    source_sha256 TEXT NOT NULL,
    legacy_mission_id UUID NOT NULL,
    owner_id TEXT NOT NULL,
    snapshot_payload JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_sha256, legacy_mission_id)
);

COMMENT ON TABLE legacy_mission_states IS
    'Read-only legacy provenance. Resume requires revalidation under the current runtime.';
