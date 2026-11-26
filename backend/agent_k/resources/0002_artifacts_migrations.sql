CREATE TABLE IF NOT EXISTS runtime_artifacts (
    artifact_id UUID PRIMARY KEY,
    mission_id UUID NOT NULL REFERENCES runtime_missions(mission_id) ON DELETE CASCADE,
    job_id UUID REFERENCES runtime_jobs(job_id) ON DELETE SET NULL,
    attempt INTEGER,
    role TEXT NOT NULL,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL,
    bucket TEXT NOT NULL,
    object_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'awaiting-upload'
        CHECK (status IN ('awaiting-upload', 'available', 'invalid', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    available_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS runtime_artifacts_mission_idx
    ON runtime_artifacts (mission_id, created_at);

CREATE TABLE IF NOT EXISTS runtime_migrations (
    migration_id UUID PRIMARY KEY,
    worker_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('mission-state', 'experiment-db')),
    source_schema_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    content_type TEXT NOT NULL
        CHECK (content_type IN ('application/json', 'application/x-sqlite3')),
    bucket TEXT NOT NULL,
    object_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'awaiting-upload'
        CHECK (status IN ('awaiting-upload', 'queued', 'running', 'completed', 'failed')),
    counts JSONB,
    error JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    queued_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS legacy_source_ledger (
    source_kind TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_schema_version TEXT NOT NULL,
    migration_id UUID NOT NULL REFERENCES runtime_migrations(migration_id) ON DELETE RESTRICT,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    counts JSONB NOT NULL DEFAULT '{}'::JSONB,
    PRIMARY KEY (source_kind, source_sha256)
);

CREATE TABLE IF NOT EXISTS legacy_experiments (
    source_sha256 TEXT NOT NULL,
    record_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_sha256, record_id)
);

CREATE TABLE IF NOT EXISTS legacy_hint_attempts (
    source_sha256 TEXT NOT NULL,
    record_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_sha256, record_id)
);

CREATE TABLE IF NOT EXISTS legacy_submissions (
    source_sha256 TEXT NOT NULL,
    record_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    history_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (history_only),
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_sha256, record_id)
);

COMMENT ON TABLE legacy_submissions IS
    'Imported submission history only. This table is intentionally not connected to runtime_jobs.';
