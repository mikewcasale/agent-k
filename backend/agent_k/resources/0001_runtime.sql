CREATE TABLE IF NOT EXISTS runtime_missions (
    mission_id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT '1.0.0',
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'paused', 'completed', 'failed', 'canceled', 'migrated')),
    state JSONB NOT NULL DEFAULT '{}'::JSONB,
    state_sha256 TEXT,
    requested_control TEXT NOT NULL DEFAULT 'continue'
        CHECK (requested_control IN ('continue', 'pause', 'cancel')),
    control_reason TEXT,
    control_requested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS runtime_jobs (
    job_id UUID PRIMARY KEY,
    mission_id UUID NOT NULL REFERENCES runtime_missions(mission_id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'leased', 'running', 'paused', 'completed', 'failed', 'canceled')),
    priority INTEGER NOT NULL DEFAULT 0,
    command JSONB NOT NULL,
    workspace JSONB NOT NULL,
    requirements TEXT[] NOT NULL DEFAULT '{}',
    idempotency_key TEXT NOT NULL UNIQUE,
    attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts >= 1),
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    worker_id TEXT,
    lease_id UUID,
    lease_expires_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    progress DOUBLE PRECISION CHECK (progress IS NULL OR (progress >= 0 AND progress <= 100)),
    current_phase TEXT,
    current_task_id TEXT,
    result JSONB,
    error JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS runtime_jobs_claim_idx
    ON runtime_jobs (priority DESC, available_at, created_at)
    WHERE status IN ('queued', 'leased', 'running');

CREATE INDEX IF NOT EXISTS runtime_jobs_mission_idx
    ON runtime_jobs (mission_id, created_at);

CREATE TABLE IF NOT EXISTS runtime_job_attempts (
    job_id UUID NOT NULL REFERENCES runtime_jobs(job_id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL,
    lease_id UUID NOT NULL,
    worker_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    outcome TEXT CHECK (outcome IS NULL OR outcome IN ('completed', 'failed', 'requeued', 'paused', 'canceled', 'fenced')),
    error JSONB,
    PRIMARY KEY (job_id, attempt),
    UNIQUE (lease_id)
);

CREATE TABLE IF NOT EXISTS runtime_job_idempotency (
    job_id UUID NOT NULL REFERENCES runtime_jobs(job_id) ON DELETE CASCADE,
    operation TEXT NOT NULL CHECK (operation IN ('complete', 'fail')),
    idempotency_key TEXT NOT NULL,
    response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, operation, idempotency_key)
);

CREATE TABLE IF NOT EXISTS runtime_events (
    stream_sequence BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL UNIQUE,
    mission_id UUID NOT NULL REFERENCES runtime_missions(mission_id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES runtime_jobs(job_id) ON DELETE CASCADE,
    job_sequence INTEGER NOT NULL CHECK (job_sequence >= 0),
    event_type TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, job_sequence)
);

CREATE INDEX IF NOT EXISTS runtime_events_replay_idx
    ON runtime_events (mission_id, stream_sequence);

CREATE TABLE IF NOT EXISTS runtime_snapshots (
    snapshot_id UUID PRIMARY KEY,
    mission_id UUID NOT NULL REFERENCES runtime_missions(mission_id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES runtime_jobs(job_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    schema_version TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type = 'application/json'),
    payload JSONB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, sequence)
);

CREATE INDEX IF NOT EXISTS runtime_snapshots_mission_idx
    ON runtime_snapshots (mission_id, sequence);
