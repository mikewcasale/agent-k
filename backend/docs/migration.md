# Legacy state migration

The clean cutover provides no old import-path shims. Legacy mission JSON and experiment SQLite data
must pass through the explicit migration workflow in `agent_k.migration`.

## Inspect before upload

The inspect command is read-only. It reports counts, schema claims, node identifiers, and SHA-256
digests without printing saved prompts, dataset values, credentials, submission content, or model
outputs.

```bash
cd backend
.venv/bin/python -m agent_k.migration inspect --legacy-root /explicit/legacy/root
```

Use an explicit dedicated legacy root. Do not point migration commands at a home directory,
repository root, or filesystem root.

## Upload and import

The upload command bundles validated state, hashes each source, requests a short-lived signed PUT,
uploads directly to S3, and calls the completion endpoint. The API performs an S3 `HEAD` with
checksum verification before setting the migration to `queued`.

Queued imports are claimed with a database processing lease and transactionally written through a
source ledger. Reprocessing the same source digest is idempotent. Imported records preserve source
provenance and distinguish historical submission records from permission to submit again.

An authorized operator processes each queued identifier from a dedicated private work root:

```bash
cd backend
.venv/bin/python -m agent_k.migration process \
  --migration-id <verified-migration-id> \
  --work-root /explicit/private/migration-work
```

The processing role requires the database and storage configuration; ordinary mission workers do
not. A `queued` response proves only that upload verification succeeded, not that import finished.

## Sensitive saved state

- Keep source and temporary files mode `0600` beneath a mode `0700` directory.
- Never log or return raw legacy state or SQLite rows.
- Treat prompts, paths, submission history, owner mappings, and artifact references as sensitive.
- Reject unrecognized schemas, path traversal, oversized bundles, malformed JSON, and invalid
  SQLite structure before any import.
- Rebase only validated workspace paths during resume; never trust an archived absolute path.
- Remove local temporary copies after checksum validation and transactional completion.

Migration completion counts are operational evidence of imported rows, not proof that an old
mission can safely resume. Resume acceptance additionally requires schema validation, artifact
availability, and an owner-authorized real worker run.
