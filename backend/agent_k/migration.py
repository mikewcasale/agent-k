"""Read-only legacy inventory, upload, and idempotent import commands.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .runner import WorkerApiClient
from .stores import (
    JobFailure,
    MigrationCreateRequest,
    MigrationSource,
    MigrationSourceKind,
    PostgresRuntimeStore,
    S3ArtifactStore,
)

__all__ = (
    "LegacyInspection",
    "inspect_legacy_root",
    "main",
    "process_migration",
    "upload_legacy_sources",
)

_CHUNK_SIZE = 1024 * 1024
_MISSION_STATE_SCHEMA = "1.0.0"
_EXPERIMENT_TABLES = {
    "experiments": "record_id",
    "hint_attempts": "id",
    "kaggle_submissions": "record_id",
}


class LegacyInspection(BaseModel):
    """Content-safe dry-run report for a legacy AGENT-K root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    mission_state_files: int = Field(ge=0)
    mission_ids: int = Field(ge=0)
    bare_node_ids: tuple[str, ...]
    completed_phase_claims: dict[str, int]
    experiment_rows: int = Field(ge=0)
    hint_attempt_rows: int = Field(ge=0)
    submission_history_rows: int = Field(ge=0)
    experiment_database_sha256: str | None = None
    warnings: tuple[str, ...] = ()


def inspect_legacy_root(legacy_root: Path) -> LegacyInspection:
    """Inspect legacy files without modifying or exposing their payloads."""
    root = _validate_legacy_root(legacy_root)
    states = _read_legacy_states(root)
    node_ids = sorted(
        {
            str(item["id"])
            for entry in states
            for item in entry["graphSnapshot"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    )
    phase_counts: dict[str, int] = {}
    for entry in states:
        for phase in entry["legacyState"].get("phases_completed", []):
            name = str(phase)
            phase_counts[name] = phase_counts.get(name, 0) + 1
    database = root / "experiments" / "experiments.sqlite"
    counts = {table: 0 for table in _EXPERIMENT_TABLES}
    digest = None
    warnings: list[str] = []
    if database.is_file():
        counts = _sqlite_counts(database)
        digest = _sha256(database)
    else:
        warnings.append("experiment database not found")
    return LegacyInspection(
        mission_state_files=len(states),
        mission_ids=len({entry["legacyMissionId"] for entry in states}),
        bare_node_ids=tuple(node_ids),
        completed_phase_claims=dict(sorted(phase_counts.items())),
        experiment_rows=counts["experiments"],
        hint_attempt_rows=counts["hint_attempts"],
        submission_history_rows=counts["kaggle_submissions"],
        experiment_database_sha256=digest,
        warnings=tuple(warnings),
    )


async def upload_legacy_sources(
    legacy_root: Path,
    *,
    owner_id: str,
    client: WorkerApiClient,
) -> tuple[uuid.UUID, ...]:
    """Upload immutable legacy sources through scoped S3 URLs for later import."""
    if not owner_id.strip():
        raise ValueError("owner_id is required to attach imported missions safely")
    root = _validate_legacy_root(legacy_root)
    migration_ids: list[uuid.UUID] = []
    with tempfile.TemporaryDirectory(prefix="agent-k-migration-") as temporary_name:
        temporary = Path(temporary_name)
        state_bundle = temporary / "mission-states.json"
        _write_state_bundle(root, owner_id, state_bundle)
        migration_ids.append(
            await _upload_source(
                client,
                state_bundle,
                kind=MigrationSourceKind.MISSION_STATE,
                schema_version=_MISSION_STATE_SCHEMA,
                content_type="application/json",
            )
        )
        database = root / "experiments" / "experiments.sqlite"
        if database.is_file():
            database_backup = temporary / "experiments.sqlite"
            _backup_sqlite(database, database_backup)
            _validate_sqlite(database_backup)
            migration_ids.append(
                await _upload_source(
                    client,
                    database_backup,
                    kind=MigrationSourceKind.EXPERIMENT_DB,
                    schema_version="legacy-sqlite-v1",
                    content_type="application/x-sqlite3",
                )
            )
    return tuple(migration_ids)


async def process_migration(
    migration_id: uuid.UUID,
    *,
    store: PostgresRuntimeStore,
    artifacts: S3ArtifactStore,
    work_root: Path,
) -> dict[str, int]:
    """Claim and transactionally import one S3-verified queued migration."""
    work_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    claimed = await store.claim_migration(migration_id)
    token = claimed["processing_token"]
    suffix = ".json" if claimed["source_kind"] == "mission-state" else ".sqlite3"
    source = work_root / f"{migration_id}{suffix}"
    try:
        await artifacts.download_claimed_migration(claimed, source)
        if claimed["source_kind"] == "mission-state":
            mission_states = _migration_mission_records(source, claimed["source_sha256"])
            return await store.complete_legacy_import(
                migration_id,
                token,
                mission_states=mission_states,
            )
        experiments, hints, submissions = _migration_database_records(source)
        return await store.complete_legacy_import(
            migration_id,
            token,
            experiments=experiments,
            hints=hints,
            submissions=submissions,
        )
    except Exception:
        failure = JobFailure(
            code="legacy_import_failed",
            message="Legacy source failed structural validation or transactional import",
            retryable=False,
            category="migration",
        )
        await store.fail_migration_import(migration_id, token, failure)
        raise
    finally:
        source.unlink(missing_ok=True)


async def _upload_source(
    client: WorkerApiClient,
    path: Path,
    *,
    kind: MigrationSourceKind,
    schema_version: str,
    content_type: Literal["application/json", "application/x-sqlite3"],
) -> uuid.UUID:
    size = path.stat().st_size
    digest = _sha256(path)
    idempotency_key = hashlib.sha256(f"{kind.value}:{digest}".encode()).hexdigest()
    request = MigrationCreateRequest(
        worker_id=client.worker_id,
        idempotency_key=idempotency_key,
        source=MigrationSource(
            kind=kind,
            schema_version=schema_version,
            sha256=digest,
            size_bytes=size,
        ),
        content_type=content_type,
    )
    response = await client.create_migration_upload(request)
    await client.upload_signed_file(response.upload, path)
    await client.complete_migration_upload(
        response.migration_id,
        idempotency_key=idempotency_key,
        sha256=digest,
        size_bytes=size,
    )
    return response.migration_id


def _write_state_bundle(root: Path, owner_id: str, destination: Path) -> None:
    bundle = {
        "schemaVersion": _MISSION_STATE_SCHEMA,
        "ownerId": owner_id,
        "resumePolicy": {
            "completedPhaseClaimsAreHistorical": True,
            "independentCandidateValidationRequired": True,
            "restartUnfinishedEvolution": True,
            "submissionHistoryOnly": True,
        },
        "stateEntries": _read_legacy_states(root),
    }
    with destination.open("x", encoding="utf-8") as target:
        os.chmod(destination, 0o600)
        json.dump(bundle, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")


def _read_legacy_states(root: Path) -> list[dict[str, Any]]:
    checkpoint_root = root / "checkpoints"
    if not checkpoint_root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for state_path in sorted(checkpoint_root.glob("*/state.json")):
        if state_path.is_symlink():
            raise ValueError(f"legacy state may not be a symlink: {state_path}")
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"legacy state must be a non-empty graph snapshot: {state_path}")
        latest = next(
            (
                item["state"]
                for item in reversed(raw)
                if isinstance(item, dict) and isinstance(item.get("state"), dict)
            ),
            None,
        )
        if latest is None:
            raise ValueError(f"legacy state has no mission state payload: {state_path}")
        mission_id = uuid.UUID(str(latest.get("mission_id")))
        schema_version = str(latest.get("schema_version", ""))
        if schema_version != _MISSION_STATE_SCHEMA:
            raise ValueError(f"unsupported mission state schema: {schema_version}")
        entries.append(
            {
                "legacyMissionId": str(mission_id),
                "graphSnapshot": raw,
                "legacyState": latest,
            }
        )
    return entries


def _migration_mission_records(
    source: Path,
    source_sha256: str,
) -> tuple[dict[str, Any], ...]:
    bundle = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or bundle.get("schemaVersion") != _MISSION_STATE_SCHEMA:
        raise ValueError("unsupported mission-state migration bundle")
    owner_id = bundle.get("ownerId")
    entries = bundle.get("stateEntries")
    if not isinstance(owner_id, str) or not owner_id or not isinstance(entries, list):
        raise ValueError("mission-state migration bundle is incomplete")
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("mission-state entry must be an object")
        mission_id = uuid.UUID(str(entry.get("legacyMissionId")))
        state = entry.get("legacyState")
        graph = entry.get("graphSnapshot")
        if not isinstance(state, dict) or not isinstance(graph, list):
            raise ValueError("mission-state entry is incomplete")
        if str(state.get("mission_id")) != str(mission_id):
            raise ValueError("legacy mission identity does not match its state")
        prototype = state.get("prototype_code")
        candidate_sha256 = (
            hashlib.sha256(prototype.encode()).hexdigest()
            if isinstance(prototype, str) and prototype
            else None
        )
        bare_node_ids = [
            item["id"]
            for item in graph
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        snapshot_payload = {
            "migration": {
                "kind": "legacy-v1",
                "sourceSha256": source_sha256,
                "legacyMissionId": str(mission_id),
                "competitionId": state.get("competition_id"),
                "criteria": state.get("criteria", {}),
                "legacyState": state,
                "graphSnapshot": graph,
                "bareNodeIds": bare_node_ids,
                "phaseClaims": state.get("phases_completed", []),
                "prototypeCodeSha256": candidate_sha256,
                "resumePolicy": bundle["resumePolicy"],
            }
        }
        records.append(
            {
                "legacyMissionId": str(mission_id),
                "ownerId": owner_id,
                "snapshotPayload": snapshot_payload,
            }
        )
    return tuple(records)


def _migration_database_records(
    source: Path,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    _validate_sqlite(source)
    connection = _open_sqlite(source)
    try:
        groups: list[tuple[dict[str, Any], ...]] = []
        for table, identity_column in _EXPERIMENT_TABLES.items():
            cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {identity_column}")
            groups.append(
                tuple(
                    {
                        "recordId": str(row[identity_column]),
                        "payload": dict(row),
                    }
                    for row in cursor
                )
            )
        return groups[0], groups[1], groups[2]
    finally:
        connection.close()


def _sqlite_counts(path: Path) -> dict[str, int]:
    _validate_sqlite(path)
    connection = _open_sqlite(path)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in _EXPERIMENT_TABLES
        }
    finally:
        connection.close()


def _validate_sqlite(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("legacy experiment source must be a regular SQLite file")
    connection = _open_sqlite(path)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise ValueError("legacy experiment database failed integrity_check")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not set(_EXPERIMENT_TABLES).issubset(tables):
            raise ValueError("legacy experiment database is missing required tables")
        for table, identity_column in _EXPERIMENT_TABLES.items():
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if identity_column not in columns:
                raise ValueError(f"legacy table {table} is missing {identity_column}")
    finally:
        connection.close()


def _open_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _backup_sqlite(source: Path, destination: Path) -> None:
    source_connection = _open_sqlite(source)
    destination_connection = sqlite3.connect(destination)
    os.chmod(destination, 0o600)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _validate_legacy_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir() or root in {Path("/"), Path.home().resolve()}:
        raise ValueError("legacy_root must be a dedicated existing directory")
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


async def _upload_command(arguments: argparse.Namespace) -> int:
    async with WorkerApiClient(
        arguments.api_url,
        arguments.worker_id,
        os.environ["AGENT_K_WORKER_TOKEN"],
    ) as client:
        migration_ids = await upload_legacy_sources(
            arguments.legacy_root,
            owner_id=arguments.owner_id,
            client=client,
        )
    print(json.dumps({"migrationIds": [str(item) for item in migration_ids]}))
    return 0


async def _process_command(arguments: argparse.Namespace) -> int:
    store = PostgresRuntimeStore(os.environ["AGENT_K_DATABASE_URL"])
    await store.open()
    try:
        artifacts = S3ArtifactStore(
            store,
            bucket=os.environ["AGENT_K_S3_BUCKET"],
            region=os.environ.get("AGENT_K_S3_REGION"),
        )
        counts = await process_migration(
            arguments.migration_id,
            store=store,
            artifacts=artifacts,
            work_root=arguments.work_root,
        )
    finally:
        await store.close()
    print(json.dumps({"migrationId": str(arguments.migration_id), "counts": counts}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_k.migration")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--legacy-root", required=True, type=Path)
    upload = commands.add_parser("upload")
    upload.add_argument("--legacy-root", required=True, type=Path)
    upload.add_argument("--owner-id", required=True)
    upload.add_argument("--api-url", required=True)
    upload.add_argument("--worker-id", required=True)
    process = commands.add_parser("process")
    process.add_argument("--migration-id", required=True, type=uuid.UUID)
    process.add_argument("--work-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Inspect, upload, or explicitly process legacy sources."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "inspect":
        report = inspect_legacy_root(arguments.legacy_root)
        print(report.model_dump_json(by_alias=True))
        return 0
    if arguments.command == "upload":
        return asyncio.run(_upload_command(arguments))
    return asyncio.run(_process_command(arguments))


if __name__ == "__main__":
    sys.exit(main())
