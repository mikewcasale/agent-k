"""Acquire a bounded, verified KonIQ-10k MOS acceptance fixture.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import argparse
import binascii
import csv
import hashlib
import json
import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

__all__ = ()

_ARCHIVE_URL: Final = "https://darus.uni-stuttgart.de/api/access/datafile/72286?format=original"
_ARCHIVE_SIZE: Final = 766_965_996
_ARCHIVE_MD5: Final = "dc213332574e8431a86c7ff34e1fa924"
_DATASET_URL: Final = (
    "https://darus.uni-stuttgart.de/dataset.xhtml?"
    "persistentId=doi%3A10.18419%2FDARUS-2435&version=1.0"
)
_METADATA_URL: Final = (
    "https://darus.uni-stuttgart.de/api/datasets/:persistentId/?"
    "persistentId=doi:10.18419/DARUS-2435&version=1.0"
)
_SCORES_URL: Final = "https://darus.uni-stuttgart.de/api/access/datafile/72288?format=original"
_METADATA_SHA256: Final = "0c9bc4c767de8df0e7cad18b27a10d3c1c9a6c0413da46d578783dc35f051f88"
_SCORES_SHA256: Final = "a5173fb66831b2049a581d2396ef6e50d341f2af509e1f9b252d4bd49fa03b5f"
_DEFAULT_ROWS: Final = 120


@dataclass(frozen=True)
class _ZipEntry:
    """Central-directory metadata required for one ranged extraction."""

    path: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one local artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_verified(
    client: httpx.Client,
    url: str,
    destination: Path,
    expected_sha256: str,
) -> None:
    """Download one small publisher artifact and enforce its pinned digest."""
    response = client.get(url)
    response.raise_for_status()
    destination.write_bytes(response.content)
    if _sha256(destination) != expected_sha256:
        raise RuntimeError(f"publisher artifact digest changed: {destination.name}")


def _fetch_range(client: httpx.Client, archive_url: str, start: int, end: int) -> bytes:
    """Fetch one exact archive interval and reject a server ignoring Range."""
    response = client.get(
        archive_url,
        headers={"Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"},
    )
    response.raise_for_status()
    expected_range = f"bytes {start}-{end}/{_ARCHIVE_SIZE}"
    if response.status_code != 206 or response.headers.get("content-range") != expected_range:
        raise RuntimeError(
            f"archive server did not honor the exact byte range: {response.status_code}, "
            f"{response.headers.get('content-range')}"
        )
    expected_size = end - start + 1
    if len(response.content) != expected_size:
        raise RuntimeError("archive range length differs from Content-Range")
    return response.content


def _resolve_archive_url(client: httpx.Client) -> str:
    """Resolve the publisher endpoint once without logging its signed URL."""
    response = client.get(
        _ARCHIVE_URL,
        headers={"Accept-Encoding": "identity", "Range": "bytes=0-0"},
    )
    response.raise_for_status()
    if response.status_code != 206 or response.headers.get("content-range") != (
        f"bytes 0-0/{_ARCHIVE_SIZE}"
    ):
        raise RuntimeError("archive endpoint does not support exact byte ranges")
    return str(response.url)


def _read_central_directory(
    client: httpx.Client,
    archive_url: str,
) -> dict[str, _ZipEntry]:
    """Read the standard ZIP central directory without downloading the archive."""
    tail_size = min(65_557, _ARCHIVE_SIZE)
    tail_start = _ARCHIVE_SIZE - tail_size
    tail = _fetch_range(client, archive_url, tail_start, _ARCHIVE_SIZE - 1)
    end_offset = tail.rfind(b"PK\x05\x06")
    if end_offset < 0:
        raise RuntimeError("ZIP end-of-central-directory record is missing")
    fields = struct.unpack_from("<4s4H2LH", tail, end_offset)
    _, disk, directory_disk, disk_entries, total_entries, size, offset, comment = fields
    if disk or directory_disk or disk_entries != total_entries:
        raise RuntimeError("multi-disk ZIP archives are unsupported")
    if end_offset + 22 + comment != len(tail):
        raise RuntimeError("ZIP end-of-central-directory record is inconsistent")
    directory = _fetch_range(client, archive_url, offset, offset + size - 1)
    entries: dict[str, _ZipEntry] = {}
    position = 0
    for _ in range(total_entries):
        header = struct.unpack_from("<4s6H3L5H2L", directory, position)
        if header[0] != b"PK\x01\x02":
            raise RuntimeError("ZIP central-directory signature is invalid")
        flags, method = header[3], header[4]
        crc32, compressed_size, uncompressed_size = header[7:10]
        name_length, extra_length, comment_length = header[10:13]
        local_offset = header[16]
        name_start = position + 46
        name_bytes = directory[name_start : name_start + name_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        archive_path = name_bytes.decode(encoding)
        entries[Path(archive_path).name] = _ZipEntry(
            path=archive_path,
            flags=flags,
            method=method,
            crc32=crc32,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_offset=local_offset,
        )
        position = name_start + name_length + extra_length + comment_length
    if position != len(directory):
        raise RuntimeError("ZIP central-directory size is inconsistent")
    return entries


def _extract_entry(client: httpx.Client, archive_url: str, entry: _ZipEntry) -> bytes:
    """Fetch, decompress, and CRC-check one central-directory entry."""
    if entry.flags & 0x1:
        raise RuntimeError(f"encrypted ZIP entry is unsupported: {entry.path}")
    local = _fetch_range(client, archive_url, entry.local_offset, entry.local_offset + 29)
    header = struct.unpack("<4s5H3L2H", local)
    if header[0] != b"PK\x03\x04":
        raise RuntimeError(f"local ZIP signature is invalid: {entry.path}")
    name_length, extra_length = header[-2:]
    data_start = entry.local_offset + 30 + name_length + extra_length
    compressed = _fetch_range(
        client,
        archive_url,
        data_start,
        data_start + entry.compressed_size - 1,
    )
    if entry.method == 0:
        content = compressed
    elif entry.method == 8:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"unsupported ZIP compression method {entry.method}: {entry.path}")
    if len(content) != entry.uncompressed_size:
        raise RuntimeError(f"uncompressed size mismatch: {entry.path}")
    if binascii.crc32(content) & 0xFFFFFFFF != entry.crc32:
        raise RuntimeError(f"CRC-32 mismatch: {entry.path}")
    return content


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    """Write one deterministic UTF-8 CSV file."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _select_rows(
    scores_path: Path,
    row_count: int,
    selection_offset: int,
) -> list[dict[str, str]]:
    """Select deterministic MOS-quantile rows spanning the publisher table."""
    with scores_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not 5 <= row_count <= len(rows):
        raise ValueError(f"rows must be between 5 and {len(rows)}")
    rows.sort(key=lambda row: (float(row["MOS"]), row["image_name"]))
    base_positions = [
        round(index * (len(rows) - 1) / (row_count - 1)) for index in range(row_count)
    ]
    positions = _offset_positions(base_positions, len(rows), selection_offset)
    selected = [rows[position] for position in positions]
    if len({row["image_name"] for row in selected}) != row_count:
        raise RuntimeError("quantile selection produced duplicate images")
    if any(not math.isfinite(float(row["MOS"])) for row in selected):
        raise RuntimeError("selected MOS contains a non-finite value")
    return selected


def _offset_positions(
    base_positions: list[int],
    source_size: int,
    selection_offset: int,
) -> list[int]:
    """Shift a quantile selection to a deterministic disjoint neighboring fold."""
    if selection_offset < 0:
        raise ValueError("selection_offset must be non-negative")
    if selection_offset == 0:
        return base_positions
    base = set(base_positions)
    shifted: list[int] = []
    for position in base_positions:
        candidates = (position + selection_offset, position - selection_offset)
        selected = next(
            (
                candidate
                for candidate in candidates
                if 0 <= candidate < source_size
                and candidate not in base
                and candidate not in shifted
            ),
            None,
        )
        if selected is None:
            raise ValueError("selection_offset cannot produce a disjoint fixture")
        shifted.append(selected)
    return shifted


def _prepare_output(output_root: Path) -> tuple[Path, Path, Path]:
    """Create the empty fixture directory structure without overwriting data."""
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_root}")
    train_images = output_root / "train_images"
    test_images = output_root / "test_images"
    source = output_root / "source"
    train_images.mkdir()
    test_images.mkdir()
    source.mkdir()
    return train_images, test_images, source


def _build_fixture(output_root: Path, row_count: int, selection_offset: int) -> None:
    """Build a verified natural-image MOS fixture using bounded range reads."""
    train_images, test_images, source = _prepare_output(output_root)
    metadata_path = source / "dataverse-metadata.json"
    scores_path = source / "koniq10k_scores_and_distributions.csv.txt"
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        _download_verified(client, _METADATA_URL, metadata_path, _METADATA_SHA256)
        _download_verified(client, _SCORES_URL, scores_path, _SCORES_SHA256)
        selected = _select_rows(scores_path, row_count, selection_offset)
        archive_url = _resolve_archive_url(client)
        entries = _read_central_directory(client, archive_url)
        train_rows: list[dict[str, object]] = []
        predict_rows: list[dict[str, object]] = []
        labels: list[dict[str, object]] = []
        for index, row in enumerate(selected):
            image_name = row["image_name"]
            entry = entries.get(image_name)
            if entry is None:
                raise RuntimeError(f"official archive is missing labeled image: {image_name}")
            row_id = f"koniq-{Path(image_name).stem}"
            target = float(row["MOS"])
            is_prediction = index % 5 == 0
            destination = (test_images if is_prediction else train_images) / image_name
            destination.write_bytes(_extract_entry(client, archive_url, entry))
            if is_prediction:
                predict_rows.append({"row_id": row_id, "image": image_name})
                labels.append({"row_id": row_id, "target": target})
            else:
                train_rows.append({"row_id": row_id, "image": image_name, "target": target})

    _write_csv(output_root / "train.csv", ("row_id", "image", "target"), train_rows)
    _write_csv(output_root / "test.csv", ("row_id", "image"), predict_rows)
    _write_csv(output_root / "sample_submission.csv", ("row_id", "target"), labels)
    files = {
        path.relative_to(output_root).as_posix(): _sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    provenance = {
        "task": "image_regression",
        "dataset": "KonIQ-10k IQA Database",
        "dataset_doi": "10.18419/DARUS-2435",
        "dataset_version": "1.0",
        "source_url": _DATASET_URL,
        "label_provenance": (
            "Publisher MOS column from Dataverse data file 72288, "
            "koniq10k_scores_and_distributions.csv"
        ),
        "license": (
            "Dataset metadata and scores are CC BY 4.0; individual YFCC100M-derived "
            "images retain their source Creative Commons licenses per publisher terms."
        ),
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive": {
            "datafile_id": 72286,
            "filename": "koniq10k_512x384.zip",
            "size_bytes": _ARCHIVE_SIZE,
            "md5": _ARCHIVE_MD5,
            "selection": (
                f"{row_count} evenly spaced rows after sorting the complete MOS table by MOS "
                "and image_name; every fifth selected row is prediction-only"
                + ("" if selection_offset == 0 else f"; selection offset is {selection_offset}")
            ),
        },
        "counts": {"train": len(train_rows), "predict": len(predict_rows)},
        "mos_range": {
            "minimum": min(float(row["MOS"]) for row in selected),
            "maximum": max(float(row["MOS"]) for row in selected),
        },
        "files": files,
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(train_rows)} training and {len(predict_rows)} held-out rows to {output_root}"
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=_DEFAULT_ROWS)
    parser.add_argument("--selection-offset", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Acquire the requested bounded fixture."""
    arguments = _parser().parse_args(argv)
    _build_fixture(
        arguments.output.resolve(),
        arguments.rows,
        arguments.selection_offset,
    )


if __name__ == "__main__":
    main()
