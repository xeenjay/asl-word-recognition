#!/usr/bin/env python3
"""Fetch a deterministic train-only ASL Citizen deployment sample, never its ZIP.

Requires requests. Example: python scripts/fetch_asl_sample.py --count 8
All paths in sample_manifest.json are relative to that manifest's directory.
The complete official split CSVs are preserved; only selected train videos are
downloaded. The archive and metadata hashes are pinned to the 2026-09-22 audit.
An archive change fails closed and needs a new metadata audit, not a fallback to
downloading the whole ZIP. This sample is for deployment, not benchmark scores.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile
import zlib


URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
ARCHIVE_BYTES = 45924134223
CENTRAL_OFFSET = 45913456045
CENTRAL_BYTES = 10678080
ENTRY_COUNT = 83406
MAX_RANGE_BYTES = 64 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
METADATA = {
    "tail": (ARCHIVE_BYTES - 65536, ARCHIVE_BYTES - 1,
             "61e93d1a45ca329684e9331400b96f36c50f056c940902245bad6f167eb113be"),
    "central": (CENTRAL_OFFSET, CENTRAL_OFFSET + CENTRAL_BYTES - 1,
                "1add51fae2692b363a72fa7480f371fd7bfabe486c94ff69bead3da5e4f53e11"),
    "splits": (0, 1661160,
               "2f0d300394fd860f9d12fbd19bfa46e16c7b8496a022492faed309114a08a70f"),
}


class AuditError(RuntimeError):
    """The remote response or archive did not satisfy the pinned contract."""


class TransientHTTPError(AuditError):
    """Retryable server status, rejected before consuming its body."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_manifest(path, manifest):
    atomic_write(path, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())


class RangeReader:
    """Validate headers BEFORE reading body; enforce a bound while streaming."""

    def __init__(self, session, max_download_bytes=MAX_DOWNLOAD_BYTES):
        self.session = session
        self.bytes_received = 0
        self.bytes_requested = 0
        self.max_download_bytes = max_download_bytes

    def read(self, start, end):
        length = end - start + 1
        require(0 <= start <= end < ARCHIVE_BYTES, "Invalid archive range")
        require(length <= MAX_RANGE_BYTES, "Refusing oversized range")
        require(self.bytes_requested + length <= self.max_download_bytes,
                f"This run exceeded the {self.max_download_bytes:,}-byte download budget")
        self.bytes_requested += length
        with self.session.get(
            URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            stream=True, timeout=(20, 90), allow_redirects=False,
        ) as response:
            # In particular, reject a server ignoring Range with HTTP 200 before
            # consuming any response body. Redirects also fail closed.
            if response.status_code in (429, 500, 502, 503, 504):
                raise TransientHTTPError(f"Transient HTTP {response.status_code}; body not read")
            require(response.status_code == 206,
                    f"Expected HTTP 206, received {response.status_code}; body not read")
            require(response.headers.get("Content-Range") == f"bytes {start}-{end}/{ARCHIVE_BYTES}",
                    "Unexpected Content-Range; body not read")
            require(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                    "Unexpected HTTP content encoding; body not read")
            if "Content-Length" in response.headers:
                require(response.headers["Content-Length"] == str(length),
                        "Unexpected Content-Length; body not read")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=min(length + 1, 64 * 1024)):
                self.bytes_received += len(chunk)
                require(len(body) + len(chunk) <= length, "Response exceeds requested range")
                body.extend(chunk)
            require(len(body) == length, "Truncated range response")
            return bytes(body)


def load_metadata(reader, cache_dir):
    buffers = {}
    for label, (start, end, digest) in METADATA.items():
        path = cache_dir / f"asl-citizen-{label}-audit.bin"
        # The earlier audit cached a slightly longer prefix. Read only this
        # exact interval and validate its pinned hash before reusing that cache.
        data = None
        if path.is_file():
            with path.open("rb") as handle:
                data = handle.read(end - start + 1)
        if data is None or len(data) != end - start + 1 or sha256(data) != digest:
            print(f"Fetching pinned {label} metadata ({end - start + 1:,} bytes)", flush=True)
            data = reader.read(start, end)
            require(sha256(data) == digest, f"{label} metadata SHA256 differs from audited archive")
            atomic_write(path, data)
        buffers[label] = data
    tail = buffers["tail"]
    offset = tail.rfind(b"PK\x06\x06")
    require(offset >= 0, "Missing ZIP64 end record")
    end_record = struct.unpack_from("<4sQ2H2I4Q", tail, offset)
    require(end_record[7:10] == (ENTRY_COUNT, CENTRAL_BYTES, CENTRAL_OFFSET),
            "ZIP64 directory differs from the audited archive")
    return buffers


def parse_directory(data):
    position, records = 0, {}
    while position < len(data):
        require(position + 46 <= len(data), "Truncated ZIP central header")
        header = struct.unpack_from("<4s6H3I5H2I", data, position)
        (signature, _, _, flags, method, _, _, crc, compressed, size,
         name_length, extra_length, comment_length, disk, _, _, offset) = header
        require(signature == b"PK\x01\x02" and disk == 0, "Invalid or multipart ZIP")
        finish = position + 46 + name_length + extra_length + comment_length
        require(finish <= len(data), "Truncated ZIP directory entry")
        name = data[position + 46:position + 46 + name_length].decode(
            "utf-8" if flags & 0x800 else "cp437")
        extra = data[position + 46 + name_length:position + 46 + name_length + extra_length]
        extra_position = 0
        while extra_position < len(extra):
            require(extra_position + 4 <= len(extra), "Truncated ZIP extra header")
            tag, length = struct.unpack_from("<HH", extra, extra_position)
            value = extra[extra_position + 4:extra_position + 4 + length]
            require(len(value) == length, "Truncated ZIP extra value")
            extra_position += 4 + length
            if tag == 1:
                cursor = 0
                values = [size, compressed, offset]
                for index, current in enumerate(values):
                    if current == 0xFFFFFFFF:
                        require(cursor + 8 <= len(value), "Incomplete ZIP64 field")
                        values[index] = struct.unpack_from("<Q", value, cursor)[0]
                        cursor += 8
                size, compressed, offset = values
        require(name not in records, f"Duplicate ZIP path: {name}")
        require(not flags & 1, "Encrypted ZIP entries are unsupported")
        require(method in (0, 8), f"Unsupported ZIP method: {method}")
        require(0 <= offset < CENTRAL_OFFSET, "ZIP local entry outside payload")
        records[name] = dict(name=name, size=size, compressed=compressed, offset=offset,
                             method=method, flags=flags, crc=crc)
        position = finish
    require(len(records) == ENTRY_COUNT, "Unexpected ZIP entry count")
    return records


def local_payload_offset(header, name_extra, record):
    fields = struct.unpack("<4s5H3I2H", header)
    signature, _, flags, method, _, _, crc, compressed, size, name_length, extra_length = fields
    require(signature == b"PK\x03\x04", "Invalid local ZIP header")
    require(flags == record["flags"] and method == record["method"], "ZIP header mismatch")
    require(len(name_extra) == name_length + extra_length, "Truncated local ZIP name/extra")
    name = name_extra[:name_length].decode("utf-8" if flags & 0x800 else "cp437")
    require(name == record["name"], "ZIP entry name mismatch")
    if not flags & 8:  # When bit 3 is set, authoritative sizes/CRC are in the directory.
        require(crc == record["crc"], "Local ZIP CRC differs from directory")
        require(compressed in (record["compressed"], 0xFFFFFFFF) and
                size in (record["size"], 0xFFFFFFFF), "Local ZIP sizes differ from directory")
    return 30 + name_length + extra_length


def decompress_verified(compressed, record):
    require(len(compressed) == record["compressed"], "Compressed payload length mismatch")
    require(record["size"] <= MAX_RANGE_BYTES, "Refusing oversized uncompressed entry")
    if record["method"] == 8:
        decompressor = zlib.decompressobj(-15)
        raw = decompressor.decompress(compressed, record["size"] + 1)
        require(decompressor.eof and not decompressor.unused_data and
                not decompressor.unconsumed_tail, "Invalid or oversized deflate payload")
    else:
        raw = compressed
    require(len(raw) == record["size"], "Uncompressed ZIP size mismatch")
    require(zlib.crc32(raw) == record["crc"], f"ZIP CRC32 failed: {record['name']}")
    return raw


def extract_buffer_entry(buffer, record):
    start = record["offset"]
    header = buffer[start:start + 30]
    require(len(header) == 30, "Local ZIP header outside metadata buffer")
    name_length, extra_length = struct.unpack_from("<HH", header, 26)
    name_extra = buffer[start + 30:start + 30 + name_length + extra_length]
    payload_start = start + local_payload_offset(header, name_extra, record)
    return decompress_verified(buffer[payload_start:payload_start + record["compressed"]], record)


def download_entry(reader, record):
    require(0 < record["compressed"] <= MAX_RANGE_BYTES and
            0 < record["size"] <= MAX_RANGE_BYTES, "Video exceeds per-entry limit")
    start = record["offset"]
    header = reader.read(start, start + 29)
    name_length, extra_length = struct.unpack_from("<HH", header, 26)
    require(name_length > 0, "Empty ZIP entry name")
    name_extra = reader.read(start + 30, start + 29 + name_length + extra_length)
    payload_start = start + local_payload_offset(header, name_extra, record)
    payload_end = payload_start + record["compressed"] - 1
    require(payload_end < CENTRAL_OFFSET, "ZIP payload exceeds archive data section")
    return decompress_verified(reader.read(payload_start, payload_end), record)


def parse_splits(buffers, records):
    raw_splits, rows_by_split = {}, {}
    expected = {"train": (40154, 35), "val": (10304, 6), "test": (32941, 11)}
    all_names, signer_sets = set(), {}
    for split, (row_count, signer_count) in expected.items():
        raw = extract_buffer_entry(buffers["splits"], records[f"ASL_Citizen/splits/{split}.csv"])
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        require(reader.fieldnames == ["Participant ID", "Video file", "Gloss", "ASL-LEX Code"],
                "Unexpected official split fields")
        rows = list(reader)
        names = {row["Video file"] for row in rows}
        signers = {row["Participant ID"] for row in rows}
        require(len(rows) == row_count and len(signers) == signer_count, "Split counts differ")
        require(len({row["Gloss"] for row in rows}) == 2731, "Split gloss count differs")
        require(len(names) == len(rows) and not names & all_names, "Duplicated split filenames")
        require(all(PurePosixPath(name).name == name and "\\" not in name for name in names),
                "Unsafe video filename")
        all_names.update(names)
        signer_sets[split] = signers
        raw_splits[split], rows_by_split[split] = raw, rows
    require(all(not signer_sets[a] & signer_sets[b]
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
            "Official signer splits overlap")
    videos = {PurePosixPath(name).name: entry for name, entry in records.items()
              if name.endswith(".mp4")}
    require(len(videos) == 83399 and set(videos) == all_names, "Labels and ZIP videos differ")
    return raw_splits, rows_by_split, videos


def choose_train(rows, count, seed):
    # A hash ranking has a stable specification across Python versions. Increasing
    # --count preserves the existing prefix. No label or inference result is used.
    return sorted(rows, key=lambda row: (
        sha256(f"{seed}\0{row['Video file']}".encode("utf-8")), row["Video file"]
    ))[:count]


def reusable_video(path, record, previous):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return None
    raw = path.read_bytes()
    digest = sha256(raw)
    if zlib.crc32(raw) != record["crc"]:
        return None
    if previous is not None and previous.get("sha256") != digest:
        return None
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=8, help="Number of train videos, 1–32 (default: 8)")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--cache-dir", type=Path, help="Optional directory of pinned metadata ranges")
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 32:
        parser.error("--count must be between 1 and 32")
    import requests

    output = args.workspace_root.resolve() / "data" / "asl_citizen"
    cache = args.cache_dir or output / ".range-cache"
    manifest_path = output / "sample_manifest.json"
    old_samples = {}
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text())
            if old.get("source_url") == URL and old.get("archive_bytes") == ARCHIVE_BYTES:
                old_samples = {sample["video_file"]: sample for sample in old.get("samples", [])}
        except (ValueError, KeyError, TypeError):
            pass  # Existing files still need an authoritative ZIP CRC check.

    with requests.Session() as session:
        reader = RangeReader(session)
        buffers = load_metadata(reader, cache)
        records = parse_directory(buffers["central"])
        raw_splits, rows_by_split, videos = parse_splits(buffers, records)
        split_manifest = {}
        for split, raw in raw_splits.items():
            relative = f"splits/{split}.csv"
            atomic_write(output / relative, raw)
            record = records[f"ASL_Citizen/splits/{split}.csv"]
            split_manifest[split] = dict(path=relative, rows=len(rows_by_split[split]),
                                        sha256=sha256(raw), zip_crc32=f"{record['crc']:08x}")
        selected = choose_train(rows_by_split["train"], args.count, args.seed)
        manifest = dict(
            schema_version=1, status="in_progress", purpose="deployment_smoke_test_not_benchmark",
            source_url=URL, archive_bytes=ARCHIVE_BYTES,
            central_directory_sha256=METADATA["central"][2], path_base="manifest_directory",
            selection=dict(split="train", count=args.count, seed=args.seed,
                           algorithm="ascending SHA256(UTF8(str(seed) + NUL + video_filename)); filename tie-break"),
            splits=split_manifest, samples=[], download_bytes_this_run=reader.bytes_received,
        )
        write_manifest(manifest_path, manifest)
        for index, row in enumerate(selected):
            filename = row["Video file"]
            record = videos[filename]
            relative = f"videos/{filename}"
            destination = output / relative
            digest = reusable_video(destination, record, old_samples.get(filename))
            reused = digest is not None
            if digest is None:
                print(f"[{index + 1}/{len(selected)}] Fetching train video {filename}", flush=True)
                raw = download_entry(reader, record)
                digest = sha256(raw)
                atomic_write(destination, raw)
            else:
                print(f"[{index + 1}/{len(selected)}] Reusing verified train video {filename}", flush=True)
            manifest["samples"].append(dict(
                index=index, split="train", video_file=filename, path=relative,
                gloss=row["Gloss"], signer_id=row["Participant ID"], asl_lex_code=row["ASL-LEX Code"],
                zip_entry=record["name"], zip_crc32=f"{record['crc']:08x}",
                size_bytes=record["size"], sha256=digest, reused=reused,
            ))
            manifest["download_bytes_this_run"] = reader.bytes_received
            write_manifest(manifest_path, manifest)
        manifest["status"] = "complete"
        write_manifest(manifest_path, manifest)
    print(f"Saved {len(selected)} verified train videos and full official splits: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditError, OSError, ValueError, struct.error, zlib.error) as error:
        sys.exit(f"ASL sample download stopped: {error}")
