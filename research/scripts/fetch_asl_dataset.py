#!/usr/bin/env python3
"""Resumable, bounded ASL Citizen train/val downloads from the pinned official ZIP.

Examples (paths are relative to the current working directory):
  python scripts/fetch_asl_dataset.py --split train --count 500 --plan-only
  python scripts/fetch_asl_dataset.py --split train --count 500 --workers 4
  python scripts/fetch_asl_dataset.py --split val --count all --max-download-mib 8192

Only the requested train or validation videos are fetched; test video downloads
are deliberately unsupported. This script preserves sample_manifest.json and
reuses the existing videos/ and .range-cache/. Completed video files are verified
against the pinned ZIP CRC and any previously recorded SHA256 before reuse.
Each range retains strict size/header/TLS checks; HTTP 200 never downloads a ZIP.
"""

import argparse
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

import fetch_asl_sample as base


ALGORITHM = "ascending SHA256(UTF8(str(seed) + NUL + video_filename)); filename tie-break"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class DownloadTerminated(BaseException):
    """SIGTERM exits through the interrupted-manifest cleanup path."""


@contextmanager
def termination_handler():
    def terminate(signum, frame):
        raise DownloadTerminated("Download received SIGTERM")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class DownloadBudget:
    """A shared bound on requested bytes, including failed requests and retries."""

    def __init__(self, maximum):
        base.require(maximum > 0, "Download budget must be positive")
        self.maximum = maximum
        self.requested = 0
        self.received = 0
        self.requests = 0
        self.lock = threading.Lock()

    def reserve(self, length):
        with self.lock:
            base.require(self.requested + length <= self.maximum,
                         f"Shared download budget exceeded ({self.maximum:,} bytes); resume with a larger budget")
            self.requested += length
            self.requests += 1

    def record(self, length):
        with self.lock:
            self.received += length

    def snapshot(self):
        with self.lock:
            return dict(max_bytes=self.maximum, requested_bytes=self.requested,
                        received_bytes=self.received, request_count=self.requests)


class RetryingReader(base.RangeReader):
    def __init__(self, session, budget, retries=2, sleep=time.sleep):
        super().__init__(session, max_download_bytes=budget.maximum)
        self.budget, self.retries, self.sleep = budget, retries, sleep

    def read(self, start, end):
        import requests
        base.require(0 <= start <= end < base.ARCHIVE_BYTES, "Invalid archive range")
        base.require(end - start + 1 <= base.MAX_RANGE_BYTES, "Refusing oversized range")
        for attempt in range(self.retries + 1):
            self.budget.reserve(end - start + 1)
            before = self.bytes_received
            try:
                return super().read(start, end)
            except requests.exceptions.SSLError:
                raise  # Certificate failures need a verified download host, not retries/insecure TLS.
            except (base.TransientHTTPError, requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
                if attempt >= self.retries:
                    raise
                self.sleep(min(2 ** attempt, 8))
            finally:
                self.budget.record(self.bytes_received - before)


@contextmanager
def exclusive_download_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".dataset-download.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise base.AuditError("Another dataset downloader owns this data directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def select_rows(rows_by_split, split, count, seed, selection_path=None):
    base.require(split in ("train", "val"), "Only train and val video downloads are allowed")
    rows = rows_by_split[split]
    if selection_path is None:
        number = len(rows) if count == "all" else int(count)
        base.require(1 <= number <= len(rows), "Count must be positive and within the official split")
        return base.choose_train(rows, number, seed)
    document = json.loads(selection_path.read_text())
    requested = document.get("samples")
    base.require(isinstance(requested, list) and requested, "Selection manifest needs a nonempty samples list")
    lookup = {row["Video file"]: row for row in rows}
    selected, seen = [], set()
    for item in requested:
        base.require(isinstance(item, dict) and item.get("split") == split,
                     "Every selected item must declare the requested train/val split")
        filename = item.get("video_file")
        base.require(isinstance(filename, str) and filename in lookup and filename not in seen,
                     f"Duplicate, unsafe or out-of-split selected video: {filename}")
        row = lookup[filename]
        for given, official in (("gloss", "Gloss"), ("signer_id", "Participant ID"),
                                ("asl_lex_code", "ASL-LEX Code")):
            base.require(given not in item or item[given] == row[official],
                         f"Selection label differs from official metadata: {filename}/{given}")
        seen.add(filename)
        selected.append(row)
    return selected


def selection_fingerprint(rows, split):
    data = [dict(split=split, video_file=row["Video file"], gloss=row["Gloss"],
                 signer_id=row["Participant ID"], asl_lex_code=row["ASL-LEX Code"]) for row in rows]
    return base.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def previous_records(manifest_path, fingerprint, output):
    previous = {}
    # The deployment manifest remains read-only. Its eight hashes can validate
    # files reused in the larger pilot without overwriting that historical sample.
    deployment = output / "sample_manifest.json"
    for path, own_manifest in ((deployment, False), (manifest_path, True)):
        if not path.is_file():
            continue
        document = json.loads(path.read_text())
        same_source = document.get("source_url") == base.URL and document.get("archive_bytes") == base.ARCHIVE_BYTES
        if own_manifest:
            base.require(same_source and document.get("selection", {}).get("sha256") == fingerprint,
                         "Existing manifest belongs to another source/selection; use a new manifest path")
        elif not same_source:
            continue
        # Carry earlier hashes through an interrupted rescan. Otherwise writing
        # an initially empty samples list would discard old SHA protection.
        for filename, digest in document.get("resume_sha256", {}).items():
            previous[filename] = {"sha256": digest}
        for item in document.get("samples", []):
            if item.get("sha256"):
                previous[item["video_file"]] = item
    return previous


def sample_record(index, row, split, record, manifest_path, output, digest, reused):
    destination = output / "videos" / row["Video file"]
    return dict(index=index, split=split, video_file=row["Video file"],
                path=os.path.relpath(destination, manifest_path.parent), gloss=row["Gloss"],
                signer_id=row["Participant ID"], asl_lex_code=row["ASL-LEX Code"],
                zip_entry=record["name"], zip_crc32=f"{record['crc']:08x}",
                size_bytes=record["size"], sha256=digest, reused=reused)


def download_one(index, row, split, record, previous, output, manifest_path, budget, retries):
    import requests
    destination = output / "videos" / row["Video file"]
    digest = base.reusable_video(destination, record, previous)
    reused = digest is not None
    if digest is None:
        with requests.Session() as session:
            # requests verifies TLS by default. Never consult an insecure curl configuration.
            reader = RetryingReader(session, budget, retries)
            raw = base.download_entry(reader, record)
        digest = base.sha256(raw)
        base.atomic_write(destination, raw)
    return sample_record(index, row, split, record, manifest_path, output, digest, reused)


def download_selected(selected, split, videos, previous, output, manifest_path, budget,
                      retries, workers, max_failures, manifest,
                      manifest_flush_every=100, manifest_flush_seconds=10.0):
    base.require(manifest_flush_every > 0 and manifest_flush_seconds > 0,
                 "Manifest flush intervals must be positive")
    completed, failures = {}, []
    next_index = 0
    pending = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    interrupted = False
    last_flush = time.monotonic()
    unflushed = 0

    def save():
        nonlocal last_flush, unflushed
        manifest["samples"] = [completed[index] for index in sorted(completed)]
        manifest["failures"] = failures
        manifest["download_budget"] = budget.snapshot()
        manifest["download_bytes_this_run"] = budget.snapshot()["received_bytes"]
        manifest["updated_utc"] = utc_now()
        base.write_manifest(manifest_path, manifest)
        last_flush = time.monotonic()
        unflushed = 0

    def maybe_save():
        if unflushed and (unflushed >= manifest_flush_every or
                          time.monotonic() - last_flush >= manifest_flush_seconds):
            save()

    try:
        while pending or next_index < len(selected):
            while next_index < len(selected) and len(pending) < workers and len(failures) < max_failures:
                row = selected[next_index]
                future = executor.submit(download_one, next_index, row, split,
                                         videos[row["Video file"]], previous.get(row["Video file"]),
                                         output, manifest_path, budget, retries)
                pending[future] = next_index
                next_index += 1
            if not pending:
                break
            timeout = (max(0.01, manifest_flush_seconds - (time.monotonic() - last_flush))
                       if unflushed else manifest_flush_seconds)
            done, _ = concurrent.futures.wait(pending, timeout=timeout,
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    item = future.result()
                    completed[index] = item
                    unflushed += 1
                    print(f"[{len(completed)}/{len(selected)}] {'Reused' if item['reused'] else 'Downloaded'} "
                          f"{split} {item['video_file']}", flush=True)
                except Exception as error:
                    failures.append(dict(index=index, video_file=selected[index]["Video file"],
                                         split=split, error_type=type(error).__name__, error=str(error)))
                    print(f"Failed {selected[index]['Video file']}: {error}", file=sys.stderr, flush=True)
                    save()  # Errors are durable immediately, irrespective of throttling.
                maybe_save()
            maybe_save()  # Flush timed progress even while remaining requests are slow.
    except BaseException:
        interrupted = True
        manifest["status"] = "interrupted"
        save()
        raise
    finally:
        # Only at most `workers` requests are active. Future invocations verify
        # completed files even if interrupted before their manifest entry is saved.
        executor.shutdown(wait=True, cancel_futures=True)
        if not interrupted:
            manifest["status"] = "complete" if len(completed) == len(selected) else "failed"
            if manifest["status"] == "complete":
                manifest.pop("resume_sha256", None)  # All original hashes now live in samples.
            manifest["not_completed"] = [dict(index=index, split=split, video_file=row["Video file"])
                                         for index, row in enumerate(selected) if index not in completed]
            manifest["finished_utc"] = utc_now()
            save()
    return 0 if manifest["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--count", default="500", help="Positive count or 'all'; ignored with --selection-manifest")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--manifest", type=Path, help="Independent manifest; never sample_manifest.json")
    parser.add_argument("--selection-manifest", type=Path, help="JSON samples[].split/video_file; metadata checked against official split")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--retries", type=int, default=2, choices=range(0, 5), help="Retries after the initial attempt per range")
    parser.add_argument("--max-download-mib", type=int, default=512)
    parser.add_argument("--max-failures", type=int, default=5)
    parser.add_argument("--manifest-flush-every", type=int, default=100,
                        help="Checkpoint manifest every N completed videos; each video is still atomic")
    parser.add_argument("--manifest-flush-seconds", type=float, default=10.0,
                        help="Checkpoint pending progress at this interval; errors/end/interruption flush immediately")
    parser.add_argument("--plan-only", action="store_true", help="Verify metadata/selection and emit manifest; no video downloads")
    args = parser.parse_args(argv)
    if (args.max_download_mib <= 0 or args.max_failures <= 0 or args.manifest_flush_every <= 0 or
            not 0 < args.manifest_flush_seconds < float("inf")):
        parser.error("Download budget, max-failures and finite manifest flush intervals must be positive")
    if args.count != "all" and (not args.count.isdigit() or int(args.count) < 1):
        parser.error("--count must be a positive integer or all")
    import requests

    output = args.workspace_root.resolve() / "data" / "asl_citizen"
    manifest_path = (args.manifest or output / f"{args.split}_{args.count}_manifest.json").resolve()
    base.require(manifest_path != output / "sample_manifest.json", "The eight-sample deployment manifest is immutable here")
    base.require(args.selection_manifest is None or args.selection_manifest.resolve() != manifest_path,
                 "Output manifest must differ from selection input")
    with exclusive_download_lock(output):
        budget = DownloadBudget(args.max_download_mib * 1024 * 1024)
        with requests.Session() as session:
            buffers = base.load_metadata(RetryingReader(session, budget, args.retries), output / ".range-cache")
        records = base.parse_directory(buffers["central"])
        raw_splits, rows_by_split, videos = base.parse_splits(buffers, records)
        selected = select_rows(rows_by_split, args.split, args.count, args.seed, args.selection_manifest)
        fingerprint = selection_fingerprint(selected, args.split)
        previous = previous_records(manifest_path, fingerprint, output)
        split_manifest = {}
        for split, raw in raw_splits.items():
            destination = output / "splits" / f"{split}.csv"
            base.atomic_write(destination, raw)
            split_manifest[split] = dict(path=os.path.relpath(destination, manifest_path.parent),
                                        rows=len(rows_by_split[split]), sha256=base.sha256(raw))
        # The budget includes metadata requests and retries. This conservative
        # estimate is informative; exact limits are enforced before every request.
        total_compressed = sum(videos[row["Video file"]]["compressed"] for row in selected)
        total_raw = sum(videos[row["Video file"]]["size"] for row in selected)
        missing_raw = sum(videos[row["Video file"]]["size"] for row in selected
                          if not (output / "videos" / row["Video file"]).is_file())
        manifest = dict(schema_version=2, status="planned" if args.plan_only else "in_progress",
                        purpose="data_preparation_not_benchmark", created_utc=utc_now(),
                        source_url=base.URL, archive_bytes=base.ARCHIVE_BYTES,
                        central_directory_sha256=base.METADATA["central"][2], path_base="manifest_directory",
                        selection=dict(split=args.split, count=len(selected), seed=args.seed,
                                       algorithm=ALGORITHM if args.selection_manifest is None else "explicit_manifest_order",
                                       sha256=fingerprint,
                                       source_manifest_sha256=base.sha256(args.selection_manifest.read_bytes())
                                       if args.selection_manifest else None,
                                       video_files=[row["Video file"] for row in selected]),
                        splits=split_manifest, samples=[], failures=[],
                        configuration=dict(workers=args.workers, retries_per_range=args.retries,
                                           max_failures=args.max_failures, tls_verify=True,
                                           manifest_flush_every=args.manifest_flush_every,
                                           manifest_flush_seconds=args.manifest_flush_seconds),
                        resume_sha256={row["Video file"]: previous[row["Video file"]]["sha256"]
                                       for row in selected if row["Video file"] in previous},
                        resume_note="Manifest checkpoints may lag completed atomic videos; resume verifies pinned ZIP CRC and any recorded SHA256 before reuse",
                        size_estimate=dict(selected_compressed_bytes=total_compressed,
                                           selected_uncompressed_bytes=total_raw,
                                           absent_video_bytes=missing_raw,
                                           caveat="Headers, metadata, corrupt-file replacement and retries add traffic/space"),
                        download_budget=budget.snapshot(), download_bytes_this_run=budget.received)
        if args.plan_only:
            # Planning must not destroy a partially complete download manifest.
            if manifest_path.exists():
                print(json.dumps(manifest, indent=2))
            else:
                base.write_manifest(manifest_path, manifest)
            print(f"Plan: {len(selected)} {args.split} videos, {total_compressed:,} compressed bytes, "
                  f"{total_raw:,} uncompressed bytes; manifest={manifest_path}", flush=True)
            return 0
        base.require(shutil.disk_usage(output).free >= missing_raw + args.workers * base.MAX_RANGE_BYTES * 2 + 256 * 1024 * 1024,
                     "Insufficient data-volume free space for missing videos and bounded worker buffers")
        base.write_manifest(manifest_path, manifest)
        with termination_handler():
            result = download_selected(selected, args.split, videos, previous, output, manifest_path,
                                       budget, args.retries, args.workers, args.max_failures, manifest,
                                       args.manifest_flush_every, args.manifest_flush_seconds)
        print(f"{manifest['status']}: {len(manifest['samples'])}/{len(selected)} verified {args.split} videos; "
              f"manifest={manifest_path}", flush=True)
        return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DownloadTerminated:
        sys.exit(143)
    except (base.AuditError, OSError, ValueError) as error:
        sys.exit(f"ASL dataset download stopped: {error}")
