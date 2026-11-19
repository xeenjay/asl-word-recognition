#!/usr/bin/env python3
"""Download the complete ASL Citizen test split for the frozen final evaluation.

Authorized only by docs/environment/final-test-preregistration-20261012.json:
the file must exist with the pinned SHA256 and status "frozen". All other logic
(pinned ZIP, HTTP Range, TLS verification, CRC32/SHA256 checks, budgets, atomic
writes, resumable manifests) is imported unchanged from fetch_asl_dataset.py and
fetch_asl_sample.py; only the train/val selection guard is replaced by a
test-only selection of every official test row.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import fetch_asl_dataset as fd
import fetch_asl_sample as base

PREREGISTRATION = "docs/environment/final-test-preregistration-20261012.json"
PREREGISTRATION_SHA256 = "e241b1bbb6a7de21187d31a914a3308321132d2f10ed3e0952fbd652b2664286"
TEST_COUNT = 32941


def require_preregistration(workspace: Path) -> dict:
    path = workspace / PREREGISTRATION
    base.require(path.is_file(), "The frozen final-test preregistration is missing")
    raw = path.read_bytes()
    base.require(base.sha256(raw) == PREREGISTRATION_SHA256, "Final-test preregistration SHA256 differs")
    document = json.loads(raw)
    base.require(document.get("status") == "frozen" and document["data"]["videos"] == TEST_COUNT,
                 "Preregistration is not the frozen 32,941-video test protocol")
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=16, choices=range(1, 25))
    parser.add_argument("--retries", type=int, default=3, choices=range(0, 5))
    parser.add_argument("--max-download-mib", type=int, required=True)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    import requests

    workspace = args.workspace_root.resolve()
    require_preregistration(workspace)
    output = workspace / "data" / "asl_citizen"
    manifest_path = output / "test_all_manifest.json"
    with fd.exclusive_download_lock(output):
        budget = fd.DownloadBudget(args.max_download_mib * 1024 * 1024)
        with requests.Session() as session:
            buffers = base.load_metadata(fd.RetryingReader(session, budget, args.retries), output / ".range-cache")
        records = base.parse_directory(buffers["central"])
        raw_splits, rows_by_split, videos = base.parse_splits(buffers, records)
        rows = rows_by_split["test"]
        base.require(len(rows) == TEST_COUNT, f"Official test split has {len(rows)} rows, expected {TEST_COUNT}")
        selected = base.choose_train(rows, len(rows), args.seed)  # deterministic hash order over all test rows
        base.require(len(selected) == TEST_COUNT and len({r["Video file"] for r in selected}) == TEST_COUNT,
                     "Test selection must contain every official test video once")
        fingerprint = fd.selection_fingerprint(selected, "test")
        previous = fd.previous_records(manifest_path, fingerprint, output)
        split_manifest = {}
        for split, raw in raw_splits.items():
            destination = output / "splits" / f"{split}.csv"
            if destination.is_file():
                base.require(base.sha256(destination.read_bytes()) == base.sha256(raw), f"Existing {split}.csv differs")
            else:
                base.atomic_write(destination, raw)
            split_manifest[split] = dict(path=os.path.relpath(destination, manifest_path.parent),
                                         rows=len(rows_by_split[split]), sha256=base.sha256(raw))
        total_compressed = sum(videos[r["Video file"]]["compressed"] for r in selected)
        total_raw = sum(videos[r["Video file"]]["size"] for r in selected)
        missing_raw = sum(videos[r["Video file"]]["size"] for r in selected
                          if not (output / "videos" / r["Video file"]).is_file())
        manifest = dict(schema_version=2, status="planned" if args.plan_only else "in_progress",
                        purpose="final_test_evaluation_data", preregistration=PREREGISTRATION,
                        preregistration_sha256=PREREGISTRATION_SHA256, created_utc=fd.utc_now(),
                        source_url=base.URL, archive_bytes=base.ARCHIVE_BYTES,
                        central_directory_sha256=base.METADATA["central"][2], path_base="manifest_directory",
                        selection=dict(split="test", count=len(selected), seed=args.seed, algorithm=fd.ALGORITHM,
                                       sha256=fingerprint, video_files=[r["Video file"] for r in selected]),
                        splits=split_manifest, samples=[], failures=[],
                        configuration=dict(workers=args.workers, retries_per_range=args.retries,
                                           max_failures=args.max_failures, tls_verify=True,
                                           manifest_flush_every=200, manifest_flush_seconds=15.0),
                        resume_sha256={r["Video file"]: previous[r["Video file"]]["sha256"]
                                       for r in selected if r["Video file"] in previous},
                        size_estimate=dict(selected_compressed_bytes=total_compressed,
                                           selected_uncompressed_bytes=total_raw, absent_video_bytes=missing_raw),
                        download_budget=budget.snapshot(), download_bytes_this_run=budget.received)
        print(f"Plan: {len(selected)} test videos, {total_compressed:,} compressed bytes, "
              f"{total_raw:,} uncompressed bytes, {missing_raw:,} absent bytes", flush=True)
        if args.plan_only:
            return 0
        base.require(shutil.disk_usage(output).free >= missing_raw + args.workers * base.MAX_RANGE_BYTES * 2 + 2 * 1024 ** 3,
                     "Insufficient free space for the test videos")
        base.write_manifest(manifest_path, manifest)
        with fd.termination_handler():
            result = fd.download_selected(selected, "test", videos, previous, output, manifest_path, budget,
                                          args.retries, args.workers, args.max_failures, manifest, 200, 15.0)
        print(f"{manifest['status']}: {len(manifest['samples'])}/{len(selected)} verified test videos; "
              f"manifest={manifest_path}", flush=True)
        return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except fd.DownloadTerminated:
        sys.exit(143)
    except (base.AuditError, OSError, ValueError) as error:
        sys.exit(f"ASL test download stopped: {error}")
