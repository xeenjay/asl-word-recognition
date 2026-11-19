#!/usr/bin/env python3
"""Pose caches for the frozen final test evaluation (original and flipped video).

Authorized only by docs/environment/final-test-preregistration-20261012.json.
Every clip is processed by the unchanged prepare_poses.process_one; the worker
initializer and flip installation come unchanged from prepare_poet_mirror_cache.
This file only replaces the train/validation selection guard with a test-only
selection and records split="test" in the index. Pipeline hashes must equal the
train/validation v2 pipeline (original) and the preregistered mirror pipeline.

Modes:
  val-identity  22 validation clips (min(4, n) per signer, label-free hash order)
                re-extracted with this wrapper must be array-identical to the
                frozen v2 validation cache. Required before test extraction.
  original      all 32,941 test clips, unflipped.
  flipped       all 32,941 test clips, horizontally flipped video (mirror TTA).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import prepare_poses as base
import prepare_poet_mirror_cache as mirror

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = "docs/environment/final-test-preregistration-20261012.json"
PREREGISTRATION_SHA256 = "e241b1bbb6a7de21187d31a914a3308321132d2f10ed3e0952fbd652b2664286"
MIRROR_PIPELINE_SHA256 = "8af5b135fdc1ed6defc8666dfaffe0450caada1d57ab6f1f4b9a894818fb2ed9"
TEST_COUNT = 32941
VAL_MANIFEST = WORKSPACE / "data/asl_citizen/val_all_manifest.json"
VAL_INDEX = WORKSPACE / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json"
IDENTITY_SALT = "final-test-wrapper-identity-20261012:"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def check_preregistration() -> None:
    path = WORKSPACE / PREREGISTRATION
    require(path.is_file() and base.file_hash(path) == PREREGISTRATION_SHA256, "frozen final-test preregistration missing or changed")


def select_items(manifest: dict, manifest_path: Path, split: str) -> list[dict]:
    """prepare_poses.select_samples with the split guard replaced by an exact split."""
    samples = manifest.get("samples")
    require(isinstance(samples, list) and samples, "manifest has no samples")
    require(manifest.get("status") == "complete" and manifest.get("selection", {}).get("count") == len(samples),
            "download manifest must be complete with selection.count == len(samples)")
    items, seen = [], set()
    for index, sample in enumerate(samples):
        fields = ("path", "sha256", "gloss", "signer_id", "split")
        require(all(isinstance(sample.get(k), str) and sample[k] for k in fields), f"sample {index} lacks {fields}")
        require(sample["split"] == split, f"sample {index} split {sample['split']!r} is not {split!r}")
        digest = sample["sha256"].lower()
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"sample {index} SHA256 invalid")
        relative = Path(sample["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, f"sample {index} path escapes manifest dir")
        source = (manifest_path.parent / relative).resolve()
        require(str(source) not in seen, f"duplicate video path {source}")
        seen.add(str(source))
        identity = {key: sample[key] for key in fields}
        identity["sha256"] = digest
        items.append({"index": index, "identity": identity, "source_path": str(source), "video_sha256": digest,
                      "gloss": sample["gloss"], "signer_id": sample["signer_id"], "split": sample["split"],
                      "cache_key": base.canonical_hash({"path": relative.as_posix(), "sha256": digest})})
    return items


def run_extraction(items, cache_root: Path, report: Path, pipeline: dict, mode: str, workers: int,
                   cpus_per_worker: int, manifest_path: Path, source_count: int, split: str):
    """prepare_poet_mirror_cache.run_extraction with the true split recorded in the index."""
    groups = base.cpu_groups(workers, cpus_per_worker)
    configuration = {
        "schema_version": base.SCHEMA_VERSION, "status": "running", "created_at_utc": base.now(),
        "source_manifest_path": str(manifest_path), "source_manifest_sha256": base.file_hash(manifest_path),
        "selection": {"indices": [item["index"] for item in items], "offset": None, "limit": None,
                      "source_count": source_count, "selected_count": len(items), "rule": f"final test protocol, mode {mode}"},
        "cache_root": str(cache_root.resolve()), "pipeline": pipeline, "pipeline_sha256": base.canonical_hash(pipeline),
        "workers": workers, "cpus_per_worker": cpus_per_worker, "cpu_groups": groups, "start_method": "spawn",
        "worker_library_threads": 1, "split": split, "mode": mode,
        "preregistration": PREREGISTRATION, "preregistration_sha256": PREREGISTRATION_SHA256}
    base.atomic_json(report / "configuration.json", configuration)
    cache_root.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    ordinal = context.Value("i", 0)
    worker_mode = "mirror" if mode == "flipped" else "identity"
    started = time.perf_counter()
    results, interrupted = [], False
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=mirror.init_worker,
                                   initargs=(pipeline, groups, ordinal, worker_mode))
    try:
        futures = {executor.submit(base.process_one, item, str(cache_root.resolve())): item for item in items}
        with (report / "results.jsonl").open("a", encoding="utf-8") as stream:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # keep the identity in the denominator
                    result = {**item, "status": "error", "error_type": type(error).__name__, "error": str(error)}
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if len(results) % 200 == 0 or len(results) == len(items):
                    print(f"[{len(results)}/{len(items)}] {mode} extraction", flush=True)
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        interrupted = True
        for process in list(getattr(executor, "_processes", {}).values()):
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
    done = {row["index"] for row in results}
    for item in items:
        if item["index"] not in done:
            results.append({**item, "status": "interrupted" if interrupted else "error", "error": "No per-sample result returned"})
    results.sort(key=lambda row: row["index"])
    summary = base.summarize(results, time.perf_counter() - started, workers, interrupted)
    summary["completed_at_utc"] = base.now()
    index = {**configuration, "status": summary["status"], "completed_at_utc": summary["completed_at_utc"],
             "samples": results, "summary": summary}
    base.atomic_json(report / "summary.json", summary)
    base.atomic_json(report / "cache_manifest.json", index)
    return index, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("val-identity", "original", "flipped"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, typically {run_dir}/extraction")
    parser.add_argument("--test-manifest", type=Path, default=WORKSPACE / "data/asl_citizen/test_all_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "assets/checkpoints/ASL_Citizen_PoET.pth")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--cpus-per-worker", type=int, default=2)
    args = parser.parse_args()
    require(1 <= args.workers <= 32 and args.cpus_per_worker >= 1, "--workers must be 1..32")
    check_preregistration()
    base.configure_cpu_environment()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pipeline = base.make_pipeline(args.checkpoint.resolve())
    require(base.canonical_hash(pipeline) == mirror.REFERENCE_PIPELINE_SHA256,
            "frozen extraction pipeline differs from the train/validation v2 pipeline")
    if args.mode == "val-identity":
        manifest_path = VAL_MANIFEST.resolve()
        items = select_items(json.loads(manifest_path.read_text()), manifest_path, "val")
        by_signer: dict[str, list[dict]] = {}
        for item in items:
            by_signer.setdefault(item["signer_id"], []).append(item)
        chosen = []
        for signer in sorted(by_signer):  # min(4, n) clips per validation signer, label-free hash order
            ranked = sorted(by_signer[signer], key=lambda it: hashlib.sha256((IDENTITY_SALT + it["video_sha256"]).encode()).hexdigest())
            chosen.extend(ranked[:4])
        reference_index, reference = mirror.load_reference(VAL_INDEX)
        index, results = run_extraction(chosen, output / "cache", output, pipeline, "val-identity", min(args.workers, 16),
                                        args.cpus_per_worker, manifest_path, len(items), "val")
        comparisons = [mirror.compare_identity(row, reference[row["index"]]) for row in results]
        passed = all(c["identical"] for c in comparisons) and len(comparisons) == len(chosen) == 22
        base.atomic_json(output / "identity_gate.json", {"gate": "wrapper_val_identity", "passed": passed,
                                                          "clips": len(comparisons), "comparisons": comparisons,
                                                          "pipeline_sha256": index["pipeline_sha256"]})
        print(json.dumps({"gate": "wrapper_val_identity", "passed": passed, "clips": len(comparisons)}), flush=True)
        return 0 if passed else 2
    manifest_path = args.test_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("preregistration_sha256") == PREREGISTRATION_SHA256, "test manifest is not bound to the protocol")
    items = select_items(manifest, manifest_path, "test")
    require(len(items) == TEST_COUNT and len({i["signer_id"] for i in items}) == 11, "test cohort must be 32,941 clips / 11 signers")
    if args.mode == "flipped":
        pipeline = mirror.mirror_pipeline(pipeline)
        require(base.canonical_hash(pipeline) == MIRROR_PIPELINE_SHA256, "mirror pipeline differs from the preregistered one")
    index, _ = run_extraction(items, output / "cache", output, pipeline, args.mode, args.workers,
                              args.cpus_per_worker, manifest_path, len(items), "test")
    print(json.dumps({"mode": args.mode, "status": index["status"], "summary": index["summary"]}), flush=True)
    return 0 if index["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
