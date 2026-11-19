#!/usr/bin/env python3
"""Re-extract official PoET poses from horizontally flipped validation videos.

Inference-only handedness check, fixed by
docs/environment/poet-handedness-preregistration-20261010.json.

The frozen extraction stack is reused unchanged: prepare_poses.process_one ->
infer_video.extract_raw_pose -> vendor extract_mediapipe_points_from_video ->
articulators_to_array -> preprocess(mode="eval"). In ``--mode mirror`` the only
change is that the vendor extractor receives ``flip=True``; it then applies
``cv2.flip(image, 1)`` to every decoded frame before MediaPipe Holistic, which
is exactly how upstream ``data.py --flip`` builds its ``_flip`` keypoints for the
paper's horizontal-flip augmentation. ``--mode identity`` re-extracts a small
fixed sample through the untouched frozen path and requires array-identical
agreement with the v2 cache before any mirrored extraction may start.

Cohort selection reads only signer IDs and video SHA256 values; no gloss,
prediction or score is used. Test data is refused. No model is loaded here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import signal
import sys
import time

import prepare_poses as base

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE = "scripts/prepare_poet_mirror_cache.py"
PREREGISTRATION = "docs/environment/poet-handedness-preregistration-20261010.json"
SOURCE_MANIFEST_SHA256 = "b59fa41ee85955a310d665a3af09a67274a6ddb48b4d18cae1faaa06c6fb7a4f"
REFERENCE_INDEX_SHA256 = "a829029d700e50dd07d0f88454d9f3d42a65e946586a1188e7dfe9a81d5a5c86"
REFERENCE_PIPELINE_SHA256 = "ceca202ce01a6327c7433b1a29a7ba5ea482f6b8335a87d7732938463dd1941b"
LEFT_DOMINANT = ("P12", "P5")
CONTROL_SIGNERS = ("P21", "P26", "P39")
CONTROL_PER_SIGNER = 500
CONTROL_SALT = "poet-handedness-control-20261010:"
IDENTITY_SIGNERS = ("P12", "P5", "P21", "P26", "P39")
IDENTITY_PER_SIGNER = 4
IDENTITY_SALT = "poet-handedness-identity-20261010:"
MOTION_CONSISTENCY_GATE = 0.75  # preregistered all-clip value; reported, no longer binding (amendment 1)
CLEAR_MOTION_RATIO = 0.5  # amendment 1: original wrist-motion min/max <= 0.5
CLEAR_CONSISTENCY_GATE = 0.90
AMENDMENT = "docs/environment/poet-handedness-preregistration-amendment1-20261010.json"
FLIP_MARK = "_poet_handedness_hflip"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rank_key(salt: str, video_sha256: str) -> str:
    return hashlib.sha256((salt + video_sha256).encode("ascii")).hexdigest()


def select_cohorts(samples: list[dict]) -> dict:
    """Deterministic, label-free cohorts from signer IDs and video hashes only."""
    by_signer: dict[str, list[dict]] = {}
    for position, sample in enumerate(samples):
        if sample.get("index") != position:
            raise ValueError("Source manifest sample indices must equal their positions")
        if sample.get("split") not in {"val", "validation"}:
            raise ValueError(f"Only validation clips are allowed, got split {sample.get('split')!r}")
        by_signer.setdefault(sample["signer_id"], []).append(sample)

    def ranked(signer: str, salt: str, count: int) -> list[dict]:
        rows = sorted(by_signer.get(signer, []), key=lambda s: (rank_key(salt, s["sha256"]), s["index"]))
        if len(rows) < count:
            raise ValueError(f"Signer {signer} has {len(rows)} clips, fewer than {count}")
        return rows[:count]

    left = [s for signer in LEFT_DOMINANT for s in by_signer.get(signer, [])]
    if not left:
        raise ValueError("No left-dominant validation clips found")
    control = [s for signer in CONTROL_SIGNERS for s in ranked(signer, CONTROL_SALT, CONTROL_PER_SIGNER)]
    identity = [s for signer in IDENTITY_SIGNERS for s in ranked(signer, IDENTITY_SALT, IDENTITY_PER_SIGNER)]

    def rows(group: list[dict], cohort: str, salt: str | None) -> list[dict]:
        return sorted(({"index": s["index"], "cohort": cohort, "signer_id": s["signer_id"],
                        "video_file": s["video_file"], "video_sha256": s["sha256"],
                        **({"rank_key": rank_key(salt, s["sha256"])} if salt else {})} for s in group),
                      key=lambda row: row["index"])

    result = {"left_dominant": rows(left, "left_dominant", None),
              "control": rows(control, "control", CONTROL_SALT),
              "identity": rows(identity, "identity", IDENTITY_SALT)}
    main = [row["index"] for row in result["left_dominant"] + result["control"]]
    if len(set(main)) != len(main):
        raise ValueError("Left-dominant and control cohorts overlap")
    return result


def selection_document(cohorts: dict) -> dict:
    counts = {name: len(rows) for name, rows in cohorts.items()}
    per_signer = {name: {signer: sum(row["signer_id"] == signer for row in rows)
                         for signer in sorted({row["signer_id"] for row in rows})}
                  for name, rows in cohorts.items()}
    return {"schema_version": 1, "preregistration": PREREGISTRATION,
            "rule": {"left_dominant": f"all validation clips of {list(LEFT_DOMINANT)} (signers mirrored by the frozen label-free classical rule)",
                     "control": f"per signer in {list(CONTROL_SIGNERS)}, the {CONTROL_PER_SIGNER} clips with the smallest SHA256('{CONTROL_SALT}' + video_sha256)",
                     "identity": f"per signer in {list(IDENTITY_SIGNERS)}, the {IDENTITY_PER_SIGNER} clips with the smallest SHA256('{IDENTITY_SALT}' + video_sha256)",
                     "inputs": "signer_id and video SHA256 only; no gloss, prediction, score or test data"},
            "counts": counts, "per_signer": per_signer, "cohorts": cohorts}


def install_flip(module) -> None:
    """Make the frozen infer_video call the vendor extractor with flip=True."""
    original = module.extract_mediapipe_points_from_video
    if getattr(original, FLIP_MARK, False):
        return

    def flipped(video_path, flip=False):
        if flip is not False:
            raise RuntimeError("The frozen extractor wrapper expected infer_video to request flip=False")
        return original(video_path, flip=True)

    setattr(flipped, FLIP_MARK, True)
    flipped.wrapped = original
    module.extract_mediapipe_points_from_video = flipped


def init_worker(pipeline, groups, ordinal, mode):
    base.init_worker(pipeline, groups, ordinal)
    if mode in {"mirror", "smoke"}:
        import infer_video
        import mediapipe_extraction
        if infer_video.extract_mediapipe_points_from_video is not mediapipe_extraction.extract_mediapipe_points_from_video:
            raise RuntimeError("infer_video no longer binds the unmodified vendor extractor")
        install_flip(infer_video)


def mirror_pipeline(pipeline: dict) -> dict:
    result = json.loads(json.dumps(pipeline))
    result["name"] = "poet-official-raw-normalized-eval-v2-hflip"
    result["source_sha256"][SCRIPT_RELATIVE] = base.file_hash(WORKSPACE / SCRIPT_RELATIVE)
    result["extractor"] = {
        "implementation": ("infer_video.extract_raw_pose (unchanged) whose vendor extract_mediapipe_points_from_video call "
                           "is forced to flip=True by scripts/prepare_poet_mirror_cache.py; the vendor function applies "
                           "cv2.flip(image, 1) to every decoded frame before MediaPipe Holistic, as upstream data.py --flip does"),
        "static_image_mode": True, "model_complexity": 2, "min_detection_confidence": 0.5, "flip": True}
    result["orientation"] = "horizontally_flipped_video"
    return result


def load_reference(reference_index: Path) -> tuple[dict, dict[int, dict]]:
    if base.file_hash(reference_index) != REFERENCE_INDEX_SHA256:
        raise ValueError("Reference v2 validation cache index SHA256 differs from the preregistered file")
    index = json.loads(reference_index.read_text(encoding="utf-8"))
    if index.get("status") != "completed" or index.get("pipeline_sha256") != REFERENCE_PIPELINE_SHA256:
        raise ValueError("Reference index is not the completed v2 validation cache")
    samples = {item["index"]: item for item in index["samples"]}
    if len(samples) != len(index["samples"]) or any(item["status"] not in {"ok", "reused"} for item in samples.values()):
        raise ValueError("Reference index has duplicate or unsuccessful samples")
    return index, samples


def make_configuration(items: list[dict], cache_root: Path, pipeline: dict, mode: str, workers: int,
                       cpus_per_worker: int, groups: list, manifest_path: Path, manifest_sha256: str, source_count: int) -> dict:
    """Same index layout as prepare_poses.run, so evaluate_pose_cache.py can audit it."""
    return {
        "schema_version": base.SCHEMA_VERSION, "status": "running", "created_at_utc": now(),
        "source_manifest_path": str(manifest_path), "source_manifest_sha256": manifest_sha256,
        "selection": {"indices": [item["index"] for item in items], "offset": None, "limit": None,
                      "source_count": source_count, "selected_count": len(items),
                      "rule": "explicit preregistered cohort; see selection.json"},
        "cache_root": str(cache_root), "pipeline": pipeline, "pipeline_sha256": base.canonical_hash(pipeline),
        "workers": workers, "cpus_per_worker": cpus_per_worker, "cpu_groups": groups,
        "start_method": "spawn", "worker_library_threads": 1, "split": "val", "mode": mode}


def finalize_index(configuration: dict, results: list[dict], summary: dict) -> dict:
    return {**configuration, "status": summary["status"], "completed_at_utc": summary["completed_at_utc"],
            "samples": sorted(results, key=lambda row: row["index"]), "summary": summary}


def run_extraction(items: list[dict], cache_root: Path, report: Path, pipeline: dict, mode: str,
                   workers: int, cpus_per_worker: int, manifest_path: Path, source_count: int) -> tuple[dict, list[dict]]:
    groups = base.cpu_groups(workers, cpus_per_worker)
    configuration = make_configuration(items, cache_root.resolve(), pipeline, mode, workers, cpus_per_worker, groups,
                                       manifest_path, base.file_hash(manifest_path), source_count)
    base.atomic_json(report / "configuration.json", configuration)
    cache_root.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    ordinal = context.Value("i", 0)
    started = time.perf_counter()
    results, interrupted = [], False
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker,
                                   initargs=(pipeline, groups, ordinal, mode))
    try:
        futures = {executor.submit(base.process_one, item, str(cache_root.resolve())): item for item in items}
        with (report / "results.jsonl").open("a", encoding="utf-8") as stream:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # pool-level failure; keep the identity in the denominator
                    result = {**item, "status": "error", "error_type": type(error).__name__, "error": str(error)}
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if len(results) % 50 == 0 or len(results) == len(items):
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
            results.append({**item, "status": "interrupted" if interrupted else "error",
                            "error": "No per-sample result returned"})
    results.sort(key=lambda row: row["index"])
    summary = base.summarize(results, time.perf_counter() - started, workers, interrupted)
    summary["completed_at_utc"] = now()
    index = finalize_index(configuration, results, summary)
    base.atomic_json(report / "summary.json", summary)
    base.atomic_json(report / "cache_manifest.json", index)
    return index, results


def np_equal(a, b) -> bool:
    import numpy as np
    return a.shape == b.shape and bool(np.array_equal(a, b))


def load_arrays(path: str) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare_identity(result: dict, reference: dict) -> dict:
    """Array-level comparison; npz bytes differ by zip timestamps, so not hashed."""
    import numpy as np
    checks = {}
    for artifact, key in (("raw_pose.npz", "pose_path"), ("normalized_features.npz", "features_path")):
        if result["status"] != "ok":
            checks[artifact] = {"identical": False, "reason": f"extraction status {result['status']}"}
            continue
        expected = reference["pose_sha256" if key == "pose_path" else "features_sha256"]
        if base.file_hash(reference[key]) != expected:
            raise ValueError(f"Reference cache file changed: {reference[key]}")
        new, old = load_arrays(result[key]), load_arrays(reference[key])
        if set(new) != set(old):
            checks[artifact] = {"identical": False, "reason": f"keys differ: {sorted(set(new) ^ set(old))}"}
            continue
        differing = [name for name in sorted(old)
                     if new[name].dtype != old[name].dtype or new[name].shape != old[name].shape
                     or not np.array_equal(new[name], old[name])]
        checks[artifact] = {"identical": not differing, "arrays": sorted(old), "differing": differing}
    return {"index": result["index"], "signer_id": result["signer_id"], "status": result["status"],
            "identical": result["status"] == "ok" and all(c["identical"] for c in checks.values()), "checks": checks}


def dominance_indicators(raw: dict) -> dict:
    """Whole-clip coverage and wrist-motion comparisons of the classical descriptor."""
    import numpy as np
    from classical_features import extract_features
    from classical_handedness import COVERAGE_LEFT, COVERAGE_RIGHT, MOTION_LEFT, MOTION_RIGHT
    x = extract_features(raw).astype(np.float64)
    left_motion, right_motion = float(x[list(MOTION_LEFT)].sum()), float(x[list(MOTION_RIGHT)].sum())
    larger = max(left_motion, right_motion)
    return {"left_coverage": bool(x[COVERAGE_LEFT] > x[COVERAGE_RIGHT]), "right_coverage": bool(x[COVERAGE_RIGHT] > x[COVERAGE_LEFT]),
            "left_motion": bool(left_motion > right_motion), "right_motion": bool(right_motion > left_motion),
            "motion_ratio": min(left_motion, right_motion) / larger if larger > 0 else 1.0}


def flip_sanity(results: list[dict], reference: dict[int, dict], cohorts: dict) -> dict:
    """Label-free check that the dominant-hand labels follow the horizontal flip."""
    cohort_of = {row["index"]: name for name in ("left_dominant", "control") for row in cohorts[name]}
    rows = []
    for result in results:
        original = dominance_indicators(load_arrays(reference[result["index"]]["pose_path"]))
        mirrored = dominance_indicators(load_arrays(result["pose_path"]))
        rows.append({"index": result["index"], "signer_id": result["signer_id"], "cohort": cohort_of[result["index"]],
                     "original_left_coverage": original["left_coverage"], "original_right_coverage": original["right_coverage"],
                     "original_left_motion": original["left_motion"], "original_right_motion": original["right_motion"],
                     "original_motion_ratio": original["motion_ratio"],
                     "mirrored_left_coverage": mirrored["left_coverage"], "mirrored_left_motion": mirrored["left_motion"]})

    def summarize(subset: list[dict]) -> dict:
        n = len(subset)
        clear = [r for r in subset if r["original_motion_ratio"] <= CLEAR_MOTION_RATIO]
        return {"clips": n, "clear_clips": len(clear),
                "clear_motion_consistency": (sum(r["mirrored_left_motion"] == r["original_right_motion"] for r in clear) / len(clear)) if clear else None,
                "motion_consistency": sum(r["mirrored_left_motion"] == r["original_right_motion"] for r in subset) / n,
                "coverage_consistency": sum(r["mirrored_left_coverage"] == r["original_right_coverage"] for r in subset) / n,
                "original_left_coverage_fraction": sum(r["original_left_coverage"] for r in subset) / n,
                "original_left_motion_fraction": sum(r["original_left_motion"] for r in subset) / n,
                "mirrored_left_coverage_fraction": sum(r["mirrored_left_coverage"] for r in subset) / n,
                "mirrored_left_motion_fraction": sum(r["mirrored_left_motion"] for r in subset) / n}

    by_cohort = {name: summarize([r for r in rows if r["cohort"] == name]) for name in ("left_dominant", "control")}
    by_signer = {signer: summarize([r for r in rows if r["signer_id"] == signer])
                 for signer in sorted({r["signer_id"] for r in rows})}
    passed = all(value["clear_motion_consistency"] is not None and value["clear_motion_consistency"] >= CLEAR_CONSISTENCY_GATE
                 for value in by_cohort.values())
    return {"schema_version": 1, "gate": "G4", "status": "passed" if passed else "failed", "amendment": AMENDMENT,
            "rule": (f"amendment 1: per cohort, among clips whose original wrist-motion min/max <= {CLEAR_MOTION_RATIO}, the fraction whose "
                     f"mirrored 'left wrist motion > right' equals the original 'right wrist motion > left' (classical-pose-motion-v1 "
                     f"x[976]+x[977] vs x[978]+x[979]) must be >= {CLEAR_CONSISTENCY_GATE}"),
            "preregistered_v1_all_clip_rule": f"all-clip motion consistency >= {MOTION_CONSISTENCY_GATE} per cohort (reported only)",
            "preregistered_v1_all_clip_rule_met": all(v["motion_consistency"] >= MOTION_CONSISTENCY_GATE for v in by_cohort.values()),
            "label_free": True, "by_cohort": by_cohort, "by_signer": by_signer, "clips": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "smoke", "mirror"), required=True,
                        help="identity: G1 gate; smoke: 2-clip flip pre-flight (no model); mirror: preregistered cohorts")
    parser.add_argument("--source-manifest", type=Path, default=WORKSPACE / "data/asl_citizen/val_all_manifest.json")
    parser.add_argument("--reference-index", type=Path, default=WORKSPACE / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "assets/checkpoints/ASL_Citizen_PoET.pth")
    parser.add_argument("--identity-gate", type=Path, help="identity_gate.json of a passed identity run (required for --mode mirror)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory, typically {run_dir}/extraction")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpus-per-worker", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.cpus_per_worker < 1:
        parser.error("--workers must be 1..32 and --cpus-per-worker positive")

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)

    base.configure_cpu_environment()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.source_manifest.resolve()
    if base.file_hash(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise SystemExit("ERROR: validation source manifest SHA256 differs from the preregistered file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohorts = select_cohorts(manifest["samples"])
    base.atomic_json(output / "selection.json", selection_document(cohorts))
    _, reference = load_reference(args.reference_index.resolve())
    all_items, _ = base.select_samples(manifest, manifest_path)
    items_by_index = {item["index"]: item for item in all_items}
    for item in all_items:
        ref = reference[item["index"]]
        if ref["identity"] != item["identity"] or ref["cache_key"] != item["cache_key"]:
            raise SystemExit(f"ERROR: reference cache identity mismatch at index {item['index']}")

    pipeline = base.make_pipeline(args.checkpoint.resolve())
    if base.canonical_hash(pipeline) != REFERENCE_PIPELINE_SHA256:
        raise SystemExit("ERROR: the frozen extraction pipeline no longer matches the v2 cache pipeline")

    if args.mode == "identity":
        items = [items_by_index[row["index"]] for row in cohorts["identity"]]
        index, results = run_extraction(items, output / "cache", output, pipeline, "identity",
                                        args.workers, args.cpus_per_worker, manifest_path, len(all_items))
        comparisons = [compare_identity(result, reference[result["index"]]) for result in results]
        passed = index["status"] == "completed" and all(row["identical"] for row in comparisons)
        gate = {"schema_version": 1, "gate": "G1", "status": "passed" if passed else "failed",
                "rule": "every identity clip re-extracted through the untouched frozen path (flip=False) is array-identical to the v2 cache (all raw_pose arrays, normalized_features, model_input, source_frame_indices)",
                "pipeline_sha256": index["pipeline_sha256"], "reference_pipeline_sha256": REFERENCE_PIPELINE_SHA256,
                "clips": len(comparisons), "identical": sum(row["identical"] for row in comparisons),
                "comparisons": comparisons, "completed_at_utc": now()}
        base.atomic_json(output / "identity_gate.json", gate)
        print(json.dumps({key: gate[key] for key in ("gate", "status", "clips", "identical")}), flush=True)
        return 0 if passed else 2

    if args.mode == "smoke":
        # Pre-flight of the spawned-worker flip patch on two identity clips; label-free, no model.
        chosen = [next(row for row in cohorts["identity"] if row["signer_id"] == signer) for signer in ("P12", "P21")]
        items = [items_by_index[row["index"]] for row in chosen]
        index, results = run_extraction(items, output / "cache", output, mirror_pipeline(pipeline), "smoke",
                                        2, args.cpus_per_worker, manifest_path, len(all_items))
        rows = []
        for result in results:
            if result["status"] != "ok":
                rows.append({"index": result["index"], "passed": False, "status": result["status"]})
                continue
            original_raw, mirrored_raw = load_arrays(reference[result["index"]]["pose_path"]), load_arrays(result["pose_path"])
            original, mirrored = dominance_indicators(original_raw), dominance_indicators(mirrored_raw)
            changed = any(not np_equal(original_raw[key], mirrored_raw[key]) for key in ("left_hand", "right_hand", "body_pose"))
            rows.append({"index": result["index"], "signer_id": result["signer_id"], "original": original, "mirrored": mirrored,
                         "detection_counts_original": reference[result["index"]]["video_metadata"]["detection_counts"],
                         "detection_counts_mirrored": result["video_metadata"]["detection_counts"],
                         "arrays_changed": changed, "passed": changed and mirrored["left_motion"] == original["right_motion"]})
        smoke = {"schema_version": 1, "status": "passed" if rows and all(r["passed"] for r in rows) else "failed",
                 "rule": "both clips extract without failure, arrays differ from the original, and the wrist-motion dominance label swaps",
                 "pipeline_sha256": index["pipeline_sha256"], "clips": rows, "completed_at_utc": now()}
        base.atomic_json(output / "smoke.json", smoke)
        print(json.dumps({"smoke": smoke["status"], "clips": [(r["index"], r["passed"]) for r in rows]}), flush=True)
        return 0 if smoke["status"] == "passed" else 2

    if args.identity_gate is None:
        parser.error("--mode mirror requires --identity-gate")
    gate = json.loads(args.identity_gate.read_text(encoding="utf-8"))
    expected = sorted(row["index"] for row in cohorts["identity"])
    if (gate.get("gate") != "G1" or gate.get("status") != "passed" or gate.get("pipeline_sha256") != REFERENCE_PIPELINE_SHA256
            or sorted(row["index"] for row in gate.get("comparisons", [])) != expected):
        raise SystemExit("ERROR: identity gate G1 has not passed for the preregistered identity sample")
    flipped = mirror_pipeline(pipeline)
    main_rows = sorted(cohorts["left_dominant"] + cohorts["control"], key=lambda row: row["index"])
    items = [items_by_index[row["index"]] for row in main_rows]
    index, results = run_extraction(items, output / "cache", output, flipped, "mirror",
                                    args.workers, args.cpus_per_worker, manifest_path, len(all_items))
    ok = [row for row in results if row["status"] == "ok"]
    extraction_gate = {"gate": "G5", "status": "passed" if index["status"] == "completed" and len(ok) == len(items) else "failed",
                       "rule": "every selected clip is re-extracted from flipped frames without failure", "selected": len(items), "ok": len(ok)}
    base.atomic_json(output / "extraction_gate.json", extraction_gate)
    if extraction_gate["status"] != "passed":
        print(json.dumps(extraction_gate), flush=True)
        return 2
    sanity = flip_sanity(results, reference, cohorts)
    base.atomic_json(output / "flip_sanity.json", sanity)
    print(json.dumps({"G5": extraction_gate["status"], "G4": sanity["status"],
                      "by_cohort": {k: round(v["motion_consistency"], 4) for k, v in sanity["by_cohort"].items()}}), flush=True)
    return 0 if sanity["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
