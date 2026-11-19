#!/usr/bin/env python3
"""Flipped-video PoET poses for the complete validation split (mirror TTA evaluation).

Inference-only preparation, fixed by
docs/environment/poet-mirror-tta-fullval-preregistration-20261011.json.

Thin wrapper around the unchanged scripts/prepare_poet_mirror_cache.py: its
worker flip patch, mirror pipeline (and therefore the pipeline SHA256),
extraction loop, array comparison and dominance indicators are imported, not
copied. Modes:

  identity  G1: 22 label-free clips re-extracted with flip=False must be
            array-identical to the frozen v2 cache.
  reuse     R1: audit the 2,806 flipped clips of the 2026-10-10 handedness run
            (index, pipeline hash, per-clip identity and file SHA256) and
            re-extract 20 of them with flip=True; they must be array-identical.
  mirror    G5/G4: extract the remaining 7,498 validation clips from flipped
            video, then write one 10,304-row index (reused + new rows, same
            pipeline hash) that both evaluators can audit.

Selection reads only signer IDs and video SHA256 values; no gloss, prediction
or score. Test data is refused. No model is loaded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

import prepare_poses as base
import prepare_poet_mirror_cache as mirror

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = "docs/environment/poet-mirror-tta-fullval-preregistration-20261011.json"
MIRROR_PIPELINE_SHA256 = "8af5b135fdc1ed6defc8666dfaffe0450caada1d57ab6f1f4b9a894818fb2ed9"
REUSED_RUN = "runs/20261010T145620Z-poet-handedness-mirror-extraction-4e92ea7e"
REUSED_INDEX_SHA256 = "651582c4b11c04c6c3ad523cbc1566a9a64522babce00e862430a68c987a9d7b"
VALIDATION_CLIPS = 10304
REUSED_CLIPS = 2806
IDENTITY_SALT = "poet-mirror-tta-fullval-identity-20261011:"
REUSE_SALT = "poet-mirror-tta-fullval-reuse-20261011:"
PER_SIGNER = 4
LEFT_GROUP = ("P12", "P5")
G4_CLEAR_RATIO = 0.5
G4_GATE = 0.85
G4_AMENDMENT1_VALUE = 0.90
COMBINED_INDEX = "fullval_cache_manifest.json"


def per_signer_first(rows: list[dict], salt: str, sha_key: str, per_signer: int = PER_SIGNER) -> list[int]:
    """For every signer, the min(per_signer, n) rows with the smallest SHA256(salt + video SHA256)."""
    by_signer: dict[str, list[dict]] = {}
    for row in rows:
        by_signer.setdefault(row["signer_id"], []).append(row)
    chosen = []
    for signer in sorted(by_signer):
        ranked = sorted(by_signer[signer], key=lambda row: (mirror.rank_key(salt, row[sha_key]), row["index"]))
        chosen.extend(row["index"] for row in ranked[:per_signer])
    return sorted(chosen)


def validation_rows(samples: list[dict]) -> list[dict]:
    for position, sample in enumerate(samples):
        if sample.get("index") != position:
            raise ValueError("Source manifest sample indices must equal their positions")
        if sample.get("split") not in {"val", "validation"}:
            raise ValueError(f"Only validation clips are allowed, got split {sample.get('split')!r}")
    return samples


def identity_indices(samples: list[dict]) -> list[int]:
    return per_signer_first(validation_rows(samples), IDENTITY_SALT, "sha256")


def reuse_indices(reused_rows: list[dict]) -> list[int]:
    return per_signer_first(reused_rows, REUSE_SALT, "video_sha256")


def remaining_indices(count: int, reused: list[int]) -> list[int]:
    reused_set = set(reused)
    if len(reused_set) != len(reused) or not reused_set <= set(range(count)):
        raise ValueError("Reused indices must be unique validation positions")
    return [index for index in range(count) if index not in reused_set]


def combine_index(reused_index: dict, reused_path: Path, new_index: dict, manifest_path: Path,
                  manifest_sha256: str, source_count: int) -> dict:
    """One complete-validation index whose rows point into the reused and the new cache."""
    if reused_index["pipeline_sha256"] != new_index["pipeline_sha256"]:
        raise ValueError("Reused and new caches have different pipelines")
    rows = [{**row, "mirror_source": f"reused:{reused_path}"} for row in reused_index["samples"]]
    rows += [{**row, "mirror_source": "new"} for row in new_index["samples"]]
    rows.sort(key=lambda row: row["index"])
    indices = [row["index"] for row in rows]
    if indices != list(range(source_count)):
        raise ValueError("Combined rows must cover every validation index exactly once")
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("Combined index may only contain successful extractions")
    return {
        "schema_version": base.SCHEMA_VERSION, "status": "completed", "created_at_utc": mirror.now(),
        "completed_at_utc": mirror.now(), "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "selection": {"indices": indices, "offset": None, "limit": None, "source_count": source_count,
                      "selected_count": len(rows),
                      "rule": f"complete validation split: {len(reused_index['samples'])} reused + {len(new_index['samples'])} new flipped clips; {PREREGISTRATION}"},
        "cache_root": None, "pipeline": new_index["pipeline"], "pipeline_sha256": new_index["pipeline_sha256"],
        "split": "val", "mode": "mirror", "preregistration": PREREGISTRATION,
        "composition": {"reused_index": str(reused_path), "reused_index_sha256": base.file_hash(reused_path),
                        "reused": len(reused_index["samples"]), "new": len(new_index["samples"])},
        "samples": rows}


def audit_reused(reused_path: Path, reused_index: dict, reference: dict[int, dict], flipped_hash: str) -> dict:
    """R1 (a) and (b): the reused cache is the completed handedness cache of the same pipeline."""
    problems = []
    if base.file_hash(reused_path) != REUSED_INDEX_SHA256:
        problems.append("reused index SHA256 differs from the preregistered file")
    if (reused_index.get("status") != "completed" or reused_index.get("mode") != "mirror"
            or reused_index.get("split") != "val"):
        problems.append("reused index is not a completed validation mirror cache")
    if not (reused_index.get("pipeline_sha256") == base.canonical_hash(reused_index.get("pipeline")) == flipped_hash == MIRROR_PIPELINE_SHA256):
        problems.append("pipeline hash differs between reused index, its pipeline, the current mirror pipeline or the preregistration")
    rows = reused_index.get("samples", [])
    if len(rows) != REUSED_CLIPS or len({row["index"] for row in rows}) != len(rows):
        problems.append(f"expected {REUSED_CLIPS} unique reused rows, found {len(rows)}")
    verified = 0
    for row in rows:
        ref = reference.get(row["index"])
        if row.get("status") != "ok" or ref is None or row.get("identity") != ref["identity"] or row.get("cache_key") != ref["cache_key"]:
            problems.append(f"row {row['index']}: status or identity differs from the v2 reference")
            continue
        metadata = json.loads(Path(row["cache_manifest_path"]).read_text(encoding="utf-8"))
        artifacts = metadata.get("artifacts", {})
        if (metadata.get("status") != "ok" or metadata.get("pipeline_sha256") != flipped_hash
                or metadata.get("identity") != ref["identity"]
                or artifacts.get("raw_pose.npz", {}).get("sha256") != row["pose_sha256"]
                or artifacts.get("normalized_features.npz", {}).get("sha256") != row["features_sha256"]):
            problems.append(f"row {row['index']}: per-clip manifest differs")
            continue
        if base.file_hash(row["pose_path"]) != row["pose_sha256"] or base.file_hash(row["features_path"]) != row["features_sha256"]:
            problems.append(f"row {row['index']}: artifact SHA256 differs")
            continue
        verified += 1
    return {"rows": len(rows), "verified_rows": verified, "problems": problems[:50], "problem_count": len(problems)}


def flip_check(rows: list[dict], reference: dict[int, dict]) -> dict:
    """G4 on the combined index: dominance labels follow the flip (label-free)."""
    clips = []
    for row in rows:
        original = mirror.dominance_indicators(mirror.load_arrays(reference[row["index"]]["pose_path"]))
        flipped = mirror.dominance_indicators(mirror.load_arrays(row["pose_path"]))
        clips.append({"index": row["index"], "signer_id": row["signer_id"],
                      "group": "left_dominant_signers" if row["signer_id"] in LEFT_GROUP else "right_dominant_signers",
                      "source": "reused" if row["mirror_source"] != "new" else "new",
                      "original_right_motion": original["right_motion"], "original_right_coverage": original["right_coverage"],
                      "original_motion_ratio": original["motion_ratio"],
                      "mirrored_left_motion": flipped["left_motion"], "mirrored_left_coverage": flipped["left_coverage"]})

    def summarize(subset: list[dict]) -> dict:
        clear = [c for c in subset if c["original_motion_ratio"] <= G4_CLEAR_RATIO]
        return {"clips": len(subset), "clear_clips": len(clear),
                "clear_motion_consistency": sum(c["mirrored_left_motion"] == c["original_right_motion"] for c in clear) / len(clear) if clear else None,
                "motion_consistency": sum(c["mirrored_left_motion"] == c["original_right_motion"] for c in subset) / len(subset),
                "coverage_consistency": sum(c["mirrored_left_coverage"] == c["original_right_coverage"] for c in subset) / len(subset)}

    groups = {name: summarize([c for c in clips if c["group"] == name]) for name in ("left_dominant_signers", "right_dominant_signers")}
    passed = all(g["clear_motion_consistency"] is not None and g["clear_motion_consistency"] >= G4_GATE for g in groups.values())
    return {"schema_version": 1, "gate": "G4", "status": "passed" if passed else "failed", "preregistration": PREREGISTRATION,
            "rule": (f"per group (P12/P5; all other signers), among clips whose original wrist-motion min/max <= {G4_CLEAR_RATIO}, "
                     f"the fraction whose mirrored 'left wrist motion > right' equals the original 'right wrist motion > left' must be >= {G4_GATE}"),
            "amendment1_value_reported": G4_AMENDMENT1_VALUE,
            "amendment1_value_met": all(g["clear_motion_consistency"] is not None and g["clear_motion_consistency"] >= G4_AMENDMENT1_VALUE for g in groups.values()),
            "label_free": True, "by_group": groups,
            "by_signer": {s: summarize([c for c in clips if c["signer_id"] == s]) for s in sorted({c["signer_id"] for c in clips})},
            "by_source": {s: summarize([c for c in clips if c["source"] == s]) for s in ("reused", "new")},
            "clips": clips}


def load_gate(path: Path, gate: str, expected: dict) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("gate") != gate or document.get("status") != "passed":
        raise SystemExit(f"ERROR: gate {gate} has not passed: {path}")
    for key, value in expected.items():
        if document.get(key) != value:
            raise SystemExit(f"ERROR: gate {gate} field {key} differs from this run's expectation")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "reuse", "mirror"), required=True)
    parser.add_argument("--source-manifest", type=Path, default=WORKSPACE / "data/asl_citizen/val_all_manifest.json")
    parser.add_argument("--reference-index", type=Path, default=WORKSPACE / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json")
    parser.add_argument("--reused-index", type=Path, default=WORKSPACE / REUSED_RUN / "extraction/cache_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "assets/checkpoints/ASL_Citizen_PoET.pth")
    parser.add_argument("--identity-gate", type=Path, help="identity_gate.json of this experiment's passed G1 run (mirror mode)")
    parser.add_argument("--reuse-gate", type=Path, help="reuse_gate.json of this experiment's passed R1 run (mirror mode)")
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
    if base.file_hash(manifest_path) != mirror.SOURCE_MANIFEST_SHA256:
        raise SystemExit("ERROR: validation source manifest SHA256 differs from the preregistered file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["samples"]) != VALIDATION_CLIPS:
        raise SystemExit("ERROR: expected the complete 10,304-clip validation manifest")
    _, reference = mirror.load_reference(args.reference_index.resolve())
    all_items, _ = base.select_samples(manifest, manifest_path)
    items_by_index = {item["index"]: item for item in all_items}
    for item in all_items:
        if item["split"] not in {"val", "validation"}:
            raise SystemExit("ERROR: only validation clips are allowed")
        ref = reference[item["index"]]
        if ref["identity"] != item["identity"] or ref["cache_key"] != item["cache_key"]:
            raise SystemExit(f"ERROR: reference cache identity mismatch at index {item['index']}")
    pipeline = base.make_pipeline(args.checkpoint.resolve())
    if base.canonical_hash(pipeline) != mirror.REFERENCE_PIPELINE_SHA256:
        raise SystemExit("ERROR: the frozen extraction pipeline no longer matches the v2 cache pipeline")
    flipped = mirror.mirror_pipeline(pipeline)
    flipped_hash = base.canonical_hash(flipped)
    if flipped_hash != MIRROR_PIPELINE_SHA256:
        raise SystemExit("ERROR: the mirror pipeline hash differs from the preregistered 8af5b135...")
    identity = identity_indices(manifest["samples"])
    reused_path = args.reused_index.resolve()
    reused_index = json.loads(reused_path.read_text(encoding="utf-8"))
    reuse_sample = reuse_indices(reused_index["samples"])
    selection = {"schema_version": 1, "preregistration": PREREGISTRATION, "mode": args.mode,
                 "inputs": "signer_id and video SHA256 only; no gloss, prediction, score or test data",
                 "identity_rule": f"per validation signer, the min({PER_SIGNER}, n) clips with the smallest SHA256('{IDENTITY_SALT}' + video_sha256)",
                 "identity_indices": identity,
                 "reuse_rule": f"per signer among the reused clips, the {PER_SIGNER} clips with the smallest SHA256('{REUSE_SALT}' + video_sha256)",
                 "reuse_indices": reuse_sample}
    base.atomic_json(output / "selection.json", selection)

    if args.mode == "identity":
        items = [items_by_index[index] for index in identity]
        index, results = mirror.run_extraction(items, output / "cache", output, pipeline, "identity",
                                               args.workers, args.cpus_per_worker, manifest_path, len(all_items))
        comparisons = [mirror.compare_identity(result, reference[result["index"]]) for result in results]
        passed = index["status"] == "completed" and len(comparisons) == len(identity) and all(row["identical"] for row in comparisons)
        gate = {"schema_version": 1, "gate": "G1", "status": "passed" if passed else "failed", "preregistration": PREREGISTRATION,
                "rule": "every identity clip re-extracted through the untouched frozen path (flip=False) is array-identical to the v2 cache",
                "pipeline_sha256": index["pipeline_sha256"], "reference_pipeline_sha256": mirror.REFERENCE_PIPELINE_SHA256,
                "indices": identity, "clips": len(comparisons), "identical": sum(row["identical"] for row in comparisons),
                "comparisons": comparisons, "completed_at_utc": mirror.now()}
        base.atomic_json(output / "identity_gate.json", gate)
        print(json.dumps({key: gate[key] for key in ("gate", "status", "clips", "identical")}), flush=True)
        return 0 if passed else 2

    if args.mode == "reuse":
        audit = audit_reused(reused_path, reused_index, reference, flipped_hash)
        reused_rows = {row["index"]: row for row in reused_index["samples"]}
        items = [items_by_index[index] for index in reuse_sample]
        index, results = mirror.run_extraction(items, output / "cache", output, flipped, "mirror",
                                               args.workers, args.cpus_per_worker, manifest_path, len(all_items))
        comparisons = [mirror.compare_identity(result, reused_rows[result["index"]]) for result in results]
        determinism = index["status"] == "completed" and len(comparisons) == len(reuse_sample) and all(row["identical"] for row in comparisons)
        passed = audit["problem_count"] == 0 and audit["verified_rows"] == REUSED_CLIPS and determinism
        gate = {"schema_version": 1, "gate": "R1", "status": "passed" if passed else "failed", "preregistration": PREREGISTRATION,
                "rule": ("(a) reused index is the preregistered completed mirror cache with the current mirror pipeline hash; "
                         "(b) every reused per-clip manifest and artifact matches; (c) the reuse sample re-extracted with flip=True "
                         "is array-identical to the reused cache"),
                "reused_index": str(reused_path), "reused_index_sha256": base.file_hash(reused_path),
                "pipeline_sha256": flipped_hash, "audit": audit,
                "determinism": {"indices": reuse_sample, "clips": len(comparisons), "identical": sum(row["identical"] for row in comparisons),
                                "comparisons": comparisons},
                "completed_at_utc": mirror.now()}
        base.atomic_json(output / "reuse_gate.json", gate)
        print(json.dumps({"gate": "R1", "status": gate["status"], "verified_rows": audit["verified_rows"],
                          "determinism_identical": gate["determinism"]["identical"]}), flush=True)
        return 0 if passed else 2

    if args.identity_gate is None or args.reuse_gate is None:
        parser.error("--mode mirror requires --identity-gate and --reuse-gate")
    load_gate(args.identity_gate, "G1", {"pipeline_sha256": mirror.REFERENCE_PIPELINE_SHA256, "indices": identity,
                                         "preregistration": PREREGISTRATION})
    load_gate(args.reuse_gate, "R1", {"pipeline_sha256": flipped_hash, "reused_index_sha256": REUSED_INDEX_SHA256,
                                      "preregistration": PREREGISTRATION})
    reused_index_rows = sorted(row["index"] for row in reused_index["samples"])
    remaining = remaining_indices(len(all_items), reused_index_rows)
    if len(remaining) != VALIDATION_CLIPS - REUSED_CLIPS:
        raise SystemExit("ERROR: unexpected number of remaining clips")
    items = [items_by_index[index] for index in remaining]
    new_index, results = mirror.run_extraction(items, output / "cache", output, flipped, "mirror",
                                               args.workers, args.cpus_per_worker, manifest_path, len(all_items))
    ok = [row for row in results if row["status"] == "ok"]
    extraction_gate = {"schema_version": 1, "gate": "G5", "preregistration": PREREGISTRATION,
                       "status": "passed" if new_index["status"] == "completed" and len(ok) == len(items) else "failed",
                       "rule": "every remaining validation clip is re-extracted from flipped frames without failure",
                       "selected": len(items), "ok": len(ok),
                       "failed_indices": [row["index"] for row in results if row["status"] != "ok"],
                       "wall_seconds": new_index["summary"]["wall_seconds"],
                       "clips_per_wall_second": new_index["summary"]["new_videos_per_wall_second"]}
    base.atomic_json(output / "extraction_gate.json", extraction_gate)
    if extraction_gate["status"] != "passed":
        print(json.dumps({key: extraction_gate[key] for key in ("gate", "status", "selected", "ok")}), flush=True)
        return 2
    combined = combine_index(reused_index, reused_path, new_index, manifest_path, base.file_hash(manifest_path), len(all_items))
    base.atomic_json(output / COMBINED_INDEX, combined)
    sanity = flip_check(combined["samples"], reference)
    base.atomic_json(output / "flip_sanity.json", sanity)
    print(json.dumps({"G5": extraction_gate["status"], "G4": sanity["status"],
                      "clear_motion_consistency": {k: round(v["clear_motion_consistency"], 4) for k, v in sanity["by_group"].items()},
                      "combined_rows": len(combined["samples"])}), flush=True)
    return 0 if sanity["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
