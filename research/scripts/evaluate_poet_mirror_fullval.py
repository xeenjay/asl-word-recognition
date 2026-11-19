#!/usr/bin/env python3
"""Official PoET logits on original and flipped poses for the complete validation split.

Inference only; no parameter is updated. Fixed by
docs/environment/poet-mirror-tta-fullval-preregistration-20261011.json.

The cache verification and float64 top-k helpers of the unchanged
scripts/evaluate_poet_mirror.py are imported, so both arms are prepared exactly
as in the 2026-10-10 handedness check. Arms, on all 10,304 clips in index
order and the same batch layout:
  original  frozen v2 validation cache (unflipped video)
  mirrored  combined flipped index (2,806 reused + 7,498 new clips)
  tta       0.5 * (original + mirrored) logits in float64

G2:  the original arm reproduces the historical official predictions.
G2b: on the reused clips the mirrored arm reproduces the 2026-10-10 mirrored
     predictions.
Metrics and the verdict are left to the independent analysis. Test data is
refused; the official checkpoint and preprocessing are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import evaluate_poet_mirror as helpers

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = "docs/environment/poet-mirror-tta-fullval-preregistration-20261011.json"
VALIDATION_CLIPS = 10304
MIRROR_PIPELINE_SHA256 = "8af5b135fdc1ed6defc8666dfaffe0450caada1d57ab6f1f4b9a894818fb2ed9"
PRIOR_MIRRORED_SHA256 = "086103f9d96f60a8ca1d485ab661c30a3dd042542470da11774d41a869eb8f73"
TOLERANCE = helpers.G2_PROBABILITY_TOLERANCE
ARMS = helpers.ARMS


def check_combined_index(index: dict, reference_samples: dict[int, dict]) -> dict[int, dict]:
    pipeline = index.get("pipeline", {})
    if (index.get("status") != "completed" or index.get("split") != "val" or index.get("mode") != "mirror"
            or pipeline.get("extractor", {}).get("flip") is not True
            or pipeline.get("name") != "poet-official-raw-normalized-eval-v2-hflip"
            or index.get("pipeline_sha256") != MIRROR_PIPELINE_SHA256):
        raise ValueError("Mirror index is not the preregistered complete flipped-video validation cache")
    samples = {item["index"]: item for item in index["samples"]}
    if len(samples) != len(index["samples"]) or sorted(samples) != list(range(VALIDATION_CLIPS)):
        raise ValueError("Mirror index must contain every validation clip exactly once")
    for position, item in samples.items():
        reference = reference_samples[position]
        if item["status"] != "ok" or item["identity"] != reference["identity"]:
            raise ValueError(f"Mirror row {position} failed or differs from the v2 reference identity")
        if reference["split"] != "val" or reference["identity"]["split"] != "val":
            raise ValueError("Only validation clips may be evaluated")
    return samples


def compare_top5(records: list[dict], expected: dict[int, dict], name: str, rule: str) -> dict:
    mismatches, largest = [], 0.0
    for record in records:
        past = expected[record["index"]]
        now_ids = [p["class_id"] for p in record["top5"]]
        past_ids = [p["class_id"] for p in past["top5"]]
        difference = max(abs(a["probability"] - b["probability"]) for a, b in zip(record["top5"], past["top5"]))
        largest = max(largest, difference)
        if now_ids != past_ids or difference > TOLERANCE or past["class_id"] != record["class_id"] or past["signer_id"] != record["signer_id"]:
            mismatches.append({"index": record["index"], "now": now_ids, "expected": past_ids, "max_abs_probability_difference": difference})
    return {"gate": name, "status": "passed" if records and not mismatches else "failed", "rule": rule,
            "clips": len(records), "mismatches": mismatches, "max_abs_probability_difference": largest}


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from _poet_runtime import load_bundle, require_finite, resolve_device, synchronize

    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "status": "running", "created_at_utc": helpers.now(), "preregistration": PREREGISTRATION,
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    helpers.write_json(output / "summary.json", summary)
    if helpers.sha256(args.reference_index) != helpers.REFERENCE_INDEX_SHA256:
        raise ValueError("Reference v2 cache index SHA256 differs from the preregistered file")
    if helpers.sha256(args.historical_predictions) != helpers.HISTORICAL_PREDICTIONS_SHA256:
        raise ValueError("Historical official predictions SHA256 differs from the preregistered file")
    if helpers.sha256(args.prior_mirrored_predictions) != PRIOR_MIRRORED_SHA256:
        raise ValueError("2026-10-10 mirrored predictions SHA256 differs from the preregistered file")
    reference = json.loads(args.reference_index.read_text(encoding="utf-8"))
    if reference.get("pipeline_sha256") != helpers.REFERENCE_PIPELINE_SHA256 or reference.get("split") != "val":
        raise ValueError("Reference index is not the v2 validation cache")
    reference_samples = {item["index"]: item for item in reference["samples"]}
    mirror = json.loads(args.mirror_index.read_text(encoding="utf-8"))
    mirror_samples = check_combined_index(mirror, reference_samples)
    indices = list(range(VALIDATION_CLIPS))
    historical = {row["index"]: row for row in helpers.read_jsonl(args.historical_predictions)}
    prior_mirrored = {row["index"]: row for row in helpers.read_jsonl(args.prior_mirrored_predictions)}
    reused = sorted(i for i, item in mirror_samples.items() if item.get("mirror_source", "").startswith("reused:"))
    if sorted(prior_mirrored) != reused:
        raise ValueError("Reused rows of the combined index differ from the 2026-10-10 mirrored clips")
    for index in indices:
        if historical[index]["video_sha256"] != reference_samples[index]["video_sha256"] or historical[index]["status"] != "ok":
            raise ValueError(f"Historical prediction differs from the v2 reference at index {index}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    bundle = load_bundle(args.checkpoint_dir.resolve(), device)
    if bundle.checkpoint["num_classes"] != helpers.NUM_CLASSES or bundle.metadata["label_map_sha256"] != helpers.OFFICIAL_LABEL_MAP_SHA256:
        raise ValueError("Requires the official 2,731-class checkpoint and unmodified label maps")

    logits, timings = {}, {}
    for arm, samples, index_doc in (("original", reference_samples, reference), ("mirrored", mirror_samples, mirror)):
        stamp = time.perf_counter()
        rows = []
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch = torch.stack([helpers.load_input(samples[i], index_doc, bundle.preprocessing) for i in batch_indices])
            with torch.inference_mode():
                values = bundle.model(batch, device)
                require_finite(f"{arm} logits", values)
                if tuple(values.shape) != (len(batch_indices), helpers.NUM_CLASSES):
                    raise ValueError(f"Unexpected logits shape {tuple(values.shape)}")
                probabilities, classes = values.softmax(dim=1).topk(5, dim=1)
            rows.append((values.float().cpu().numpy(), classes.cpu().tolist(), probabilities.cpu().tolist()))
            if (start // args.batch_size) % 20 == 0:
                print(f"{arm}: {start + len(batch_indices)}/{len(indices)}", flush=True)
        synchronize(device)
        timings[f"{arm}_seconds"] = time.perf_counter() - stamp
        logits[arm] = (np.concatenate([r[0] for r in rows]), [c for r in rows for c in r[1]], [p for r in rows for p in r[2]])

    tta_logits = 0.5 * (logits["original"][0].astype(np.float64) + logits["mirrored"][0].astype(np.float64))
    tta_classes, tta_probabilities = helpers.topk_rows(tta_logits)
    top5 = {"original": logits["original"][1:], "mirrored": logits["mirrored"][1:], "tta": (tta_classes, tta_probabilities)}

    records = {arm: [] for arm in ARMS}
    for position, index in enumerate(indices):
        ref = reference_samples[index]
        target = bundle.label_to_id[ref["gloss"]]
        source = "reused" if index in prior_mirrored else "new"
        for arm in ARMS:
            classes, probabilities = top5[arm][0][position], top5[arm][1][position]
            records[arm].append({
                "index": index, "signer_id": ref["signer_id"], "gloss": ref["gloss"], "class_id": target, "arm": arm,
                "mirror_source": source,
                "top5": [{"rank": rank + 1, "class_id": c, "gloss": bundle.id_to_label[c], "probability": p}
                         for rank, (c, p) in enumerate(zip(classes, probabilities))]})
    for arm in ARMS:
        with (output / f"predictions_{arm}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records[arm]:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(output / "logits.npz", indices=np.asarray(indices, dtype=np.int64),
                        original=logits["original"][0], mirrored=logits["mirrored"][0])

    g2 = compare_top5(records["original"], historical, "G2",
                      f"original-arm top-5 class IDs identical to runs/20260923T050910Z-official-val-v2-evaluation-019ecd54 for all {VALIDATION_CLIPS} clips, |probability difference| <= {TOLERANCE}")
    g2b = compare_top5([r for r in records["mirrored"] if r["index"] in prior_mirrored], prior_mirrored, "G2b",
                       f"mirrored-arm top-5 class IDs on the {len(reused)} reused clips identical to runs/20261010T154553Z-poet-handedness-inference-cd7e0157 predictions_mirrored.jsonl, |probability difference| <= {TOLERANCE}")
    helpers.write_json(output / "gate_g2.json", g2)
    helpers.write_json(output / "gate_g2b.json", g2b)
    passed = g2["status"] == "passed" and g2b["status"] == "passed"
    gpu = None
    if device.type == "cuda":
        gpu = {"name": torch.cuda.get_device_name(device), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    summary.update({"status": "completed" if passed else "gate_failed", "completed_at_utc": helpers.now(),
                    "clips": len(indices), "reused_mirrored_clips": len(reused), "new_mirrored_clips": len(indices) - len(reused),
                    "model": bundle.metadata,
                    "gates": {g["gate"]: {k: g[k] for k in ("status", "clips", "max_abs_probability_difference")} for g in (g2, g2b)},
                    "inputs": {"reference_index": str(args.reference_index), "reference_index_sha256": helpers.REFERENCE_INDEX_SHA256,
                               "mirror_index": str(args.mirror_index), "mirror_index_sha256": helpers.sha256(args.mirror_index),
                               "mirror_pipeline_sha256": mirror["pipeline_sha256"],
                               "historical_predictions": str(args.historical_predictions),
                               "historical_predictions_sha256": helpers.HISTORICAL_PREDICTIONS_SHA256,
                               "prior_mirrored_predictions": str(args.prior_mirrored_predictions),
                               "prior_mirrored_predictions_sha256": PRIOR_MIRRORED_SHA256},
                    "tta_definition": "0.5 * (original_logits + mirrored_logits) in float64; top-5 by descending value, ties to the lower class ID",
                    "metrics": "not computed here; see the independent analysis run",
                    "timings": {**timings, "total_seconds": time.perf_counter() - started}, "gpu": gpu,
                    "evaluation_code_sha256": helpers.sha256(Path(__file__)), "helpers_code_sha256": helpers.sha256(Path(helpers.__file__)),
                    "artifact_sha256": {name: helpers.sha256(output / name) for name in
                                        ["logits.npz", "gate_g2.json", "gate_g2b.json"] + [f"predictions_{arm}.jsonl" for arm in ARMS]},
                    "interpretation": "Validation-only, inference-only; not a test result."})
    helpers.write_json(output / "summary.json", summary)
    print(json.dumps({"G2": g2["status"], "G2b": g2b["status"], "clips": len(indices),
                      "max_abs_probability_difference": {"G2": g2["max_abs_probability_difference"], "G2b": g2b["max_abs_probability_difference"]}}), flush=True)
    return 0 if passed else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-index", type=Path, default=WORKSPACE / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json")
    parser.add_argument("--historical-predictions", type=Path, default=WORKSPACE / "runs/20260923T050910Z-official-val-v2-evaluation-019ecd54/evaluation/predictions.jsonl")
    parser.add_argument("--prior-mirrored-predictions", type=Path, default=WORKSPACE / "runs/20261010T154553Z-poet-handedness-inference-cd7e0157/inference/predictions_mirrored.jsonl")
    parser.add_argument("--mirror-index", type=Path, required=True, help="fullval_cache_manifest.json of the mirror run")
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        code = run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    raise SystemExit(code)


if __name__ == "__main__":
    main()
