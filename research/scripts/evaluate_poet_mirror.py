#!/usr/bin/env python3
"""Official PoET logits on original and horizontally flipped validation poses.

Inference only; no parameter is updated. Fixed by
docs/environment/poet-handedness-preregistration-20261010.json.

Arms, evaluated on the same preregistered clips in the same batch layout:
  original  frozen v2 validation cache (unflipped video)
  mirrored  poses re-extracted from horizontally flipped video
            (scripts/prepare_poet_mirror_cache.py --mode mirror)
  tta       mean of the original and mirrored logits (float64), secondary

Gate G2: the original arm must reproduce the historical official predictions
(identical top-5 class IDs, probability difference <= 1e-5) for every clip.
Test data is refused; the official checkpoint and preprocessing are unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
NUM_CLASSES = 2731
REFERENCE_INDEX_SHA256 = "a829029d700e50dd07d0f88454d9f3d42a65e946586a1188e7dfe9a81d5a5c86"
REFERENCE_PIPELINE_SHA256 = "ceca202ce01a6327c7433b1a29a7ba5ea482f6b8335a87d7732938463dd1941b"
HISTORICAL_PREDICTIONS_SHA256 = "8d7e6fe2d347ef7a23ef5b8b59a0534602109d83148fcbedeb81545651aaaca5"
OFFICIAL_LABEL_MAP_SHA256 = {
    "d_LId2Label.pkl": "cd29470b041e1a029e62bc66f47313b08744f669b283b58cf647d532482268e6",
    "d_Label2LId.pkl": "a2d0f112d9c26cdd75fe9beaa10b54747df3793f4334e95539d70f2dde9e86dd",
}
G2_PROBABILITY_TOLERANCE = 1e-5
ARMS = ("original", "mirrored", "tta")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def topk_rows(logits, k: int = 5) -> tuple[list[list[int]], list[list[float]]]:
    """Float64 top-k by descending logit, ties broken by the lower class ID."""
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    order = np.lexsort((np.broadcast_to(np.arange(values.shape[1]), values.shape), -values), axis=1)[:, :k]
    return order.tolist(), np.take_along_axis(probabilities, order, axis=1).tolist()


def selected_indices(selection: dict) -> tuple[list[int], dict[int, str]]:
    cohort_of = {}
    for name in ("left_dominant", "control"):
        for row in selection["cohorts"][name]:
            if row["index"] in cohort_of:
                raise ValueError("Cohorts overlap")
            cohort_of[row["index"]] = name
    return sorted(cohort_of), cohort_of


def check_mirror_index(index: dict, indices: list[int]) -> dict[int, dict]:
    pipeline = index.get("pipeline", {})
    if (index.get("status") != "completed" or index.get("split") != "val" or index.get("mode") != "mirror"
            or pipeline.get("extractor", {}).get("flip") is not True or pipeline.get("name") != "poet-official-raw-normalized-eval-v2-hflip"):
        raise ValueError("Mirror index is not a completed flipped-video validation cache")
    samples = {item["index"]: item for item in index["samples"]}
    if sorted(samples) != indices or any(item["status"] != "ok" for item in samples.values()):
        raise ValueError("Mirror cache does not contain exactly the preregistered clips, all successful")
    return samples


def load_input(item: dict, index: dict, preprocessing):
    """Verify one cache entry exactly as evaluate_pose_cache.py does, then prepare it."""
    import numpy as np
    from _poet_runtime import prepare_features
    metadata = json.loads(Path(item["cache_manifest_path"]).read_text(encoding="utf-8"))
    if (metadata.get("status") != "ok" or metadata.get("pipeline_sha256") != index["pipeline_sha256"]
            or metadata.get("identity") != item.get("identity") or not item.get("identity")):
        raise ValueError(f"Cache manifest identity/pipeline mismatch at index {item['index']}")
    artifacts = metadata["artifacts"]
    if (artifacts["normalized_features.npz"]["sha256"] != item["features_sha256"]
            or artifacts["raw_pose.npz"]["sha256"] != item["pose_sha256"]):
        raise ValueError(f"Cache artifact hashes differ from index at {item['index']}")
    if sha256(Path(item["features_path"])) != item["features_sha256"] or sha256(Path(item["pose_path"])) != item["pose_sha256"]:
        raise ValueError(f"Cache file SHA256 mismatch at index {item['index']}")
    with np.load(item["features_path"], allow_pickle=False) as archive:
        prepared, frames = prepare_features(archive["normalized_features"], preprocessing, mode="eval")
        if not np.array_equal(prepared.numpy(), archive["model_input"]) or not np.array_equal(frames, archive["source_frame_indices"]):
            raise ValueError(f"Recomputed eval preprocessing differs from cached input at {item['index']}")
    return prepared


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from _poet_runtime import load_bundle, require_finite, resolve_device, synchronize

    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "status": "running", "created_at_utc": now(),
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    write_json(output / "summary.json", summary)
    if sha256(args.reference_index) != REFERENCE_INDEX_SHA256:
        raise ValueError("Reference v2 cache index SHA256 differs from the preregistered file")
    if sha256(args.historical_predictions) != HISTORICAL_PREDICTIONS_SHA256:
        raise ValueError("Historical official predictions SHA256 differs from the preregistered file")
    reference = json.loads(args.reference_index.read_text(encoding="utf-8"))
    if reference.get("pipeline_sha256") != REFERENCE_PIPELINE_SHA256 or reference.get("split") != "val":
        raise ValueError("Reference index is not the v2 validation cache")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    indices, cohort_of = selected_indices(selection)
    mirror = json.loads(args.mirror_index.read_text(encoding="utf-8"))
    mirror_samples = check_mirror_index(mirror, indices)
    reference_samples = {item["index"]: item for item in reference["samples"]}
    historical = {row["index"]: row for row in read_jsonl(args.historical_predictions)}
    for index in indices:
        ref, mir = reference_samples[index], mirror_samples[index]
        if mir["identity"] != ref["identity"] or historical[index]["video_sha256"] != ref["video_sha256"]:
            raise ValueError(f"Identity mismatch between caches/history at index {index}")
        if ref["split"] != "val" or ref["identity"]["split"] != "val":
            raise ValueError("Only validation clips may be evaluated")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    bundle = load_bundle(args.checkpoint_dir.resolve(), device)
    if bundle.checkpoint["num_classes"] != NUM_CLASSES or bundle.metadata["label_map_sha256"] != OFFICIAL_LABEL_MAP_SHA256:
        raise ValueError("Requires the official 2,731-class checkpoint and unmodified label maps")

    logits = {}
    timings = {}
    for arm, samples, index_doc in (("original", reference_samples, reference), ("mirrored", mirror_samples, mirror)):
        stamp = time.perf_counter()
        rows = []
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            batch = torch.stack([load_input(samples[i], index_doc, bundle.preprocessing) for i in batch_indices])
            with torch.inference_mode():
                values = bundle.model(batch, device)
                require_finite(f"{arm} logits", values)
                if tuple(values.shape) != (len(batch_indices), NUM_CLASSES):
                    raise ValueError(f"Unexpected logits shape {tuple(values.shape)}")
                probabilities, classes = values.softmax(dim=1).topk(5, dim=1)
            rows.append((values.float().cpu().numpy(), classes.cpu().tolist(), probabilities.cpu().tolist()))
        synchronize(device)
        timings[f"{arm}_seconds"] = time.perf_counter() - stamp
        logits[arm] = (np.concatenate([r[0] for r in rows]), [c for r in rows for c in r[1]], [p for r in rows for p in r[2]])

    tta_logits = 0.5 * (logits["original"][0].astype(np.float64) + logits["mirrored"][0].astype(np.float64))
    tta_classes, tta_probabilities = topk_rows(tta_logits)
    top5 = {"original": logits["original"][1:], "mirrored": logits["mirrored"][1:], "tta": (tta_classes, tta_probabilities)}

    records = {arm: [] for arm in ARMS}
    for position, index in enumerate(indices):
        ref = reference_samples[index]
        target = bundle.label_to_id[ref["gloss"]]
        for arm in ARMS:
            classes, probabilities = top5[arm][0][position], top5[arm][1][position]
            records[arm].append({
                "index": index, "cohort": cohort_of[index], "signer_id": ref["signer_id"], "gloss": ref["gloss"],
                "class_id": target, "arm": arm, "top1_correct": classes[0] == target, "top5_correct": target in classes,
                "top5": [{"rank": rank + 1, "class_id": c, "gloss": bundle.id_to_label[c], "probability": p}
                         for rank, (c, p) in enumerate(zip(classes, probabilities))]})
    for arm in ARMS:
        with (output / f"predictions_{arm}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records[arm]:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    np.savez_compressed(output / "logits.npz", indices=np.asarray(indices, dtype=np.int64),
                        original=logits["original"][0], mirrored=logits["mirrored"][0])

    mismatches, max_difference = [], 0.0
    for record in records["original"]:
        past = historical[record["index"]]
        ids_now = [p["class_id"] for p in record["top5"]]
        ids_past = [p["class_id"] for p in past["top5"]]
        difference = max(abs(a["probability"] - b["probability"]) for a, b in zip(record["top5"], past["top5"]))
        max_difference = max(max_difference, difference)
        if (ids_now != ids_past or difference > G2_PROBABILITY_TOLERANCE or past["class_id"] != record["class_id"]
                or past["status"] != "ok" or past["signer_id"] != record["signer_id"]):
            mismatches.append({"index": record["index"], "now": ids_now, "historical": ids_past, "max_abs_probability_difference": difference})
    gate = {"gate": "G2", "status": "passed" if not mismatches else "failed",
            "rule": f"original-arm top-5 class IDs identical to runs/20260923T050910Z-official-val-v2-evaluation-019ecd54 and |probability difference| <= {G2_PROBABILITY_TOLERANCE}",
            "clips": len(indices), "mismatches": mismatches, "max_abs_probability_difference": max_difference}
    write_json(output / "gate_g2.json", gate)

    def rate(rows, key):
        return sum(r[key] for r in rows) / len(rows)
    quick = {arm: {cohort: {"clips": len(rows), "top1": rate(rows, "top1_correct"), "top5": rate(rows, "top5_correct")}
                   for cohort in ("left_dominant", "control") for rows in [[r for r in records[arm] if r["cohort"] == cohort]]}
             for arm in ARMS}
    gpu = None
    if device.type == "cuda":
        gpu = {"name": torch.cuda.get_device_name(device), "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    summary.update({"status": "completed" if gate["status"] == "passed" else "gate_failed", "completed_at_utc": now(),
                    "clips": len(indices), "cohort_counts": {c: sum(v == c for v in cohort_of.values()) for c in ("left_dominant", "control")},
                    "model": bundle.metadata, "gate_g2": {k: gate[k] for k in ("status", "clips", "max_abs_probability_difference")},
                    "inputs": {"reference_index": str(args.reference_index), "reference_index_sha256": REFERENCE_INDEX_SHA256,
                               "mirror_index": str(args.mirror_index), "mirror_index_sha256": sha256(args.mirror_index),
                               "mirror_pipeline_sha256": mirror["pipeline_sha256"],
                               "selection": str(args.selection), "selection_sha256": sha256(args.selection),
                               "historical_predictions": str(args.historical_predictions),
                               "historical_predictions_sha256": HISTORICAL_PREDICTIONS_SHA256},
                    "tta_definition": "0.5 * (original_logits + mirrored_logits) in float64; top-5 by descending value, ties to the lower class ID",
                    "quick_metrics_not_the_analysis": quick, "timings": {**timings, "total_seconds": time.perf_counter() - started},
                    "gpu": gpu, "evaluation_code_sha256": sha256(Path(__file__)),
                    "artifact_sha256": {name: sha256(output / name) for name in
                                        ["logits.npz", "gate_g2.json"] + [f"predictions_{arm}.jsonl" for arm in ARMS]},
                    "interpretation": "Validation-only, inference-only diagnostic of input orientation; not a test result."})
    write_json(output / "summary.json", summary)
    print(json.dumps({"G2": gate["status"], "max_abs_probability_difference": max_difference, "clips": len(indices)}), flush=True)
    return 0 if gate["status"] == "passed" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-index", type=Path, default=WORKSPACE / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json")
    parser.add_argument("--historical-predictions", type=Path, default=WORKSPACE / "runs/20260923T050910Z-official-val-v2-evaluation-019ecd54/evaluation/predictions.jsonl")
    parser.add_argument("--mirror-index", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
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
