#!/usr/bin/env python3
"""Frozen PoET inference for the final test, with a mandatory validation replay.

Authorized by docs/environment/final-test-preregistration-20261012.json. No model
is trained or changed. Two families run in separate processes so that each keeps
the numerics of the historical run it must reproduce:

  official   official checkpoint on the original cache and on the horizontally
             flipped cache (mirror TTA); numerics of evaluate_poet_mirror_fullval.py
             (seeded, batch 128, default CUDA math). Saves full float32 logits,
             the classifier-input CLS embedding, and top-5 for original, mirrored
             and TTA = 0.5 * (original + mirrored) logits in float64.
  finetuned  the six val-selected B/G checkpoints of the factorial preregistration;
             numerics of evaluate_methods.py (deterministic algorithms, no TF32,
             CUBLAS_WORKSPACE_CONFIG, 4 threads, batch 128). Geometry is computed
             on the fly with build_geometry_features exactly as the cache builder
             and read_geometry do. Saves CLS, top-5 IDs/logits/probabilities and
             per-clip full-logit SHA256.

Cohorts:
  val   frozen v2 validation cache, the mirror-TTA full-validation flipped cache;
        outputs are compared with the historical outputs and must agree exactly
        (gate G3); nothing here reads test data.
  test  final test caches; refused unless the matching val gate file passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = WORKSPACE / "docs/environment/final-test-preregistration-20261012.json"
PREREGISTRATION_SHA256 = "e241b1bbb6a7de21187d31a914a3308321132d2f10ed3e0952fbd652b2664286"
FACTORIAL_PREREGISTRATION = WORKSPACE / "docs/environment/factorial-retrieval-preregistration-20260926.json"
OFFICIAL_SHA256 = "5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0"
V2_PIPELINE = "ceca202ce01a6327c7433b1a29a7ba5ea482f6b8335a87d7732938463dd1941b"
MIRROR_PIPELINE = "8af5b135fdc1ed6defc8666dfaffe0450caada1d57ab6f1f4b9a894818fb2ed9"
COUNTS = {"val": 10304, "test": 32941}
VAL = {
    "original_index": "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json",
    "flipped_index": "runs/20261011T024119Z-poet-mirror-tta-r2-extraction-0600c862/extraction/fullval_cache_manifest.json",
    "tta_inference": "runs/20261011T052015Z-poet-mirror-tta-r2-inference-0d9b12e9/inference",
    "official_cls": "runs/20261011T071432Z-cls-representation-dataset-73a92db2/features/val_features.npy",
    "factorial_val_features": "runs/20260926T224535Z-factorial-retrieval-val-extraction-f573dc76/features",
}
BATCH = 128


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_index(path: Path, split: str, pipeline: str) -> tuple[dict, list[dict]]:
    index = json.loads(path.read_text(encoding="utf-8"))
    require(index.get("status") == "completed", f"{path} is not a completed cache index")
    require(index.get("pipeline_sha256") == pipeline, f"{path} pipeline differs from {pipeline[:12]}")
    samples = sorted(index["samples"], key=lambda item: item["index"])
    require([item["index"] for item in samples] == list(range(COUNTS[split])), f"{path} must hold indices 0..{COUNTS[split]-1}")
    require(all(item["split"] == split and item["status"] in {"ok", "reused"} for item in samples),
            f"{path} has wrong-split or failed samples; failures must be handled explicitly, not dropped")
    for item in samples:
        for key in ("features_path", "pose_path", "cache_manifest_path"):
            value = Path(item[key])
            item[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return index, samples


def read_input(item: dict, preprocessing) -> tuple[np.ndarray, np.ndarray]:
    from extract_retrieval_features import read_evaluation_input
    return read_evaluation_input(item, preprocessing)


def geometry_for(item: dict, indices: np.ndarray) -> np.ndarray:
    """build_geometry_cache + read_geometry, without the train/val-only cache contract."""
    from poet_geometry import GEOMETRY_DIM, build_geometry_features
    from prepare_geometry_cache import _validate_array
    require(sha256_file(Path(item["pose_path"])) == item["pose_sha256"], f"raw pose SHA256 changed at {item['index']}")
    with np.load(item["pose_path"], allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    frames = len(raw["body_pose"])
    require(frames == item["video_metadata"]["decoded_frames"], f"frame count differs at {item['index']}")
    full = build_geometry_features(raw, np.arange(frames, dtype=np.int64))
    _validate_array(full, frames)
    require(indices.ndim == 1 and np.all(indices >= -1) and np.all(indices < frames), "bad source indices")
    aligned = np.zeros((len(indices), GEOMETRY_DIM), dtype=np.float32)
    real = indices >= 0
    aligned[real] = full[indices[real]]
    return aligned


def run_official(args, cohort: str, output: Path) -> dict:
    import torch
    from _poet_runtime import load_bundle, resolve_device
    from evaluate_poet_mirror import topk_rows
    seed = 20260922
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(args.device)
    bundle = load_bundle(WORKSPACE / "assets/checkpoints", device)
    require(bundle.metadata["checkpoint_sha256"] == OFFICIAL_SHA256, "official checkpoint differs")
    bundle.model.eval()
    results = {}
    for arm, index_path, pipeline in (("original", args.original_index, V2_PIPELINE), ("mirrored", args.flipped_index, MIRROR_PIPELINE)):
        _, samples = load_index(index_path, cohort, pipeline)
        logits = np.zeros((len(samples), 2731), dtype=np.float32)
        embeddings = np.zeros((len(samples), 128), dtype=np.float32)
        for start in range(0, len(samples), BATCH):
            batch_items = samples[start:start + BATCH]
            batch = torch.stack([torch.from_numpy(read_input(item, bundle.preprocessing)[0]) for item in batch_items])
            captured = []
            hook = bundle.model.mlp_classifier.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
            try:
                with torch.inference_mode():
                    values = bundle.model(batch, device)
            finally:
                hook.remove()
            require(tuple(values.shape) == (len(batch_items), 2731) and bool(torch.isfinite(values).all()), "bad logits")
            require(len(captured) == 1 and tuple(captured[0].shape) == (len(batch_items), 128), "CLS capture failed")
            logits[start:start + len(batch_items)] = values.cpu().numpy()
            embeddings[start:start + len(batch_items)] = captured[0].cpu().numpy()
            if (start // BATCH) % 20 == 0:
                print(f"official {arm}: {start + len(batch_items)}/{len(samples)}", flush=True)
        results[arm] = {"samples": samples, "logits": logits, "embeddings": embeddings}
    rows = {}
    for arm in ("original", "mirrored"):
        probabilities, classes = torch.from_numpy(results[arm]["logits"]).softmax(dim=1).topk(5, dim=1)
        rows[arm] = (classes.tolist(), probabilities.tolist())
    tta = 0.5 * (results["original"]["logits"].astype(np.float64) + results["mirrored"]["logits"].astype(np.float64))
    rows["tta"] = topk_rows(tta)
    np.savez(output / "official_logits.npz", original=results["original"]["logits"], mirrored=results["mirrored"]["logits"],
             embeddings=results["original"]["embeddings"])
    samples = results["original"]["samples"]
    for arm, (classes, probabilities) in rows.items():
        with (output / f"predictions_official_{arm}.jsonl").open("w", encoding="utf-8") as stream:
            for i, item in enumerate(samples):
                record = {key: item[key] for key in ("index", "gloss", "signer_id", "split", "video_sha256")}
                record.update(status="ok", top5_ids=classes[i], top5_scores=probabilities[i])
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return {"arms": list(rows), "clips": len(samples)}


def run_finetuned(args, cohort: str, output: Path) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from _poet_runtime import resolve_device
    from extract_factorial_retrieval_features import extract_method_batch, load_frozen_models
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    entries = json.loads(FACTORIAL_PREREGISTRATION.read_text())["models"]
    models = load_frozen_models(entries, device)
    _, samples = load_index(args.original_index, cohort, V2_PIPELINE)
    summary = {}
    for entry in entries:
        model_id = entry["model_id"]
        frozen = models[model_id]
        model, bundle = frozen["model"], frozen["bundle"]
        is_geometry = entry["method"] == "geometry"
        embeddings = np.zeros((len(samples), 128), dtype=np.float32)
        ids = np.zeros((len(samples), 5), dtype=np.int64)
        top_logits = np.zeros((len(samples), 5), dtype=np.float32)
        records = []
        for start in range(0, len(samples), BATCH):
            batch_items = samples[start:start + BATCH]
            prepared, geometry = [], []
            for item in batch_items:
                values, indices = read_input(item, bundle.preprocessing)
                prepared.append(values)
                if is_geometry:
                    geometry.append(geometry_for(item, indices))
            arrays, details = extract_method_batch(model, np.stack(prepared), np.stack(geometry) if is_geometry else None, device)
            span = slice(start, start + len(batch_items))
            embeddings[span], ids[span], top_logits[span] = arrays["embeddings"], arrays["top5_class_ids"], arrays["top5_logits"]
            for i, item in enumerate(batch_items):
                records.append({**{key: item[key] for key in ("index", "gloss", "signer_id", "split", "video_sha256")},
                                "status": "ok", "top5_ids": arrays["top5_class_ids"][i].tolist(),
                                "top5_scores": details["top5_probabilities"][i], "logits_sha256": details["logits_sha256"][i]})
            if (start // BATCH) % 20 == 0:
                print(f"{model_id}: {start + len(batch_items)}/{len(samples)}", flush=True)
        np.savez(output / f"{model_id}_features.npz", embeddings=embeddings, top5_class_ids=ids, top5_logits=top_logits)
        with (output / f"predictions_{model_id}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary[model_id] = {"clips": len(records), "geometry": is_geometry}
    return summary


def replay_gate(family: str, output: Path) -> dict:
    """Compare validation outputs with the historical runs; every check must be exact."""
    checks = {}
    if family == "official":
        reference = np.load(WORKSPACE / VAL["tta_inference"] / "logits.npz")
        ours = np.load(output / "official_logits.npz")
        checks["reference_row_order_is_index_order"] = bool(np.array_equal(reference["indices"], np.arange(COUNTS["val"])))
        for arm in ("original", "mirrored"):
            ref_key = [k for k in reference.files if arm in k]
            require(len(ref_key) == 1, f"reference logits key for {arm}: {reference.files}")
            checks[f"logits_{arm}_exact"] = bool(np.array_equal(ours[arm], reference[ref_key[0]]))
        cls_reference = np.load(WORKSPACE / VAL["official_cls"])
        checks["official_cls_exact"] = bool(np.array_equal(ours["embeddings"], cls_reference))
        checks["official_cls_max_abs_diff"] = float(np.max(np.abs(ours["embeddings"].astype(np.float64) - cls_reference)))
        for arm in ("original", "mirrored", "tta"):
            ref = [json.loads(line) for line in (WORKSPACE / VAL["tta_inference"] / f"predictions_{arm}.jsonl").read_text().splitlines()]
            new = [json.loads(line) for line in (output / f"predictions_official_{arm}.jsonl").read_text().splitlines()]
            ref_ids = [r.get("top5_ids") or [c["class_id"] for c in r["top5"]] for r in ref]
            checks[f"top5_{arm}_exact"] = len(ref) == len(new) and all(r == n["top5_ids"] for r, n in zip(ref_ids, new))
    else:
        entries = json.loads(FACTORIAL_PREREGISTRATION.read_text())["models"]
        for entry in entries:
            model_id = entry["model_id"]
            ref = [json.loads(line) for line in (WORKSPACE / entry["reference_predictions_path"]).read_text().splitlines()]
            new = [json.loads(line) for line in (output / f"predictions_{model_id}.jsonl").read_text().splitlines()]
            same_ids = len(ref) == len(new) and all([c["class_id"] for c in r["top5"]] == n["top5_ids"] for r, n in zip(ref, new))
            same_sha = len(ref) == len(new) and all(r["logits_sha256"] == n["logits_sha256"] for r, n in zip(ref, new))
            reference_features = np.load(WORKSPACE / VAL["factorial_val_features"] / model_id / "features.npz")
            ours = np.load(output / f"{model_id}_features.npz")
            same_features = all(np.array_equal(ours[k], reference_features[k]) for k in ("embeddings", "top5_class_ids", "top5_logits"))
            checks[model_id] = {"top5_ids_exact": same_ids, "logits_sha256_exact": same_sha, "features_exact": same_features}
    flat = [v for v in checks.values() if isinstance(v, bool)] + [x for v in checks.values() if isinstance(v, dict) for x in v.values()]
    return {"gate": f"G3_{family}", "passed": all(flat), "checks": checks, "verified_at_utc": now()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=("official", "finetuned"), required=True)
    parser.add_argument("--cohort", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-index", type=Path)
    parser.add_argument("--flipped-index", type=Path)
    parser.add_argument("--val-gate", type=Path, help="Passed gate JSON of the same family (required for test)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    require(sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256, "final-test preregistration changed")
    if args.cohort == "val":
        args.original_index = WORKSPACE / VAL["original_index"]
        args.flipped_index = WORKSPACE / VAL["flipped_index"]
    else:
        require(args.original_index is not None and (args.family == "finetuned" or args.flipped_index is not None),
                "test cohort needs explicit cache indices")
        require(args.val_gate is not None, "test inference requires the passed validation replay gate")
        gate = json.loads(args.val_gate.read_text())
        require(gate.get("passed") is True and gate.get("gate") == f"G3_{args.family}", "validation replay gate did not pass")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = run_official(args, args.cohort, output) if args.family == "official" else run_finetuned(args, args.cohort, output)
    result = {"family": args.family, "cohort": args.cohort, "summary": summary, "seconds": time.perf_counter() - started,
              "preregistration_sha256": PREREGISTRATION_SHA256, "completed_at_utc": now(),
              "inputs": {"original_index": str(args.original_index), "original_index_sha256": sha256_file(args.original_index),
                         "flipped_index": str(args.flipped_index) if args.flipped_index else None,
                         "flipped_index_sha256": sha256_file(args.flipped_index) if args.flipped_index else None}}
    if args.cohort == "val":
        gate = replay_gate(args.family, output)
        write_json(output / f"gate_G3_{args.family}.json", gate)
        result["gate"] = {"passed": gate["passed"]}
        print(json.dumps({"gate": gate["gate"], "passed": gate["passed"]}), flush=True)
    write_json(output / "summary.json", result)
    return 0 if args.cohort == "test" or result["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
