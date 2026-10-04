#!/usr/bin/env python3
"""Evaluate the pinned 2,731-class PoET checkpoint on an audited pose cache.

Default: the complete official validation split. Training data requires
--allow-train-smoke; a validation subset requires --allow-subset-val. Test data
is deliberately unsupported by this development entry point. Failed extraction
or inference remains in both instance and per-class denominators.

Run through scripts/run_logged.py and use --output-dir '{run_dir}/evaluation'.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
NUM_CLASSES = 2731
OFFICIAL_LABEL_MAP_SHA256 = {
    "d_LId2Label.pkl": "cd29470b041e1a029e62bc66f47313b08744f669b283b58cf647d532482268e6",
    "d_Label2LId.pkl": "a2d0f112d9c26cdd75fe9beaa10b54747df3793f4334e95539d70f2dde9e86dd",
}
SPLIT_SHA256 = {
    "train": "87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329",
    "val": "6ac78c9d551dfcc81d225a5279d7637c2891a4490851234be6e4674666e21366",
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def split_name(value: str) -> str:
    if value == "validation":
        return "val"
    if value not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {value!r}")
    return value


def require_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def referenced_path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing artifact path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def verify_file(path: Path, expected: str, name: str) -> None:
    require_sha(expected, name)
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch for {name}: {path}")


class MetricAccumulator:
    """Pure-Python metrics: every declared input contributes exactly once."""

    def __init__(self, id_to_label: dict[int, str]):
        if set(id_to_label) != set(range(len(id_to_label))) or len(set(id_to_label.values())) != len(id_to_label):
            raise ValueError("Vocabulary IDs must be contiguous and labels unique")
        self.id_to_label = id_to_label
        self.label_to_id = {label: i for i, label in id_to_label.items()}
        self.rows = [{"class_id": i, "gloss": id_to_label[i], "attempted": 0, "inferred": 0,
                      "failed": 0, "top1_correct": 0, "top5_correct": 0} for i in range(len(id_to_label))]

    def add(self, gloss: str, predictions: list[int] | None) -> None:
        if gloss not in self.label_to_id:
            raise ValueError(f"Unknown ground-truth gloss: {gloss!r}")
        if predictions is not None:
            if not predictions or len(predictions) > 5 or len(set(predictions)) != len(predictions):
                raise ValueError("Predictions must contain 1–5 distinct class IDs")
            if any(type(i) is not int or i not in self.id_to_label for i in predictions):
                raise ValueError("Unknown predicted class ID")
        target = self.label_to_id[gloss]
        row = self.rows[target]
        row["attempted"] += 1
        if predictions is None:
            row["failed"] += 1
        else:
            row["inferred"] += 1
            row["top1_correct"] += int(predictions[0] == target)
            row["top5_correct"] += int(target in predictions)

    def summary(self) -> tuple[dict, list[dict]]:
        counts = {key: sum(row[key] for row in self.rows)
                  for key in ("attempted", "inferred", "failed", "top1_correct", "top5_correct")}
        rows = [{**row,
                 "errors_top1": row["attempted"] - row["top1_correct"],
                 "errors_top5": row["attempted"] - row["top5_correct"],
                 "top1": row["top1_correct"] / row["attempted"] if row["attempted"] else None,
                 "top5": row["top5_correct"] / row["attempted"] if row["attempted"] else None}
                for row in self.rows]
        present = [row for row in rows if row["attempted"]]
        missing = [row["gloss"] for row in rows if not row["attempted"]]
        macro = {key: sum(row[key] for row in present) / len(present) if present else None
                 for key in ("top1", "top5")}
        result = {**counts, "denominator_policy": "all selected inputs; extraction/load/inference failures count as incorrect",
                  "per_instance": {key: counts[key + "_correct"] / counts["attempted"] if counts["attempted"] else None
                                   for key in ("top1", "top5")},
                  "per_class_observed": {**macro, "denominator_classes": len(present)},
                  "per_class_full_vocabulary": {key: macro[key] if not missing else None for key in ("top1", "top5")},
                  "vocabulary_classes": len(rows), "missing_classes": missing,
                  "missing_class_policy": "Full-vocabulary macro is null when any class has no selected input; observed-class macro is labeled separately."}
        return result, rows


def validate_cache_index(index: dict, label_to_id: dict[str, int], allow_train_smoke: bool) -> str:
    """Check cohort structure before any inference or sample-error handling."""
    if index.get("schema_version") != 1 or index.get("status") not in {"completed", "failed"}:
        raise ValueError("Cache index must be finalized (schema_version=1, status=completed|failed)")
    samples = index.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Cache index has no samples")
    splits = {split_name(item["split"]) for item in samples}
    if "test" in splits:
        raise ValueError("Test data is forbidden in this development evaluator")
    if len(splits) != 1:
        raise ValueError(f"Mixed splits are forbidden: {sorted(splits)}")
    split = next(iter(splits))
    if split_name(index["split"]) != split:
        raise ValueError("Top-level split disagrees with sample splits")
    if split == "train" and not allow_train_smoke:
        raise ValueError("Training data requires explicit --allow-train-smoke")
    indices, sources = set(), set()
    for item in samples:
        if item.get("gloss") not in label_to_id:
            raise ValueError(f"Unknown ground-truth gloss: {item.get('gloss')!r}")
        if type(item.get("index")) is not int or item["index"] < 0 or item["index"] in indices:
            raise ValueError("Cache item indices must be unique nonnegative integers")
        indices.add(item["index"])
        source = item.get("source_path")
        if not isinstance(source, str) or not source or source in sources:
            raise ValueError("Source paths must be present and unique")
        sources.add(source)
        require_sha(item.get("video_sha256"), "video_sha256")
        if not isinstance(item.get("signer_id"), str) or not item["signer_id"]:
            raise ValueError("Each sample needs a signer_id")
        if item.get("status") not in {"ok", "reused", "error", "interrupted"}:
            raise ValueError(f"Unfinished/unknown cache status: {item.get('status')!r}")
    selection = index.get("selection", {})
    if selection.get("indices") != [item["index"] for item in samples] or selection.get("selected_count") != len(samples):
        raise ValueError("Cache samples do not exactly match recorded selection order/count")
    return split


def audit_provenance(index_path: Path, index: dict, preprocessing: dict,
                     label_to_id: dict[str, int], split: str, allow_subset_val: bool) -> dict:
    pipeline = index["pipeline"]
    if canonical_hash(pipeline) != require_sha(index.get("pipeline_sha256"), "pipeline_sha256"):
        raise ValueError("Pipeline identity hash mismatch")
    if pipeline.get("preprocessing") != preprocessing:
        raise ValueError("Cache preprocessing differs from checkpoint preprocessing")
    code_hashes = pipeline.get("source_sha256", {})
    required = {"scripts/_poet_runtime.py", "vendor/poet/mediapipe_extraction.py", "vendor/poet/model.py"}
    if not required.issubset(code_hashes):
        raise ValueError("Pipeline does not identify all normalization/preprocessing sources")
    for name, digest in code_hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"scripts", "vendor"}:
            raise ValueError(f"Invalid pipeline source path: {name!r}")
        verify_file(WORKSPACE / relative, digest, f"pipeline source {name}")
    source_path = referenced_path(index_path.parent, index["source_manifest_path"])
    verify_file(source_path, index["source_manifest_sha256"], "source manifest")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_samples = source["samples"]
    if source.get("status") != "complete" or source.get("selection", {}).get("count") != len(source_samples):
        raise ValueError("Source download manifest is incomplete; omitted download failures must not disappear from evaluation")
    if index["selection"]["source_count"] != len(source_samples):
        raise ValueError("Source manifest count changed")
    split_info = source["splits"][split]
    csv_path = referenced_path(source_path.parent, split_info["path"])
    if split_info["sha256"] != SPLIT_SHA256[split]:
        raise ValueError("Split CSV does not identify the pinned official ASL Citizen release")
    verify_file(csv_path, split_info["sha256"], "official split CSV")
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        split_rows = list(csv.DictReader(stream))
    official = {row["Video file"]: row for row in split_rows}
    if len(official) != len(split_rows) or len(official) != split_info["rows"]:
        raise ValueError("Official split CSV count/uniqueness mismatch")
    selected_videos = []
    for item in index["samples"]:
        original = source_samples[item["index"]]
        expected_path = referenced_path(source_path.parent, original["path"])
        if (Path(item["source_path"]).resolve() != expected_path or item["video_sha256"] != original["sha256"]
                or item["gloss"] != original["gloss"] or item["signer_id"] != original["signer_id"]
                or split_name(original["split"]) != split):
            raise ValueError(f"Cache row does not match original source manifest: {item['index']}")
        filename = original["video_file"]
        if expected_path.name != filename or filename not in official:
            raise ValueError(f"Video is absent from official split: {filename}")
        row = official[filename]
        if item["gloss"] != row["Gloss"] or item["signer_id"] != row["Participant ID"]:
            raise ValueError(f"Official split label/signer mismatch: {filename}")
        selected_videos.append(filename)
    if len(set(selected_videos)) != len(selected_videos):
        raise ValueError("Repeated videos in evaluation cohort")
    full_split = set(selected_videos) == set(official)
    if split == "val" and not full_split and not allow_subset_val:
        raise ValueError("Validation cohort is incomplete; only a diagnostic run may use --allow-subset-val")
    unknown_official = sorted({row["Gloss"] for row in split_rows} - set(label_to_id))
    if unknown_official:
        raise ValueError(f"Official split has unknown vocabulary labels: {unknown_official}")
    return {"cache_manifest_path": str(index_path), "cache_manifest_sha256": sha256(index_path),
            "source_manifest_path": str(source_path), "source_manifest_sha256": index["source_manifest_sha256"],
            "pipeline_sha256": index["pipeline_sha256"], "pipeline": pipeline,
            "official_split_path": str(csv_path), "official_split_sha256": split_info["sha256"],
            "official_split_count": len(official), "selected_count": len(selected_videos),
            "complete_official_split": full_split,
            "video_integrity_policy": "Original source and extraction identities verified; video bytes are not re-read during pose-only evaluation."}


def run(args: argparse.Namespace) -> dict:
    # Lazy imports keep metric/cohort tests usable without a local CUDA stack.
    import numpy as np
    import torch
    from _poet_runtime import load_bundle, prepare_features, require_finite, resolve_device, synchronize

    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    write_json(output / "summary.json", summary)
    try:
        if args.batch_size < 1 or args.warmup_batches < 0:
            raise ValueError("batch-size must be positive and warmup-batches nonnegative")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        device = resolve_device(args.device)
        stamp = time.perf_counter()
        bundle = load_bundle(args.checkpoint_dir, device, args.checkpoint)
        synchronize(device)
        load_seconds = time.perf_counter() - stamp
        if bundle.checkpoint["num_classes"] != NUM_CLASSES:
            raise ValueError(f"This ASL Citizen evaluator requires exactly {NUM_CLASSES} official classes")
        if bundle.metadata["label_map_sha256"] != OFFICIAL_LABEL_MAP_SHA256:
            raise ValueError("Class-ID order must use the unmodified official label maps")
        index_path = args.cache_manifest.resolve()
        index = json.loads(index_path.read_text(encoding="utf-8"))
        split = validate_cache_index(index, bundle.label_to_id, args.allow_train_smoke)
        provenance = audit_provenance(index_path, index, bundle.checkpoint["data_preprocessing"],
                                     bundle.label_to_id, split, args.allow_subset_val)
        summary.update({"split": split, "purpose": "train_smoke_not_benchmark" if split == "train" else
                        ("official_validation" if provenance["complete_official_split"] else "validation_subset_diagnostic"),
                        "model": bundle.metadata, "data": provenance,
                        "evaluation_code_sha256": sha256(Path(__file__)),
                        "determinism": {"seed": args.seed, "eval_mode": True, "augmentation": False,
                                        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                        "claim": "Fixed eval preprocessing and seeds; cross-device bitwise determinism is not claimed."}})
        records = []
        accumulator = MetricAccumulator(bundle.id_to_label)
        timing = {"load_model_seconds": load_seconds, "cache_integrity_and_preprocessing_seconds": 0.0,
                  "warmup_seconds": 0.0, "warmup_batches": args.warmup_batches, "warmup_actual_batch_size": None,
                  "timed_model_forward_seconds": 0.0, "timed_batches": 0, "timed_examples": 0}
        queue = []
        warmed = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def record_failure(record: dict, stage: str, error: Exception | str) -> None:
            record.update({"status": "error", "failure_stage": stage, "error": str(error),
                           "error_type": type(error).__name__ if isinstance(error, Exception) else "UpstreamCacheError",
                           "top5": [], "top1_correct": False, "top5_correct": False})
            accumulator.add(record["gloss"], None)

        def infer_batch() -> None:
            nonlocal warmed
            if not queue:
                return
            batch = torch.stack([item[1] for item in queue])
            try:
                with torch.inference_mode():
                    if not warmed:
                        synchronize(device)
                        start = time.perf_counter()
                        for _ in range(args.warmup_batches):
                            warm_logits = bundle.model(batch, device)
                            require_finite("warmup logits", warm_logits)
                        synchronize(device)
                        timing["warmup_seconds"] = time.perf_counter() - start
                        timing["warmup_actual_batch_size"] = len(queue)
                        warmed = True
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                    synchronize(device)
                    start = time.perf_counter()
                    logits = bundle.model(batch, device)
                    synchronize(device)
                    timing["timed_model_forward_seconds"] += time.perf_counter() - start
                    timing["timed_batches"] += 1
                    timing["timed_examples"] += len(queue)
                    require_finite("model logits", logits)
                    if tuple(logits.shape) != (len(queue), NUM_CLASSES):
                        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
                    probabilities, class_ids = logits.softmax(dim=1).topk(5, dim=1)
                    values, classes = probabilities.cpu().tolist(), class_ids.cpu().tolist()
                for (record, _), ids, probs in zip(queue, classes, values):
                    target = bundle.label_to_id[record["gloss"]]
                    record.update({"status": "ok", "top1_correct": ids[0] == target, "top5_correct": target in ids,
                                   "top5": [{"rank": rank + 1, "class_id": i, "gloss": bundle.id_to_label[i], "probability": p}
                                            for rank, (i, p) in enumerate(zip(ids, probs))]})
                    accumulator.add(record["gloss"], ids)
            except Exception as error:
                for record, _ in queue:
                    record_failure(record, "model_forward", error)
            finally:
                queue.clear()

        for item in index["samples"]:
            record = {key: item[key] for key in ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")}
            record["class_id"] = bundle.label_to_id[item["gloss"]]
            records.append(record)
            if item["status"] not in {"ok", "reused"}:
                record_failure(record, "pose_extraction", item.get("error", f"Cache status={item['status']}"))
                continue
            stamp = time.perf_counter()
            try:
                cache_metadata_path = referenced_path(index_path.parent, item["cache_manifest_path"])
                cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
                if (cache_metadata.get("status") != "ok" or cache_metadata.get("pipeline_sha256") != index["pipeline_sha256"]
                        or cache_metadata.get("identity") != item.get("identity") or not item.get("identity")):
                    raise ValueError("Per-video cache manifest identity/pipeline differs from evaluation index")
                artifacts = cache_metadata["artifacts"]
                if (artifacts["normalized_features.npz"]["sha256"] != item["features_sha256"]
                        or artifacts["raw_pose.npz"]["sha256"] != item["pose_sha256"]):
                    raise ValueError("Per-video cache artifact hashes differ from evaluation index")
                features_path = referenced_path(index_path.parent, item["features_path"])
                verify_file(features_path, item["features_sha256"], "normalized feature cache")
                # raw_pose is not decoded, but its bytes are verified so later
                # geometry runs share the same intact cache lineage.
                pose_path = referenced_path(index_path.parent, item["pose_path"])
                verify_file(pose_path, item["pose_sha256"], "raw pose cache")
                with np.load(features_path, allow_pickle=False) as archive:
                    features = archive["normalized_features"]
                    prepared, indices = prepare_features(features, bundle.preprocessing, mode="eval")
                    if not np.array_equal(prepared.numpy(), archive["model_input"]) or not np.array_equal(indices, archive["source_frame_indices"]):
                        raise ValueError("Recomputed official eval preprocessing differs from cached inputs")
                record.update({"features_path": str(features_path), "features_sha256": item["features_sha256"],
                               "pose_path": str(pose_path), "pose_sha256": item["pose_sha256"],
                               "original_frames": len(features), "input_frames": len(prepared),
                               "source_frame_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()})
                queue.append((record, prepared))
            except Exception as error:
                record_failure(record, "cache_integrity_or_preprocessing", error)
            finally:
                timing["cache_integrity_and_preprocessing_seconds"] += time.perf_counter() - stamp
            if len(queue) >= args.batch_size:
                infer_batch()
                if timing["timed_batches"] % 10 == 0:
                    print(f"Evaluated {len(records)}/{len(index['samples'])} selected inputs", flush=True)
        infer_batch()
        metrics, classes = accumulator.summary()
        if metrics["attempted"] != len(index["samples"]):
            raise RuntimeError("Internal error: metric denominator differs from selected cohort")
        timing.update({"total_seconds": time.perf_counter() - started,
                       "model_examples_per_second": timing["timed_examples"] / timing["timed_model_forward_seconds"] if timing["timed_model_forward_seconds"] else None,
                       "timing_scope": "synchronized official model.forward on CPU-prepared batches, including its CPU-to-device transfer; excludes softmax/topk, cache IO, extraction, and warmup"})
        gpu = None
        if device.type == "cuda":
            gpu = {"name": torch.cuda.get_device_name(device),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                   "memory_scope": "PyTorch allocator including resident model; peak reset after first warmup; excludes other processes"}
        summary.update({"status": "completed" if not metrics["failed"] else "completed_with_failures", "metrics": metrics,
                        "timings": timing, "gpu": gpu, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "interpretation": "Train smoke/subset diagnostic scores are not official validation or test results. Softmax probabilities are not calibrated confidence."})
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(classes[0]))
            writer.writeheader()
            writer.writerows(classes)
        write_json(output / "failures.json", {"count": metrics["failed"], "samples": [r for r in records if r["status"] != "ok"]})
        write_json(output / "metrics.json", metrics)
        summary["artifacts"] = {name: {"path": name, "sha256": sha256(output / name)}
                                for name in ("predictions.jsonl", "per_class.csv", "failures.json", "metrics.json")}
        write_json(output / "summary.json", summary)
        console_metrics = {key: metrics[key] for key in ("attempted", "inferred", "failed", "per_instance", "per_class_observed")}
        print(json.dumps({"status": summary["status"], "purpose": summary["purpose"], "metrics": console_metrics,
                          "output_dir": str(output)}, ensure_ascii=False), flush=True)
        return summary
    except BaseException as error:
        summary.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        "error_type": type(error).__name__, "error": str(error),
                        "elapsed_seconds": time.perf_counter() - started})
        write_json(output / "summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    default_output = Path(os.environ["POET_RUN_DIR"]) / "evaluation" if "POET_RUN_DIR" in os.environ else None
    parser.add_argument("--output-dir", type=Path, default=default_output, required=default_output is None)
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--checkpoint", type=Path, help="Optional alternative checkpoint; official vocabulary maps still required")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--allow-train-smoke", action="store_true")
    parser.add_argument("--allow-subset-val", action="store_true", help="Explicit diagnostic subset; cannot be labeled full validation")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
