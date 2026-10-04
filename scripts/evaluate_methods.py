#!/usr/bin/env python3
"""Independent word-only evaluator for strict PoET method checkpoints.

Development accepts official train (explicit smoke) and validation only.
A broken cohort audit fails the entire run; forward failures retain every input
in the metric denominator. Attribute target files are not accepted or read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from finetune_baseline import atomic_json, audit_cohort, now, tensor_state_hash
from evaluate_pose_cache import (MetricAccumulator, NUM_CLASSES, OFFICIAL_LABEL_MAP_SHA256,
                                 canonical_hash, sha256, verify_file)

from method_optimizer import SUPPORTED_PROTOCOLS, validate_checkpoint_optimizer_contract

WORKSPACE = Path(__file__).resolve().parents[1]


def run(args):
    if min(args.batch_size, args.num_threads) < 1:
        raise ValueError("Batch size and num-threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise ValueError("Invalid deterministic CUBLAS workspace configuration")
    import numpy as np
    import torch
    from _poet_runtime import load_bundle, resolve_device, require_finite, synchronize
    from method_runtime import restore_method
    from prepare_geometry_cache import audit_geometry_index, read_geometry

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {"schema_version": 1, "status": "running", "created_at_utc": now(),
               "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    atomic_json(output / "summary.json", summary)
    try:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = resolve_device(args.device)
        bundle = load_bundle(args.checkpoint_dir, device)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        protocol = checkpoint.get("protocol", {})
        if (checkpoint.get("schema_version") != 1
                or protocol.get("version") not in SUPPORTED_PROTOCOLS
                or checkpoint.get("protocol_sha256") != canonical_hash(protocol)):
            raise ValueError("A valid unified-method checkpoint/protocol fingerprint is required")
        if checkpoint.get("num_classes") != NUM_CLASSES or bundle.metadata["label_map_sha256"] != OFFICIAL_LABEL_MAP_SHA256:
            raise ValueError("Official 2731-class vocabulary is required")
        for key in ("data_preprocessing", "model_parameters"):
            if checkpoint[key] != protocol["model"][key]:
                raise ValueError(f"Checkpoint {key} disagrees with its protocol")
        if checkpoint["training_parameters"]["hidden_dropout_prob"] != protocol["model"]["hidden_dropout_prob"]:
            raise ValueError("Checkpoint dropout disagrees with protocol")
        bundle.model, method_metadata = restore_method(bundle, checkpoint)
        if (method_metadata != protocol.get("method_metadata")
                or method_metadata["method"] != protocol.get("method")):
            raise ValueError("Checkpoint method configuration disagrees with its protocol")
        validate_checkpoint_optimizer_contract(checkpoint, bundle.model)
        if tensor_state_hash(bundle.model.state_dict()) != checkpoint.get("model_state_sha256"):
            raise ValueError("Checkpoint model tensor fingerprint differs")
        is_geometry = protocol["method"] == "geometry"
        if is_geometry != (args.geometry_manifest is not None):
            raise ValueError("Only geometry checkpoints require a geometry manifest")
        cohort = audit_cohort(args.cache_manifest, bundle, args.engineering_smoke, "validation")
        geometry_index = audit_geometry_index(args.geometry_manifest, cohort) if is_geometry else None
        scope = "smoke_only" if args.engineering_smoke or protocol["scope"] == "smoke_only" else "official_validation"
        summary.update(method=protocol["method"], method_metadata=method_metadata, scope=scope,
                       checkpoint_sha256=sha256(args.checkpoint), checkpoint_epoch=checkpoint["completed_epoch"],
                       protocol_sha256=checkpoint["protocol_sha256"], optimizer_protocol_version=protocol["version"],
                       data=cohort["provenance"],
                       geometry_manifest_sha256=None if geometry_index is None else geometry_index["sha256"],
                       evaluator_sha256=sha256(Path(__file__)), attribute_targets_read=False,
                       numerics={"dtype": "float32", "tf32": False, "deterministic_algorithms": True,
                                 "device": str(device), "num_threads": args.num_threads, "batch_size": args.batch_size})
        bundle.model.eval()
        accumulator = MetricAccumulator(bundle.id_to_label)
        failures = []
        forward_seconds = 0.0
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for offset in range(0, len(cohort["samples"]), args.batch_size):
                items = cohort["samples"][offset:offset + args.batch_size]
                # An independently implemented batch path: labels are used only
                # below for scoring, never supplied to the model forward call.
                inputs, geometry = [], []
                for item in items:
                    path = Path(item["features_path"])
                    verify_file(path, item["features_sha256"], "evaluation features")
                    with np.load(path, allow_pickle=False) as archive:
                        inputs.append(torch.from_numpy(archive["model_input"]))
                        if is_geometry:
                            geometry.append(torch.from_numpy(read_geometry(item, archive["source_frame_indices"], geometry_index)))
                batch = torch.stack(inputs)
                geo = torch.stack(geometry) if is_geometry else None
                error = None
                try:
                    with torch.inference_mode():
                        synchronize(device)
                        stamp = time.perf_counter()
                        logits = bundle.model(batch, device, geometry=geo)
                        synchronize(device)
                        forward_seconds += time.perf_counter() - stamp
                        require_finite("evaluation logits", logits)
                        if tuple(logits.shape) != (len(items), NUM_CLASSES):
                            raise ValueError("Unexpected method logits shape")
                        values, classes = logits.softmax(dim=1).topk(5, dim=1)
                        values, classes, arrays = values.cpu().tolist(), classes.cpu().tolist(), logits.cpu().numpy()
                except Exception as exc:
                    error = {"error_type": type(exc).__name__, "error": str(exc)}
                for i, item in enumerate(items):
                    record = {key: item[key] for key in ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")}
                    if error is None:
                        accumulator.add(item["gloss"], classes[i])
                        record.update(status="ok", top5=[{"class_id": label, "gloss": bundle.id_to_label[label], "probability": probability}
                            for label, probability in zip(classes[i], values[i])],
                            logits_sha256=hashlib.sha256(arrays[i].astype("<f4").tobytes()).hexdigest())
                    else:
                        accumulator.add(item["gloss"], None)
                        record.update(status="error", top5=[], failure_stage="model_forward", **error)
                        failures.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        metrics, per_class = accumulator.summary()
        if metrics["attempted"] != len(cohort["samples"]):
            raise RuntimeError("Metric denominator differs from the full selected cohort")
        with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
        atomic_json(output / "metrics.json", metrics)
        atomic_json(output / "failures.json", {"count": len(failures), "samples": failures})
        summary.update(status="completed" if not failures else "completed_with_failures", metrics=metrics,
                       completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                       model_forward_seconds_including_transfer=forward_seconds,
                       timing_note="No warmup; validation timing is not an end-to-end or latency benchmark.",
                       artifacts={name: {"sha256": sha256(output / name)} for name in ("predictions.jsonl", "metrics.json", "per_class.csv", "failures.json")})
        atomic_json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__,
                       error=str(error), elapsed_seconds=time.perf_counter() - started,
                       denominator_policy="Cohort/cache errors abort evaluation; no reduced-cohort result is published.")
        atomic_json(output / "summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"status": result["status"], "method": result["method"], "scope": result["scope"],
                      "attempted": result["metrics"]["attempted"], "failed": result["metrics"]["failed"]}), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
