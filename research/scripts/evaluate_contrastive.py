#!/usr/bin/env python3
"""Independent word-only evaluator for strict cross-signer pilot checkpoints.

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

from contrastive_components import PROTOCOL_VERSION, create_optimizer, optimizer_protocol
from finetune_methods import validate_optimizer_resume

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
    from contrastive_components import ContrastivePoET
    from _poet_runtime import plain_metadata, OFFICIAL_ASL_CHECKPOINT_SHA256

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
                or protocol.get("version") != PROTOCOL_VERSION
                or checkpoint.get("protocol_sha256") != canonical_hash(protocol)):
            raise ValueError("A valid cross-signer checkpoint/protocol fingerprint is required")
        if checkpoint.get("num_classes") != NUM_CLASSES or bundle.metadata["label_map_sha256"] != OFFICIAL_LABEL_MAP_SHA256:
            raise ValueError("Official 2731-class vocabulary is required")
        for key in ("data_preprocessing", "model_parameters"):
            if checkpoint[key] != protocol["model"][key]:
                raise ValueError(f"Checkpoint {key} disagrees with its protocol")
        if checkpoint["training_parameters"]["hidden_dropout_prob"] != protocol["model"]["hidden_dropout_prob"]:
            raise ValueError("Checkpoint dropout disagrees with protocol")
        if (bundle.metadata["checkpoint_sha256"] != OFFICIAL_ASL_CHECKPOINT_SHA256
                or protocol["model"]["checkpoint_sha256"] != OFFICIAL_ASL_CHECKPOINT_SHA256):
            raise ValueError("Only the pinned official PoET initialization is supported")
        for key in ("data_preprocessing", "model_parameters"):
            if checkpoint[key] != plain_metadata(bundle.checkpoint[key]):
                raise ValueError("Checkpoint base architecture/preprocessing differs from official")
        if checkpoint["training_parameters"]["hidden_dropout_prob"] != float(bundle.checkpoint["training_parameters"]["hidden_dropout_prob"]):
            raise ValueError("Checkpoint dropout differs from official")
        for name, expected in protocol["code_sha256"].items():
            if sha256(WORKSPACE / name) != expected:
                raise ValueError(f"Checkpoint source contract differs: {name}")
        bundle.model = ContrastivePoET(bundle.model, head_seed=protocol["method_metadata"]["head_seed"])
        method_metadata = bundle.model.protocol()
        if method_metadata != protocol["method_metadata"] or checkpoint.get("method_metadata") != method_metadata:
            raise ValueError("Projection model metadata disagrees with protocol")
        settings = checkpoint["training_parameters"]
        loss_weight = protocol["loss"]["supcon_lambda"]
        if (protocol["method"] not in {"balanced_ce", "cross_signer_supcon"}
                or (protocol["method"] == "balanced_ce") != (loss_weight == 0)
                or settings["supcon_lambda"] != loss_weight
                or settings["temperature"] != protocol["loss"]["temperature"]):
            raise ValueError("Checkpoint loss settings disagree with protocol")
        expected_state = bundle.model.state_dict()
        restored_state = checkpoint["model_state_dict"]
        if set(restored_state) != set(expected_state):
            raise ValueError("Checkpoint model state inventory differs")
        for name, value in restored_state.items():
            if (not torch.is_tensor(value) or value.dtype != expected_state[name].dtype
                    or value.shape != expected_state[name].shape):
                raise ValueError(f"Invalid model tensor: {name}")
            require_finite(f"checkpoint model {name}", value)
        bundle.model.load_state_dict(restored_state, strict=True)
        if tensor_state_hash(bundle.model.state_dict()) != checkpoint.get("model_state_sha256"):
            raise ValueError("Checkpoint model tensor fingerprint differs")
        config = dict(lr=settings["lr"], projection_lr=settings["projection_lr"],
                      weight_decay=settings["weight_decay"], supcon_lambda=loss_weight)
        if optimizer_protocol(bundle.model, **config) != protocol["optimizer"]:
            raise ValueError("Checkpoint optimizer configuration differs")
        optimizer = create_optimizer(bundle.model, **config)
        steps = protocol["training"]["steps_per_epoch"]
        epoch = checkpoint["completed_epoch"]
        if (type(epoch) is not int or epoch < 0 or checkpoint["next_epoch"] != epoch + 1
                or checkpoint["global_optimizer_step"] != epoch * steps
                or checkpoint["samples_seen"] != epoch * protocol["training"]["draws_per_epoch"]
                or checkpoint.get("scheduler_state_dict") is not None or checkpoint.get("scaler_state_dict") is not None):
            raise ValueError("Checkpoint epoch/step/sample counters disagree")
        validate_optimizer_resume(optimizer, checkpoint["optimizer_state_dict"], checkpoint["global_optimizer_step"])
        cohort = audit_cohort(args.cache_manifest, bundle, args.engineering_smoke, "validation")
        expected_data = protocol["data"]["validation"]
        actual_data = {"manifest_sha256": cohort["sha256"], "split": cohort["split"],
                       "samples": len(cohort["samples"]), "pipeline_sha256": cohort["provenance"]["pipeline_sha256"],
                       "source_manifest_sha256": cohort["provenance"]["source_manifest_sha256"]}
        if actual_data != expected_data:
            raise ValueError("Independent evaluation must reproduce the checkpoint's validation cohort")
        scope = "smoke_only" if args.engineering_smoke or protocol["scope"] == "smoke_only" else "official_validation"
        summary.update(method=protocol["method"], method_metadata=method_metadata, scope=scope,
                       checkpoint_sha256=sha256(args.checkpoint), checkpoint_epoch=checkpoint["completed_epoch"],
                       protocol_sha256=checkpoint["protocol_sha256"], optimizer_protocol_version=protocol["version"],
                       data=cohort["provenance"],
                       evaluator_sha256=sha256(Path(__file__)), projection_used_at_inference=False,
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
                inputs = []
                for item in items:
                    path = Path(item["features_path"])
                    verify_file(path, item["features_sha256"], "evaluation features")
                    with np.load(path, allow_pickle=False) as archive:
                        inputs.append(torch.from_numpy(archive["model_input"]))
                batch = torch.stack(inputs)
                error = None
                try:
                    with torch.inference_mode():
                        synchronize(device)
                        stamp = time.perf_counter()
                        logits = bundle.model(batch, device)
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
