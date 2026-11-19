#!/usr/bin/env python3
"""Fixed-checkpoint geometry inference ablation, without training or test data.

The normal condition must exactly reproduce a specified completed independent
evaluation before off/shuffled inference is allowed. Off supplies all-zero
geometry, including masks, so the existing model applies zero residual. A
shuffled recipient receives one other video's entire prepared geometry sequence
(including that donor's masks and temporal sampling). This is an intervention
on a coadapted model, not a separately trained baseline or an OOD-free causal
test. Run through run_logged.py; never overwrite an existing output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from finetune_baseline import atomic_json, audit_cohort, now, tensor_state_hash
from evaluate_pose_cache import (MetricAccumulator, NUM_CLASSES, OFFICIAL_LABEL_MAP_SHA256,
                                 canonical_hash, sha256, verify_file)
from method_optimizer import SUPPORTED_PROTOCOLS, validate_checkpoint_optimizer_contract

WORKSPACE = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")
CONDITIONS = ("normal", "off", "shuffled")
LIMITATIONS = [
    "Same-checkpoint off is not an independently trained baseline; base and branch were coadapted.",
    "Shuffling changes geometry values, presence masks, and temporal alignment together; it may create out-of-distribution inputs.",
    "One fixed label-independent permutation is a mechanism diagnostic, not a permutation significance test.",
    "Validation was used to select this checkpoint; these are not test results or independent-seed evidence.",
]


def donor_positions(count: int, seed: int) -> list[int]:
    """Private-RNG single cycle: every donor once, no sample assigned itself."""
    if type(count) is not int or count < 2 or type(seed) is not int:
        raise ValueError("Derangement requires an integer count >= 2 and integer seed")
    order = list(range(count))
    random.Random(seed).shuffle(order)
    donors = [None] * count
    for position, recipient in enumerate(order):
        donors[recipient] = order[(position + 1) % count]
    return donors


def make_permutation(samples: list[dict], seed: int) -> dict:
    donors = donor_positions(len(samples), seed)
    if len({item["index"] for item in samples}) != len(samples):
        raise ValueError("Permutation sample indices must be unique")
    mapping = [{"recipient_position": i, "donor_position": donor,
                "recipient_index": samples[i]["index"], "donor_index": samples[donor]["index"],
                "recipient_video_sha256": samples[i]["video_sha256"],
                "donor_video_sha256": samples[donor]["video_sha256"]}
               for i, donor in enumerate(donors)]
    return {"schema_version": 1, "seed": seed,
            "algorithm": "private_random_shuffle_then_cycle_successor_v1",
            "sample_count": len(samples), "labels_used": False,
            "mapping": mapping, "mapping_sha256": canonical_hash(mapping)}


def load_geometry_bank(samples: list[dict], geometry_index: dict, reader):
    """Read every donor using its own audited evaluation source-frame indices."""
    import numpy as np
    import torch
    aligned = []
    for position, item in enumerate(samples, 1):
        path = Path(item["features_path"])
        verify_file(path, item["features_sha256"], "ablation donor evaluation features")
        with np.load(path, allow_pickle=False) as archive:
            geometry = reader(item, archive["source_frame_indices"], geometry_index)
            if geometry.shape != (archive["model_input"].shape[0], 28):
                raise ValueError("Donor geometry differs from its own prepared sequence length")
            aligned.append(torch.from_numpy(geometry))
        if position % 1000 == 0:
            print(f"Prepared own-index geometry {position}/{len(samples)}", flush=True)
    return torch.stack(aligned)


def geometry_for_condition(bank, positions: list[int], condition: str, donors: list[int]):
    import torch
    if condition not in CONDITIONS:
        raise ValueError("Unknown geometry intervention")
    if condition == "shuffled":
        return torch.stack([bank[donors[i]] for i in positions])
    geometry = torch.stack([bank[i] for i in positions])
    return torch.zeros_like(geometry) if condition == "off" else geometry


def load_reference(path: Path) -> dict:
    """Require finalized standard independent-evaluator artifacts and hashes."""
    directory = path if path.is_dir() else path.parent
    if not path.is_dir() and path.name != "predictions.jsonl":
        raise ValueError("Normal reference must be an evaluation directory or predictions.jsonl")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("method") != "geometry":
        raise ValueError("Normal reference must be a completed geometry evaluation")
    names = ("predictions.jsonl", "metrics.json", "per_class.csv", "failures.json")
    digests = {name: sha256(directory / name) for name in names}
    for name, digest in digests.items():
        if summary.get("artifacts", {}).get(name, {}).get("sha256") != digest:
            raise ValueError(f"Normal reference artifact hash mismatch: {name}")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics != summary.get("metrics") or metrics.get("failed") != 0:
        raise ValueError("Normal reference metrics are inconsistent or include failures")
    return {"directory": str(directory.resolve()), "summary": summary, "metrics": metrics,
            "artifact_sha256": digests, "summary_sha256": sha256(directory / "summary.json")}


def compare_normal_reference(output: Path, reference: dict) -> dict:
    """Check record equality (identities/probabilities/logits), metrics and classes."""
    directory = Path(reference["directory"])
    compared = 0
    with (output / "predictions.jsonl").open(encoding="utf-8") as actual, (directory / "predictions.jsonl").open(encoding="utf-8") as expected:
        from itertools import zip_longest
        for compared, (left, right) in enumerate(zip_longest(actual, expected), 1):
            if left is None or right is None or json.loads(left) != json.loads(right):
                raise ValueError(f"Normal/reference prediction differs at record {compared}")
    if compared != reference["metrics"]["attempted"]:
        raise ValueError("Normal/reference prediction count disagrees with its denominator")
    if json.loads((output / "metrics.json").read_text()) != reference["metrics"]:
        raise ValueError("Normal/reference metrics are not exactly equal")
    if (output / "per_class.csv").read_bytes() != (directory / "per_class.csv").read_bytes():
        raise ValueError("Normal/reference per-class metrics are not exactly equal")
    for name, digest in reference["artifact_sha256"].items():
        verify_file(directory / name, digest, "unchanged normal reference " + name)
    verify_file(directory / "summary.json", reference["summary_sha256"], "unchanged normal reference summary")
    return {"exact": True, "records_compared": compared, "identity_top5_probability_logits_exact": True,
            "metrics_exact": True, "per_class_exact": True}


def evaluate_condition(condition: str, output: Path, bundle, cohort: dict, bank, donors: list[int], args, common: dict) -> dict:
    """Same batch construction and inference numerics as evaluate_methods.py."""
    import numpy as np
    import torch
    from _poet_runtime import require_finite, synchronize

    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    summary = {**common, "condition": condition, "status": "running", "created_at_utc": now()}
    atomic_json(output / "summary.json", summary)
    accumulator = MetricAccumulator(bundle.id_to_label)
    failures, forward_seconds = [], 0.0
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for offset in range(0, len(cohort["samples"]), args.batch_size):
                items = cohort["samples"][offset:offset + args.batch_size]
                inputs = []
                for item in items:
                    path = Path(item["features_path"])
                    verify_file(path, item["features_sha256"], "evaluation features")
                    with np.load(path, allow_pickle=False) as archive:
                        inputs.append(torch.from_numpy(archive["model_input"]))
                batch = torch.stack(inputs)
                positions = list(range(offset, offset + len(items)))
                geo = geometry_for_condition(bank, positions, condition, donors)
                error = None
                try:
                    with torch.inference_mode():
                        synchronize(bundle.device)
                        stamp = time.perf_counter()
                        logits = bundle.model(batch, bundle.device, geometry=geo)
                        synchronize(bundle.device)
                        forward_seconds += time.perf_counter() - stamp
                        require_finite("evaluation logits", logits)
                        if tuple(logits.shape) != (len(items), NUM_CLASSES):
                            raise ValueError("Unexpected method logits shape")
                        values, classes = logits.softmax(dim=1).topk(5, dim=1)
                        values, classes, arrays = values.cpu().tolist(), classes.cpu().tolist(), logits.cpu().numpy()
                except Exception as exc:
                    error = {"error_type": type(exc).__name__, "error": str(exc)}
                for i, item in enumerate(items):
                    record = {key: item[key] for key in IDENTITY_FIELDS}
                    if error is None:
                        accumulator.add(item["gloss"], classes[i])
                        record.update(status="ok", top5=[{"class_id": label, "gloss": bundle.id_to_label[label], "probability": probability}
                            for label, probability in zip(classes[i], values[i])],
                            logits_sha256=hashlib.sha256(arrays[i].astype("<f4").tobytes()).hexdigest())
                    else:
                        accumulator.add(item["gloss"], None)
                        record.update(status="error", top5=[], failure_stage="model_forward", **error)
                    if condition == "shuffled":
                        donor = cohort["samples"][donors[offset + i]]
                        record.update(geometry_donor_index=donor["index"], geometry_donor_video_sha256=donor["video_sha256"])
                    if error is not None:
                        failures.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if offset == 0 or offset + len(items) >= len(cohort["samples"]) or (offset // args.batch_size + 1) % 20 == 0:
                    print(f"{condition}: inferred {offset + len(items)}/{len(cohort['samples'])}", flush=True)
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
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started,
                       denominator_policy="Cohort/cache errors abort evaluation; no reduced-cohort result is published.")
        atomic_json(output / "summary.json", summary)
        raise


def run(args):
    if min(args.batch_size, args.num_threads) < 1:
        raise ValueError("Batch size and num-threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise ValueError("Invalid deterministic CUBLAS workspace configuration")
    import torch
    from _poet_runtime import load_bundle, resolve_device
    from method_runtime import restore_method
    from prepare_geometry_cache import audit_geometry_index, read_geometry

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {"schema_version": 1, "status": "running", "created_at_utc": now(),
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "optimizer_steps": 0, "backward_calls": 0, "attribute_targets_read": False,
               "normal_reference_exact": False, "limitations": LIMITATIONS, "conditions": {}}
    atomic_json(output / "summary.json", summary)
    try:
        verify_file(args.checkpoint, args.expected_checkpoint_sha256, "fixed ablation checkpoint")
        reference = load_reference(args.normal_reference)
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
        verify_file(args.checkpoint, args.expected_checkpoint_sha256, "fixed ablation checkpoint after load")
        protocol = checkpoint.get("protocol", {})
        if (checkpoint.get("schema_version") != 1 or protocol.get("version") not in SUPPORTED_PROTOCOLS
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
        if method_metadata != protocol.get("method_metadata") or method_metadata["method"] != "geometry" or protocol.get("method") != "geometry":
            raise ValueError("A strictly restored geometry checkpoint and matching method protocol are required")
        validate_checkpoint_optimizer_contract(checkpoint, bundle.model)
        initial_state = tensor_state_hash(bundle.model.state_dict())
        if initial_state != checkpoint.get("model_state_sha256"):
            raise ValueError("Checkpoint model tensor fingerprint differs")
        if not args.engineering_smoke and protocol.get("scope") != "formal_finetuning":
            raise ValueError("A formal ablation requires a formal fine-tuning checkpoint")
        cohort = audit_cohort(args.cache_manifest, bundle, args.engineering_smoke, "validation")
        geometry_index = audit_geometry_index(args.geometry_manifest, cohort)
        scope = "smoke_only" if args.engineering_smoke else "official_validation"
        numerics = {"dtype": "float32", "tf32": False, "deterministic_algorithms": True,
                    "device": str(device), "num_threads": args.num_threads, "batch_size": args.batch_size}
        common = {"schema_version": 1, "method": "geometry", "method_metadata": method_metadata,
                  "scope": scope, "checkpoint_sha256": args.expected_checkpoint_sha256,
                  "checkpoint_epoch": checkpoint["completed_epoch"], "protocol_sha256": checkpoint["protocol_sha256"],
                  "optimizer_protocol_version": protocol["version"], "data": cohort["provenance"],
                  "geometry_manifest_sha256": geometry_index["sha256"], "evaluator_sha256": sha256(Path(__file__)),
                  "attribute_targets_read": False, "optimizer_steps": 0, "backward_calls": 0,
                  "numerics": numerics, "limitations": LIMITATIONS}
        for key in ("scope", "checkpoint_sha256", "checkpoint_epoch", "protocol_sha256", "data", "geometry_manifest_sha256", "numerics"):
            if reference["summary"].get(key) != common[key]:
                raise ValueError(f"Normal reference {key} disagrees with the fixed ablation protocol")
        if reference["metrics"]["attempted"] != len(cohort["samples"]):
            raise ValueError("Normal reference denominator disagrees with the audited cohort")
        permutation = make_permutation(cohort["samples"], args.shuffle_seed)
        atomic_json(output / "permutation.json", permutation)
        donors = [entry["donor_position"] for entry in permutation["mapping"]]
        bank = load_geometry_bank(cohort["samples"], geometry_index, read_geometry)
        bank_hash = tensor_state_hash({"aligned_geometry_bank": bank})
        summary.update(**common, initial_model_state_sha256=initial_state,
                       cache_manifest_sha256=cohort["sha256"], permutation_sha256=sha256(output / "permutation.json"),
                       permutation_mapping_sha256=permutation["mapping_sha256"],
                       geometry_bank_sha256=bank_hash, geometry_bank_shape=list(bank.shape),
                       normal_reference={key: value for key, value in reference.items() if key != "summary"})
        atomic_json(output / "summary.json", summary)
        bundle.model.eval()
        for condition in CONDITIONS:
            summary["active_condition"] = condition
            atomic_json(output / "summary.json", summary)
            result = evaluate_condition(condition, output / condition, bundle, cohort, bank, donors, args, common)
            summary["conditions"][condition] = result
            if tensor_state_hash(bundle.model.state_dict()) != initial_state:
                raise ValueError(f"Model tensors changed during {condition} inference")
            if condition == "normal":
                summary["normal_reference_comparison"] = compare_normal_reference(output / condition, reference)
                summary["normal_reference_exact"] = True
            atomic_json(output / "summary.json", summary)
        final_state = tensor_state_hash(bundle.model.state_dict())
        if final_state != initial_state:
            raise ValueError("Model tensors changed during ablation")
        if tensor_state_hash({"aligned_geometry_bank": bank}) != bank_hash:
            raise ValueError("Aligned geometry bank changed during ablation")
        if any(parameter.grad is not None for parameter in bundle.model.parameters()):
            raise ValueError("Unexpected gradients were created during inference-only ablation")
        verify_file(args.checkpoint, args.expected_checkpoint_sha256, "unchanged fixed ablation checkpoint")
        verify_file(args.cache_manifest, cohort["sha256"], "unchanged pose cache manifest")
        verify_file(args.geometry_manifest, geometry_index["sha256"], "unchanged geometry manifest")
        summary.update(status="completed" if all(result["status"] == "completed" for result in summary["conditions"].values()) else "completed_with_failures",
                       active_condition=None, completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                       weights_unchanged=True, checkpoint_file_unchanged=True, geometry_bank_unchanged=True,
                       final_model_state_sha256=final_state, checkpoint_sha256_after=sha256(args.checkpoint),
                       artifacts={"permutation.json": {"sha256": sha256(output / "permutation.json")},
                                  **{f"{condition}/summary.json": {"sha256": sha256(output / condition / "summary.json")} for condition in CONDITIONS}})
        atomic_json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started,
                       denominator_policy="Cohort/cache errors abort evaluation; no reduced-cohort result is published.")
        atomic_json(output / "summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--normal-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--shuffle-seed", type=int, default=20260925)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "normal_reference_exact": result["normal_reference_exact"],
                      "conditions": {key: value["metrics"] for key, value in result["conditions"].items()}}), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
