#!/usr/bin/env python3
"""Single-device PoET warm-start fine-tuning with epoch-boundary resume only.

Official train + validation cohorts are required unless --engineering-smoke is
explicitly set. Smoke runs may reuse train data for the validation role; every
result then remains smoke_only. Run this command through run_logged.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "poet-baseline-fp32-weight1-epoch-resume-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_seed(run_seed: int, epoch: int, video_sha256: str, purpose: str = "temporal") -> int:
    payload = json.dumps([purpose, run_seed, epoch, video_sha256], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def weight1_from_labels(labels: list[int], num_classes: int) -> tuple[list[int], list[float]]:
    """The actual upstream weight1 formula: 1 - count / N, not 1 / count."""
    if not labels or num_classes < 1:
        raise ValueError("Nonempty training labels and positive class count are required")
    counts = [0] * num_classes
    for label in labels:
        if type(label) is not int or not 0 <= label < num_classes:
            raise ValueError(f"Unknown training class ID: {label!r}")
        counts[label] += 1
    weights = [1.0 - count / len(labels) for count in counts]
    if any(weights[label] <= 0 for label in labels):
        raise ValueError("weight1 needs at least two observed training classes; a one-class smoke has zero CE denominator")
    return counts, weights


def validate_resume_contract(checkpoint: dict, protocol: dict, epochs: int, steps_per_epoch: int, train_count: int) -> int:
    if checkpoint.get("schema_version") != 1 or checkpoint.get("protocol") != protocol:
        raise ValueError("Resume protocol/data/code/environment differs; this is not an exact epoch-boundary resume")
    completed = checkpoint.get("completed_epoch")
    if type(completed) is not int or completed < 0 or checkpoint.get("next_epoch") != completed + 1:
        raise ValueError("Checkpoint is not a completed epoch boundary")
    if epochs <= completed:
        raise ValueError("--epochs must extend beyond the last completed epoch")
    if (checkpoint.get("global_optimizer_step") != completed * steps_per_epoch
            or checkpoint.get("samples_seen") != completed * train_count):
        raise ValueError("Checkpoint step/sample counters do not match complete epochs")
    if checkpoint.get("scheduler_state_dict") is not None or checkpoint.get("scaler_state_dict") is not None:
        raise ValueError("This protocol has no scheduler or AMP scaler")
    required = {"model_state_dict", "optimizer_state_dict", "rng_state", "sampler_state",
                "best", "metrics_history", "data_preprocessing", "model_parameters", "training_parameters", "num_classes"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Resume checkpoint is incomplete: {sorted(required - set(checkpoint))}")
    return completed


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    import torch
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


@contextmanager
def isolated_numpy_seed(seed: int):
    import numpy as np
    previous = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(previous)


def capture_rng(device) -> dict:
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state: dict, device) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["algorithm"], np.array(value["keys"], dtype=np.uint32),
                         value["position"], value["has_gauss"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if state["torch_cuda"] is None:
            raise ValueError("CUDA resume is missing its RNG state")
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    elif state["torch_cuda"] is not None:
        raise ValueError("CPU resume cannot consume CUDA RNG state")


def tensor_state_hash(state: dict) -> str:
    """Portable digest of model values, independent of torch.save ZIP metadata."""
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        result.update(descriptor.encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def check_data_role(train: dict, validation: dict, engineering: bool) -> None:
    if train["split"] != "train":
        raise ValueError("Training cache must use the official train split")
    if engineering:
        return
    if validation["split"] != "val" or not train["provenance"]["complete_official_split"] or not validation["provenance"]["complete_official_split"]:
        raise ValueError("Formal training requires complete official train and validation; subsets require --engineering-smoke")
    if ({item["signer_id"] for item in train["samples"]} & {item["signer_id"] for item in validation["samples"]}
            or {item["video_sha256"] for item in train["samples"]} & {item["video_sha256"] for item in validation["samples"]}):
        raise ValueError("Formal train and validation overlap in signer or video identity")


def audit_cohort(path: Path, bundle, engineering: bool, role: str) -> dict:
    import numpy as np
    from _poet_runtime import prepare_features
    from evaluate_pose_cache import audit_provenance, validate_cache_index, referenced_path, sha256
    from prepare_poses import verify_cache

    path = path.resolve()
    index = json.loads(path.read_text(encoding="utf-8"))
    split = validate_cache_index(index, bundle.label_to_id, allow_train_smoke=role == "train" or engineering)
    provenance = audit_provenance(path, index, bundle.checkpoint["data_preprocessing"],
                                 bundle.label_to_id, split, allow_subset_val=engineering)
    failed = [item for item in index["samples"] if item["status"] not in {"ok", "reused"}]
    if failed:
        raise ValueError(f"{role} has {len(failed)} failed cache entries; first indices={[item['index'] for item in failed[:10]]}. No implicit exclusions are allowed.")
    for number, item in enumerate(index["samples"], 1):
        metadata_path = referenced_path(path.parent, item["cache_manifest_path"])
        target = metadata_path.parent
        if metadata_path.name != "manifest.json":
            raise ValueError("Per-video cache must use manifest.json")
        for key, filename in (("pose_path", "raw_pose.npz"), ("features_path", "normalized_features.npz")):
            if referenced_path(path.parent, item[key]) != target / filename:
                raise ValueError(f"Cache path disagrees with its per-video manifest: {item[key]}")
        metadata = verify_cache(target, item, index["pipeline_sha256"])
        if (item["pose_sha256"] != metadata["artifacts"]["raw_pose.npz"]["sha256"]
                or item["features_sha256"] != metadata["artifacts"]["normalized_features.npz"]["sha256"]):
            raise ValueError("Cache index/artifact SHA256 mismatch")
        with np.load(target / "normalized_features.npz", allow_pickle=False) as archive:
            prepared, indices = prepare_features(archive["normalized_features"], bundle.preprocessing, mode="eval")
            if not np.array_equal(prepared.numpy(), archive["model_input"]) or not np.array_equal(indices, archive["source_frame_indices"]):
                raise ValueError("Cached evaluation input differs from official preprocessing")
        item["features_path"] = str(target / "normalized_features.npz")
        if number % 500 == 0:
            print(f"Audited {role} cache {number}/{len(index['samples'])}", flush=True)
    return {"path": str(path), "sha256": sha256(path), "split": split, "samples": index["samples"], "provenance": provenance}


def read_features(item: dict, eval_mode: bool = False):
    import numpy as np
    from evaluate_pose_cache import verify_file
    path = Path(item["features_path"])
    # Caches are immutable; reject concurrent mutation rather than changing the
    # training cohort after the epoch-boundary checkpoint was validated.
    verify_file(path, item["features_sha256"], "training/evaluation features")
    with np.load(path, allow_pickle=False) as archive:
        return archive["model_input" if eval_mode else "normalized_features"]


def evaluate_epoch(bundle, cohort: dict, output: Path, epoch: int, batch_size: int) -> dict:
    import numpy as np
    import torch
    from _poet_runtime import require_finite, synchronize
    from evaluate_pose_cache import MetricAccumulator, sha256
    target = output / "evaluation" / f"epoch-{epoch:04d}"
    target.mkdir(parents=True, exist_ok=False)
    bundle.model.eval()
    accumulator = MetricAccumulator(bundle.id_to_label)
    started = time.perf_counter()
    forward_seconds = 0.0
    with (target / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        with torch.inference_mode():
            for offset in range(0, len(cohort["samples"]), batch_size):
                items = cohort["samples"][offset:offset + batch_size]
                inputs = torch.from_numpy(np.stack([read_features(item, eval_mode=True) for item in items]))
                synchronize(bundle.device)
                stamp = time.perf_counter()
                logits = bundle.model(inputs, bundle.device)
                synchronize(bundle.device)
                forward_seconds += time.perf_counter() - stamp
                require_finite("validation logits", logits)
                values, classes = logits.softmax(dim=1).topk(5, dim=1)
                for item, ids, probabilities, raw_logits in zip(items, classes.cpu().tolist(), values.cpu().tolist(), logits.cpu().numpy()):
                    accumulator.add(item["gloss"], ids)
                    record = {key: item[key] for key in ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")}
                    record.update({"top5": [{"class_id": class_id, "gloss": bundle.id_to_label[class_id], "probability": probability}
                                            for class_id, probability in zip(ids, probabilities)],
                                   "logits_sha256": hashlib.sha256(raw_logits.astype("<f4").tobytes()).hexdigest()})
                    predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    metrics, classes = accumulator.summary()
    with (target / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)
    result = {"epoch": epoch, "split": cohort["split"], "metrics": metrics,
              "seconds": time.perf_counter() - started, "model_forward_seconds_including_transfer": forward_seconds,
              "batch_size": batch_size, "warmup_batches": 0,
              "timing_note": "Training-time validation, no warmup; not a latency benchmark.",
              "predictions_path": str(target / "predictions.jsonl"), "predictions_sha256": sha256(target / "predictions.jsonl")}
    atomic_json(target / "metrics.json", result)
    return result


def make_protocol(args, bundle, train: dict, validation: dict, counts: list[int], class_weights) -> dict:
    import torch
    from evaluate_pose_cache import sha256
    code_files = ["scripts/finetune_baseline.py", "scripts/evaluate_pose_cache.py", "scripts/_poet_runtime.py",
                  "scripts/prepare_poses.py", "scripts/infer_video.py"]
    code_files += sorted(str(path.relative_to(WORKSPACE)) for path in (WORKSPACE / "vendor/poet").glob("*.py"))
    device = bundle.device
    gpu = None if device.type != "cuda" else {"name": torch.cuda.get_device_name(device),
                                                "capability": list(torch.cuda.get_device_capability(device))}
    return {"version": PROTOCOL_VERSION, "method": "baseline", "scope": "smoke_only" if args.engineering_smoke else "formal_finetuning",
            "model": {key: bundle.metadata[key] for key in ("checkpoint_sha256", "label_map_sha256", "num_classes", "num_parameters",
                                                               "data_preprocessing", "model_parameters", "hidden_dropout_prob")},
            "data": {role: {"manifest_sha256": cohort["sha256"], "split": cohort["split"], "samples": len(cohort["samples"]),
                             "pipeline_sha256": cohort["provenance"]["pipeline_sha256"],
                             "source_manifest_sha256": cohort["provenance"]["source_manifest_sha256"]}
                     for role, cohort in (("train", train), ("validation", validation))},
            "code_sha256": {name: sha256(WORKSPACE / name) for name in code_files},
            "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay, "betas": [0.9, 0.999],
                          "eps": 1e-8, "foreach": False, "fused": False},
            "loss": {"name": "weighted_cross_entropy", "formula": "weight1[c] = 1 - train_count[c] / N", "train_class_counts": counts,
                     "weights_sha256": hashlib.sha256(class_weights.cpu().numpy().astype("<f4").tobytes()).hexdigest()},
            "training": {"batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size, "seed": args.seed,
                         "scheduler": None, "amp": False, "dtype": "float32", "trainable": "all original model parameters",
                         "drop_last": False, "gradient_accumulation": 1, "num_workers": 0, "pin_memory": False,
                         "spatial_augmentation": None, "temporal_augmentation": "unmodified prepare_features(mode=train)",
                         "temporal_seed": "SHA256([temporal,run_seed,epoch,video_sha256]) first 4 bytes big endian; NumPy RNG isolated",
                         "sampler": "independent torch CPU Generator randperm; state restored at epoch boundaries"},
            "selection": {"metric": "per_class_observed.top1" if args.engineering_smoke else "per_class_full_vocabulary.top1",
                          "ties": "keep earlier checkpoint", "initial_best": "epoch 0", "early_stopping": False},
            "numerics": {"device": str(device), "gpu": gpu, "packages": bundle.metadata["packages"], "torch_cuda": torch.version.cuda,
                         "num_threads": args.num_threads, "num_interop_threads": 1, "deterministic_algorithms": True,
                         "cudnn_deterministic": True, "cudnn_benchmark": False, "cuda_matmul_tf32": False, "cudnn_tf32": False,
                         "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}


def run(args: argparse.Namespace) -> dict:
    if args.epochs < 1 or min(args.batch_size, args.eval_batch_size, args.num_threads) < 1:
        raise ValueError("Epochs, batch sizes and num-threads must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Learning rate must be finite positive; weight decay finite nonnegative")
    if not 0 <= args.seed <= 2 ** 32 - 1:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    import numpy as np
    import torch
    from _poet_runtime import load_bundle, resolve_device, require_finite, prepare_features, synchronize, plain_metadata, OFFICIAL_ASL_CHECKPOINT_SHA256
    from evaluate_pose_cache import NUM_CLASSES, OFFICIAL_LABEL_MAP_SHA256, sha256, canonical_hash

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    started = time.perf_counter()
    summary = {"schema_version": 1, "state": "running", "started_at_utc": now(), "scope": "smoke_only" if args.engineering_smoke else "formal_finetuning",
               "command": sys.argv, "pid": os.getpid(), "python": sys.executable, "completed_epoch": None,
               "requested_epochs": args.epochs, "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    atomic_json(output / "summary.json", summary)
    try:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = resolve_device(args.device)
        bundle = load_bundle(args.checkpoint_dir, device)
        if bundle.checkpoint["num_classes"] != NUM_CLASSES or bundle.metadata["label_map_sha256"] != OFFICIAL_LABEL_MAP_SHA256:
            raise ValueError("Training requires the unmodified official 2,731-class label maps")
        if bundle.metadata["checkpoint_sha256"] != OFFICIAL_ASL_CHECKPOINT_SHA256:
            raise ValueError("Warm-start initialization must be the pinned official checkpoint")
        train = audit_cohort(args.train_cache_manifest, bundle, args.engineering_smoke, "train")
        validation = audit_cohort(args.val_cache_manifest, bundle, args.engineering_smoke, "validation")
        check_data_role(train, validation, args.engineering_smoke)
        counts, weights = weight1_from_labels([bundle.label_to_id[item["gloss"]] for item in train["samples"]], NUM_CLASSES)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        protocol = plain_metadata(make_protocol(args, bundle, train, validation, counts, class_weights))
        protocol_hash = canonical_hash(protocol)
        atomic_json(output / "protocol.json", {"sha256": protocol_hash, "protocol": protocol})
        atomic_json(output / "data_manifest.json", {"train": train["provenance"], "validation": validation["provenance"],
                                                   "excluded_samples": [], "validation_on_train": validation["split"] == "train"})
        atomic_json(output / "class_weights.json", {"counts": counts, "weights": class_weights.cpu().tolist(), "formula": "1 - count / N"})
        optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                     betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
        sampler = torch.Generator(device="cpu")
        sampler.manual_seed(stable_seed(args.seed, 0, "", purpose="sampler"))
        steps_per_epoch = math.ceil(len(train["samples"]) / args.batch_size)
        completed_epoch, global_step, samples_seen = 0, 0, 0
        history, best, best_artifact = [], None, None
        resume_evidence = None

        def selection_score(result: dict) -> float:
            field = "per_class_observed" if args.engineering_smoke else "per_class_full_vocabulary"
            value = result["metrics"][field]["top1"]
            if value is None or not math.isfinite(value) or result["metrics"]["failed"]:
                raise ValueError("Validation cannot select a model with missing metrics or failed samples")
            return value

        def checkpoint_payload() -> dict:
            for name, value in bundle.model.state_dict().items():
                require_finite(f"checkpoint model {name}", value)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        require_finite(f"optimizer {key}", value)
            return {"schema_version": 1, "protocol": protocol, "protocol_sha256": protocol_hash,
                    "num_classes": NUM_CLASSES, "data_preprocessing": plain_metadata(bundle.checkpoint["data_preprocessing"]),
                    "model_parameters": plain_metadata(bundle.checkpoint["model_parameters"]),
                    "training_parameters": {"hidden_dropout_prob": float(bundle.checkpoint["training_parameters"]["hidden_dropout_prob"]),
                                            "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                                            "epochs": args.epochs, "scheduler": "none", "method": "warm_start_finetuning"},
                    "model_state_dict": bundle.model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None, "scaler_state_dict": None,
                    "completed_epoch": completed_epoch, "next_epoch": completed_epoch + 1,
                    "global_optimizer_step": global_step, "samples_seen": samples_seen,
                    "best": best, "best_artifact": best_artifact, "metrics_history": history,
                    "rng_state": capture_rng(device), "sampler_state": sampler.get_state(),
                    "source_run": str(output), "resume_from": resume_evidence,
                    "model_state_sha256": tensor_state_hash(bundle.model.state_dict())}

        def save_boundary(improved: bool) -> dict:
            nonlocal best_artifact
            if improved:
                # A best file describes itself by best epoch/score. Its hash
                # is recorded in last.pth and the external checkpoint index.
                best_artifact = None
                # Keep immutable versions: if interrupted between best and
                # last updates, the previous last still points to a valid best.
                best_path = output / "checkpoints" / f"best-epoch-{completed_epoch:04d}.pth"
                if best_path.exists():
                    raise FileExistsError(f"Refusing to overwrite immutable best checkpoint: {best_path}")
                atomic_checkpoint(best_path, checkpoint_payload())
                best_artifact = {"path": str(best_path), "sha256": sha256(best_path), "epoch": best["epoch"]}
                atomic_copy(best_path, output / "checkpoints/best.pth")
            last_path = output / "checkpoints/last.pth"
            atomic_checkpoint(last_path, checkpoint_payload())
            artifacts = {"last": {"path": str(last_path), "sha256": sha256(last_path), "epoch": completed_epoch}, "best": best_artifact}
            atomic_json(output / "checkpoints/index.json", artifacts)
            return artifacts

        if args.resume is not None:
            resume_path = args.resume.resolve()
            restored = torch.load(resume_path, map_location="cpu", weights_only=True)
            completed_epoch = validate_resume_contract(restored, protocol, args.epochs, steps_per_epoch, len(train["samples"]))
            if restored.get("protocol_sha256") != protocol_hash:
                raise ValueError("Resume protocol fingerprint is inconsistent")
            bundle.model.load_state_dict(restored["model_state_dict"], strict=True)
            if tensor_state_hash(bundle.model.state_dict()) != restored["model_state_sha256"]:
                raise ValueError("Resumed model tensor fingerprint differs")
            optimizer.load_state_dict(restored["optimizer_state_dict"])
            global_step, samples_seen = restored["global_optimizer_step"], restored["samples_seen"]
            history, best = restored["metrics_history"], restored["best"]
            if restored["best_artifact"] is None and best["epoch"] == completed_epoch:
                old_best = resume_path
                expected_best_hash = sha256(resume_path)
            else:
                old_best = Path(restored["best_artifact"]["path"])
                expected_best_hash = restored["best_artifact"]["sha256"]
            if sha256(old_best) != expected_best_hash:
                raise ValueError("Prior best checkpoint is missing or changed")
            copied_best = output / "checkpoints" / f"best-epoch-{best['epoch']:04d}.pth"
            atomic_copy(old_best, copied_best)
            atomic_copy(copied_best, output / "checkpoints/best.pth")
            best_artifact = {"path": str(copied_best), "sha256": sha256(copied_best), "epoch": best["epoch"]}
            resume_evidence = {"checkpoint": str(resume_path), "sha256": sha256(resume_path),
                               "parent_output": restored["source_run"], "completed_epoch": completed_epoch,
                               "policy": "exact epoch boundary; unfinished next-epoch work is repeated, not resumed mid-step"}
            sampler.set_state(restored["sampler_state"])
            restore_rng(restored["rng_state"], device)
            # Preserve a local last checkpoint even if the next epoch fails.
            checkpoints = save_boundary(False)
            print(f"Resumed completed epoch {completed_epoch}, optimizer step {global_step}", flush=True)
        else:
            initial = evaluate_epoch(bundle, validation, output, 0, args.eval_batch_size)
            history.append(initial)
            best = {"epoch": 0, "score": selection_score(initial)}
            checkpoints = save_boundary(True)
            print(f"Epoch 0: macro Top-1={best['score']:.8f}", flush=True)
        atomic_json(output / "metrics_history.json", {"scope": summary["scope"], "epochs": history})
        summary.update(protocol_sha256=protocol_hash, completed_epoch=completed_epoch, resume=resume_evidence,
                       trainable_parameters=sum(p.numel() for p in bundle.model.parameters() if p.requires_grad),
                       steps_per_epoch=steps_per_epoch, checkpoints=checkpoints)
        atomic_json(output / "summary.json", summary)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with (output / "steps.jsonl").open("w", encoding="utf-8") as trace:
            for epoch in range(completed_epoch + 1, args.epochs + 1):
                epoch_start = time.perf_counter()
                bundle.model.train()
                order = torch.randperm(len(train["samples"]), generator=sampler).tolist()
                weighted_loss_sum, weight_sum = 0.0, 0.0
                for offset in range(0, len(order), args.batch_size):
                    items = [train["samples"][i] for i in order[offset:offset + args.batch_size]]
                    prepared, indices_hashes, temporal_seeds = [], [], []
                    for item in items:
                        features = read_features(item)
                        seed = stable_seed(args.seed, epoch, item["video_sha256"])
                        with isolated_numpy_seed(seed):
                            value, indices = prepare_features(features, bundle.preprocessing, mode="train")
                        prepared.append(value)
                        temporal_seeds.append(seed)
                        indices_hashes.append(hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest())
                    batch = torch.stack(prepared)
                    labels = torch.tensor([bundle.label_to_id[item["gloss"]] for item in items], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = bundle.model(batch, device)
                    require_finite("training logits", logits)
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
                    require_finite("training loss", loss)
                    loss.backward()
                    gradient_squared = 0.0
                    for name, parameter in bundle.model.named_parameters():
                        if parameter.grad is None:
                            raise ValueError(f"Trainable parameter received no gradient: {name}")
                        require_finite(f"gradient {name}", parameter.grad)
                        gradient_squared += float(parameter.grad.detach().double().square().sum().item())
                    optimizer.step()
                    global_step += 1
                    samples_seen += len(items)
                    batch_weight = float(class_weights[labels].sum().item())
                    weighted_loss_sum += float(loss.item()) * batch_weight
                    weight_sum += batch_weight
                    trace.write(json.dumps({"epoch": epoch, "global_optimizer_step": global_step, "samples_seen": samples_seen,
                                            "indices": [item["index"] for item in items], "video_sha256": [item["video_sha256"] for item in items],
                                            "temporal_seeds": temporal_seeds, "temporal_indices_sha256": indices_hashes,
                                            "loss": float(loss.item()), "gradient_l2": math.sqrt(gradient_squared), "lr": optimizer.param_groups[0]["lr"]},
                                           allow_nan=False) + "\n")
                    trace.flush()
                    if global_step % 50 == 0:
                        print(f"Epoch {epoch}: step {offset // args.batch_size + 1}/{steps_per_epoch}, loss={loss.item():.6f}", flush=True)
                synchronize(device)
                training_seconds = time.perf_counter() - epoch_start
                result = evaluate_epoch(bundle, validation, output, epoch, args.eval_batch_size)
                result["training"] = {"weighted_ce": weighted_loss_sum / weight_sum, "seconds": training_seconds,
                                      "samples": len(order), "optimizer_steps": steps_per_epoch,
                                      "order_sha256": hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()}
                history.append(result)
                completed_epoch = epoch
                score = selection_score(result)
                improved = score > best["score"]
                if improved:
                    best = {"epoch": epoch, "score": score}
                checkpoints = save_boundary(improved)
                atomic_json(output / "metrics_history.json", {"scope": summary["scope"], "epochs": history})
                summary.update(completed_epoch=completed_epoch, global_optimizer_step=global_step, samples_seen=samples_seen,
                               best=best, checkpoints=checkpoints)
                atomic_json(output / "summary.json", summary)
                print(f"Epoch {epoch}: train loss={result['training']['weighted_ce']:.8f}, macro Top-1={score:.8f}, best epoch={best['epoch']}", flush=True)
        summary.update(state="completed", completed_at_utc=now(), elapsed_seconds=time.perf_counter() - started,
                       model_state_sha256=tensor_state_hash(bundle.model.state_dict()),
                       gpu=None if device.type != "cuda" else {"name": torch.cuda.get_device_name(device),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)},
                       interpretation="Warm-start fine-tuning, not continuation of the author's optimizer. Smoke-only results are not benchmark or improvement evidence.")
        atomic_json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        summary.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__,
                       error=str(error), elapsed_seconds=time.perf_counter() - started,
                       recovery="Use the last fully committed epoch checkpoint in a new output directory; partial epoch steps are not resumable.")
        atomic_json(output / "summary.json", summary)
        raise


def main() -> None:
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache-manifest", type=Path, required=True)
    parser.add_argument("--val-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Total completed epoch target; may increase on resume")
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = run(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps({"state": result["state"], "scope": result["scope"], "completed_epoch": result["completed_epoch"],
                      "best": result["best"], "output_dir": str(args.output_dir.resolve())}), flush=True)


if __name__ == "__main__":
    main()
