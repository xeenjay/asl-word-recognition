#!/usr/bin/env python3
"""Unified single-device PoET methods; strict, complete-epoch resume.

Formal mode requires full official train/val. Engineering runs are explicitly
smoke_only. Original pose extraction and upstream model remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from finetune_baseline import (now, stable_seed, weight1_from_labels, validate_resume_contract,
    atomic_json, atomic_checkpoint, atomic_copy, isolated_numpy_seed, capture_rng,
    restore_rng, tensor_state_hash, check_data_role, audit_cohort, read_features,
    make_protocol as baseline_protocol)

from method_optimizer import (PROTOCOL_V1, PROTOCOL_V2, create_optimizer, optimizer_protocol,
    validate_module_lr, gradient_group_metrics, validate_checkpoint_optimizer_contract)

WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = PROTOCOL_V1


def validate_method_args(args):
    args.module_lr = getattr(args, "module_lr", None)
    validate_module_lr(args.method, args.module_lr)
    if args.attribute_lambda is None:
        args.attribute_lambda = 0.1 if args.method == "attributes" else 0.0
    if not math.isfinite(args.attribute_lambda) or args.attribute_lambda < 0:
        raise ValueError("attribute-lambda must be finite and nonnegative")
    if args.method == "geometry":
        if args.train_geometry_manifest is None or args.val_geometry_manifest is None:
            raise ValueError("geometry needs both train and validation geometry manifests")
    elif args.train_geometry_manifest is not None or args.val_geometry_manifest is not None:
        raise ValueError("Geometry manifests are only valid for geometry")
    if args.method != "attributes" and (args.attribute_targets is not None or args.attribute_lambda != 0):
        raise ValueError("Attribute targets/loss are only valid for attributes")
    if args.method == "attributes" and args.attribute_targets is None:
        raise ValueError("attributes requires the pinned training target mapping")
    if not 0 <= args.head_seed <= 2 ** 32 - 1:
        raise ValueError("head-seed must be an unsigned 32-bit integer")
    if args.geometry_hidden < 1:
        raise ValueError("geometry-hidden must be positive")


def validate_optimizer_resume(optimizer, restored, global_step, inactive_parameters=()):
    """Reject optimizer mutations rather than silently overriding the protocol."""
    import torch
    expected = optimizer.state_dict()
    if not isinstance(restored, dict) or set(restored) != {"state", "param_groups"}:
        raise ValueError("Resume optimizer must contain state and param_groups")
    if restored["param_groups"] != expected["param_groups"]:
        raise ValueError("Resume optimizer parameter groups/settings differ from protocol")
    inactive = {id(parameter) for parameter in inactive_parameters}
    parameters = {identifier: parameter
                  for serialized, live in zip(expected["param_groups"], optimizer.param_groups)
                  for identifier, parameter in zip(serialized["params"], live["params"])}
    active = {identifier for identifier, parameter in parameters.items()
              if id(parameter) not in inactive} if global_step else set()
    states = restored["state"]
    if not isinstance(states, dict) or set(states) != active:
        raise ValueError("Resume optimizer state inventory differs from updated parameters")
    for identifier, state in states.items():
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Resume AdamW state schema differs")
        step = state["step"]
        if (not torch.is_tensor(step) or step.shape != torch.Size([]) or step.dtype != torch.float32
                or not bool(torch.isfinite(step)) or step.item() != global_step):
            raise ValueError("Resume AdamW step differs from completed epoch counters")
        parameter = parameters[identifier]
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if (not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Resume AdamW moment shape/dtype/finite contract differs")
        if bool((state["exp_avg_sq"] < 0).any()):
            raise ValueError("Resume AdamW second moments must be nonnegative")


def prepare_method_batch(bundle, items, *, mode, run_seed=0, epoch=0, geometry_index=None):
    """One temporal sample supplies both branches; targets never enter here."""
    import numpy as np
    import torch
    from _poet_runtime import prepare_features
    from evaluate_pose_cache import verify_file
    from prepare_geometry_cache import read_geometry
    if mode not in {"train", "eval"}:
        raise ValueError("Batch mode must be train or eval")
    if (bundle.model.method == "geometry") != (geometry_index is not None):
        raise ValueError("Method and geometry batch inputs disagree")
    prepared, geometry, traces = [], [], []
    for item in items:
        seed = stable_seed(run_seed, epoch, item["video_sha256"])
        if mode == "train":
            with isolated_numpy_seed(seed):
                value, indices = prepare_features(read_features(item), bundle.preprocessing, mode="train")
        else:
            path = Path(item["features_path"])
            verify_file(path, item["features_sha256"], "evaluation feature cache")
            with np.load(path, allow_pickle=False) as archive:
                value = torch.from_numpy(archive["model_input"])
                indices = archive["source_frame_indices"]
        prepared.append(value)
        record = {"temporal_seed": seed if mode == "train" else None,
                  "source_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                  "input_sha256": hashlib.sha256(value.numpy().astype("<f4").tobytes()).hexdigest()}
        if geometry_index is not None:
            geo = read_geometry(item, indices, geometry_index)
            geometry.append(torch.from_numpy(geo))
            record.update(geometry_sha256=hashlib.sha256(geo.astype("<f4").tobytes()).hexdigest(),
                          geometry_valid_relations=int(geo[:, 3::4].sum()),
                          geometry_total_relations=int(len(geo) * 7))
        traces.append(record)
    return torch.stack(prepared), torch.stack(geometry) if geometry else None, traces


def evaluate_epoch(bundle, cohort: dict, output: Path, epoch: int, batch_size: int, geometry_index=None) -> dict:
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
                inputs, geometry, _ = prepare_method_batch(bundle, items, mode="eval", geometry_index=geometry_index)
                synchronize(bundle.device)
                stamp = time.perf_counter()
                logits = bundle.model(inputs, bundle.device, geometry=geometry)
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


def make_protocol(args, bundle, train, validation, counts, class_weights,
                  method_metadata, train_geometry, val_geometry):
    from evaluate_pose_cache import sha256
    value = baseline_protocol(args, bundle, train, validation, counts, class_weights)
    module_lr = getattr(args, "module_lr", None)
    value.update(version=PROTOCOL_V1 if module_lr is None else PROTOCOL_V2, method=args.method, method_metadata=method_metadata)
    value["optimizer"] = optimizer_protocol(bundle.model, lr=args.lr, weight_decay=args.weight_decay, module_lr=module_lr)
    value["model"]["method_num_parameters"] = sum(p.numel() for p in bundle.model.parameters())
    value["training"]["trainable"] = "all original model and selected method parameters"
    value["loss"].update(attribute_lambda=args.attribute_lambda,
                         auxiliary="mean of masked per-head BCE means; empty heads contribute zero" if args.method == "attributes" else None)
    for name in ("finetune_methods.py", "evaluate_methods.py", "method_runtime.py",
                 "poet_methods.py", "poet_geometry.py", "prepare_geometry_cache.py", "method_optimizer.py"):
        value["code_sha256"]["scripts/" + name] = sha256(WORKSPACE / "scripts" / name)
    value["geometry_data"] = None if train_geometry is None else {
        role: {"manifest_sha256": item["sha256"],
               "geometry_spec_sha256": item["geometry_spec_sha256"],
               "source_sha256": item["source_sha256"]}
        for role, item in (("train", train_geometry), ("validation", val_geometry))}
    return value


def run(args: argparse.Namespace) -> dict:
    validate_method_args(args)
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

    from method_runtime import load_attribute_targets, create_method, compute_training_loss
    from prepare_geometry_cache import audit_geometry_index

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
        if train["provenance"]["pipeline_sha256"] != validation["provenance"]["pipeline_sha256"]:
            raise ValueError("Train and validation must share the same pose pipeline")
        train_geometry = audit_geometry_index(args.train_geometry_manifest, train) if args.method == "geometry" else None
        val_geometry = audit_geometry_index(args.val_geometry_manifest, validation) if args.method == "geometry" else None
        attribute_targets = load_attribute_targets(args.attribute_targets, bundle) if args.method == "attributes" else None
        bundle.model, method_metadata = create_method(bundle, args.method, attribute_targets=attribute_targets,
            geometry_hidden=args.geometry_hidden, head_seed=args.head_seed, attribute_lambda=args.attribute_lambda)
        counts, weights = weight1_from_labels([bundle.label_to_id[item["gloss"]] for item in train["samples"]], NUM_CLASSES)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        protocol = plain_metadata(make_protocol(args, bundle, train, validation, counts, class_weights, method_metadata, train_geometry, val_geometry))
        protocol_hash = canonical_hash(protocol)
        atomic_json(output / "protocol.json", {"sha256": protocol_hash, "protocol": protocol})
        atomic_json(output / "data_manifest.json", {"train": train["provenance"], "validation": validation["provenance"],
                                                   "excluded_samples": [], "validation_on_train": validation["split"] == "train",
                                                   "geometry_data": protocol["geometry_data"], "method_metadata": method_metadata})
        atomic_json(output / "class_weights.json", {"counts": counts, "weights": class_weights.cpu().tolist(), "formula": "1 - count / N"})
        optimizer = create_optimizer(bundle.model, lr=args.lr, weight_decay=args.weight_decay, module_lr=args.module_lr)
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
                    "method_metadata": method_metadata, "num_classes": NUM_CLASSES, "data_preprocessing": plain_metadata(bundle.checkpoint["data_preprocessing"]),
                    "model_parameters": plain_metadata(bundle.checkpoint["model_parameters"]),
                    "training_parameters": {"hidden_dropout_prob": float(bundle.checkpoint["training_parameters"]["hidden_dropout_prob"]),
                                            "lr": args.lr, "module_lr": args.module_lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
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
            if restored.get("method_metadata") != method_metadata:
                raise ValueError("Resume method metadata differs")
            for key in ("data_preprocessing", "model_parameters", "num_classes"):
                if restored[key] != plain_metadata(bundle.checkpoint[key]):
                    raise ValueError(f"Resume {key} differs from the official base")
            if restored["training_parameters"]["hidden_dropout_prob"] != float(bundle.checkpoint["training_parameters"]["hidden_dropout_prob"]):
                raise ValueError("Resume dropout differs from the official base")
            expected_state = bundle.model.state_dict()
            for name, value in restored["model_state_dict"].items():
                if not torch.is_tensor(value) or (name in expected_state and value.dtype != expected_state[name].dtype):
                    raise ValueError(f"Resume tensor dtype differs: {name}")
                require_finite(f"resume tensor {name}", value)
            bundle.model.load_state_dict(restored["model_state_dict"], strict=True)
            if tensor_state_hash(bundle.model.state_dict()) != restored["model_state_sha256"]:
                raise ValueError("Resumed model tensor fingerprint differs")
            validate_checkpoint_optimizer_contract(restored, bundle.model)
            inactive_parameters = bundle.model.attribute_heads.parameters() if args.method == "attributes" and args.attribute_lambda == 0 else ()
            validate_optimizer_resume(optimizer, restored["optimizer_state_dict"], restored["global_optimizer_step"], inactive_parameters)
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
            initial = evaluate_epoch(bundle, validation, output, 0, args.eval_batch_size, val_geometry)
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
                auxiliary_sum, total_loss_sum, epoch_batches = 0.0, 0.0, 0
                for offset in range(0, len(order), args.batch_size):
                    items = [train["samples"][i] for i in order[offset:offset + args.batch_size]]
                    batch, geometry, batch_trace = prepare_method_batch(bundle, items, mode="train",
                        run_seed=args.seed, epoch=epoch, geometry_index=train_geometry)
                    temporal_seeds = [item["temporal_seed"] for item in batch_trace]
                    indices_hashes = [item["source_indices_sha256"] for item in batch_trace]
                    labels = torch.tensor([bundle.label_to_id[item["gloss"]] for item in items], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = compute_training_loss(bundle.model, batch, labels, class_weights,
                        geometry=geometry, attribute_targets=attribute_targets, attribute_lambda=args.attribute_lambda)
                    loss, word_ce, auxiliary = losses["loss"], losses["word_ce"], losses["auxiliary"]
                    require_finite("training logits", losses["logits"])
                    require_finite("training loss", loss)
                    loss.backward()
                    gradient_squared = 0.0
                    gradient_squared_by_parameter = {}
                    for name, parameter in bundle.model.named_parameters():
                        if parameter.grad is None:
                            if args.method == "attributes" and args.attribute_lambda == 0 and name.startswith("attribute_heads."):
                                continue  # Deliberately excluded from the lambda=0 control loss.
                            raise ValueError(f"Trainable parameter received no gradient: {name}")
                        require_finite(f"gradient {name}", parameter.grad)
                        squared = float(parameter.grad.detach().double().square().sum().item())
                        gradient_squared += squared
                        gradient_squared_by_parameter[id(parameter)] = squared
                    optimizer_groups = gradient_group_metrics(optimizer, gradient_squared_by_parameter)
                    optimizer.step()
                    global_step += 1
                    samples_seen += len(items)
                    batch_weight = float(class_weights[labels].sum().item())
                    weighted_loss_sum += float(word_ce.item()) * batch_weight
                    auxiliary_sum += float(auxiliary.item())
                    total_loss_sum += float(loss.item())
                    epoch_batches += 1
                    weight_sum += batch_weight
                    trace.write(json.dumps({"epoch": epoch, "global_optimizer_step": global_step, "samples_seen": samples_seen,
                                            "indices": [item["index"] for item in items], "video_sha256": [item["video_sha256"] for item in items],
                                            "temporal_seeds": temporal_seeds, "temporal_indices_sha256": indices_hashes,
                                            "loss": float(loss.item()), "word_ce": float(word_ce.item()), "auxiliary": float(auxiliary.item()),
                                            "attribute_valid_elements": losses["valid_elements"], "attribute_valid_examples": losses["valid_examples"],
                                            "batch_trace": batch_trace, "gradient_l2": math.sqrt(gradient_squared), "lr": optimizer.param_groups[0]["lr"],
                                            "optimizer_groups": optimizer_groups},
                                           allow_nan=False) + "\n")
                    trace.flush()
                    if global_step % 50 == 0:
                        print(f"Epoch {epoch}: step {offset // args.batch_size + 1}/{steps_per_epoch}, loss={loss.item():.6f}", flush=True)
                synchronize(device)
                training_seconds = time.perf_counter() - epoch_start
                result = evaluate_epoch(bundle, validation, output, epoch, args.eval_batch_size, val_geometry)
                result["training"] = {"weighted_ce": weighted_loss_sum / weight_sum,
                                      "auxiliary_batch_mean": auxiliary_sum / epoch_batches, "total_loss_batch_mean": total_loss_sum / epoch_batches,
                                      "seconds": training_seconds,
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
    parser.add_argument("--method", choices=("baseline", "geometry", "attributes"), required=True)
    parser.add_argument("--train-geometry-manifest", type=Path)
    parser.add_argument("--val-geometry-manifest", type=Path)
    parser.add_argument("--attribute-targets", type=Path)
    parser.add_argument("--attribute-lambda", type=float, default=None)
    parser.add_argument("--geometry-hidden", type=int, default=32)
    parser.add_argument("--head-seed", type=int, default=20260922)
    parser.add_argument("--train-cache-manifest", type=Path, required=True)
    parser.add_argument("--val-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Total completed epoch target; may increase on resume")
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--module-lr", type=float, default=None,
                        help="Explicit geometry/attribute-head LR; enables strict two-group v2 protocol")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
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
