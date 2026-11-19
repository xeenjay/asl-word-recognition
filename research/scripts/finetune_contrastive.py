#!/usr/bin/env python3
"""Balanced CE versus cross-signer SupCon; audited full train/val, epoch resume."""
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

from finetune_methods import prepare_method_batch, evaluate_epoch, validate_optimizer_resume
from method_optimizer import gradient_group_metrics
from contrastive_components import (PROTOCOL_VERSION, CrossSignerBatchSampler,
    validate_sampler_state, create_optimizer, optimizer_protocol, compute_contrastive_loss)

WORKSPACE = Path(__file__).resolve().parents[1]


def validate_contrastive_args(args):
    if args.method not in {"balanced_ce", "cross_signer_supcon"}:
        raise ValueError("Unknown contrastive comparison arm")
    if (not math.isfinite(args.supcon_lambda) or args.supcon_lambda < 0
            or (args.method == "balanced_ce") != (args.supcon_lambda == 0)):
        raise ValueError("balanced_ce requires lambda=0; cross_signer_supcon requires lambda>0")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite positive")
    if not math.isfinite(args.projection_lr) or args.projection_lr <= 0:
        raise ValueError("projection-lr must be finite positive")
    for field in ("classes_per_batch", "signers_per_class", "steps_per_epoch", "head_seed"):
        value = getattr(args, field)
        if type(value) is not int or value < (0 if field == "head_seed" else 1):
            raise ValueError(f"Invalid {field}")
    if args.head_seed > 2 ** 32 - 1 or min(args.classes_per_batch, args.signers_per_class) < 2:
        raise ValueError("Need uint32 head seed, at least 2 classes and 2 signers")
    if args.batch_size != args.classes_per_batch * args.signers_per_class:
        raise ValueError("batch-size must equal classes-per-batch times signers-per-class")
    # Formal protocol is the single preregistered bounded pilot; engineering
    # smoke can use smaller cohorts/budgets but never emits benchmark claims.
    if not args.engineering_smoke:
        expected = {"classes_per_batch": 32, "signers_per_class": 4, "steps_per_epoch": 314,
                    "batch_size": 128, "eval_batch_size": 128, "seed": 20260926,
                    "head_seed": 20260926, "lr": 3e-5, "projection_lr": 3e-4,
                    "weight_decay": .01, "temperature": .1, "epochs": 10}
        if any(getattr(args, key) != value for key, value in expected.items()):
            raise ValueError("Formal run differs from the bounded preregistered contrastive pilot")
        if args.supcon_lambda != (0.0 if args.method == "balanced_ce" else .05):
            raise ValueError("Formal loss weight must be 0 or 0.05 for the corresponding arm")


def validate_rng_state(state, device):
    import numpy as np
    import torch
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("RNG state schema differs")
    try:
        random.Random().setstate(state["python"])
        value = state["numpy"]
        if (set(value) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
                or value["algorithm"] != "MT19937" or len(value["keys"]) != 624
                or any(type(key) is not int or not 0 <= key < 2 ** 32 for key in value["keys"])
                or type(value["position"]) is not int or not 0 <= value["position"] <= 624
                or value["has_gauss"] not in (0, 1) or not math.isfinite(value["cached_gaussian"])):
            raise ValueError("Invalid NumPy RNG schema")
        np.random.RandomState().set_state((value["algorithm"], np.array(value["keys"], dtype=np.uint32),
            value["position"], value["has_gauss"], value["cached_gaussian"]))
        cpu = state["torch_cpu"]
        if not torch.is_tensor(cpu) or cpu.dtype != torch.uint8 or cpu.device.type != "cpu" or cpu.ndim != 1:
            raise ValueError("Invalid torch CPU RNG state")
        torch.Generator(device="cpu").set_state(cpu)
        cuda = state["torch_cuda"]
        if device.type == "cuda":
            reference = torch.cuda.get_rng_state(device)
            if (not torch.is_tensor(cuda) or cuda.dtype != reference.dtype
                    or cuda.shape != reference.shape or cuda.device.type != "cpu"):
                raise ValueError("Invalid CUDA RNG state")
            torch.Generator(device=device).set_state(cuda)
        elif cuda is not None:
            raise ValueError("CPU resume cannot consume CUDA RNG state")
    except Exception as error:
        raise ValueError(f"Invalid RNG state: {error}") from error


def validate_contrastive_resume(checkpoint, protocol, epochs, sampler, device):
    from evaluate_pose_cache import canonical_hash
    completed = validate_resume_contract(checkpoint, protocol, epochs, sampler.steps_per_epoch,
                                         sampler.steps_per_epoch * sampler.batch_size)
    if checkpoint.get("protocol_sha256") != canonical_hash(protocol):
        raise ValueError("Resume protocol fingerprint differs")
    validate_sampler_state(checkpoint["sampler_state"], sampler, completed)
    validate_rng_state(checkpoint["rng_state"], device)
    history = checkpoint.get("metrics_history")
    if not isinstance(history, list) or [row.get("epoch") for row in history] != list(range(completed + 1)):
        raise ValueError("Resume history must include each completed epoch and epoch zero")
    key = protocol["selection"]["metric"].split(".")[0]
    best = None
    for row in history:
        metrics = row["metrics"]
        score = metrics[key]["top1"]
        if (metrics["attempted"] != protocol["data"]["validation"]["samples"]
                or metrics["failed"] != 0 or type(score) not in (float, int) or not math.isfinite(score)):
            raise ValueError("Resume validation history denominator/score differs")
        if row["epoch"]:
            training = row.get("training", {})
            expected = sampler.epoch_summary(row["epoch"])
            if (training.get("samples") != sampler.steps_per_epoch * sampler.batch_size
                    or training.get("optimizer_steps") != sampler.steps_per_epoch
                    or training.get("sampling") != expected):
                raise ValueError("Resume epoch sampling/coverage evidence differs")
        if best is None or score > best["score"]:
            best = {"epoch": row["epoch"], "score": score}
    if checkpoint.get("best") != best:
        raise ValueError("Resume best checkpoint violates macro selection or earliest tie rule")
    return completed


def make_protocol(args, bundle, train, validation, counts, class_weights, method_metadata, sampler):
    from evaluate_pose_cache import sha256
    value = baseline_protocol(args, bundle, train, validation, counts, class_weights)
    value.update(version=PROTOCOL_VERSION, method=args.method, method_metadata=method_metadata)
    value["optimizer"] = optimizer_protocol(bundle.model, lr=args.lr, projection_lr=args.projection_lr,
                                            weight_decay=args.weight_decay, supcon_lambda=args.supcon_lambda)
    value["model"]["method_num_parameters"] = sum(p.numel() for p in bundle.model.parameters())
    value["training"].update(sampler=sampler.protocol(), steps_per_epoch=args.steps_per_epoch,
                              draws_per_epoch=sampler.batch_size * sampler.steps_per_epoch,
                              trainable="all base parameters; projection participates only for positive lambda",
                              drop_last=False)
    value["loss"].update(supcon_lambda=args.supcon_lambda, temperature=args.temperature,
        auxiliary="L2-normalized projected CLS; mean over anchors of mean positive log probability",
        positives="same class and different signer; no self pairs",
        denominator="all different classes plus same-class different-signers; exclude self and same-class same-signer")
    for name in ("contrastive_components.py", "finetune_contrastive.py", "evaluate_contrastive.py",
                 "finetune_methods.py", "evaluate_methods.py", "method_runtime.py", "poet_methods.py",
                 "poet_geometry.py", "prepare_geometry_cache.py", "method_optimizer.py"):
        value["code_sha256"]["scripts/" + name] = sha256(WORKSPACE / "scripts" / name)
    return value


def run(args: argparse.Namespace) -> dict:
    validate_contrastive_args(args)
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

    from contrastive_components import ContrastivePoET

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
        bundle.model = ContrastivePoET(bundle.model, head_seed=args.head_seed)
        method_metadata = bundle.model.protocol()
        sampler = CrossSignerBatchSampler(train["samples"], bundle.label_to_id, seed=args.seed,
            classes_per_batch=args.classes_per_batch, signers_per_class=args.signers_per_class,
            steps_per_epoch=args.steps_per_epoch)
        counts, weights = weight1_from_labels([bundle.label_to_id[item["gloss"]] for item in train["samples"]], NUM_CLASSES)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        protocol = plain_metadata(make_protocol(args, bundle, train, validation, counts, class_weights, method_metadata, sampler))
        protocol_hash = canonical_hash(protocol)
        atomic_json(output / "protocol.json", {"sha256": protocol_hash, "protocol": protocol})
        atomic_json(output / "data_manifest.json", {"train": train["provenance"], "validation": validation["provenance"],
                                                   "excluded_samples": [], "validation_on_train": validation["split"] == "train",
                                                   "sampling": sampler.protocol(), "method_metadata": method_metadata})
        atomic_json(output / "class_weights.json", {"counts": counts, "weights": class_weights.cpu().tolist(), "formula": "1 - count / N"})
        optimizer = create_optimizer(bundle.model, lr=args.lr, weight_decay=args.weight_decay,
                                     projection_lr=args.projection_lr, supcon_lambda=args.supcon_lambda)
        steps_per_epoch = args.steps_per_epoch
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
                                            "lr": args.lr, "projection_lr": args.projection_lr, "supcon_lambda": args.supcon_lambda, "temperature": args.temperature,
                                            "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                                            "epochs": args.epochs, "scheduler": "none", "method": "warm_start_finetuning"},
                    "model_state_dict": bundle.model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None, "scaler_state_dict": None,
                    "completed_epoch": completed_epoch, "next_epoch": completed_epoch + 1,
                    "global_optimizer_step": global_step, "samples_seen": samples_seen,
                    "best": best, "best_artifact": best_artifact, "metrics_history": history,
                    "rng_state": capture_rng(device), "sampler_state": sampler.state_dict(completed_epoch),
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
            completed_epoch = validate_contrastive_resume(restored, protocol, args.epochs, sampler, device)
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
            if restored["training_parameters"] != checkpoint_payload()["training_parameters"]:
                # Epoch target may extend for an engineering exact-resume check.
                expected_training = checkpoint_payload()["training_parameters"]
                old_training = dict(restored["training_parameters"])
                old_training["epochs"] = expected_training["epochs"]
                if old_training != expected_training:
                    raise ValueError("Resume training settings differ from protocol")
            validate_optimizer_resume(optimizer, restored["optimizer_state_dict"], restored["global_optimizer_step"])
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
                       trainable_parameters=sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
                       method=args.method, sampling=sampler.protocol(),
                       steps_per_epoch=steps_per_epoch, checkpoints=checkpoints)
        atomic_json(output / "summary.json", summary)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with (output / "steps.jsonl").open("w", encoding="utf-8") as trace:
            for epoch in range(completed_epoch + 1, args.epochs + 1):
                epoch_start = time.perf_counter()
                bundle.model.train()
                batches = sampler.batches(epoch)
                sampling = sampler.epoch_summary(epoch)
                order = [position for batch_positions in batches for position in batch_positions]
                weighted_loss_sum, weight_sum = 0.0, 0.0
                auxiliary_sum, total_loss_sum, epoch_batches = 0.0, 0.0, 0
                for batch_number, positions in enumerate(batches, 1):
                    items = [train["samples"][i] for i in positions]
                    batch, geometry, batch_trace = prepare_method_batch(bundle, items, mode="train",
                        run_seed=args.seed, epoch=epoch)
                    temporal_seeds = [item["temporal_seed"] for item in batch_trace]
                    indices_hashes = [item["source_indices_sha256"] for item in batch_trace]
                    labels = torch.tensor([bundle.label_to_id[item["gloss"]] for item in items], dtype=torch.long, device=device)
                    signers = torch.tensor([sampler.signer_to_id[item["signer_id"]] for item in items], dtype=torch.long, device=device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = compute_contrastive_loss(bundle.model, batch, labels, signers, class_weights,
                        supcon_lambda=args.supcon_lambda, temperature=args.temperature)
                    loss, word_ce, auxiliary = losses["loss"], losses["word_ce"], losses["auxiliary"]
                    require_finite("training logits", losses["logits"])
                    require_finite("training loss", loss)
                    loss.backward()
                    gradient_squared = 0.0
                    gradient_squared_by_parameter = {}
                    for name, parameter in bundle.model.named_parameters():
                        if parameter.grad is None:
                            if args.supcon_lambda == 0 and name.startswith("projection."):
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
                                            "signer_ids": [item["signer_id"] for item in items], "class_ids": labels.cpu().tolist(),
                                            "positive_pairs_per_anchor": args.signers_per_class - 1,
                                            "batch_trace": batch_trace, "gradient_l2": math.sqrt(gradient_squared), "lr": optimizer.param_groups[0]["lr"],
                                            "optimizer_groups": optimizer_groups},
                                           allow_nan=False) + "\n")
                    trace.flush()
                    if global_step % 50 == 0:
                        print(f"Epoch {epoch}: step {batch_number}/{steps_per_epoch}, loss={loss.item():.6f}", flush=True)
                synchronize(device)
                training_seconds = time.perf_counter() - epoch_start
                result = evaluate_epoch(bundle, validation, output, epoch, args.eval_batch_size)
                result["training"] = {"weighted_ce": weighted_loss_sum / weight_sum,
                                      "auxiliary_batch_mean": auxiliary_sum / epoch_batches, "total_loss_batch_mean": total_loss_sum / epoch_batches,
                                      "seconds": training_seconds,
                                      "samples": len(order), "optimizer_steps": steps_per_epoch,
                                      "order_sha256": sampling["order_sha256"], "sampling": sampling}
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
    parser.add_argument("--method", choices=("balanced_ce", "cross_signer_supcon"), required=True)
    parser.add_argument("--supcon-lambda", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--projection-lr", type=float, default=3e-4)
    parser.add_argument("--classes-per-batch", type=int, default=32)
    parser.add_argument("--signers-per-class", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=314)
    parser.add_argument("--head-seed", type=int, default=20260926)
    parser.add_argument("--train-cache-manifest", type=Path, required=True)
    parser.add_argument("--val-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True, help="Total completed epoch target; may increase on resume")
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260926)
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
