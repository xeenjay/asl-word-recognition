"""One fixed, label-blind top-five retrieval pilot using frozen PoET features.

Only training examples enter the memory. Validation targets are used after all
three rankings have been computed, solely for descriptive metrics and audits.
There are no optimizer steps or hyperparameter searches in this program.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np


PROTOCOL = "poet-frozen-signer-balanced-top5-retrieval-v1"
CHECKPOINT_SHA256 = "5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0"
BETA = 10.0
MEMORY_WEIGHT = 0.25
MODES = ("original", "pure_memory", "fixed_blend")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_record(path, root):
    path = Path(path)
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def valid_sha(value):
    return isinstance(value, str) and bool(SHA_RE.fullmatch(value))


def load_preregistration(path, expected_sha256):
    path = Path(path).resolve()
    require(valid_sha(expected_sha256) and sha256(path) == expected_sha256, "Preregistration SHA256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == 1 and value.get("registered_before_retrieval_scores") is True,
            "Require preregistration before scores")
    for key, expected in (("test_used", False), ("manual_annotation", False), ("optimizer_steps", 0),
                          ("sweeps_authorized", False), ("primary_method", "fixed_blend"),
                          ("methods", list(MODES))):
        require(value.get(key) == expected, "Preregistration protocol mismatch: " + key)
    model, data, retrieval, gate = (value.get(key, {}) for key in ("model", "data", "retrieval", "continuation_signal"))
    require(model.get("checkpoint_sha256") == CHECKPOINT_SHA256 and model.get("frozen") is True,
            "Preregistration checkpoint/frozen mismatch")
    for key, expected in (("train_count", 40154), ("val_count", 10304), ("classes", 2731),
                          ("memory_split", "train"), ("query_split", "val"), ("drop_samples", False)):
        require(data.get(key) == expected, "Preregistration population mismatch: " + key)
    for key, expected in (("l2_normalize", True), ("min_embedding_norm", 1e-12), ("cosine_clip", [-1., 1.]),
                          ("beta", BETA), ("candidate_count", 5), ("blend_memory_weight", MEMORY_WEIGHT),
                          ("blend_base_weight", 1. - MEMORY_WEIGHT), ("math_dtype", "float64"),
                          ("aggregation", "mean of per-signer mean kernels within each class"),
                          ("memory_order", "canonical ascending video_sha256"),
                          ("ties", "preserve original PoET top5 rank")):
        require(retrieval.get(key) == expected, "Preregistration retrieval mismatch: " + key)
    for key, expected in (("primary_top1_gain_percentage_points", .5), ("minimum_net_correct", 52),
                          ("macro_top1_strictly_better", True), ("same_top5_set_and_hit_every_sample", True),
                          ("zero_failed_required", True), ("auto_learned_reranker", False),
                          ("auto_full_training", False)):
        require(gate.get(key) == expected, "Preregistration gate mismatch: " + key)
    return value


def contained_file(root, relative):
    require(isinstance(relative, str) and bool(relative), "Missing artifact path")
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, "Artifact must be relative and contained")
    resolved = (root / value).resolve()
    require(root in resolved.parents and resolved.is_file(), "Missing/escaping artifact")
    return resolved


def normalize_embeddings(embeddings):
    require(isinstance(embeddings, np.ndarray) and embeddings.ndim == 2
            and embeddings.shape[1] == 128 and embeddings.dtype == np.float32,
            "Embeddings must be float32 [N,128]")
    require(bool(np.isfinite(embeddings).all()), "Non-finite embeddings")
    values = embeddings.astype(np.float64)
    norms = np.sqrt(np.einsum("ij,ij->i", values, values, optimize=False))
    require(bool((norms > 1e-12).all()), "Zero/near-zero embedding norm")
    return values / norms[:, None]


def validate_candidates(class_ids, logits, samples, allowed_classes):
    require(class_ids.dtype == np.int64 and class_ids.shape == (samples, 5),
            "Candidates must be int64 [N,5]")
    require(logits.dtype == np.float32 and logits.shape == (samples, 5),
            "Logits must be float32 [N,5]")
    require(bool(np.isfinite(logits).all()), "Non-finite top-five logits")
    require(bool(np.isin(class_ids, allowed_classes).all()), "Unknown candidate class")
    require(bool((np.diff(np.sort(class_ids, axis=1), axis=1) > 0).all()),
            "Duplicate top-five candidate class")
    require(bool((np.diff(logits, axis=1) <= 0).all()), "Candidates must follow original descending logit order")


def load_split(manifest_path, expected_sha256, split, *, engineering_smoke=False):
    """Validate every byte, row identity and formal population before scoring."""
    require(split in ("train", "val"), "Only train and val are allowed")
    manifest_path = Path(manifest_path).resolve()
    require(valid_sha(expected_sha256), "Expected manifest SHA256 is required")
    require(sha256(manifest_path) == expected_sha256, "Manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1 and manifest.get("status") == "completed",
            "Extraction must be completed schema_version=1")
    require(manifest.get("test_used") is False, "Test data are forbidden")
    require(manifest.get("checkpoint_sha256") == CHECKPOINT_SHA256, "Require the fixed official checkpoint")
    require(manifest.get("feature_dim") == 128, "Require PoET CLS dimension 128")
    require(manifest.get("weights_unchanged") is True and manifest.get("optimizer_steps") == 0,
            "Require verified frozen weights and zero optimizer steps")
    if not engineering_smoke:
        require(manifest.get("gradients_absent") is True and manifest.get("model_eval") is True,
                "Formal extraction must verify eval mode and absent gradients")
    require(type(manifest.get("failed")) is int and manifest["failed"] == 0, "Failed rows are forbidden")
    require(manifest.get("scope") == ("engineering" if engineering_smoke else "formal"),
            "Extraction scope mismatch; engineering is not a formal result")
    require(set(manifest.get("splits", {})) == {split}, "Manifest contains wrong/additional split")
    classes = manifest.get("class_ids")
    require(isinstance(classes, list) and len(classes) >= 5
            and all(type(value) is int and 0 <= value < 2731 for value in classes)
            and classes == sorted(set(classes)), "Invalid class_ids")
    labels = manifest.get("label_map")
    require(isinstance(labels, dict) and set(labels) == {str(value) for value in classes}
            and all(isinstance(value, str) and value.strip() == value and bool(value) for value in labels.values())
            and len(set(labels.values())) == len(classes), "Invalid/broken label bijection")
    if not engineering_smoke:
        require(classes == list(range(2731)), "Formal pilot requires all 2731 classes")
    info = manifest["splits"][split]
    if not engineering_smoke:
        require(info.get("complete_official_split") is True and info.get("failed") == 0,
                "Formal extraction must declare complete split and zero failures")
    samples = info.get("samples")
    require(type(samples) is int and samples > 0, "Invalid sample count")
    if not engineering_smoke:
        require(samples == {"train": 40154, "val": 10304}[split], "Incomplete formal population")
    for key in ("features_sha256", "metadata_sha256", "source_manifest_sha256", "cache_manifest_sha256"):
        require(valid_sha(info.get(key)), "Invalid/missing " + key)
    root = manifest_path.parent
    features_path = contained_file(root, info.get("features_path"))
    metadata_path = contained_file(root, info.get("metadata_path"))
    require(features_path != metadata_path, "Features and metadata must be distinct")
    require(sha256(features_path) == info["features_sha256"], "Features SHA256 mismatch")
    require(sha256(metadata_path) == info["metadata_sha256"], "Metadata SHA256 mismatch")
    for name, path in (("features", features_path), ("metadata", metadata_path)):
        if name + "_bytes" in info:
            require(type(info[name + "_bytes"]) is int and info[name + "_bytes"] == path.stat().st_size,
                    "Artifact byte count mismatch")
    with np.load(features_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"embeddings", "top5_class_ids", "top5_logits"}, "Unexpected feature arrays")
        arrays = {name: archive[name].copy() for name in archive.files}
    require(arrays["embeddings"].shape == (samples, 128), "Embedding/sample shape mismatch")
    normalized = normalize_embeddings(arrays["embeddings"])
    validate_candidates(arrays["top5_class_ids"], arrays["top5_logits"], samples, classes)
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == samples and all(line.strip() for line in lines), "Metadata/sample count mismatch")
    rows = [json.loads(line) for line in lines]
    seen_videos, seen_indices, seen_identities, signers, observed = set(), set(), set(), set(), Counter()
    for row_number, row in enumerate(rows):
        require(isinstance(row, dict) and row.get("split") == split, "Wrong split/test record")
        require(type(row.get("row")) is int and row["row"] == row_number, "Metadata row order mismatch")
        require(type(row.get("index")) is int and row["index"] >= 0 and row["index"] not in seen_indices,
                "Invalid/duplicate source index")
        seen_indices.add(row["index"])
        identity = row.get("identity")
        require(isinstance(identity, dict) and set(identity) == {"path", "sha256", "gloss", "signer_id", "split"}
                and isinstance(identity["path"], str) and bool(identity["path"])
                and identity["sha256"] == row.get("video_sha256") and identity["gloss"] == row.get("gloss")
                and identity["signer_id"] == row.get("signer_id") and identity["split"] == split,
                "Invalid source identity binding")
        identity_hash = canonical_hash(identity)
        require(identity_hash not in seen_identities, "Invalid/duplicate identity")
        seen_identities.add(identity_hash)
        for key in ("video_sha256", "input_sha256", "source_manifest_sha256"):
            require(valid_sha(row.get(key)), "Invalid/missing record " + key)
        require(row["source_manifest_sha256"] == info["source_manifest_sha256"], "Source manifest identity mismatch")
        if not engineering_smoke or "cache_manifest_sha256" in row:
            require(row.get("cache_manifest_sha256") == info["cache_manifest_sha256"], "Record cache manifest identity mismatch")
        require(row["video_sha256"] not in seen_videos, "Duplicate video identity")
        seen_videos.add(row["video_sha256"])
        label = row.get("class_id")
        require(type(label) is int and label in classes and row.get("gloss") == labels[str(label)],
                "Class/gloss label mismatch")
        signer = row.get("signer_id")
        require(isinstance(signer, str) and bool(signer) and signer.strip() == signer, "Invalid signer_id")
        signers.add(signer)
        observed[label] += 1
    require(set(observed) == set(classes), "Missing class in split")
    if not engineering_smoke:
        require(len(signers) == {"train": 35, "val": 6}[split], "Unexpected formal signer count")
        require(seen_indices == set(range(samples)), "Incomplete formal source indices")
    # Detect replacement while loading, rather than trusting an initial stat/hash.
    for path, expected in ((manifest_path, expected_sha256), (features_path, info["features_sha256"]),
                           (metadata_path, info["metadata_sha256"])):
        require(sha256(path) == expected, "Input changed while loading")
    return {"manifest": manifest, "manifest_path": str(manifest_path), "manifest_sha256": expected_sha256,
            "files": [{"path": str(path), "sha256": expected} for path, expected in
                      ((manifest_path, expected_sha256), (features_path, info["features_sha256"]),
                       (metadata_path, info["metadata_sha256"]))],
            "class_ids": classes, "label_map": labels, "rows": rows, "signers": signers,
            "video_sha256s": seen_videos, "identities": seen_identities, "arrays": arrays,
            "normalized_embeddings": normalized, "class_counts": dict(sorted(observed.items()))}


def verify_pair(train, val):
    require(train["manifest"]["splits"].keys() == {"train"}, "Memory must come from training only")
    require(val["manifest"]["splits"].keys() == {"val"}, "Queries must come from validation only")
    require(train["class_ids"] == val["class_ids"] and train["label_map"] == val["label_map"],
            "Train/val label maps differ")
    require(not train["signers"] & val["signers"], "Train/val signer overlap")
    require(not train["video_sha256s"] & val["video_sha256s"], "Train/val video overlap")
    require(not train["identities"] & val["identities"], "Train/val identity overlap")


def build_memory(embeddings, rows, class_ids):
    """Canonical train-only memory with equal signer, then equal example weight."""
    normalized = normalize_embeddings(embeddings)
    require(len(rows) == len(normalized) and bool(rows), "Memory metadata size mismatch")
    groups = defaultdict(list)
    seen = set()
    for index, row in enumerate(rows):
        require(row.get("split") == "train", "Memory must contain training rows only")
        key, label, signer = row.get("video_sha256"), row.get("class_id"), row.get("signer_id")
        require(valid_sha(key) and key not in seen, "Duplicate/invalid memory video")
        require(type(label) is int and label in class_ids, "Unknown memory class")
        require(isinstance(signer, str) and bool(signer), "Missing memory signer")
        seen.add(key)
        groups[label].append((key, signer, index))
    require(set(groups) == set(class_ids), "Missing memory class")
    result = {}
    for label in class_ids:
        ordered = sorted(groups[label])
        counts = Counter(signer for _, signer, _ in ordered)
        result[label] = {"embeddings": normalized[[index for _, _, index in ordered]],
                         "weights": np.array([1.0 / (len(counts) * counts[signer])
                                              for _, signer, _ in ordered], dtype=np.float64),
                         "samples": len(ordered), "signers": len(counts),
                         "signer_counts": dict(sorted(counts.items())),
                         "video_order_sha256": canonical_hash([key for key, _, _ in ordered])}
    return result


def memory_scores(memory, query_embeddings, candidates, *, chunk_size=256):
    """Score only original five candidates; query labels are not an input."""
    require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be positive")
    normalized = normalize_embeddings(query_embeddings)
    require(candidates.dtype == np.int64 and candidates.shape == (len(normalized), 5),
            "Candidates must be int64 [N,5]")
    require(bool((np.diff(np.sort(candidates, axis=1), axis=1) > 0).all()), "Duplicate candidate class")
    require(set(np.unique(candidates)).issubset(memory), "Missing candidate class in training memory")
    scores = np.full(candidates.shape, np.nan, np.float64)
    for class_id in sorted(np.unique(candidates).tolist()):
        rows, columns = np.where(candidates == class_id)
        group = memory[class_id]
        for start in range(0, len(rows), chunk_size):
            selected = rows[start:start + chunk_size]
            # Fixed reduction dimension/order avoids different chunk-size BLAS paths.
            cosine = np.einsum("qd,md->qm", normalized[selected], group["embeddings"], optimize=False)
            kernels = np.exp(-BETA * (1.0 - np.clip(cosine, -1.0, 1.0)))
            weighted = np.einsum("qm,m->q", kernels, group["weights"], optimize=False)
            scores[selected, columns[start:start + chunk_size]] = weighted
    require(bool(np.isfinite(scores).all()) and bool((scores > 0).all()), "Invalid memory kernel score")
    return scores


def rank_candidates(candidates, logits, scores):
    """Three rankings; exact ties preserve the incoming PoET top-five order."""
    validate_candidates(candidates, logits, len(candidates), np.unique(candidates))
    require(scores.dtype == np.float64 and scores.shape == candidates.shape
            and bool(np.isfinite(scores).all()) and bool((scores > 0).all()), "Invalid memory scores")
    base = logits.astype(np.float64) - logits[:, :1].astype(np.float64)
    base = np.exp(base)
    base /= base.sum(axis=1, keepdims=True)
    memory = scores / scores.sum(axis=1, keepdims=True)
    blend = (1.0 - MEMORY_WEIGHT) * base + MEMORY_WEIGHT * memory
    probabilities = {"original": base, "pure_memory": memory, "fixed_blend": blend}
    rankings, positions = {}, {}
    for mode in MODES:
        positions[mode] = (np.broadcast_to(np.arange(5), candidates.shape).copy() if mode == "original"
                           else np.argsort(-probabilities[mode], axis=1, kind="stable"))
        rankings[mode] = np.take_along_axis(candidates, positions[mode], axis=1)
        require(np.array_equal(np.sort(rankings[mode], axis=1), np.sort(candidates, axis=1)),
                "Reranking changed candidate membership")
    return {"rankings": rankings, "positions": positions, "probabilities": probabilities}


def metrics(targets, rankings, class_ids):
    targets = np.asarray(targets, dtype=np.int64)
    require(len(targets) == len(rankings) and len(targets) > 0, "Invalid metric population")
    top1 = rankings[:, 0] == targets
    top5 = (rankings == targets[:, None]).any(axis=1)
    per_class = {}
    for label in class_ids:
        selected = targets == label
        total = int(selected.sum())
        if total:
            per_class[str(label)] = {"samples": total, "correct_top1": int(top1[selected].sum()),
                                     "correct_top5": int(top5[selected].sum()),
                                     "top1": float(top1[selected].mean()), "top5": float(top5[selected].mean())}
    require(bool(per_class), "No observed class")
    return {"samples": len(targets), "failed": 0, "candidate_classes": len(class_ids),
            "observed_classes": len(per_class), "correct_top1": int(top1.sum()), "correct_top5": int(top5.sum()),
            "top1": float(top1.mean()), "top5": float(top5.mean()),
            "macro_top1": float(np.mean([value["top1"] for value in per_class.values()])),
            "macro_top5": float(np.mean([value["top5"] for value in per_class.values()])), "per_class": per_class}


def paired_comparison(rows, targets, original, candidate):
    good_original, good_candidate = original[:, 0] == targets, candidate[:, 0] == targets
    fixed = np.flatnonzero(~good_original & good_candidate).tolist()
    regressed = np.flatnonzero(good_original & ~good_candidate).tolist()
    changed = np.flatnonzero(original[:, 0] != candidate[:, 0]).tolist()
    details = lambda indices: [{"row": i, "index": rows[i]["index"], "identity": rows[i]["identity"],
                                "video_sha256": rows[i]["video_sha256"], "signer_id": rows[i]["signer_id"],
                                "target": int(targets[i]), "original": int(original[i, 0]),
                                "candidate": int(candidate[i, 0])} for i in indices]
    return {"samples": len(targets), "fixed_count": len(fixed), "regressed_count": len(regressed),
            "net_correct": len(fixed) - len(regressed), "changed_top1_count": len(changed),
            "fixed": details(fixed), "regressed": details(regressed), "changed_top1": details(changed),
            "delta_top1_percentage_points": 100.0 * (len(fixed) - len(regressed)) / len(targets)}


def continuation_signal(results, *, engineering_smoke=False):
    original, blend = results["original"], results["fixed_blend"]
    delta = 100.0 * (blend["top1"] - original["top1"])
    macro_delta = 100.0 * (blend["macro_top1"] - original["macro_top1"])
    # Integer comparison enforces >= 0.5 percentage points without rounding drift.
    net = blend["correct_top1"] - original["correct_top1"]
    instance_gate = net * 200 >= original["samples"]
    macro_gate = blend["macro_top1"] > original["macro_top1"]
    return {"primary_method": "fixed_blend", "comparison": "original",
            "scope": "engineering" if engineering_smoke else "retrieval_pilot_full2731",
            "threshold_top1_percentage_points": 0.5, "net_correct": net,
            "delta_top1_percentage_points": delta, "delta_macro_top1_percentage_points": macro_delta,
            "instance_threshold_met": instance_gate, "macro_strictly_positive": macro_gate,
            "pilot_signal": bool(instance_gate and macro_gate and not engineering_smoke),
            "engineering_only": engineering_smoke, "statistical_significance_test": False,
            "automatic_next_experiment_authorized": False}


def run(args):
    start = time.monotonic()
    source_path = Path(__file__).resolve()
    source_sha = sha256(source_path)
    prereg_path = getattr(args, "preregistration", None)
    prereg_sha = getattr(args, "preregistration_sha256", None)
    require(bool(prereg_path) == bool(prereg_sha), "Preregistration path and SHA256 must both be given")
    require(args.engineering_smoke or prereg_path, "Formal execution requires preregistration")
    prereg = load_preregistration(prereg_path, prereg_sha) if prereg_path else None
    output = Path(args.output_dir).resolve()
    # Never write into an input folder or nest an output around it.
    for manifest in (args.train_features_manifest, args.val_features_manifest):
        parent = Path(manifest).resolve().parent
        require(output != parent and output not in parent.parents and parent not in output.parents,
                "Output and extraction input directories must be separate")
    output.mkdir(parents=True, exist_ok=False)
    config = {"protocol": PROTOCOL, "beta": BETA, "memory_weight": MEMORY_WEIGHT,
              "base_weight": 1.0 - MEMORY_WEIGHT, "top_k": 5,
              "normalization": "per-example L2 in float64; norm > 1e-12",
              "memory_aggregation": "mean across signers of mean example kernels within each class",
              "tie_policy": "stable original candidate order", "optimizer_steps": 0, "test_used": False,
              "engineering_smoke": bool(args.engineering_smoke), "query_chunk_size": args.query_chunk_size,
              "prereg_sha256": prereg_sha, "source_sha256": source_sha,
              "train_manifest_sha256": args.train_manifest_sha256, "val_manifest_sha256": args.val_manifest_sha256,
              "primary_method": "fixed_blend", "diagnostic_method": "pure_memory"}
    write_json(output / "config.json", config)
    state = {"schema_version": 1, "status": "running", "started_utc": now(), "protocol": PROTOCOL,
             "optimizer_steps": 0, "test_used": False,
             "scope": "engineering" if args.engineering_smoke else "retrieval_pilot_full2731"}
    write_json(output / "state.json", state)
    try:
        load_start = time.monotonic()
        train = load_split(args.train_features_manifest, args.train_manifest_sha256, "train",
                           engineering_smoke=args.engineering_smoke)
        val = load_split(args.val_features_manifest, args.val_manifest_sha256, "val",
                         engineering_smoke=args.engineering_smoke)
        verify_pair(train, val)
        if prereg is not None:
            for split, loaded in (("train", train), ("val", val)):
                require(loaded["manifest"]["splits"][split]["cache_manifest_sha256"]
                        == prereg["data"][split + "_cache_sha256"], "Preregistered source cache SHA256 mismatch")
        input_seconds = time.monotonic() - load_start
        scoring_start = time.monotonic()
        memory = build_memory(train["arrays"]["embeddings"], train["rows"], train["class_ids"])
        scores = memory_scores(memory, val["arrays"]["embeddings"], val["arrays"]["top5_class_ids"],
                               chunk_size=args.query_chunk_size)
        rankings = rank_candidates(val["arrays"]["top5_class_ids"], val["arrays"]["top5_logits"], scores)
        scoring_seconds = time.monotonic() - scoring_start
        with (output / "scores.npz").open("xb") as stream:
            np.savez_compressed(stream, candidate_ids=val["arrays"]["top5_class_ids"],
                                base_probability=rankings["probabilities"]["original"], memory_kernel_scores=scores,
                                memory_probability=rankings["probabilities"]["pure_memory"],
                                blend_probability=rankings["probabilities"]["fixed_blend"])
        # Targets enter here, strictly after every score and ranking is final.
        rows, classes = val["rows"], val["class_ids"]
        targets = np.array([row["class_id"] for row in rows], np.int64)
        results, signer_results, paired = {}, {}, {}
        original_hits = (rankings["rankings"]["original"] == targets[:, None]).any(axis=1)
        for mode in MODES:
            ordered = rankings["rankings"][mode]
            require(np.array_equal(original_hits, (ordered == targets[:, None]).any(axis=1)),
                    "Reranking changed top-five membership/hits")
            results[mode] = metrics(targets, ordered, classes)
            signer_results[mode] = {}
            for signer in sorted(val["signers"]):
                mask = np.array([row["signer_id"] == signer for row in rows])
                signer_results[mode][signer] = metrics(targets[mask], ordered[mask], classes)
            mode_dir = output / mode
            mode_dir.mkdir()
            write_json(mode_dir / "metrics.json", results[mode])
            write_json(mode_dir / "by_signer.json", signer_results[mode])
            with (mode_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for i, row in enumerate(rows):
                    value = {"row": i, "index": row["index"], "identity": row["identity"], "split": "val",
                             "video_sha256": row["video_sha256"], "signer_id": row["signer_id"],
                             "target": int(targets[i]), "gloss": row["gloss"], "method": mode,
                             "original_candidate_ids": val["arrays"]["top5_class_ids"][i].tolist(),
                             "original_logits": val["arrays"]["top5_logits"][i].tolist(),
                             "memory_kernel_scores": scores[i].tolist(),
                             "base_probability": rankings["probabilities"]["original"][i].tolist(),
                             "memory_probability": rankings["probabilities"]["pure_memory"][i].tolist(),
                             "blend_probability": rankings["probabilities"]["fixed_blend"][i].tolist(),
                             "ranking_positions": rankings["positions"][mode][i].tolist(),
                             "ranked_class_ids": ordered[i].tolist(), "prediction": int(ordered[i, 0]),
                             "correct_top1": bool(ordered[i, 0] == targets[i]), "correct_top5": bool(original_hits[i])}
                    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            if mode != "original":
                paired[mode] = paired_comparison(rows, targets, rankings["rankings"]["original"], ordered)
                write_json(mode_dir / "paired_vs_original.json", paired[mode])
        memory_audit = {str(key): {name: value for name, value in group.items()
                                  if name not in ("embeddings", "weights")} for key, group in memory.items()}
        write_json(output / "memory_audit.json", memory_audit)
        input_files = train["files"] + val["files"]
        if prereg_path:
            input_files.append({"path": str(Path(prereg_path).resolve()), "sha256": prereg_sha})
        for item in input_files:
            require(sha256(item["path"]) == item["sha256"], "Input changed during retrieval")
        require(sha256(source_path) == source_sha, "Reranker source changed during execution")
        summary = {**state, "status": "completed", "ended_utc": now(), "failed": 0,
                   "config_sha256": sha256(output / "config.json"),
                   "code": {"path": str(source_path), "sha256": source_sha}, "source_sha256": source_sha,
                   "prereg_sha256": prereg_sha, "train_manifest_sha256": args.train_manifest_sha256,
                   "val_manifest_sha256": args.val_manifest_sha256, "train_samples": len(train["rows"]),
                   "val_samples": len(rows), "candidate_classes": len(classes), "topk": 5,
                   "top5_unchanged": True, "input_bytes_unchanged": True,
                   "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                   "platform": platform.platform()}, "inputs": input_files,
                   "population": {"train": len(train["rows"]), "val": len(rows), "classes": len(classes),
                                  "train_signers": len(train["signers"]), "val_signers": len(val["signers"])},
                   "metrics": results, "by_signer": signer_results,
                   "paired_comparisons": paired, "candidate_membership_unchanged": True,
                   "top5_hits_unchanged": True, "train_val_video_and_signer_disjoint": True,
                   "continuation_signal": continuation_signal(results, engineering_smoke=args.engineering_smoke),
                   "timing": {"input_validation_seconds": input_seconds, "scoring_seconds": scoring_seconds,
                              "elapsed_seconds": time.monotonic() - start},
                   "artifacts": [file_record(path, output) for path in sorted(output.rglob("*"))
                                 if path.is_file() and path.name not in ("summary.json", "state.json")]}
        write_json(output / "summary.json", summary)
        write_json(output / "state.json", {**state, "status": "completed", "ended_utc": summary["ended_utc"],
                                            "summary_sha256": sha256(output / "summary.json")})
        print(json.dumps({"status": "completed", "scope": summary["scope"], "output_dir": str(output),
                          "top1": {mode: value["top1"] for mode, value in results.items()},
                          "continuation_signal": summary["continuation_signal"]}, sort_keys=True))
        return summary
    except BaseException as error:
        write_json(output / "state.json", {**state, "status": "failed", "ended_utc": now(),
                                            "error": type(error).__name__ + ": " + str(error),
                                            "elapsed_seconds": time.monotonic() - start})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features-manifest", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--val-features-manifest", required=True)
    parser.add_argument("--val-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preregistration")
    parser.add_argument("--preregistration-sha256")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--engineering-smoke", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
