"""Frozen six-checkpoint 2x2 geometry/retrieval comparison; no training or sweep.

Each checkpoint gets its own train memory and val queries. Only pure numerical
retrieval functions are reused from the immutable official-checkpoint pilot.
The new manifests and preregistration retain their actual checkpoint identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import time
import numpy as np

import rerank_retrieval_pilot as core
from rerank_retrieval_pilot import (require, sha256, canonical_hash, now, write_json,
    file_record, valid_sha, contained_file, normalize_embeddings, validate_candidates,
    build_memory, memory_scores, rank_candidates, metrics, paired_comparison, verify_pair)

PROTOCOL = "poet-factorial-retrieval-v1"
SEEDS = (20260922, 20260923, 20260924)
METHODS = ("baseline", "geometry")
CELLS = ("B", "B_plus_R", "G", "G_plus_R")
MODES = core.MODES
COMPARISONS = (("B_plus_R_vs_B", "B_plus_R", "B"), ("G_vs_B", "G", "B"),
               ("G_plus_R_vs_G", "G_plus_R", "G"),
               ("G_plus_R_vs_B_plus_R", "G_plus_R", "B_plus_R"),
               ("G_plus_R_vs_B", "G_plus_R", "B"))


def validate_preregistration(value):
    expected = {"schema_version": 1, "protocol": PROTOCOL,
                "registered_before_factorial_scores": True, "seeds": list(SEEDS),
                "methods": list(CELLS), "diagnostic_methods": ["pure_memory"],
                "primary_method": "geometry_fixed_blend", "optimizer_steps": 0,
                "test_used": False, "manual_annotation": False, "sweeps_authorized": False}
    for key, wanted in expected.items():
        require(value.get(key) == wanted, "Preregistration protocol mismatch: " + key)
    models = value.get("models")
    require(isinstance(models, list) and len(models) == 6, "Exactly six frozen models required")
    wanted = {method + "_seed" + str(seed) for seed in SEEDS for method in METHODS}
    require({model.get("model_id") for model in models} == wanted, "Six model identities mismatch")
    checkpoints = set()
    for model in models:
        require(model.get("method") in METHODS and model.get("seed") in SEEDS
                and model["model_id"] == model["method"] + "_seed" + str(model["seed"]),
                "Model method/seed identity mismatch")
        for key in ("checkpoint_sha256", "reference_predictions_sha256", "model_state_sha256", "protocol_sha256"):
            require(valid_sha(model.get(key)), "Invalid model " + key)
        for key in ("checkpoint_path", "reference_predictions_path"):
            require(isinstance(model.get(key), str) and bool(model[key]), "Missing model " + key)
        require(type(model.get("checkpoint_epoch")) is int and model["checkpoint_epoch"] > 0, "Invalid selected checkpoint epoch")
        checkpoints.add(model["checkpoint_sha256"])
    require(len(checkpoints) == 6, "Duplicate checkpoint identity across models")
    for key, wanted in {"train_count": 40154, "val_count": 10304, "classes": 2731,
                        "memory_split": "train", "query_split": "val", "drop_samples": False}.items():
        require(value.get("data", {}).get(key) == wanted, "Preregistered population mismatch: " + key)
    for split in ("train", "val"):
        require(valid_sha(value["data"].get(split + "_cache_sha256")), "Missing cache SHA256")
    retrieval = {"l2_normalize": True, "min_embedding_norm": 1e-12, "cosine_clip": [-1., 1.],
                 "beta": 10., "candidate_count": 5, "blend_memory_weight": .25,
                 "blend_base_weight": .75, "math_dtype": "float64",
                 "aggregation": "mean of per-signer mean kernels within each class",
                 "memory_order": "canonical ascending video_sha256",
                 "ties": "preserve original PoET top5 rank"}
    for key, wanted in retrieval.items():
        require(value.get("retrieval", {}).get(key) == wanted, "Fixed retrieval mismatch: " + key)
    gate = {"comparisons": ["G_plus_R_vs_G", "G_plus_R_vs_B_plus_R"],
            "metrics": ["top1", "macro_top1"], "all_seeds_strictly_positive": True,
            "zero_failed_required": True, "historical_predictions_exact_required": True,
            "same_top5_within_model_required": True, "independent_verification_required": True,
            "automatic_further_experiments": False}
    require(value.get("retention_gate") == gate, "Preregistered retention gate mismatch")
    return value


def load_preregistration(path, expected_sha):
    path = Path(path).resolve()
    require(valid_sha(expected_sha) and sha256(path) == expected_sha, "Preregistration SHA256 mismatch")
    value = validate_preregistration(json.loads(path.read_text(encoding="utf-8")))
    require(sha256(path) == expected_sha, "Preregistration changed while loading")
    return value


def validate_model_binding(manifest, model, prereg_sha):
    for key, wanted in (("protocol", PROTOCOL), ("model_id", model["model_id"]),
                        ("method", model["method"]), ("seed", model["seed"]),
                        ("prereg_sha256", prereg_sha), ("checkpoint_sha256", model["checkpoint_sha256"]),
                        ("checkpoint_epoch", model["checkpoint_epoch"])):
        require(manifest.get(key) == wanted, "Feature model binding mismatch: " + key)
    require(manifest.get("model_state_sha256") == model["model_state_sha256"], "Feature model state mismatch")
    require(manifest.get("training_protocol_sha256") == model["protocol_sha256"],
            "Feature training protocol mismatch")


def load_collection(path, expected_sha, split, prereg, prereg_sha):
    require(split in ("train", "val"), "Only train/val collections allowed")
    path = Path(path).resolve()
    require(valid_sha(expected_sha) and sha256(path) == expected_sha, "Collection SHA256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, wanted in (("schema_version", 1), ("protocol", PROTOCOL), ("scope", "formal"),
                        ("status", "completed"), ("split", split), ("prereg_sha256", prereg_sha)):
        require(value.get(key) == wanted, "Collection contract mismatch: " + key)
    models = value.get("models")
    require(isinstance(models, dict) and set(models) == {m["model_id"] for m in prereg["models"]},
            "Collection model identities mismatch")
    specs = {}
    for model_id, info in models.items():
        require(isinstance(info, dict) and valid_sha(info.get("manifest_sha256")), "Invalid model manifest pointer")
        child = contained_file(path.parent, info.get("manifest_path"))
        require(sha256(child) == info["manifest_sha256"], "Model manifest SHA256 mismatch")
        specs[model_id] = (child, info["manifest_sha256"])
    require(len({str(spec[0]) for spec in specs.values()}) == 6, "Duplicate model manifest path")
    require(sha256(path) == expected_sha, "Collection changed while loading")
    return specs, {"path": str(path), "sha256": expected_sha}


def load_split(manifest_path, expected_sha256, split, model, prereg, prereg_sha):
    """Validate every byte, row identity and formal population before scoring."""
    engineering_smoke = False
    require(split in ("train", "val"), "Only train and val are allowed")
    manifest_path = Path(manifest_path).resolve()
    require(valid_sha(expected_sha256), "Expected manifest SHA256 is required")
    require(sha256(manifest_path) == expected_sha256, "Manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1 and manifest.get("status") == "completed",
            "Extraction must be completed schema_version=1")
    require(manifest.get("test_used") is False, "Test data are forbidden")
    validate_model_binding(manifest, model, prereg_sha)
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
    expected_geometry = prereg["data"][split + "_geometry_sha256"] if model["method"] == "geometry" else None
    require(manifest.get("geometry_manifest_sha256") == expected_geometry, "Geometry manifest binding mismatch")
    info = manifest["splits"][split]
    require(info.get("cache_manifest_sha256") == prereg["data"][split + "_cache_sha256"],
            "Preregistered source cache mismatch")
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
    for row_number, row in enumerate(rows):
        require(row["index"] == row_number, "Formal source index order mismatch")
        require(row.get("model_id") == model["model_id"]
                and row.get("checkpoint_sha256") == model["checkpoint_sha256"], "Metadata model binding mismatch")
        if model["method"] == "geometry":
            require(valid_sha(row.get("geometry_input_sha256")), "Missing geometry input SHA256")
        else:
            require(row.get("geometry_input_sha256") is None, "Unexpected baseline geometry input")
        require(valid_sha(row.get("logits_sha256")), "Missing full logits SHA256")
        probabilities = row.get("top5_probabilities")
        require(isinstance(probabilities, list) and len(probabilities) == 5
                and all(type(x) is float and np.isfinite(x) and 0 <= x <= 1 for x in probabilities)
                and probabilities == sorted(probabilities, reverse=True), "Invalid full softmax probabilities")
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

def common_rows_fingerprint(loaded):
    """Geometry inputs vary by method; shared pose and video order must not."""
    return canonical_hash([{key: row[key] for key in
        ("row", "index", "identity", "class_id", "gloss", "signer_id", "split", "video_sha256", "input_sha256")}
        for row in loaded["rows"]])


def verify_model_pair(train, val):
    verify_pair(train, val)
    for key in ("model_id", "method", "seed", "checkpoint_sha256", "checkpoint_epoch",
                "model_state_sha256", "training_protocol_sha256", "prereg_sha256"):
        require(train["manifest"].get(key) == val["manifest"].get(key),
                "Train/val feature space mismatch: " + key)


def verify_reference(val, model):
    """Full historical float32-logit hash, identities and ranks must reproduce."""
    path = Path(model["reference_predictions_path"]).resolve()
    expected = model["reference_predictions_sha256"]
    require(sha256(path) == expected, "Historical prediction SHA256 mismatch")
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(val["rows"]), "Historical prediction population mismatch")
    for i, (line, row) in enumerate(zip(lines, val["rows"])):
        reference = json.loads(line)
        require(reference.get("status") == "ok", "Historical failed prediction")
        for key in ("index", "gloss", "signer_id", "split", "source_path", "video_sha256", "logits_sha256"):
            require(reference.get(key) == row.get(key), "Historical prediction mismatch: " + key)
        top5 = reference.get("top5")
        require(isinstance(top5, list) and len(top5) == 5, "Historical top-five shape mismatch")
        require([item["class_id"] for item in top5] == val["arrays"]["top5_class_ids"][i].tolist(),
                "Historical top-five ranks mismatch")
        require([item["gloss"] for item in top5] == [val["label_map"][str(k)] for k in
                val["arrays"]["top5_class_ids"][i].tolist()], "Historical top-five gloss mismatch")
        require([item["probability"] for item in top5] == row["top5_probabilities"],
                "Historical full-softmax probabilities mismatch")
    require(sha256(path) == expected, "Historical predictions changed while loading")
    return {"samples": len(lines), "identity_exact": True, "top5_ranks_exact": True,
            "full_logits_sha256_exact": True, "top5_probabilities_exact": True,
            "path": str(path), "sha256": expected}


def score_model(train, val, output, model, prereg_sha, chunk_size):
    """Compute all rankings before validation targets enter the scoring path."""
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    memory = build_memory(train["arrays"]["embeddings"], train["rows"], train["class_ids"])
    scores = memory_scores(memory, val["arrays"]["embeddings"], val["arrays"]["top5_class_ids"],
                           chunk_size=chunk_size)
    ranked = rank_candidates(val["arrays"]["top5_class_ids"], val["arrays"]["top5_logits"], scores)
    with (output / "scores.npz").open("xb") as stream:
        np.savez_compressed(stream, candidate_ids=val["arrays"]["top5_class_ids"],
                            base_probability=ranked["probabilities"]["original"], memory_kernel_scores=scores,
                            memory_probability=ranked["probabilities"]["pure_memory"],
                            blend_probability=ranked["probabilities"]["fixed_blend"])
    rows, classes = val["rows"], val["class_ids"]
    targets = np.array([row["class_id"] for row in rows], np.int64)
    original_hits = (ranked["rankings"]["original"] == targets[:, None]).any(axis=1)
    results, by_signer, paired = {}, {}, {}
    for mode in MODES:
        ranking = ranked["rankings"][mode]
        require(np.array_equal(original_hits, (ranking == targets[:, None]).any(axis=1)),
                "Reranking changed top-five hits")
        results[mode] = metrics(targets, ranking, classes)
        by_signer[mode] = {}
        for signer in sorted(val["signers"]):
            mask = np.array([row["signer_id"] == signer for row in rows])
            by_signer[mode][signer] = metrics(targets[mask], ranking[mask], classes)
        directory = output / mode
        directory.mkdir()
        write_json(directory / "metrics.json", results[mode])
        write_json(directory / "by_signer.json", by_signer[mode])
        with (directory / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for i, row in enumerate(rows):
                item = {"row": i, "index": row["index"], "identity": row["identity"], "split": "val",
                        "model_id": model["model_id"], "checkpoint_sha256": model["checkpoint_sha256"],
                        "video_sha256": row["video_sha256"], "signer_id": row["signer_id"],
                        "target": int(targets[i]), "gloss": row["gloss"], "method": mode,
                        "original_candidate_ids": val["arrays"]["top5_class_ids"][i].tolist(),
                        "original_logits": val["arrays"]["top5_logits"][i].tolist(),
                        "original_full_logits_sha256": row["logits_sha256"],
                        "memory_kernel_scores": scores[i].tolist(),
                        "base_probability": ranked["probabilities"]["original"][i].tolist(),
                        "memory_probability": ranked["probabilities"]["pure_memory"][i].tolist(),
                        "blend_probability": ranked["probabilities"]["fixed_blend"][i].tolist(),
                        "ranking_positions": ranked["positions"][mode][i].tolist(),
                        "ranked_class_ids": ranking[i].tolist(), "prediction": int(ranking[i, 0]),
                        "correct_top1": bool(ranking[i, 0] == targets[i]), "correct_top5": bool(original_hits[i])}
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        if mode != "original":
            paired[mode] = paired_comparison(rows, targets, ranked["rankings"]["original"], ranking)
            paired[mode]["delta_macro_top1_percentage_points"] = 100. * (
                results[mode]["macro_top1"] - results["original"]["macro_top1"])
            write_json(directory / "paired_vs_original.json", paired[mode])
    audit = {str(k): {name: value for name, value in group.items() if name not in ("embeddings", "weights")}
             for k, group in memory.items()}
    write_json(output / "memory_audit.json", audit)
    summary = {"schema_version": 1, "protocol": PROTOCOL, "status": "completed",
               "scope": "factorial_retrieval_full2731", "model_id": model["model_id"],
               "method": model["method"], "seed": model["seed"], "checkpoint_sha256": model["checkpoint_sha256"],
               "checkpoint_epoch": model["checkpoint_epoch"], "prereg_sha256": prereg_sha,
               "train_samples": len(train["rows"]), "val_samples": len(rows), "candidate_classes": len(classes), "topk": 5,
               "failed": 0, "optimizer_steps": 0, "test_used": False, "top5_unchanged": True,
               "candidate_membership_unchanged": True, "top5_hits_unchanged": True,
               "train_val_video_and_signer_disjoint": True,
               "train_manifest_sha256": train["manifest_sha256"], "val_manifest_sha256": val["manifest_sha256"],
               "metrics": results, "by_signer": by_signer, "paired_comparisons": paired,
               "scoring_and_output_seconds": time.monotonic() - started}
    return summary, {mode: ranked["rankings"][mode] for mode in ("original", "fixed_blend")}


def moments(values):
    values = np.asarray(values, np.float64)
    require(values.shape == (3,) and bool(np.isfinite(values).all()), "Exactly three finite seed values required")
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)), "n": 3}


def factorial_results(model_results, rankings, rows, classes):
    targets = np.array([row["class_id"] for row in rows], np.int64)
    by_seed = {}
    for seed in SEEDS:
        baseline_id, geometry_id = "baseline_seed" + str(seed), "geometry_seed" + str(seed)
        assignments = {"B": (baseline_id, "original"), "B_plus_R": (baseline_id, "fixed_blend"),
                       "G": (geometry_id, "original"), "G_plus_R": (geometry_id, "fixed_blend")}
        cells = {cell: model_results[model_id]["metrics"][mode] for cell, (model_id, mode) in assignments.items()}
        orders = {cell: rankings[model_id][mode] for cell, (model_id, mode) in assignments.items()}
        comparisons = {}
        for name, candidate, reference in COMPARISONS:
            comparison = paired_comparison(rows, targets, orders[reference], orders[candidate])
            comparison["delta_macro_top1_percentage_points"] = 100. * (
                cells[candidate]["macro_top1"] - cells[reference]["macro_top1"])
            comparisons[name] = comparison
        interaction = {"net_correct": comparisons["G_plus_R_vs_G"]["net_correct"] - comparisons["B_plus_R_vs_B"]["net_correct"],
                       "top1_percentage_points": comparisons["G_plus_R_vs_G"]["delta_top1_percentage_points"]
                       - comparisons["B_plus_R_vs_B"]["delta_top1_percentage_points"],
                       "macro_top1_percentage_points": comparisons["G_plus_R_vs_G"]["delta_macro_top1_percentage_points"]
                       - comparisons["B_plus_R_vs_B"]["delta_macro_top1_percentage_points"]}
        positive = all(comparisons[name]["net_correct"] > 0 and
                       comparisons[name]["delta_macro_top1_percentage_points"] > 0
                       for name in ("G_plus_R_vs_G", "G_plus_R_vs_B_plus_R"))
        by_seed[str(seed)] = {"cells": cells, "comparisons": comparisons, "interaction": interaction,
                              "combination_strictly_better_than_both": bool(positive)}
    aggregate = {"cells": {}, "comparisons": {}, "interaction": {}}
    for cell in CELLS:
        aggregate["cells"][cell] = {metric: moments([v["cells"][cell][metric] for v in by_seed.values()])
            for metric in ("top1", "macro_top1", "top5", "macro_top5", "correct_top1")}
    for name, _, _ in COMPARISONS:
        aggregate["comparisons"][name] = {metric: moments([v["comparisons"][name][metric] for v in by_seed.values()])
            for metric in ("net_correct", "delta_top1_percentage_points", "delta_macro_top1_percentage_points")}
    for metric in ("net_correct", "top1_percentage_points", "macro_top1_percentage_points"):
        aggregate["interaction"][metric] = moments([v["interaction"][metric] for v in by_seed.values()])
    per_seed = {seed: item["combination_strictly_better_than_both"] for seed, item in by_seed.items()}
    signal = {"mathematical_conditions_met": all(per_seed.values()), "per_seed": per_seed,
              "all_seeds_strictly_positive": True, "independent_verification_required": True,
              "independent_verification_completed": False, "combination_retained": False,
              "automatic_further_experiments": False, "statistical_significance_test": False}
    return {"by_seed": by_seed, "aggregate": aggregate, "retention_signal": signal,
            "interpretation": "descriptive paired validation comparisons; same videos repeated across seeds; no significance claim"}


def run(args):
    started = time.monotonic()
    source_files = [{"path": str(path), "sha256": sha256(path)}
                    for path in (Path(__file__).resolve(), Path(core.__file__).resolve())]
    prereg = load_preregistration(args.preregistration, args.preregistration_sha256)
    output = Path(args.output_dir).resolve()
    for manifest in (args.train_features_manifest, args.val_features_manifest):
        parent = Path(manifest).resolve().parent
        require(output != parent and output not in parent.parents and parent not in output.parents,
                "Output and extraction directories must be separate")
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "protocol": PROTOCOL, "status": "running", "started_utc": now(),
             "scope": "factorial_retrieval_full2731", "optimizer_steps": 0, "test_used": False}
    write_json(output / "state.json", state)
    config = {"protocol": PROTOCOL, "prereg_sha256": args.preregistration_sha256,
              "source_files": source_files, "beta": 10., "memory_weight": .25, "base_weight": .75,
              "query_chunk_size": args.query_chunk_size, "train_manifest_sha256": args.train_manifest_sha256,
              "val_manifest_sha256": args.val_manifest_sha256, "model_ids": [m["model_id"] for m in prereg["models"]]}
    write_json(output / "config.json", config)
    try:
        train_specs, train_collection = load_collection(args.train_features_manifest, args.train_manifest_sha256,
                                                        "train", prereg, args.preregistration_sha256)
        val_specs, val_collection = load_collection(args.val_features_manifest, args.val_manifest_sha256,
                                                    "val", prereg, args.preregistration_sha256)
        inputs = [train_collection, val_collection,
                  {"path": str(Path(args.preregistration).resolve()), "sha256": args.preregistration_sha256}]
        results, all_rankings, fingerprints = {}, {}, {}
        reference_rows, classes = None, None
        for model in prereg["models"]:
            model_id = model["model_id"]
            print(json.dumps({"status": "loading", "model_id": model_id}), flush=True)
            train = load_split(*train_specs[model_id], "train", model, prereg, args.preregistration_sha256)
            val = load_split(*val_specs[model_id], "val", model, prereg, args.preregistration_sha256)
            verify_model_pair(train, val)
            for split, loaded in (("train", train), ("val", val)):
                fingerprint = common_rows_fingerprint(loaded)
                if split in fingerprints:
                    require(fingerprints[split] == fingerprint, "Cross-model shared row/pose order mismatch")
                fingerprints[split] = fingerprint
            historical = verify_reference(val, model)
            inputs.extend(train["files"] + val["files"] + [{"path": historical["path"], "sha256": historical["sha256"]}])
            model_output = output / "models" / model_id
            summary, ordered = score_model(train, val, model_output, model, args.preregistration_sha256, args.query_chunk_size)
            summary["historical_reference"] = historical
            results[model_id], all_rankings[model_id] = summary, ordered
            write_json(model_output / "summary.json", summary)
            if reference_rows is None:
                reference_rows, classes = val["rows"], val["class_ids"]
            print(json.dumps({"status": "scored", "model_id": model_id}), flush=True)
            del train, val
        factorial = factorial_results(results, all_rankings, reference_rows, classes)
        write_json(output / "factorial.json", factorial)
        for item in inputs + source_files:
            require(sha256(item["path"]) == item["sha256"], "Input/source changed during factorial retrieval")
        summary = {**state, "status": "completed", "ended_utc": now(), "failed": 0,
                   "prereg_sha256": args.preregistration_sha256, "config_sha256": sha256(output / "config.json"),
                   "train_manifest_sha256": args.train_manifest_sha256, "val_manifest_sha256": args.val_manifest_sha256,
                   "train_samples_per_model": 40154, "val_samples_per_model": 10304, "candidate_classes": 2731,
                   "models": results, "factorial": factorial, "retention_signal": factorial["retention_signal"],
                   "all_historical_predictions_exact": True, "top5_unchanged_within_each_model": True,
                   "independent_feature_spaces": True, "cross_model_shared_identity_order_exact": True,
                   "input_bytes_unchanged": True, "common_rows_sha256": fingerprints, "inputs": inputs,
                   "source_files": source_files,
                   "source_sha256": {item["path"]: item["sha256"] for item in source_files + [inputs[2]]}, "environment": {"python": platform.python_version(), "numpy": np.__version__},
                   "elapsed_seconds": time.monotonic() - started,
                   "artifacts": [file_record(path, output) for path in sorted(output.rglob("*"))
                                 if path.is_file() and path.name not in ("state.json",)
                                 and path != output / "summary.json"]}
        write_json(output / "summary.json", summary)
        write_json(output / "state.json", {**state, "status": "completed", "ended_utc": summary["ended_utc"],
                                           "summary_sha256": sha256(output / "summary.json")})
        print(json.dumps({"status": "completed", "retention_signal": summary["retention_signal"]}), flush=True)
        return summary
    except BaseException as error:
        write_json(output / "state.json", {**state, "status": "failed", "ended_utc": now(),
                                           "error": type(error).__name__ + ": " + str(error)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features-manifest", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--val-features-manifest", required=True)
    parser.add_argument("--val-manifest-sha256", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
