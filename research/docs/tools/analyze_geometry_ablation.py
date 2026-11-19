#!/usr/bin/env python3
"""Read-only, preregistered analysis of the fixed epoch-10 geometry ablation.

Inputs are standard evaluation directories, not checkpoints or model code.
The CLI always requires the full 10,304-video, 2,731-class validation cohort.
An optional output is created exclusively; existing evidence is never replaced.
Exit 0 means valid analysis, including an unsupported gate; invalid evidence is 1.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sys

SAMPLES = 10304
CLASSES = 2731
SHUFFLE_SEED = 20260925
ALGORITHM = "private_random_shuffle_then_cycle_successor_v1"
GEOMETRY_SHA = "b3c4b6babf6b91f91970aa2a172a57ac8f31b85d3c48916ede6cfe194a35fc87"
BASELINE_SHA = "504f7bfe9be5fa538239dd1a0fa754a516fc55c0dd22676a343dc5812caddb36"
CACHE_SHA = "a829029d700e50dd07d0f88454d9f3d42a65e946586a1188e7dfe9a81d5a5c86"
IDENTITY = ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def identity(row):
    return tuple(row[key] for key in IDENTITY)


def is_correct(row, k):
    return row["gloss"] in [item["gloss"] for item in row["top5"][:k]]


def summarize(rows, vocabulary):
    """Recount predictions independently, in fixed class-ID order."""
    counts = {label: [0, 0, 0] for label in vocabulary.values()}
    for row in rows:
        require(row["gloss"] in counts, "Unknown ground-truth gloss")
        cell = counts[row["gloss"]]
        cell[0] += 1
        cell[1] += is_correct(row, 1)
        cell[2] += is_correct(row, 5)
    observed = [cell for cell in counts.values() if cell[0]]
    require(bool(rows) and bool(observed), "Empty prediction cohort")
    top1, top5 = (sum(cell[k] for cell in observed) for k in (1, 2))
    macro = [sum(cell[k] / cell[0] for cell in observed) / len(observed) for k in (1, 2)]
    result = {"attempted": len(rows), "inferred": len(rows), "failed": 0,
              "top1_correct": top1, "top5_correct": top5,
              "per_instance": {"top1": top1 / len(rows), "top5": top5 / len(rows)},
              "per_class_observed": {"top1": macro[0], "top5": macro[1], "denominator_classes": len(observed)},
              "per_class_full_vocabulary": {"top1": macro[0] if len(observed) == len(counts) else None,
                                            "top5": macro[1] if len(observed) == len(counts) else None}}
    return result, counts


def exact_macro_top1(counts):
    require(all(cell[0] > 0 for cell in counts.values()), "Gate requires every vocabulary class")
    return sum((Fraction(cell[1], cell[0]) for cell in counts.values()), Fraction()) / len(counts)


def load_evaluation(directory):
    directory = Path(directory).resolve()
    summary = read(directory / "summary.json")
    metrics = read(directory / "metrics.json")
    require(summary["status"] == "completed" and summary["scope"] == "official_validation", "Evaluation not complete official validation")
    require(summary.get("attribute_targets_read") is False, "Evaluation must not read attribute targets")
    require(summary["metrics"] == metrics, "Summary/metrics disagreement")
    provenance = summary["data"]
    require(provenance["complete_official_split"] is True and provenance["official_split_count"] == SAMPLES
            and provenance["selected_count"] == SAMPLES and provenance["cache_manifest_sha256"] == CACHE_SHA,
            "Evaluation does not identify the fixed complete validation cache")
    fingerprints = {}
    for name in ("predictions.jsonl", "metrics.json", "per_class.csv", "failures.json"):
        fingerprints[name] = sha(directory / name)
        require(summary["artifacts"][name]["sha256"] == fingerprints[name], f"Artifact SHA mismatch: {directory / name}")
    failures = read(directory / "failures.json")
    require(failures == {"count": 0, "samples": []}, "Ablation gate requires zero failures")
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    require(len(rows) == SAMPLES and len({row["index"] for row in rows}) == SAMPLES
            and len({row["video_sha256"] for row in rows}) == SAMPLES, "Incomplete or duplicate prediction identities")
    with (directory / "per_class.csv").open(newline="", encoding="utf-8") as stream:
        per_class = list(csv.DictReader(stream))
    require(len(per_class) == CLASSES and [int(row["class_id"]) for row in per_class] == list(range(CLASSES)), "Invalid class inventory/order")
    vocabulary = {int(row["class_id"]): row["gloss"] for row in per_class}
    require(len(set(vocabulary.values())) == CLASSES, "Duplicated class glosses")
    for row in rows:
        require(row["status"] == "ok" and row["split"] == "val", "Failed or non-validation prediction")
        require(len(row["top5"]) == 5 and len({p["class_id"] for p in row["top5"]}) == 5, "Invalid Top-5 inventory")
        probabilities = []
        for prediction in row["top5"]:
            require(type(prediction["class_id"]) is int and vocabulary.get(prediction["class_id"]) == prediction["gloss"], "Predicted class ID/gloss mismatch")
            value = prediction["probability"]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, "Invalid prediction probability")
            probabilities.append(value)
        require(probabilities == sorted(probabilities, reverse=True) and sum(probabilities) <= 1 + 1e-6, "Top-5 probabilities are invalid")
        for field in ("logits_sha256", "video_sha256"):
            require(isinstance(row[field], str) and len(row[field]) == 64 and all(c in "0123456789abcdef" for c in row[field]), "Invalid prediction digest")
    recomputed, counts = summarize(rows, vocabulary)
    for field in ("attempted", "inferred", "failed", "top1_correct", "top5_correct", "per_instance", "per_class_observed", "per_class_full_vocabulary"):
        require(metrics[field] == recomputed[field], f"Prediction/metric disagreement: {field}")
    require(metrics["vocabulary_classes"] == CLASSES and metrics["missing_classes"] == [], "Missing vocabulary classes")
    for row in per_class:
        n, one, five = counts[row["gloss"]]
        expected = {"attempted": n, "inferred": n, "failed": 0, "top1_correct": one, "top5_correct": five,
                    "errors_top1": n - one, "errors_top5": n - five}
        require(n > 0, "Missing validation class")
        require(all(int(row[key]) == value for key, value in expected.items())
                and float(row["top1"]) == one / n and float(row["top5"]) == five / n, "Per-class table differs from predictions")
    fingerprints["summary.json"] = sha(directory / "summary.json")
    return {"directory": str(directory), "summary": summary, "metrics": recomputed, "rows": rows,
            "vocabulary": vocabulary, "counts": counts, "sha256": fingerprints}


def paired(candidate, reference):
    require([identity(row) for row in candidate] == [identity(row) for row in reference], "Paired comparison identities/order differ")
    result = {}
    for k in (1, 5):
        fixed = sum(is_correct(a, k) and not is_correct(b, k) for a, b in zip(candidate, reference))
        regressed = sum(not is_correct(a, k) and is_correct(b, k) for a, b in zip(candidate, reference))
        result[f"top{k}"] = {"fixed": fixed, "regressed": regressed, "net": fixed - regressed,
                               "delta_percentage_points": (fixed - regressed) / len(candidate) * 100}
    return result


def verify_permutation(path, recipient_rows, shuffled_rows):
    document = read(path)
    n = len(recipient_rows)
    require(document["schema_version"] == 1 and document["seed"] == SHUFFLE_SEED
            and document["algorithm"] == ALGORITHM and document["sample_count"] == n
            and document.get("labels_used") is False, "Unregistered donor permutation specification")
    mapping = document["mapping"]
    require(len(mapping) == n and canonical_hash(mapping) == document["mapping_sha256"], "Donor mapping digest/count mismatch")
    order = list(range(n))
    random.Random(SHUFFLE_SEED).shuffle(order)
    donors = {order[i]: order[(i + 1) % n] for i in range(n)}
    expected = []
    for position, recipient in enumerate(recipient_rows):
        donor_position = donors[position]
        donor = recipient_rows[donor_position]
        expected.append({"recipient_position": position, "donor_position": donor_position,
                         "recipient_index": recipient["index"], "donor_index": donor["index"],
                         "recipient_video_sha256": recipient["video_sha256"], "donor_video_sha256": donor["video_sha256"]})
        require(donor_position != position, "Donor permutation has a fixed point")
        require(shuffled_rows[position].get("geometry_donor_index") == donor["index"]
                and shuffled_rows[position].get("geometry_donor_video_sha256") == donor["video_sha256"], "Prediction donor annotation differs from fixed permutation")
    require(mapping == expected, "Donors were not generated by the fixed label-independent algorithm")
    return {"status": "verified", "seed": SHUFFLE_SEED, "algorithm": ALGORITHM,
            "sample_count": n, "fixed_points": 0, "unique_donors": len(set(donors.values())),
            "labels_used": False, "mapping_sha256": document["mapping_sha256"], "file_sha256": sha(path),
            "interpretation": "One fixed label-independent derangement, not a permutation significance test. Donor masks and temporal information are also mismatched."}


def evaluate_gate(packs):
    normal = packs["normal"]
    comparisons = {}
    for condition in ("off", "shuffled"):
        control = packs[condition]
        comparisons[condition] = {
            "instance_top1_strictly_higher": normal["metrics"]["top1_correct"] > control["metrics"]["top1_correct"],
            "macro_top1_strictly_higher": exact_macro_top1(normal["counts"]) > exact_macro_top1(control["counts"])}
    supported = all(all(row.values()) for row in comparisons.values())
    return {"supported": supported, "comparisons": comparisons,
            "decision": "eligible_for_two_additional_paired_seeds" if supported else "stop_before_additional_seed_training",
            "rule": "After complete-cohort, zero-failure and exact normal-reference checks, normal must strictly exceed both off and shuffled on instance Top-1 and full-vocabulary macro Top-1. Macro inequalities use exact rational counts; no rounding or tolerance.",
            "statistical_significance_claimed": False,
            "interpretation": "This fixed-checkpoint mechanism gate is not evidence of stable improvement over baseline. No training is launched by this analyzer."}


def verify_engineering_acceptance(args, packs):
    parents = {Path(getattr(args, key)).resolve().parent for key in ("normal", "off", "shuffled")}
    require(len(parents) == 1, "Three conditions must belong to one finalized ablation")
    parent = next(iter(parents))
    require(Path(args.permutation).resolve() == parent / "permutation.json", "Permutation must belong to the same ablation")
    summary = read(parent / "summary.json")
    require(summary["status"] == "completed", "Ablation engineering acceptance has not completed")
    for field in ("normal_reference_exact", "weights_unchanged", "checkpoint_file_unchanged", "geometry_bank_unchanged"):
        require(summary.get(field) is True, f"Ablation engineering invariant failed: {field}")
    for field in ("optimizer_steps", "backward_calls"):
        require(type(summary.get(field)) is int and summary[field] == 0, "Ablation must perform no training or backward calls")
    model_hash = summary["initial_model_state_sha256"]
    require(model_hash == summary["final_model_state_sha256"] and len(model_hash) == 64, "Ablation model state changed")
    require(summary["checkpoint_sha256"] == summary["checkpoint_sha256_after"] == GEOMETRY_SHA, "Ablation checkpoint file changed")
    require(summary["geometry_bank_shape"] == [SAMPLES, 128, 28] and len(summary["geometry_bank_sha256"]) == 64,
            "Geometry bank is not the complete fixed evaluation sequence inventory")
    comparison = summary["normal_reference_comparison"]
    require(comparison["records_compared"] == SAMPLES and all(comparison.get(key) is True
            for key in ("exact", "identity_top5_probability_logits_exact", "metrics_exact", "per_class_exact")),
            "Parent normal-reference comparison is incomplete")
    for key in ("normal", "off", "shuffled"):
        require(Path(getattr(args, key)).resolve().name == key, "Unexpected condition directory name")
        require(summary["conditions"][key] == packs[key]["summary"], "Parent/child condition summary disagreement")
        require(summary["artifacts"][key + "/summary.json"]["sha256"] == packs[key]["sha256"]["summary.json"],
                "Parent condition summary SHA mismatch")
    permutation_hash = sha(args.permutation)
    require(summary["permutation_sha256"] == summary["artifacts"]["permutation.json"]["sha256"] == permutation_hash,
            "Parent permutation SHA mismatch")
    require(summary["permutation_mapping_sha256"] == read(args.permutation)["mapping_sha256"], "Parent mapping SHA mismatch")
    reference_hashes = packs["normal_reference"]["sha256"]
    require(summary["normal_reference"]["summary_sha256"] == reference_hashes["summary.json"]
            and summary["normal_reference"]["artifact_sha256"] == {key: value for key, value in reference_hashes.items() if key != "summary.json"},
            "Parent used a different normal reference")
    return {"status": "verified", "summary_path": str(parent / "summary.json"),
            "summary_sha256": sha(parent / "summary.json"), "weights_unchanged": True,
            "checkpoint_file_unchanged": True, "geometry_bank_unchanged": True,
            "initial_and_final_model_state_sha256": model_hash, "optimizer_steps": 0, "backward_calls": 0,
            "scope": "Finalized evaluator invariants and artifact hashes checked; outer run manifest/source freeze are verified separately."}


def analyze(args):
    paths = {key: getattr(args, key) for key in ("normal", "off", "shuffled", "normal_reference", "baseline_reference")}
    require(len({str(Path(path).resolve()) for path in paths.values()}) == len(paths), "Evaluation directories must be distinct")
    packs = {key: load_evaluation(path) for key, path in paths.items()}
    engineering = verify_engineering_acceptance(args, packs)
    reference = packs["normal_reference"]
    baseline = packs["baseline_reference"]
    require(reference["summary"]["method"] == "geometry" and reference["summary"]["checkpoint_sha256"] == GEOMETRY_SHA
            and reference["summary"]["checkpoint_epoch"] == 10, "Wrong pinned normal reference")
    require(baseline["summary"]["method"] == "baseline" and baseline["summary"]["checkpoint_sha256"] == BASELINE_SHA
            and baseline["summary"]["checkpoint_epoch"] == 2, "Wrong pinned baseline reference")
    identities = [identity(row) for row in reference["rows"]]
    for key, pack in packs.items():
        require([identity(row) for row in pack["rows"]] == identities, f"Cohort identity/order differs: {key}")
        require(pack["vocabulary"] == reference["vocabulary"], f"Vocabulary differs: {key}")
        if key in ("normal", "off", "shuffled"):
            summary = pack["summary"]
            require(summary.get("condition") == key, f"Condition metadata mismatch: {key}")
            for field in ("method", "method_metadata", "checkpoint_sha256", "checkpoint_epoch", "protocol_sha256",
                          "data", "geometry_manifest_sha256", "numerics"):
                require(summary[field] == reference["summary"][field], f"Fixed checkpoint/data/numerics differs: {key}.{field}")
    require(packs["normal"]["rows"] == reference["rows"], "Normal predictions/probabilities/logits do not exactly reproduce the existing best evaluation")
    require(read(Path(args.normal) / "metrics.json") == read(Path(args.normal_reference) / "metrics.json")
            and (Path(args.normal) / "per_class.csv").read_bytes() == (Path(args.normal_reference) / "per_class.csv").read_bytes(), "Normal metric artifacts do not exactly reproduce the reference")
    permutation = verify_permutation(args.permutation, packs["normal"]["rows"], packs["shuffled"]["rows"])
    comparisons = {"normal_vs_off": paired(packs["normal"]["rows"], packs["off"]["rows"]),
                   "normal_vs_shuffled": paired(packs["normal"]["rows"], packs["shuffled"]["rows"])}
    for key in ("normal", "off", "shuffled"):
        comparisons[key + "_vs_actual_baseline"] = paired(packs[key]["rows"], baseline["rows"])
    by_signer = {}
    for signer in sorted({row["signer_id"] for row in reference["rows"]}):
        slices = {key: [row for row in pack["rows"] if row["signer_id"] == signer]
                  for key, pack in packs.items() if key != "normal_reference"}
        by_signer[signer] = {"metrics": {key: summarize(rows, reference["vocabulary"])[0] for key, rows in slices.items()},
                             "normal_vs_off": paired(slices["normal"], slices["off"]),
                             "normal_vs_shuffled": paired(slices["normal"], slices["shuffled"]),
                             "normal_vs_actual_baseline": paired(slices["normal"], slices["baseline_reference"])}
    return {"schema_version": 1, "status": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analyzer_sha256": sha(Path(__file__)), "normal_reference_exact": True,
            "full_cohort_identity_and_order_equal": True, "samples": SAMPLES, "classes": CLASSES,
            "identity_sha256": canonical_hash(identities), "donor_permutation": permutation, "engineering_acceptance": engineering,
            "sources": {key: {"directory": pack["directory"], "sha256": pack["sha256"],
                              "checkpoint_sha256": pack["summary"]["checkpoint_sha256"]} for key, pack in packs.items()},
            "metrics": {key: pack["metrics"] for key, pack in packs.items() if key != "normal_reference"},
            "paired_comparisons": comparisons, "by_signer": by_signer, "gate": evaluate_gate(packs),
            "limitations": ["Off is the jointly trained geometry model with its branch removed, not the independently trained baseline.",
                            "Ablation loss can reflect coadaptation; mismatched geometry, masks and timing are out of distribution.",
                            "One chosen checkpoint, one training seed and one fixed donor permutation do not establish reproducible, causal or statistically significant gains.",
                            "Signer breakdowns are descriptive and are not additional selection or continuation gates.",
                            "All development is on validation; there are no test results or new training in this analysis."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal", "off", "shuffled", "normal-reference", "baseline-reference", "permutation"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Create a new report; existing files are never overwritten")
    args = parser.parse_args()
    try:
        report = analyze(args)
        code = 0
    except Exception as error:
        report = {"schema_version": 1, "status": "invalid_evidence", "error_type": type(error).__name__,
                  "error": str(error), "gate": {"supported": False, "decision": "do_not_train_with_invalid_evidence"}}
        code = 1
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
