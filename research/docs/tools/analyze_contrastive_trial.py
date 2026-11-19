"""Verify the completed two-arm pilot and matched batches before reporting scores.

Read-only except the explicitly requested new report. Requires the remote
checkpoints and complete run artifacts; never starts or resumes training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
import json
from pathlib import Path

from analyze_paired_seeds import score, paired, jsonl
from run_contrastive_lane import (ROOT, read, sha, inside, validate_plan,
                                 check_frozen, verify_training, verify_evaluation)

MATCHED_STEP_FIELDS = (
    "epoch", "global_optimizer_step", "samples_seen", "indices", "video_sha256",
    "temporal_seeds", "temporal_indices_sha256", "signer_ids", "class_ids",
    "positive_pairs_per_anchor", "batch_trace",
)


def compare_steps(left_path, right_path, *, expected_steps=3140, batch_size=128,
                  steps_per_epoch=314, classes_per_batch=32, signers_per_class=4):
    seen, by_epoch, draws = set(), {}, 0
    with Path(left_path).open() as left, Path(right_path).open() as right:
        for number, (a, b) in enumerate(zip_longest(left, right), 1):
            if a is None or b is None:
                raise ValueError("Comparison arms have different step counts")
            a, b = json.loads(a), json.loads(b)
            if any(a.get(key) != b.get(key) for key in MATCHED_STEP_FIELDS):
                raise ValueError(f"Batch/augmentation evidence differs at optimizer step {number}")
            epoch = (number - 1) // steps_per_epoch + 1
            if (a["global_optimizer_step"] != number or a["epoch"] != epoch
                    or a["samples_seen"] != number * batch_size):
                raise ValueError("Unexpected contiguous epoch/step/sample counters")
            hashes, labels, signers = a["video_sha256"], a["class_ids"], a["signer_ids"]
            vectors = (hashes, labels, signers, a["indices"], a["temporal_seeds"],
                       a["temporal_indices_sha256"], a["batch_trace"])
            if any(len(values) != batch_size for values in vectors) or len(set(hashes)) != batch_size:
                raise ValueError("Batch length or distinct-video contract differs")
            counts = Counter(labels)
            if len(counts) != classes_per_batch or set(counts.values()) != {signers_per_class}:
                raise ValueError("Batch class balance differs")
            for label in counts:
                if len({s for c, s in zip(labels, signers) if c == label}) != signers_per_class:
                    raise ValueError("Same-class batch examples do not have distinct signers")
            if a["positive_pairs_per_anchor"] != signers_per_class - 1:
                raise ValueError("Positive-pair count differs")
            seen.update(hashes)
            by_epoch.setdefault(epoch, set()).update(hashes)
            draws += batch_size
    if draws != expected_steps * batch_size:
        raise ValueError("Fixed pilot step/sample budget is incomplete")
    return {"optimizer_steps": expected_steps, "sample_visits": draws,
            "batch_and_temporal_augmentation_exactly_equal": True,
            "all_batch_classes_and_cross_signer_positives_valid": True,
            "unique_training_videos_across_budget": len(seen),
            "unique_training_videos_by_epoch": {str(k): len(v) for k, v in by_epoch.items()},
            "note": "Repeated visits are intentional; the training cohort size is not the unique count of sampled videos."}


def analyze(plan_path, root=ROOT):
    plan_path = inside(plan_path, root)
    plan = read(plan_path)
    plan_hash = sha(plan_path)
    check_frozen(plan_path, plan_hash, plan, root)
    cohort = read(root / plan["validation_manifest"])["samples"]
    train_cohort = read(root / plan["training_manifest"])["samples"]
    method_results, predictions, training_paths, inputs = {}, {}, {}, {}
    for lane_name in ("gpu0", "gpu1"):
        lane = validate_plan(plan, lane_name, root)
        state_path = plan_path.parent / lane_name / "state.json"
        state = read(state_path)
        if state["state"] != "completed" or state["plan_sha256"] != plan_hash:
            raise ValueError("Both independently verified lanes must be completed")
        item, job = lane["jobs"][0], state["jobs"][0]
        if job["job_id"] != item["job_id"] or job["state"] != "completed":
            raise ValueError("Completed lane job identity differs")
        training = inside(job["training"]["run_dir"], root)
        evaluation = inside(job["evaluation"]["run_dir"], root)
        evidence = verify_training(training, item, plan, root)
        proof = verify_evaluation(training, evaluation, item, evidence, plan, root)
        rows = jsonl(evaluation / "evaluation/predictions.jsonl")
        metrics = score(rows, cohort, 2731)
        method = item["method"]
        method_results[method] = {
            "training_run": str(training.relative_to(root)),
            "evaluation_run": str(evaluation.relative_to(root)),
            "selected_epoch": evidence["selected_epoch"],
            "best_checkpoint_sha256": evidence["best_checkpoint_sha256"],
            "recomputed": metrics, "independent_evaluation_verification": proof,
        }
        predictions[method], training_paths[method] = rows, training
        for path in [state_path, training / "training/summary.json", training / "training/protocol.json",
                     training / "training/metrics_history.json", training / "training/steps.jsonl",
                     evaluation / "evaluation/predictions.jsonl", evaluation / "evaluation/metrics.json"]:
            inputs[str(path.relative_to(root))] = sha(path)
    matched = compare_steps(training_paths["balanced_ce"] / "training/steps.jsonl",
                            training_paths["cross_signer_supcon"] / "training/steps.jsonl")
    allowed_train = {row["video_sha256"] for row in train_cohort}
    with (training_paths["balanced_ce"] / "training/steps.jsonl").open() as stream:
        for line in stream:
            if not set(json.loads(line)["video_sha256"]).issubset(allowed_train):
                raise ValueError("A sampled video is outside the official training cohort")
    matched["all_sampled_videos_in_official_train"] = True
    comparison = paired(predictions["balanced_ce"], predictions["cross_signer_supcon"])
    for signer in comparison["by_signer_descriptive_only"].values():
        signer["balanced_ce_correct"] = signer.pop("baseline_correct")
        signer["cross_signer_supcon_correct"] = signer.pop("geometry_correct")
    a, b = (method_results[m]["recomputed"] for m in ("balanced_ce", "cross_signer_supcon"))
    comparison["macro_top1_change_percentage_points"] = b["macro_top1_percent"] - a["macro_top1_percent"]
    comparison["top5_change_percentage_points"] = b["top5_percent"] - a["top5_percent"]
    check_frozen(plan_path, plan_hash, plan, root)
    return {"schema_version": 1, "status": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan_hash, "test_used": False, "seed": 20260926,
            "matched_training": matched, "methods": method_results, "paired_comparison": comparison,
            "input_sha256": inputs,
            "interpretation": "Single-seed development validation, selected by macro Top1 including epoch0. Compare SupCon with same-sampler CE; historical geometry/random-sampling baselines are context, not matched controls. No significance or test-generalization claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "paired_comparison": result["paired_comparison"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
