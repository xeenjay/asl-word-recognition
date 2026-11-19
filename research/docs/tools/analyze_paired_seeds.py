"""Recompute the fixed three paired-seed comparisons from saved predictions.

No model loading, fitting, inference, threshold search, or test-set access occurs.
The original development seed and the two subsequent confirmation seeds are
reported both together and separately. Full remote artifact/checkpoint auditing
is a separate task; this tool verifies the precise lightweight inputs it uses.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

PLAN_SHA = "45c546d016480496a068696334684518b0a10eeb5d08819c4d757099b93bcb84"
COHORT_SHA = "a829029d700e50dd07d0f88454d9f3d42a65e946586a1188e7dfe9a81d5a5c86"
SEEDS = (20260922, 20260923, 20260924)
KEYS = ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")
OLD_EVAL = {
    "baseline": "20260925T151912Z-best-val-baseline-seed20260922-c64c6e9b",
    "geometry": "20260925T185353Z-grouped-lr-geometry-best-val-cce9cd41",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(row):
    return tuple(row[key] for key in KEYS)


def score(rows, cohort, expected_classes):
    require(len(rows) == len(cohort) and rows, "Prediction denominator differs or is empty")
    require([identity(row) for row in rows] == [identity(row) for row in cohort],
            "Prediction identities or order differ from the frozen complete cohort")
    require(len({row["index"] for row in rows}) == len(rows), "Duplicate sample index")
    classes = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        require(row.get("status") == "ok" and len(row["top5"]) == 5, "Failed/malformed prediction")
        require(row["split"] == "val", "Only validation predictions are allowed")
        labels = [item["gloss"] for item in row["top5"]]
        require(len(set(labels)) == 5, "Duplicate top-5 label")
        probabilities = [item["probability"] for item in row["top5"]]
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), "Invalid probability")
        bucket = classes[row["gloss"]]
        bucket[0] += 1
        bucket[1] += labels[0] == row["gloss"]
        bucket[2] += row["gloss"] in labels
    require(len(classes) == expected_classes, "Incomplete vocabulary")
    counts = [sum(item[index] for item in classes.values()) for index in (1, 2)]
    return {"samples": len(rows), "classes": len(classes), "failed": 0,
            "top1_correct": counts[0], "top5_correct": counts[1],
            "top1_percent": 100 * counts[0] / len(rows),
            "top5_percent": 100 * counts[1] / len(rows),
            "macro_top1_percent": 100 * statistics.fmean(item[1] / item[0] for item in classes.values()),
            "macro_top5_percent": 100 * statistics.fmean(item[2] / item[0] for item in classes.values())}


def paired(before, after):
    require(len(before) == len(after) and len(before) > 0, "Paired denominator mismatch")
    fixed = regressed = changed = 0
    signers = defaultdict(lambda: {"samples": 0, "baseline_correct": 0, "geometry_correct": 0,
                                   "incorrect_to_correct": 0, "correct_to_incorrect": 0})
    for left, right in zip(before, after):
        require(identity(left) == identity(right), "Paired identities differ")
        a, b = left["top5"][0]["gloss"], right["top5"][0]["gloss"]
        target = left["gloss"]
        fixed += a != target and b == target
        regressed += a == target and b != target
        changed += a != b
        bucket = signers[left["signer_id"]]
        bucket["samples"] += 1
        bucket["baseline_correct"] += a == target
        bucket["geometry_correct"] += b == target
        bucket["incorrect_to_correct"] += a != target and b == target
        bucket["correct_to_incorrect"] += a == target and b != target
    for bucket in signers.values():
        bucket["net_correct"] = bucket["geometry_correct"] - bucket["baseline_correct"]
    return {"incorrect_to_correct": fixed, "correct_to_incorrect": regressed,
            "net_correct": fixed - regressed, "top1_prediction_changes": changed,
            "top1_change_percentage_points": 100 * (fixed - regressed) / len(before),
            "by_signer_descriptive_only": dict(sorted(signers.items()))}


def aggregate(results):
    require(len(results) >= 2, "At least two paired seeds are required for sample SD")
    fields = ("top1_percent", "macro_top1_percent", "top5_percent", "macro_top5_percent")
    out = {"seeds": [row["seed"] for row in results], "n_seeds": len(results),
           "standard_deviation_definition": "sample standard deviation across seeds (ddof=1)",
           "units": "percentage points; method means are percentages"}
    for method in ("baseline", "geometry"):
        out[method] = {}
        for field in fields:
            values = [row[method]["recomputed"][field] for row in results]
            out[method][field] = {"mean": statistics.fmean(values), "sample_sd": statistics.stdev(values)}
    out["paired_difference"] = {}
    for field in fields:
        values = [row["geometry"]["recomputed"][field] - row["baseline"]["recomputed"][field]
                  for row in results]
        out["paired_difference"][field] = {"mean": statistics.fmean(values), "sample_sd": statistics.stdev(values),
                                            "per_seed": values, "all_strictly_positive": all(v > 0 for v in values),
                                            "has_sign_reversal": min(values) < 0 < max(values)}
    return out


def local_path(path, root, remote_root):
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(remote_root)
    require(".." not in path.parts, "Unsafe relative input path")
    return root / path


def analyze(root, plan_path):
    require(sha(plan_path) == PLAN_SHA, "Unexpected frozen plan")
    plan = read(plan_path)
    remote_root = Path(plan["workspace"])
    require(plan["seeds"] == list(SEEDS[1:]) and plan["test_used"] is False, "Unexpected seed/test scope")
    cohort_path = root / plan["validation_manifest"]
    require(sha(cohort_path) == COHORT_SHA, "Frozen validation cohort changed")
    cohort = read(cohort_path)["samples"]
    require(len(cohort) == 10304 and len({row["gloss"] for row in cohort}) == 2731,
            "Incomplete official validation cohort")
    references = {SEEDS[0]: {method: (root / "runs" / plan["seed20260922_references"][method],
                                     root / "runs" / OLD_EVAL[method]) for method in OLD_EVAL}}
    for lane in ("gpu0", "gpu1"):
        state = read(plan_path.parent / lane / "state.json")
        require(state["state"] == "completed" and state["plan_sha256"] == PLAN_SHA, "Lane not completed")
        require([job["seed"] for job in state["jobs"]] == list(SEEDS[1:]), "Lane seed list differs")
        for job in state["jobs"]:
            require(job["state"] == job["training"]["state"] == job["evaluation"]["state"] == "completed",
                    "Job training or evaluation incomplete")
            method = "baseline" if lane == "gpu0" else "geometry"
            require(job["method"] == method, "Wrong lane method")
            references.setdefault(job["seed"], {})[method] = tuple(
                local_path(job[stage]["run_dir"], root, remote_root) for stage in ("training", "evaluation"))
    inputs = []

    def checked(run, relative):
        index = read(run / "artifacts.json")["files"]
        entry = next(item for item in index if item["path"] == relative)
        path = run / relative
        require(path.stat().st_size == entry["size_bytes"] and sha(path) == entry["sha256"],
                "Analysis artifact changed: " + str(path))
        inputs.append({"path": str(path.relative_to(root)), "sha256": entry["sha256"]})
        return path

    results = []
    for seed in SEEDS:
        result = {"seed": seed, "role": "development_seed" if seed == SEEDS[0] else "additional_confirmation_seed"}
        predictions = {}
        for method in ("baseline", "geometry"):
            train, evaluation = references[seed][method]
            for run in (train, evaluation):
                manifest = read(run / "manifest.json")
                require(manifest["state"] == "completed" and manifest["returncode"] == 0
                        and (run / "exit_code.txt").read_text().strip() == "0", "Run incomplete")
            training = read(checked(train, "training/summary.json"))
            history = read(checked(train, "training/metrics_history.json"))["epochs"]
            require([row["epoch"] for row in history] == list(range(11)), "Incomplete epoch history")
            require((training["completed_epoch"], training["global_optimizer_step"], training["samples_seen"])
                    == (10, 3140, 401540), "Unequal training budget")
            cfg = training["config"]
            require(cfg["method"] == method and cfg["seed"] == cfg["head_seed"] == seed, "Wrong method/seed")
            best = max(history, key=lambda row: row["metrics"]["per_class_full_vocabulary"]["top1"])
            require(training["best"]["epoch"] == best["epoch"], "Best selection differs")
            summary = read(checked(evaluation, "evaluation/summary.json"))
            metrics = read(checked(evaluation, "evaluation/metrics.json"))
            checkpoint_index = read(checked(train, "training/checkpoints/index.json"))
            require(summary["checkpoint_sha256"] == checkpoint_index["best"]["sha256"]
                    and summary["checkpoint_epoch"] == best["epoch"]
                    and summary["protocol_sha256"] == training["protocol_sha256"]
                    and summary["method"] == method and summary["scope"] == "official_validation"
                    and summary["attribute_targets_read"] is False, "Evaluation linkage differs")
            require(metrics == summary["metrics"] == best["metrics"], "Stored metrics differ")
            rows = jsonl(checked(evaluation, "evaluation/predictions.jsonl"))
            internal = jsonl(checked(train, "training/evaluation/epoch-%04d/predictions.jsonl" % best["epoch"]))
            require([{k: v for k, v in row.items() if k != "status"} for row in rows] == internal,
                    "Independent predictions/probabilities/logits differ from selected epoch")
            measured = score(rows, cohort, 2731)
            require(metrics["attempted"] == metrics["inferred"] == 10304 and metrics["failed"] == 0,
                    "Stored denominator/failed count invalid")
            for level, prefix in (("per_instance", ""), ("per_class_full_vocabulary", "macro_")):
                for top in ("top1", "top5"):
                    require(abs(100 * metrics[level][top] - measured[prefix + top + "_percent"]) < 1e-9,
                            "Independent metric recomputation differs")
            require(metrics["top1_correct"] == measured["top1_correct"]
                    and metrics["top5_correct"] == measured["top5_correct"], "Correct counts differ")
            predictions[method] = rows
            result[method] = {"training_run": train.name, "evaluation_run": evaluation.name,
                              "best_epoch": best["epoch"], "checkpoint_sha256": summary["checkpoint_sha256"],
                              "recomputed": measured}
        result["paired"] = paired(predictions["baseline"], predictions["geometry"])
        result["paired"]["macro_top1_change_percentage_points"] = (
            result["geometry"]["recomputed"]["macro_top1_percent"] - result["baseline"]["recomputed"]["macro_top1_percent"])
        results.append(result)
    return {"schema_version": 1, "status": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "saved_predictions_of_validation_selected_best_checkpoints", "test_used": False,
            "analyzer_sha256": sha(__file__), "plan_sha256": PLAN_SHA, "validation_cohort_sha256": COHORT_SHA,
            "verified_input_artifacts": inputs, "paired_seeds": results,
            "three_seeds_descriptive": aggregate(results), "two_additional_seeds_descriptive": aggregate(results[1:]),
            "interpretation_limits": ["Same fixed validation cohort, not independent datasets or test scores.",
                "Seed 20260922 informed development; additional two seeds are shown separately.",
                "Best epoch selected by validation macro Top1; epochs are not independent seeds.",
                "Means and sample SD are descriptive, not a significance test or confidence interval.",
                "Positive branch ablation effects are not gains over an independently trained baseline.",
                "The fixed budget is exhausted; no automatic search, extra seeds, epochs, combined model or test.",
                "No quantitative small-gain cutoff was preregistered; do not invent one after results.",
                "This verifies referenced lightweight analysis inputs; full remote payload audit is separate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", "--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--plan", type=Path, default=Path("runs/paired-seed-launches/20260925T221436Z/plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    plan = args.plan if args.plan.is_absolute() else root / args.plan
    report = analyze(root, plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "sha256": sha(args.output),
                      "three_seeds_descriptive": report["three_seeds_descriptive"],
                      "two_additional_seeds_descriptive": report["two_additional_seeds_descriptive"]}))


if __name__ == "__main__":
    main()
