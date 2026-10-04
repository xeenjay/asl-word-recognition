#!/usr/bin/env python3
"""Independently verify saved classical-baseline predictions without ML imports.

This checks membership, label order, full denominators, and reported metrics. It
does not prove the training algorithm or feature extraction implementation;
those require their own tests and model-reload prediction checks. No test split
is accepted. The resulting receipt is append-only (an existing one is refused).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import pickle

WORKSPACE = Path(__file__).resolve().parents[1]
METHODS = ("knn", "linear_svm", "random_forest")
IDENTITY_KEYS = ("index", "gloss", "signer_id", "split", "source_path", "video_sha256")
CACHE_SHA256 = {
    "train": "0403e8ab7b2eaeae589749a30f67f68a6157615a2140623f3af9d40f02d8881c",
    "val": "a829029d700e50dd07d0f88454d9f3d42a65e946586a1188e7dfe9a81d5a5c86",
}
LABEL_MAP_SHA256 = {
    "d_LId2Label.pkl": "cd29470b041e1a029e62bc66f47313b08744f669b283b58cf647d532482268e6",
    "d_Label2LId.pkl": "a2d0f112d9c26cdd75fe9beaa10b54747df3793f4334e95539d70f2dde9e86dd",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes | str):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, parse_constant=reject, object_pairs_hook=pairs)


class PlainMappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected label-map global: {module}.{name}")


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path: Path, expected: str | None = None) -> bytes:
        path = path.resolve()
        data = path.read_bytes()
        sha = digest(data)
        if expected is not None:
            require(sha == expected, f"SHA256 mismatch: {path}")
        row = {"path": str(path), "size_bytes": len(data), "sha256": sha}
        previous = self.records.get(str(path))
        require(previous is None or previous == row, f"Input changed during verification: {path}")
        self.records[str(path)] = row
        return data

    def json(self, path: Path, expected: str | None = None):
        return strict_json(self.read(path, expected))

    def recheck(self):
        for row in list(self.records.values()):
            self.read(Path(row["path"]), row["sha256"])


def validate_labels(labels: list, inputs: Inputs, require_complete: bool):
    require(isinstance(labels, list) and labels, "id_to_label must be a nonempty ordered list")
    require(all(isinstance(x, str) and x for x in labels), "Labels must be nonempty strings")
    require(len(set(labels)) == len(labels), "Duplicate vocabulary label")
    if require_complete:
        require(len(labels) == 2731, "Formal evaluation requires 2731 vocabulary labels")
        maps = {}
        for name, sha in LABEL_MAP_SHA256.items():
            raw = inputs.read(WORKSPACE / "assets/checkpoints" / name, sha)
            maps[name] = PlainMappingUnpickler(io.BytesIO(raw)).load()
        require(maps["d_LId2Label.pkl"] == dict(enumerate(labels)), "Official class-ID order mismatch")
        require(maps["d_Label2LId.pkl"] == {label: i for i, label in enumerate(labels)},
                "Official inverse vocabulary mismatch")


def load_cohort(experiment: Path, role: str, spec: dict, labels: list,
                inputs: Inputs, require_complete: bool) -> list:
    require(isinstance(spec, dict), f"Missing dataset {role} record")
    name = spec.get("cache_manifest_path")
    require(isinstance(name, str) and name, f"Missing {role} cache_manifest_path")
    expected = spec.get("sha256")
    require(isinstance(expected, str) and len(expected) == 64, f"Missing {role} SHA256")
    if require_complete:
        require(expected == CACHE_SHA256[role], f"Unrecognized formal {role} cache manifest")
    path = Path(name)
    if not path.is_absolute():
        path = experiment / path
    index = inputs.json(path, expected)
    require(index.get("schema_version") == 1 and index.get("status") == "completed",
            f"{role} cache manifest must be finalized and completed")
    require(index.get("split") == role, f"Wrong split in {role} cache; test is forbidden")
    rows = index.get("samples")
    require(isinstance(rows, list) and rows, f"Missing {role} samples")
    selection = index.get("selection", {})
    require(selection.get("selected_count") == len(rows), f"{role} selection count mismatch")
    require(selection.get("indices") == [x.get("index") for x in rows], f"{role} selection order mismatch")
    seen = {key: set() for key in ("index", "source_path", "video_sha256")}
    vocabulary = set(labels)
    for row in rows:
        require(row.get("split") == role, f"Wrong sample split in {role}; test is forbidden")
        require(row.get("status") in {"ok", "reused"}, f"Failed {role} cache row cannot be excluded")
        require(row.get("gloss") in vocabulary, f"Unknown {role} target label")
        require(type(row.get("index")) is int and row["index"] >= 0, f"Invalid {role} index")
        for key in ("source_path", "signer_id", "video_sha256"):
            require(isinstance(row.get(key), str) and row[key], f"Missing {role} {key}")
        sha = row["video_sha256"]
        require(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), f"Invalid {role} video SHA256")
        for key, values in seen.items():
            require(row[key] not in values, f"Duplicate {role} {key}")
            values.add(row[key])
    if require_complete:
        require(len(rows) == {"train": 40154, "val": 10304}[role], f"Incomplete official {role} denominator")
        require({x["gloss"] for x in rows} == vocabulary, f"Incomplete {role} class coverage")
        require(len({x["signer_id"] for x in rows}) == {"train": 35, "val": 6}[role],
                f"Unexpected {role} signer count")
        require(selection.get("source_count") == len(rows), f"Formal {role} is a subset")
    return rows


def recompute_predictions(data: bytes, cohort: list, labels: list) -> tuple[dict, list]:
    lines = data.decode("utf-8").splitlines()
    require(len(lines) == len(cohort), "Prediction count differs from complete validation denominator")
    lookup = {label: i for i, label in enumerate(labels)}
    counts = [[0, 0, 0] for _ in labels]
    expected_k = min(5, len(labels))
    for position, (line, source) in enumerate(zip(lines, cohort)):
        row = strict_json(line)
        require(isinstance(row, dict), f"Prediction {position} must be an object")
        require(all(row.get(k) == source[k] for k in IDENTITY_KEYS),
                f"Prediction identity/order mismatch at row {position}")
        require(row.get("status") == "ok", f"Prediction {position} failed; cannot certify zero-failure run")
        ids, scores = row.get("top5_ids"), row.get("top5_scores")
        require(isinstance(ids, list) and 1 <= len(ids) <= expected_k,
                f"Prediction {position} needs 1 to {expected_k} ranked IDs")
        require(all(type(x) is int and 0 <= x < len(labels) for x in ids),
                f"Prediction {position} has invalid class IDs")
        require(len(set(ids)) == len(ids), f"Prediction {position} has duplicate ranked IDs")
        require(isinstance(scores, list) and len(scores) == len(ids), f"Prediction {position} score count mismatch")
        require(all(type(x) in (int, float) and math.isfinite(x) for x in scores),
                f"Prediction {position} scores must be finite numbers")
        require(all(a >= b for a, b in zip(scores, scores[1:])), f"Prediction {position} scores are not descending")
        target = lookup[source["gloss"]]
        counts[target][0] += 1
        counts[target][1] += int(ids[0] == target)
        counts[target][2] += int(target in ids)
    classes = []
    for class_id, (n, top1, top5) in enumerate(counts):
        classes.append({"class_id": class_id, "gloss": labels[class_id], "attempted": n,
                        "inferred": n, "failed": 0, "top1_correct": top1, "top5_correct": top5,
                        "errors_top1": n - top1, "errors_top5": n - top5,
                        "top1": top1 / n if n else None, "top5": top5 / n if n else None})
    observed = [x for x in classes if x["attempted"]]
    missing = [x["gloss"] for x in classes if not x["attempted"]]
    correct1 = sum(x[1] for x in counts)
    correct5 = sum(x[2] for x in counts)
    macro = {name: sum(x[name] for x in observed) / len(observed) for name in ("top1", "top5")}
    metrics = {"attempted": len(cohort), "inferred": len(cohort), "failed": 0,
               "top1_correct": correct1, "top5_correct": correct5,
               "denominator_policy": "all selected inputs; extraction/load/inference failures count as incorrect",
               "per_instance": {"top1": correct1 / len(cohort), "top5": correct5 / len(cohort)},
               "per_class_observed": {**macro, "denominator_classes": len(observed)},
               "per_class_full_vocabulary": {k: v if not missing else None for k, v in macro.items()},
               "vocabulary_classes": len(labels), "missing_classes": missing,
               "missing_class_policy": "Full-vocabulary macro is null when any class has no selected input; observed-class macro is labeled separately."}
    return metrics, classes


def compare(expected, actual, context: str):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{context} fields mismatch")
        for key in expected:
            compare(expected[key], actual[key], f"{context}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{context} length mismatch")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{context}[{i}]")
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12), f"{context} value mismatch")
    else:
        require(type(expected) is type(actual) and expected == actual, f"{context} value mismatch")


def verify_experiment(experiment_dir: Path, require_complete: bool = True) -> dict:
    root = Path(experiment_dir).resolve()
    receipt = root / "verification.json"
    if receipt.exists():
        raise FileExistsError(f"Refusing to overwrite existing verification: {receipt}")
    inputs = Inputs()
    inputs.read(Path(__file__))
    dataset = inputs.json(root / "dataset.json")
    plan = inputs.json(root / "plan.json")
    labels = dataset.get("id_to_label")
    validate_labels(labels, inputs, require_complete)
    configured = plan.get("methods")
    require(isinstance(configured, list), "Plan must list methods")
    names = [x if isinstance(x, str) else x.get("name") if isinstance(x, dict) else None for x in configured]
    require(len(names) == len(METHODS) and set(names) == set(METHODS), "Plan methods must be exactly kNN, Linear SVM and Random Forest")
    require(isinstance(plan.get("feature_spec"), dict) and plan["feature_spec"], "Plan needs a fixed feature_spec")
    train = load_cohort(root, "train", dataset.get("train"), labels, inputs, require_complete)
    val = load_cohort(root, "val", dataset.get("val"), labels, inputs, require_complete)
    for key in ("signer_id", "video_sha256", "source_path"):
        require(not ({x[key] for x in train} & {x[key] for x in val}), f"Train/validation overlap in {key}")
    results = {}
    for name in METHODS:
        directory = root / name
        metrics, classes = recompute_predictions(inputs.read(directory / "predictions.jsonl"), val, labels)
        compare(metrics, inputs.json(directory / "metrics.json"), f"{name}.metrics")
        compare(classes, inputs.json(directory / "per_class.json"), f"{name}.per_class")
        results[name] = metrics
    inputs.recheck()
    result = {"schema_version": 1, "status": "verified",
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "formal_complete_cohorts": require_complete, "test_used": False,
              "train_count": len(train), "validation_count": len(val), "num_classes": len(labels),
              "metrics": results, "inputs": sorted(inputs.records.values(), key=lambda x: x["path"]),
              "limitations": "Saved predictions and metrics independently verified; feature computation, model fitting and serialized-model reproducibility are verified separately. No model inference was performed by this verifier."}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--engineering-smoke", action="store_true",
                        help="Permit synthetic/small train/val cohorts; never permits test data")
    args = parser.parse_args()
    result = verify_experiment(args.experiment_dir, require_complete=not args.engineering_smoke)
    print(json.dumps({"status": result["status"], "validation_count": result["validation_count"],
                      "formal_complete_cohorts": result["formal_complete_cohorts"]}))


if __name__ == "__main__":
    main()
