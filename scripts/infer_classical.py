#!/usr/bin/env python3
"""Predict one isolated ASL word from an existing raw pose cache.

Example (use the project's pinned classical Python dependencies):
  python scripts/infer_classical.py --experiment-dir /path/to/formal-experiment \
      --pose-path /path/to/raw_pose.npz --method knn --device cpu

This does not decode video or extract MediaPipe landmarks. It never reads a
sample's target gloss, signer or dataset-row metadata. The experiment vocabulary
only converts predicted class IDs into readable words. A completed individual
method is usable even while sibling methods are still running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from classical_features import extract_features, feature_spec
from classical_models import load_model, predict_topk, transform_features

METHODS = ("knn", "linear_svm", "random_forest")
POSE_ARRAYS = ("body_pose", "body_pose_present", "body_visibility",
               "left_hand", "left_hand_present", "right_hand", "right_hand_present")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def infer(experiment_dir: str | Path, pose_path: str | Path, method: str,
          device: str = "cpu") -> dict:
    """Load one completed, checksum-verified model and predict without labels."""
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    experiment = Path(experiment_dir).resolve()
    pose = Path(pose_path).resolve()
    plan = _read_json(experiment / "plan.json")
    spec = feature_spec()
    if plan.get("feature_spec") != spec:
        raise ValueError("Saved feature_spec differs from current feature implementation")
    if plan.get("scaler_fit_split") != "train":
        raise ValueError("Experiment does not declare train-only scaler fitting")
    if method not in plan.get("methods", []):
        raise ValueError("Requested method is not declared in the experiment plan")
    dataset_path = experiment / "dataset.json"
    expected_dataset = plan.get("prepared_sha256", {}).get("dataset.json")
    if not isinstance(expected_dataset, str) or sha256(dataset_path) != expected_dataset:
        raise ValueError("Saved dataset/vocabulary SHA256 mismatch")
    dataset = _read_json(dataset_path)
    if dataset.get("feature_spec") != spec:
        raise ValueError("Saved dataset feature_spec differs from current feature implementation")
    labels = dataset.get("id_to_label")
    if (not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Invalid class-ID-to-word vocabulary")
    state = _read_json(experiment / method / "state.json")
    if state.get("status") != "completed" or state.get("method") != method:
        raise ValueError("Requested method must have its own completed state")
    digest = state.get("model_sha256")
    if not isinstance(digest, str):
        raise ValueError("Completed method is missing model_sha256")
    started = time.perf_counter()
    model, scaler = load_model(experiment / method / "model.joblib", expected_sha256=digest)
    if model.method != method or model.num_classes != len(labels) or model.feature_dim != spec["dimension"]:
        raise ValueError("Model method, feature dimension or class mapping disagrees with experiment")
    if scaler is None:
        raise ValueError("Saved model does not include its train-fitted scaler")
    load_seconds = time.perf_counter() - started
    pose_digest = sha256(pose)
    # Deliberately load only required numeric pose arrays. Labels or other
    # metadata in the archive are neither accessed nor passed to the model.
    with np.load(pose, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in POSE_ARRAYS}
    started = time.perf_counter()
    descriptor = extract_features(raw)
    standardized = transform_features(scaler, descriptor[None])
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = predict_topk(model, standardized, top_k=5,
                          device=device if method == "knn" else "cpu",
                          batch_size=128, reference_chunk_size=8192)
    predict_seconds = time.perf_counter() - started
    candidates = []
    for index, score, valid in zip(result["topk_indices"][0], result["topk_scores"][0], result["valid_mask"][0]):
        if valid:
            class_id = int(index)
            candidates.append({"rank": len(candidates) + 1, "class_id": class_id,
                               "word": labels[class_id], "score": float(score)})
    if not candidates:
        raise ValueError("Model produced no supported candidate")
    output = {
        "status": "completed", "task": "isolated_asl_word_classification",
        "method": method, "experiment_dir": str(experiment),
        "experiment_purpose": plan.get("purpose"), "pose_path": str(pose),
        "pose_sha256": pose_digest, "model_sha256": digest,
        "dataset_sha256": expected_dataset,
        "feature_version": spec["version"], "feature_dimension": spec["dimension"],
        "frame_count": int(len(raw["body_pose"])), "num_classes": len(labels),
        "prediction": candidates[0]["word"], "topk": candidates,
        "score_kind": result["score_kind"], "backend": result["backend"],
        "requested_top_k": 5, "returned_candidates": len(candidates),
        "fewer_than_five_supported_classes": len(candidates) < 5,
        "uses_target_label": False, "raw_video_processed": False,
        "all_pose_parts_missing": all(not raw[name].any() for name in
                                      ("body_pose_present", "left_hand_present", "right_hand_present")),
        "timings_seconds": {"model_load": load_seconds, "feature_and_scaler": feature_seconds,
                            "classifier": predict_seconds},
        "interpretation": "One isolated-word prediction, not sentence translation or an accuracy estimate. "
                          "kNN vote fractions and SVM margins are not calibrated confidence probabilities.",
    }
    if "zero_query_mask" in result:
        output["zero_standardized_query"] = bool(result["zero_query_mask"][0])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--pose-path", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="cpu", help="kNN device; SVM/RF always use CPU")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        result = infer(args.experiment_dir, args.pose_path, args.method, args.device)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x") as stream:
                stream.write(payload)
        print(payload, end="")
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
