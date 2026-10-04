#!/usr/bin/env python3
"""CPU-only learning example: synthetic poses -> descriptors -> three classifiers.

These invented trajectories are not ASL words or recorded people. The example
checks interfaces, train-only scaling, fitting and model reloading. Its output
is not evidence about recognition accuracy on ASL Citizen or user videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from classical_features import extract_features
from classical_models import (fit_classifier, fit_train_scaler, load_model,
                              predict_topk, save_model, transform_features)

LABELS = ("SYNTH_HORIZONTAL", "SYNTH_VERTICAL", "SYNTH_CIRCULAR")


def synthetic_pose(label: int, seed: int, frames: int = 16) -> dict:
    """Return correctly shaped raw-pose arrays with a different left-hand path."""
    rng = np.random.default_rng(seed)
    raw = {name: np.zeros((frames, points, 3), dtype=np.float64)
           for name, points in (("body_pose", 33), ("left_hand", 21), ("right_hand", 21))}
    raw.update({name + "_present": np.ones(frames, dtype=bool) for name in tuple(raw)})
    raw["body_visibility"] = np.ones((frames, 33), dtype=np.float64)
    for point, xy in {0: (.5, .2), 11: (.3, .5), 12: (.7, .5), 13: (.2, .6),
                      14: (.8, .6), 15: (.2, .3), 16: (.8, .3)}.items():
        raw["body_pose"][:, point, :2] = xy
    for name, x in (("left_hand", .2), ("right_hand", .8)):
        raw[name][..., :2] = [x, .3]
        for point in range(1, 21):
            raw[name][:, point, :2] += [.004 * point, -.003 * point]
    t = np.linspace(0, 1, frames)
    amplitude = .15 + rng.uniform(-.015, .015)
    if label == 0:
        trajectory = np.column_stack((amplitude * t, np.zeros(frames)))
    elif label == 1:
        trajectory = np.column_stack((np.zeros(frames), amplitude * t))
    elif label == 2:
        trajectory = amplitude * np.column_stack((np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)))
    else:
        raise ValueError("Unknown synthetic class")
    raw["left_hand"][..., :2] += trajectory[:, None, :]
    # Camera translation is unrelated to the synthetic trajectory label.
    offset = rng.uniform(-.03, .03, size=2)
    for name in ("body_pose", "left_hand", "right_hand"):
        raw[name][..., :2] += offset
    return raw


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    train_labels = np.repeat(np.arange(3, dtype=np.int64), 8)
    train = np.stack([extract_features(synthetic_pose(int(label), 100 + i))
                      for i, label in enumerate(train_labels)])
    query = np.stack([extract_features(synthetic_pose(label, 900 + label)) for label in range(3)])
    scaler = fit_train_scaler(train, split="train")
    x, q = transform_features(scaler, train), transform_features(scaler, query)
    summary = {"purpose": "synthetic_engineering_demo_only", "real_asl_data_used": False,
               "recognition_performance_claim": False, "train_shape": list(train.shape),
               "query_shape": list(query.shape), "scaler_fit_split": "train", "methods": {}}
    for method in ("knn", "linear_svm", "random_forest"):
        # Tiny examples use fewer forest trees; these are not the formal settings.
        options = {"n_estimators": 8, "min_samples_leaf": 1, "n_jobs": 1} if method == "random_forest" else {}
        model = fit_classifier(method, x, train_labels, 3, **options)
        before = predict_topk(model, q, device="cpu")
        path = output / (method + ".joblib")
        digest = save_model(model, path, scaler=scaler)
        restored, saved_scaler = load_model(path, expected_sha256=digest)
        after = predict_topk(restored, transform_features(saved_scaler, query), device="cpu")
        np.testing.assert_array_equal(before["topk_indices"], after["topk_indices"])
        np.testing.assert_array_equal(before["topk_scores"], after["topk_scores"])
        summary["methods"][method] = {
            "predicted_synthetic_labels": [LABELS[int(i)] for i in after["topk_indices"][:, 0]],
            "score_kind": after["score_kind"], "reload_exact": True,
            "model_sha256": digest, "parameters": model.parameters,
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional new directory; defaults to a temporary directory")
    args = parser.parse_args()
    if args.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="asl-synthetic-") as temporary:
            summary = run(Path(temporary) / "demo")
    else:
        summary = run(args.output_dir)
    print(json.dumps(summary, indent=2))
    print("Synthetic interface check only. No ASL recognition performance was measured.")


if __name__ == "__main__":
    main()
