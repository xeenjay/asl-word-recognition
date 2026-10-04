#!/usr/bin/env python3
"""Run official PoET ASL inference and retain raw landmarks for later research.

Example:
  python scripts/infer_video.py --video sample.mp4 --output-dir runs/sample \
      --checkpoint-dir checkpoints/asl_citizen --device cuda:0
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from _poet_runtime import (
    PARTS, load_bundle, normalize_raw_pose, prepare_features, require_finite,
    resolve_device, sha256, synchronize, validate_raw_pose, write_json,
)
from mediapipe_extraction import extract_mediapipe_points_from_video


def extract_raw_pose(video: Path) -> tuple[dict[str, np.ndarray], dict]:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open video: {video}")
        info = {
            "reported_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()
    if info["reported_frame_count"] <= 0 or not np.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError(f"Invalid video metadata for the official frame-indexed extractor: {info}")
    # Match upstream: no flip, static_image_mode=True, complexity=2, detection=.5.
    records = extract_mediapipe_points_from_video(str(video), flip=False)
    if len(records) != info["reported_frame_count"]:
        raise ValueError(f"Decoded {len(records)}/{info['reported_frame_count']} frames; refusing a partial video")
    if not records:
        raise ValueError("The video decoded no frames")
    raw = {name: np.zeros((len(records), points, 3), dtype=np.float64) for name, points in PARTS.items()}
    raw.update({name + "_present": np.zeros(len(records), dtype=np.bool_) for name in PARTS})
    raw["body_visibility"] = np.zeros((len(records), 33), dtype=np.float64)
    record_keys = {"body_pose": "body_pose", "face_mesh": "face_mesh", "right_hand": "right_hand_pose", "left_hand": "left_hand_pose"}
    for index, record in enumerate(records):
        for name, count in PARTS.items():
            points = record[record_keys[name]]
            if not points:
                continue
            if len(points) != count:
                raise ValueError(f"Frame {index}: {name} expected {count} landmarks, got {len(points)}")
            raw[name + "_present"][index] = True
            # Named fields avoid relying on protobuf dictionary insertion order.
            # Omitted scalar protobuf fields are their defined numeric default 0.
            raw[name][index] = [[point.get(axis, 0.0) for axis in ("x", "y", "z")] for point in points]
            if name == "body_pose":
                raw["body_visibility"][index] = [point.get("visibility", 0.0) for point in points]
    raw["frame_ids"] = np.array([r["frame_id"] for r in records], dtype=np.int64)
    raw["timestamps_seconds"] = raw["frame_ids"] / info["fps"]
    raw["schema_version"] = np.array(1, dtype=np.int64)
    validate_raw_pose(raw)
    require_finite("body visibility", raw["body_visibility"])
    info["decoded_frames"] = len(records)
    info["detection_counts"] = {name: int(raw[name + "_present"].sum()) for name in PARTS}
    return raw, info


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    video, output = args.video.resolve(), args.output_dir.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("manifest.json", "raw_pose.npz", "model_input.npz")) and not args.overwrite:
        raise FileExistsError(f"Output already contains a result; choose a new directory or --overwrite: {output}")
    manifest = {"schema_version": 1, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(), "video": str(video)}
    try:
        manifest["video_sha256"] = sha256(video)
        device = resolve_device(args.device)
        stamp = time.perf_counter()
        bundle = load_bundle(args.checkpoint_dir, device)
        synchronize(device)
        timings = {"load_model_seconds": time.perf_counter() - stamp}
        manifest["model"] = bundle.metadata
        stamp = time.perf_counter()
        raw, info = extract_raw_pose(video)
        timings["landmark_extraction_seconds"] = time.perf_counter() - stamp
        manifest["video_metadata"] = info
        raw_path = output / "raw_pose.npz"
        np.savez_compressed(raw_path, **raw)
        stamp = time.perf_counter()
        features = normalize_raw_pose(raw)
        prepared, indices = prepare_features(features, bundle.preprocessing, mode="eval")
        timings["normalization_and_temporal_preprocessing_seconds"] = time.perf_counter() - stamp
        input_path = output / "model_input.npz"
        np.savez_compressed(input_path, normalized_features=features, model_input=prepared.numpy(), source_frame_indices=indices)
        synchronize(device)
        stamp = time.perf_counter()
        with torch.inference_mode():
            logits = bundle.model(prepared.unsqueeze(0), device)
            require_finite("model logits", logits)
            probabilities = logits.softmax(dim=1)
            values, classes = probabilities.topk(min(5, bundle.checkpoint["num_classes"]), dim=1)
        synchronize(device)
        timings["model_forward_seconds"] = time.perf_counter() - stamp
        predictions = [{"rank": rank + 1, "class_id": class_id, "gloss": bundle.id_to_label[class_id], "probability": probability}
                       for rank, (class_id, probability) in enumerate(zip(classes[0].tolist(), values[0].tolist()))]
        manifest.update({
            "status": "ok", "top5": predictions, "model_input_shape": list(prepared.shape),
            "pose_path": str(raw_path), "pose_sha256": sha256(raw_path),
            "model_input_path": str(input_path), "model_input_sha256": sha256(input_path),
            "extraction": {"implementation": "upstream extract_mediapipe_points_from_video", "static_image_mode": True, "model_complexity": 2, "min_detection_confidence": 0.5, "flip": False, "coordinates": "MediaPipe normalized image xyz; absent articulators are zero with explicit presence masks"},
            "preprocessing": "upstream articulators_to_array and preprocess(mode=eval); temporal source-index trace checked exactly against unmodified upstream output",
            "interpretation": "Isolated-word predictions; softmax probability is not calibrated confidence. One-video smoke output is not dataset accuracy.",
        })
        timings["total_seconds"] = time.perf_counter() - started
        manifest["timings"] = timings
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update({"status": "error", "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.perf_counter() - started})
        write_json(output / "manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    for prediction in result["top5"]:
        print(f"{prediction['rank']}. {prediction['gloss']}: {prediction['probability']:.6f}")
    print(f"Manifest: {args.output_dir.resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
