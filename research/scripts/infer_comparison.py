#!/usr/bin/env python3
"""Persistent, label-free five-method inference for one segmented ASL video.

The frozen 2026-10-02 classical models, official PoET checkpoint, and the
predeclared geometry seed 20260923 + train-memory reranker are loaded once.
This is a demonstration adapter: it performs no training or dataset evaluation.
All old source files, checkpoints, caches, and completed runs remain read-only.

Use the existing project environment and classical target dependencies:
  python scripts/infer_comparison.py --root . --device cpu \
      --video word.mp4 --output-dir runs/NEW-comparison
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

CLASSICAL_EXPERIMENT = "runs/20261002T161254Z-classical-formal-three-methods-dac0c02f/experiment"
CLASSICAL_PLAN_SHA = "6b9c55e41885629bc9aff5a758f4039e40c5e6d9ca4fbd69a869b3ece676f7f0"
CLASSICAL_MODEL_SHA = {
    "knn": "ceeab09372ba8af3eb34b4d7b72a4bb1bed68937d198509b2724937151e75b38",
    "linear_svm": "3262ee1e1ca341247a7dde57930dd30967ae1b629908afb3dbe518b86deeb0cc",
    "random_forest": "50a53f9e3a2b05205aac630ee9e2b9e0cbb2cc3d928017274a9fe160e104cdb3",
}
PREREGISTRATION = "docs/environment/factorial-retrieval-preregistration-20260926.json"
PREREGISTRATION_SHA = "8ff28628da3dc07374aa06ddf5d2d612ad8a5052c21485885ba42f81367244fc"
TRAIN_BANK = "runs/20260926T223315Z-factorial-retrieval-train-extraction-2a69042a/features/geometry_seed20260923/manifest.json"
TRAIN_BANK_SHA = "929f5f47e97f6f9c2e423a5f9fc5da06531474d5eb42229f0d334faa10d9dcbb"
GEOMETRY_MODEL_ID = "geometry_seed20260923"
CLASSICAL_METHODS = ("knn", "linear_svm", "random_forest")
METHODS = (*CLASSICAL_METHODS, "poet_official", "poet_geometry_retrieval")
PART_NAMES = ("body_pose", "face_mesh", "left_hand", "right_hand")
POSE_ARRAYS = (*PART_NAMES, *(name + "_present" for name in PART_NAMES), "body_visibility")
VIDEO_LIMITS = {"bytes": 25 * 1024 * 1024, "seconds": 30, "frames": 900,
                "width": 1920, "height": 1080}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_verified_json(path: Path, expected: str) -> dict:
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def read_pose(path: Path) -> dict:
    """Access only numeric model inputs, never embedded gloss/target metadata."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in POSE_ARRAYS}


def detection_coverage(raw: dict) -> dict:
    frames = len(raw["body_pose"])
    if frames == 0:
        raise ValueError("Pose contains no frames")
    masks = {name: raw[name + "_present"] for name in PART_NAMES}
    if any(mask.shape != (frames,) or mask.dtype != np.bool_ for mask in masks.values()):
        raise ValueError("Detection presence masks must be boolean frame vectors")
    masks["both_hands"] = masks["left_hand"] & masks["right_hand"]
    masks["any_hand"] = masks["left_hand"] | masks["right_hand"]
    return {"frame_count": frames,
            **{name: {"detected_frames": int(mask.sum()), "fraction": float(mask.mean())}
               for name, mask in masks.items()},
            "all_hands_missing": not bool(masks["any_hand"].any()),
            "all_parts_missing": not any(bool(masks[name].any()) for name in PART_NAMES)}


def validate_video_metadata(info: dict, file_bytes: int) -> None:
    """Bound the demo before the expensive upstream frame-by-frame extractor."""
    if file_bytes <= 0 or file_bytes > VIDEO_LIMITS["bytes"]:
        raise ValueError("Video must be nonempty and at most 25 MiB")
    for key in ("reported_frame_count", "fps", "width", "height"):
        if not np.isfinite(info[key]) or info[key] <= 0:
            raise ValueError(f"Invalid video metadata: {key}")
    if info["reported_frame_count"] > VIDEO_LIMITS["frames"]:
        raise ValueError("The isolated-word demo accepts at most 900 frames")
    if info["reported_frame_count"] / info["fps"] > VIDEO_LIMITS["seconds"]:
        raise ValueError("The isolated-word demo accepts at most 30 seconds")
    if info["width"] > VIDEO_LIMITS["width"] or info["height"] > VIDEO_LIMITS["height"]:
        raise ValueError("Video resolution must be at most 1920 by 1080")


def inspect_video(path: Path) -> dict:
    import cv2
    size = path.stat().st_size
    if size <= 0 or size > VIDEO_LIMITS["bytes"]:
        raise ValueError("Video must be nonempty and at most 25 MiB")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video")
        info = {"reported_frame_count": float(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        capture.release()
    validate_video_metadata(info, size)
    return {**info, "duration_seconds": info["reported_frame_count"] / info["fps"],
            "file_bytes": size}


def candidate_list(ids, scores, labels: dict | list, valid=None) -> list[dict]:
    supported = [True] * len(ids) if valid is None else valid
    result = []
    for class_id, score, keep in zip(ids, scores, supported):
        if keep:
            if not np.isfinite(score):
                raise ValueError("Nonfinite candidate score")
            index = int(class_id)
            result.append({"rank": len(result) + 1, "class_id": index,
                           "word": labels[index], "score": float(score)})
    if not result:
        raise ValueError("Classifier returned no supported candidates")
    return result


class ComparisonEngine:
    """One persistent engine; predictions serialize under an internal lock.

    ``output_dir`` must not already exist. All predictions retain their numeric
    inputs and a JSON result there. Model-load cost is reported separately from
    warm per-video latency. CPU works for every model; CUDA accelerates PoET and
    kNN, while SVM/RF and MediaPipe extraction remain CPU operations.
    """

    def __init__(self, root: str | Path, device: str = "cpu"):
        started = time.perf_counter()
        self.root = Path(root).resolve()
        self._lock = threading.Lock()
        runtime = self.root / "assets/classical-runtime/site-packages"
        if runtime.is_dir() and str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        from _poet_runtime import load_bundle, resolve_device, synchronize
        from evaluate_pose_cache import OFFICIAL_LABEL_MAP_SHA256
        from method_runtime import OFFICIAL_CHECKPOINT_SHA256
        self.device = resolve_device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("ComparisonEngine supports cpu or cuda devices")
        self.bundle = load_bundle(self.root / "assets/checkpoints", self.device)
        if (self.bundle.metadata["checkpoint_sha256"] != OFFICIAL_CHECKPOINT_SHA256 or
                self.bundle.metadata["label_map_sha256"] != OFFICIAL_LABEL_MAP_SHA256):
            raise ValueError("Official PoET checkpoint or vocabulary identity mismatch")
        self.bundle.model.requires_grad_(False).eval()
        self.labels = self.bundle.id_to_label
        self._load_classical()
        self._load_geometry_retrieval()
        synchronize(self.device)
        self.load_seconds = time.perf_counter() - started

    def _load_classical(self) -> None:
        from classical_features import feature_spec
        from classical_models import load_model
        experiment = self.root / CLASSICAL_EXPERIMENT
        plan = read_verified_json(experiment / "plan.json", CLASSICAL_PLAN_SHA)
        if plan["feature_spec"] != feature_spec() or plan["scaler_fit_split"] != "train":
            raise ValueError("Classical feature definition or train-only scaler contract differs")
        dataset_sha = plan["prepared_sha256"]["dataset.json"]
        dataset = read_verified_json(experiment / "dataset.json", dataset_sha)
        if dataset["feature_spec"] != feature_spec():
            raise ValueError("Dataset feature definition differs")
        if dataset["id_to_label"] != [self.labels[index] for index in range(len(self.labels))]:
            raise ValueError("Classical/PoET vocabularies disagree")
        self.classical = {}
        for method in CLASSICAL_METHODS:
            state = json.loads((experiment / method / "state.json").read_text())
            if (state.get("status") != "completed" or state.get("method") != method or
                    state.get("model_sha256") != CLASSICAL_MODEL_SHA[method]):
                raise ValueError(f"Classical method is not completed: {method}")
            path = experiment / method / "model.joblib"
            model, scaler = load_model(path, expected_sha256=state["model_sha256"])
            if (model.method != method or model.num_classes != len(self.labels) or
                    model.feature_dim != feature_spec()["dimension"] or scaler is None):
                raise ValueError("Classical model dimensions or saved train scaler differ")
            self.classical[method] = {"model": model, "scaler": scaler,
                "provenance": {"experiment": CLASSICAL_EXPERIMENT,
                               "model_path": str(path), "model_sha256": state["model_sha256"],
                               "dataset_sha256": dataset_sha, "feature_spec": feature_spec(),
                               "fit_split": "train", "train_samples": 40154}}

    def _load_geometry_retrieval(self) -> None:
        import torch
        from _poet_runtime import load_bundle
        from extract_factorial_retrieval_features import validate_checkpoint_metadata
        from finetune_baseline import tensor_state_hash
        from method_optimizer import validate_checkpoint_optimizer_contract
        from method_runtime import restore_method
        from rerank_factorial_retrieval import load_preregistration, load_split
        from rerank_retrieval_pilot import build_memory
        prereg = load_preregistration(self.root / PREREGISTRATION, PREREGISTRATION_SHA)
        entry = next(row for row in prereg["models"] if row["model_id"] == GEOMETRY_MODEL_ID)
        path = self.root / entry["checkpoint_path"]
        if sha256(path) != entry["checkpoint_sha256"]:
            raise ValueError("Frozen geometry checkpoint SHA256 mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint_metadata(checkpoint, entry)
        base = load_bundle(self.root / "assets/checkpoints", self.device)
        model, metadata = restore_method(base, checkpoint)
        if metadata != checkpoint["protocol"]["method_metadata"]:
            raise ValueError("Restored geometry method metadata differs")
        validate_checkpoint_optimizer_contract(checkpoint, model)
        if tensor_state_hash(model.state_dict()) != entry["model_state_sha256"]:
            raise ValueError("Restored geometry model state differs")
        model.requires_grad_(False).eval()
        self.geometry_model = model
        train = load_split(self.root / TRAIN_BANK, TRAIN_BANK_SHA, "train", entry,
                           prereg, PREREGISTRATION_SHA)
        if train["label_map"] != {str(k): v for k, v in self.labels.items()}:
            raise ValueError("Retrieval memory/PoET vocabularies disagree")
        # The current shared input adapters must still match the frozen bank's
        # original preprocessing. Do not change any historical source to pass.
        pipeline = train["manifest"]["splits"]["train"]["provenance"]["pipeline"]
        for name, expected in pipeline["source_sha256"].items():
            if sha256(self.root / name) != expected:
                raise ValueError(f"Frozen pose preprocessing source differs: {name}")
        self.memory = build_memory(train["arrays"]["embeddings"], train["rows"], train["class_ids"])
        self.geometry_provenance = {
            "model_id": GEOMETRY_MODEL_ID, "seed": entry["seed"],
            "checkpoint_epoch": entry["checkpoint_epoch"], "checkpoint_path": str(path),
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "train_bank_manifest": TRAIN_BANK, "train_bank_manifest_sha256": TRAIN_BANK_SHA,
            "train_bank_features_sha256": train["manifest"]["splits"]["train"]["features_sha256"],
            "train_samples": 40154, "train_signers": 35, "memory_split": "train",
            "preregistration_sha256": PREREGISTRATION_SHA,
            "retrieval": prereg["retrieval"],
            "interpretation": "Fixed representative seed 20260923, not a three-seed ensemble or mean score."}

    def predict_video(self, video_path: str | Path, output_dir: str | Path) -> dict:
        """Decode/extract a video once, then send the same pose to all methods."""
        return self._predict(Path(video_path), Path(output_dir), "video")

    def predict_pose(self, pose_path: str | Path, output_dir: str | Path) -> dict:
        """Replay an existing raw_pose.npz, without reading any query labels."""
        return self._predict(Path(pose_path), Path(output_dir), "pose_replay")

    def _predict(self, input_path: Path, output: Path, kind: str) -> dict:
        with self._lock:
            return self._predict_locked(input_path.resolve(), output.resolve(), kind)

    def _predict_locked(self, input_path: Path, output: Path, kind: str) -> dict:
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        result = {"schema_version": 1, "status": "running",
                  "task": "isolated_asl_word_classification",
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "uses_target_label": False, "optimizer_steps": 0,
                  "input": {"kind": kind, "filename": input_path.name,
                            "sha256": sha256(input_path)},
                  "methods": {}, "warnings": [],
                  "interpretation": "One presegmented ASL word, not continuous translation or an accuracy estimate. "
                                    "Method scores have different meanings and are not calibrated or comparable confidence values."}
        manifest = output / "comparison.json"
        try:
            from _poet_runtime import validate_raw_pose
            stamp = time.perf_counter()
            if kind == "video":
                from infer_video import extract_raw_pose
                result["input"].update(inspect_video(input_path))
                raw, info = extract_raw_pose(input_path)
                result["input"].update(info)
                extraction_count = 1
            else:
                raw = read_pose(input_path)
                extraction_count = 0
            validate_raw_pose(raw)
            if sha256(input_path) != result["input"]["sha256"]:
                raise ValueError("Input changed while reading/extracting")
            preprocessing_seconds = time.perf_counter() - stamp
            result["coverage"] = detection_coverage(raw)
            result["input"]["frame_count"] = result["coverage"]["frame_count"]
            result["pose_extraction_count"] = extraction_count
            result["shared_raw_pose"] = True
            if result["coverage"]["all_hands_missing"]:
                result["warnings"].append("No hands were detected in any frame; predictions may be unreliable.")
            if result["coverage"]["all_parts_missing"]:
                result["warnings"].append("No body, face, or hands were detected; the output must not be interpreted as recognizing a visible sign.")
            raw_path = output / "raw_pose.npz"
            np.savez_compressed(raw_path, **raw)
            result["artifacts"] = {"raw_pose": {"path": str(raw_path), "sha256": sha256(raw_path)}}
            stamp = time.perf_counter()
            result["methods"], prepared_info = self._predict_methods(raw, output)
            classifiers_seconds = time.perf_counter() - stamp
            result["artifacts"].update(prepared_info)
            result["status"] = "completed"
            result["num_classes"] = len(self.labels)
            result["timings_seconds"] = {"model_load_once": self.load_seconds,
                "preprocessing": preprocessing_seconds, "classifiers": classifiers_seconds,
                "request_total": time.perf_counter() - started}
            manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            return result
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error),
                          elapsed_seconds=time.perf_counter() - started)
            manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            raise

    def _predict_methods(self, raw: dict, output: Path) -> tuple[dict, dict]:
        import torch
        from _poet_runtime import normalize_raw_pose, prepare_features, require_finite, synchronize
        from classical_features import extract_features
        from classical_models import predict_topk, transform_features
        from extract_factorial_retrieval_features import extract_method_batch
        from poet_geometry import build_geometry_features
        from rerank_retrieval_pilot import memory_scores, rank_candidates
        results = {}
        stamp = time.perf_counter()
        descriptor = extract_features(raw)[None]
        descriptor_seconds = time.perf_counter() - stamp
        for method in CLASSICAL_METHODS:
            loaded = self.classical[method]
            stamp = time.perf_counter()
            features = transform_features(loaded["scaler"], descriptor)
            feature_seconds = time.perf_counter() - stamp
            synchronize(self.device)
            stamp = time.perf_counter()
            answer = predict_topk(loaded["model"], features, top_k=5,
                                  device=str(self.device) if method == "knn" else "cpu",
                                  batch_size=128, reference_chunk_size=8192)
            synchronize(self.device)
            seconds = time.perf_counter() - stamp
            candidates = candidate_list(answer["topk_indices"][0], answer["topk_scores"][0],
                                        self.labels, answer["valid_mask"][0])
            results[method] = {"prediction": candidates[0]["word"], "topk": candidates,
                "score_type": answer["score_kind"], "backend": answer["backend"],
                "timings_seconds": {"shared_descriptor": descriptor_seconds,
                                    "scaler": feature_seconds, "classifier": seconds},
                "provenance": loaded["provenance"]}
        stamp = time.perf_counter()
        normalized = normalize_raw_pose(raw)
        prepared, indices = prepare_features(normalized, self.bundle.preprocessing, mode="eval")
        geometry = build_geometry_features(raw, indices)
        preprocess_seconds = time.perf_counter() - stamp
        model_input = output / "model_input.npz"
        np.savez_compressed(model_input, normalized_features=normalized,
                            model_input=prepared.numpy(), source_frame_indices=indices,
                            geometry=geometry, classical_descriptor=descriptor)
        synchronize(self.device)
        stamp = time.perf_counter()
        with torch.inference_mode():
            logits = self.bundle.model(prepared.unsqueeze(0), self.device)
            require_finite("Official model logits", logits)
            values, ids = logits.softmax(dim=1).topk(5, dim=1)
        synchronize(self.device)
        seconds = time.perf_counter() - stamp
        candidates = candidate_list(ids[0].tolist(), values[0].tolist(), self.labels)
        results["poet_official"] = {"prediction": candidates[0]["word"], "topk": candidates,
            "score_type": "softmax_probability", "backend": "torch_" + str(self.device),
            "timings_seconds": {"shared_preprocessing_and_geometry": preprocess_seconds,
                                "classifier": seconds},
            "provenance": self.bundle.metadata}
        synchronize(self.device)
        stamp = time.perf_counter()
        arrays, details = extract_method_batch(self.geometry_model, prepared.numpy()[None],
                                               geometry[None], self.device)
        synchronize(self.device)
        geometry_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        scores = memory_scores(self.memory, arrays["embeddings"], arrays["top5_class_ids"])
        ranked = rank_candidates(arrays["top5_class_ids"], arrays["top5_logits"], scores)
        positions = ranked["positions"]["fixed_blend"][0]
        blend = ranked["probabilities"]["fixed_blend"][0][positions]
        candidates = candidate_list(ranked["rankings"]["fixed_blend"][0], blend, self.labels)
        retrieval_seconds = time.perf_counter() - stamp
        results["poet_geometry_retrieval"] = {
            "prediction": candidates[0]["word"], "topk": candidates,
            "score_type": "top5_blend_probability", "backend": "torch_" + str(self.device) + "+numpy_memory",
            "timings_seconds": {"shared_preprocessing_and_geometry": preprocess_seconds,
                                "classifier": geometry_seconds, "retrieval": retrieval_seconds},
            "original_geometry_topk": candidate_list(arrays["top5_class_ids"][0],
                                                       details["top5_probabilities"][0], self.labels),
            "original_geometry_score_type": "softmax_probability",
            "candidate_membership_unchanged": True,
            "provenance": self.geometry_provenance}
        return results, {"model_input": {"path": str(model_input), "sha256": sha256(model_input)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cpu")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--pose-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError("Output directory already exists; choose a new run directory")
        engine = ComparisonEngine(args.root, args.device)
        result = (engine.predict_video(args.video, args.output_dir) if args.video is not None
                  else engine.predict_pose(args.pose_path, args.output_dir))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
