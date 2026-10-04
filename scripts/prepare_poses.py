#!/usr/bin/env python3
"""Extract pinned PoET poses on CPU, with atomic caches and auditable resume.

Inputs use fetch_asl_sample's manifest format. All failures remain in the output
index; a nonzero exit status forbids treating a partial cache as a complete set.
No model is created and CUDA is hidden from this preprocessing process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

WORKSPACE = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
_WORKER = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def configure_cpu_environment():
    # Set before importing NumPy/Torch/MediaPipe, including in spawned workers.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[variable] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def select_samples(manifest, manifest_path, offset=0, limit=None):
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Input manifest must contain a nonempty samples array")
    if manifest.get("status") != "complete" or manifest.get("selection", {}).get("count") != len(samples):
        raise ValueError("Input download manifest must be complete and its declared selection.count must equal len(samples); download failures cannot disappear from the preprocessing denominator")
    if offset < 0 or offset >= len(samples) or (limit is not None and limit < 1):
        raise ValueError("offset/limit selects no samples or is out of range")
    indices = list(range(offset, min(len(samples), offset + limit) if limit else len(samples)))
    selected, seen = [], set()
    for index in indices:
        sample = samples[index]
        required = ("path", "sha256", "gloss", "signer_id", "split")
        if any(not isinstance(sample.get(key), str) or not sample[key] for key in required):
            raise ValueError(f"Sample {index} must have nonempty string fields {required}")
        digest = sample["sha256"].lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Sample {index} has an invalid SHA256")
        relative = Path(sample["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Sample {index} path must stay relative to the manifest directory")
        if sample["split"] not in {"train", "val", "validation"}:
            raise ValueError(f"Sample {index} split {sample['split']!r} is outside the current train/validation preparation scope")
        source = (manifest_path.parent / relative).resolve()
        if not source.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError(f"Sample {index} resolves outside its manifest directory")
        if str(source) in seen:
            raise ValueError(f"Duplicate video path at sample {index}: {source}")
        seen.add(str(source))
        identity = {key: sample[key] for key in required}
        identity["sha256"] = digest
        selected.append({
            "index": index, "identity": identity,
            "source_path": str(source), "video_sha256": digest,
            "gloss": sample["gloss"], "signer_id": sample["signer_id"],
            "split": sample["split"],
            "cache_key": canonical_hash({"path": relative.as_posix(), "sha256": digest}),
        })
    return selected, {"indices": indices, "offset": offset, "limit": limit,
                      "source_count": len(samples), "selected_count": len(selected)}


def make_pipeline(checkpoint):
    from _poet_runtime import load_checkpoint, package_versions
    loaded = load_checkpoint(checkpoint)
    preprocessing = loaded["data_preprocessing"]
    if preprocessing.get("keypoints") != 188 or preprocessing.get("max_seq_length") != 128:
        raise ValueError("This cache version requires the official 188-feature, 128-frame configuration")
    files = ["scripts/prepare_poses.py", "scripts/infer_video.py", "scripts/_poet_runtime.py",
             "vendor/poet/model.py", "vendor/poet/mediapipe_extraction.py", "vendor/poet/protobuf_to_dict.py"]
    spec = importlib.util.find_spec("mediapipe")
    heavy = Path(spec.origin).parent / "modules/pose_landmark/pose_landmark_heavy.tflite"
    if not heavy.is_file():
        raise FileNotFoundError("Install the pinned local MediaPipe heavy asset before preprocessing; automatic network download is not allowed here")
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "poet-official-raw-normalized-eval-v2",
        "source_sha256": {relative: file_hash(WORKSPACE / relative) for relative in files},
        "packages": package_versions(),
        "checkpoint_sha256": file_hash(checkpoint),
        "heavy_model_sha256": file_hash(heavy),
        "preprocessing": preprocessing,
        "extractor": {"implementation": "infer_video.extract_raw_pose using unmodified vendor extractor",
                      "static_image_mode": True, "model_complexity": 2,
                      "min_detection_confidence": 0.5, "flip": False},
        "coordinates": "raw image-normalized xy and per-articulator original z; no cross-articulator 3D metric implied",
        "raw_dtype": "float64", "normalized_dtype": "float64", "model_input_dtype": "float32",
        "temporal_mode": "eval", "missing_articulators": "zeros plus explicit boolean presence masks",
        "all_hands_missing_policy": "official_zero_hands_v2; retain zero hands and available body/face; classify every valid decoded video",
    }


def init_worker(pipeline, cpu_groups, ordinal):
    configure_cpu_environment()
    with ordinal.get_lock():
        worker_index = ordinal.value
        ordinal.value += 1
    affinity = cpu_groups[worker_index % len(cpu_groups)]
    if affinity is not None:
        os.sched_setaffinity(0, affinity)
    # Graph construction happens after affinity is applied. This bounds CPU use
    # even when MediaPipe's native graph creates its own internal thread pools.
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER.update(pipeline=pipeline, pipeline_sha256=canonical_hash(pipeline),
                   affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def verify_cache(target, item, pipeline_hash):
    """Check identity/config, file hashes, array schema and preprocessing output."""
    import numpy as np
    from _poet_runtime import validate_raw_pose, require_finite
    metadata_path = target / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "ok" or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Cache has no valid completed manifest")
    if metadata.get("identity") != item["identity"] or metadata.get("pipeline_sha256") != pipeline_hash:
        raise ValueError("Cache identity/config mismatch; preserve it and choose a new versioned cache root")
    for name, artifact in metadata["artifacts"].items():
        path = target / name
        if name not in {"raw_pose.npz", "normalized_features.npz"}:
            raise ValueError(f"Unexpected cache artifact name: {name}")
        if path.stat().st_size != artifact["bytes"] or file_hash(path) != artifact["sha256"]:
            raise ValueError(f"Cache hash/size mismatch: {path}; existing cache is preserved")
    if set(metadata["artifacts"]) != {"raw_pose.npz", "normalized_features.npz"}:
        raise ValueError("Cache artifact inventory is incomplete")
    with np.load(target / "raw_pose.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    frames = validate_raw_pose(raw)
    if frames != metadata["video_metadata"]["decoded_frames"]:
        raise ValueError("Cached raw frame count differs from its metadata")
    for key in ("body_visibility", "timestamps_seconds", "frame_ids"):
        require_finite(key, raw[key])
    if not np.array_equal(raw["frame_ids"], np.arange(frames)):
        raise ValueError("Cached raw frame IDs are not contiguous")
    with np.load(target / "normalized_features.npz", allow_pickle=False) as archive:
        features, prepared, indices = (archive[key] for key in ("normalized_features", "model_input", "source_frame_indices"))
        if features.shape != (frames, 188) or features.dtype != np.float64:
            raise ValueError("Cached normalized features have an invalid shape/dtype")
        if prepared.shape != (128, 188) or prepared.dtype != np.float32 or indices.shape != (128,) or indices.dtype != np.int64:
            raise ValueError("Cached eval input has an invalid shape/dtype")
        require_finite("cached normalized features", features)
        require_finite("cached model input", prepared)
        if np.any(indices < -1) or np.any(indices >= frames):
            raise ValueError("Cached source-frame indices are outside the raw video")
    return metadata


def successful_result(item, target, metadata, reused, total_seconds):
    return {**item, "status": "reused" if reused else "ok", "cache_manifest_path": str(target / "manifest.json"),
            "pose_path": str(target / "raw_pose.npz"),
            "pose_sha256": metadata["artifacts"]["raw_pose.npz"]["sha256"],
            "features_path": str(target / "normalized_features.npz"),
            "features_sha256": metadata["artifacts"]["normalized_features.npz"]["sha256"],
            "cache_bytes": sum(value["bytes"] for value in metadata["artifacts"].values()),
            "video_metadata": metadata["video_metadata"],
            "timings": {"total_seconds": total_seconds} if reused else metadata["timings"],
            "cached_extraction_timings": metadata["timings"] if reused else None,
            **worker_resources()}


def worker_resources():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"worker_pid": os.getpid(), "cpu_affinity": _WORKER.get("affinity"),
            "worker_peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
            "worker_native_thread_count": len(list(Path("/proc/self/task").iterdir())) if Path("/proc/self/task").is_dir() else None}


def process_one(item, cache_root):
    import fcntl
    import numpy as np
    from infer_video import extract_raw_pose
    from _poet_runtime import normalize_raw_pose, prepare_features
    started = time.perf_counter()
    cache_root = Path(cache_root)
    key = item["cache_key"]
    target = cache_root / key[:2] / key
    staging = None
    try:
        source = Path(item["source_path"])
        stamp = time.perf_counter()
        if file_hash(source) != item["video_sha256"]:
            raise ValueError("Source video SHA256 differs from its data manifest")
        timings = {"source_verification_seconds": time.perf_counter() - stamp}
        lock_root = cache_root / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        # flock is released automatically on a process crash. A surviving lock
        # file is harmless; no stale-PID guessing or deletion is needed.
        with (lock_root / (key + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                metadata = verify_cache(target, item, _WORKER["pipeline_sha256"])
                return successful_result(item, target, metadata, True, time.perf_counter() - started)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(key + ".incomplete-" + uuid.uuid4().hex)
            staging.mkdir()
            atomic_json(staging / "manifest.json", {"status": "running", "schema_version": SCHEMA_VERSION,
                        "identity": item["identity"], "pipeline_sha256": _WORKER["pipeline_sha256"],
                        "created_at_utc": now(), "pid": os.getpid()})
            stamp = time.perf_counter()
            raw, info = extract_raw_pose(source)
            timings["extraction_seconds"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            features = normalize_raw_pose(raw)
            prepared, indices = prepare_features(features, SimpleNamespace(**_WORKER["pipeline"]["preprocessing"]), mode="eval")
            timings["normalization_temporal_seconds"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            np.savez_compressed(staging / "raw_pose.npz", **raw)
            np.savez_compressed(staging / "normalized_features.npz", normalized_features=features,
                                model_input=prepared.numpy(), source_frame_indices=indices)
            artifacts = {name: {"sha256": file_hash(staging / name), "bytes": (staging / name).stat().st_size}
                         for name in ("raw_pose.npz", "normalized_features.npz")}
            timings["write_hash_seconds"] = time.perf_counter() - stamp
            # Guard against source mutation during extraction.
            if file_hash(source) != item["video_sha256"]:
                raise ValueError("Source video changed during extraction")
            timings["total_seconds"] = time.perf_counter() - started
            metadata = {"schema_version": SCHEMA_VERSION, "status": "ok", "completed_at_utc": now(),
                        "identity": item["identity"], "pipeline_sha256": _WORKER["pipeline_sha256"],
                        "pipeline": _WORKER["pipeline"], "video_metadata": info,
                        "artifacts": artifacts, "timings": timings, "cpu_affinity": _WORKER["affinity"],
                        "worker_pid": os.getpid(), "eval_input_shape": list(prepared.shape),
                        "source_frames_ge_129": info["decoded_frames"] >= 129,
                        "temporal_source_unique_frames": int(len(np.unique(indices[indices >= 0])))}
            atomic_json(staging / "manifest.json", metadata)
            verify_cache(staging, item, _WORKER["pipeline_sha256"])
            os.replace(staging, target)
            return successful_result(item, target, metadata, False, time.perf_counter() - started)
    except Exception as error:
        result = {**item, "status": "error", "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "timings": {"total_seconds": time.perf_counter() - started},
                  **worker_resources(),
                  "incomplete_path": str(staging) if staging else None}
        if staging is not None and staging.is_dir():
            atomic_json(staging / "failure.json", result)
        return result


def cpu_groups(workers, per_worker):
    if not hasattr(os, "sched_getaffinity"):
        if workers > 1:
            raise ValueError("Parallel mode requires Linux sched_setaffinity; use --workers 1 on this platform")
        return [None]
    available = sorted(os.sched_getaffinity(0))
    if workers * per_worker > len(available):
        raise ValueError(f"Requested {workers * per_worker} dedicated CPUs but inherited affinity has only {len(available)}")
    return [available[index * per_worker:(index + 1) * per_worker] for index in range(workers)]


def summarize(results, wall_seconds, workers, interrupted=False):
    good = [item for item in results if item["status"] in {"ok", "reused"}]
    new = [item for item in good if item["status"] == "ok"]
    failed = [item for item in results if item["status"] not in {"ok", "reused"}]
    processed_frames = sum(item["video_metadata"]["decoded_frames"] for item in new)
    stages = {}
    for item in new:
        for key, seconds in item["timings"].items():
            stages[key] = stages.get(key, 0.0) + seconds
    frame_counts = sorted(item["video_metadata"]["decoded_frames"] for item in good)
    durations = sorted(item["video_metadata"]["decoded_frames"] / item["video_metadata"]["fps"] for item in good)
    return {"schema_version": SCHEMA_VERSION,
            "status": "interrupted" if interrupted else "failed" if failed else "completed",
            "selected_count": len(results), "successful_count": len(good), "new_count": len(new),
            "reused_count": len(good) - len(new), "failed_count": len(failed),
            "workers": workers, "wall_seconds": wall_seconds,
            "source_frames": sum(frame_counts), "newly_processed_frames": processed_frames,
            "new_videos_per_wall_second": len(new) / wall_seconds if wall_seconds else 0,
            "new_frames_per_wall_second": processed_frames / wall_seconds if wall_seconds else 0,
            "worker_stage_seconds_sum": stages,
            "source_frames_ge_129_count": sum(count >= 129 for count in frame_counts),
            "frame_count_min": min(frame_counts) if frame_counts else None,
            "frame_count_max": max(frame_counts) if frame_counts else None,
            "source_duration_seconds_min": min(durations) if durations else None,
            "source_duration_seconds_max": max(durations) if durations else None,
            "cache_artifact_bytes": sum(item["cache_bytes"] for item in good),
            "new_cache_artifact_bytes": sum(item["cache_bytes"] for item in new),
            "worker_peak_rss_bytes": {str(pid): max(item.get("worker_peak_rss_bytes", 0) for item in results if item.get("worker_pid") == pid)
                                      for pid in {item["worker_pid"] for item in results if "worker_pid" in item}},
            "worker_native_threads_max": max((item.get("worker_native_thread_count") or 0 for item in results), default=0),
            "interpretation": "Engineering preprocessing validation; no recognition accuracy or training result. Stage sums are worker time, not elapsed wall time."}


def run(args):
    started = time.perf_counter()
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=False)
    atomic_json(report / "summary.json", {"status": "running", "started_at_utc": now()})
    selected, results, futures = [], [], {}
    executor = None
    interrupted = False
    fatal_error = None
    configuration = {}
    try:
        configure_cpu_environment()
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text())
        selected, selection = select_samples(manifest, manifest_path, args.offset, args.limit)
        groups = cpu_groups(args.workers, args.cpus_per_worker)
        pipeline = make_pipeline(args.checkpoint.resolve())
        configuration = {
            "schema_version": SCHEMA_VERSION, "status": "running", "created_at_utc": now(),
            "source_manifest_path": str(manifest_path), "source_manifest_sha256": file_hash(manifest_path),
            "selection": selection, "cache_root": str(args.cache_root.resolve()),
            "pipeline": pipeline, "pipeline_sha256": canonical_hash(pipeline),
            "workers": args.workers, "cpus_per_worker": args.cpus_per_worker, "cpu_groups": groups,
            "start_method": "spawn", "worker_library_threads": 1,
            "split": next(iter({item["split"] for item in selected})) if len({item["split"] for item in selected}) == 1 else "mixed",
        }
        atomic_json(report / "configuration.json", configuration)
        atomic_json(report / "selection.json", selected)
        args.cache_root.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        ordinal = context.Value("i", 0)
        executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                       initializer=init_worker, initargs=(pipeline, groups, ordinal))
        futures = {executor.submit(process_one, item, str(args.cache_root.resolve())): item for item in selected}
        with (report / "results.jsonl").open("a", encoding="utf-8") as stream:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {**item, "status": "error", "error_type": type(error).__name__, "error": str(error),
                              "recovery": "Worker/pool failed; any completed atomic cache can be reused by a new run"}
                results.append(result)
                atomic_json(report / "samples" / f"{item['index']:06d}-{item['cache_key'][:12]}.json", result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"[{len(results)}/{len(selected)}] {result['status']} index={item['index']} gloss={item['gloss']}", flush=True)
        executor.shutdown(wait=True)
        executor = None
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        fatal_error = f"{type(error).__name__}: {error}"
        atomic_json(report / "fatal_error.json", {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        if not selected:
            atomic_json(report / "summary.json", {"status": "failed", "selected_count": 0, "error": str(error), "completed_at_utc": now()})
            return 1
    finally:
        if executor is not None:
            # On interrupt, terminate only the children owned by this executor.
            # Otherwise Python's exit hook would wait for all pending work.
            for process in list(getattr(executor, "_processes", {}).values()):
                if process.is_alive():
                    process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
    completed = {item["index"] for item in results}
    for item in selected:
        if item["index"] not in completed:
            results.append({**item, "status": "interrupted" if interrupted else "error",
                            "error_type": "Interrupted" if interrupted else "OrchestrationFailure",
                            "error": "No successful per-sample result was returned; inspect any atomic cache on resume"})
    results.sort(key=lambda item: item["index"])
    summary = summarize(results, time.perf_counter() - started, args.workers, interrupted)
    if fatal_error is not None:
        summary.update(status="failed", fatal_error=fatal_error)
    summary["completed_at_utc"] = now()
    atomic_json(report / "summary.json", summary)
    atomic_json(report / "failures.json", [item for item in results if item["status"] not in {"ok", "reused"}])
    atomic_json(report / "cache_manifest.json", {**configuration, "status": summary["status"],
                "completed_at_utc": summary["completed_at_utc"], "samples": results, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 130 if interrupted else 0 if summary["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True, help="New directory, typically {run_dir}/preprocessing")
    parser.add_argument("--checkpoint", type=Path, default=WORKSPACE / "assets/checkpoints/ASL_Citizen_PoET.pth")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--cpus-per-worker", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.cpus_per_worker < 1:
        parser.error("--cpus-per-worker must be positive")
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(run(args))
    except (FileExistsError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
