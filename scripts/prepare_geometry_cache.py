#!/usr/bin/env python3
"""Immutable full-frame geometry cache for an already-audited pose cohort.

The CLI performs finetune_baseline.audit_cohort first. Python callers must pass
that audited result, not an unvalidated pose index. This module never modifies
raw pose data and never resamples time when building the T x 28 cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np

from evaluate_pose_cache import canonical_hash, require_sha, sha256, verify_file
from finetune_baseline import atomic_json
from poet_geometry import GEOMETRY_DIM, build_geometry_features, geometry_spec

WORKSPACE = Path(__file__).resolve().parents[1]
CACHE_VERSION = "poet-full-frame-geometry-v1"
IDENTITY_FIELDS = ("index", "identity", "video_sha256", "gloss", "signer_id", "split", "pose_sha256")


def source_hashes() -> dict:
    return {name: sha256(WORKSPACE / name) for name in
            ("scripts/poet_geometry.py", "scripts/prepare_geometry_cache.py")}


def _path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing cache artifact path")
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if path.is_symlink():
        raise ValueError(f"Cache artifact cannot be a symlink: {path}")
    return path.resolve()


def _stamp(path: Path) -> tuple:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular immutable cache file: {path}")
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _identity(item: dict) -> dict:
    return {name: item[name] for name in IDENTITY_FIELDS}


def _contract(cohort: dict) -> dict:
    """Recheck that the caller's audited cohort still identifies its source."""
    if cohort.get("split") not in {"train", "val"}:
        raise ValueError("Geometry cache only supports train and val; test is held out")
    path = _path(Path.cwd(), cohort["path"])
    verify_file(path, cohort["sha256"], "audited pose index")
    original = json.loads(path.read_text(encoding="utf-8"))
    samples = cohort["samples"]
    if not samples or len(samples) != len(original["samples"]):
        raise ValueError("Audited cohort denominator differs from the source pose index")
    provenance = cohort["provenance"]
    if (provenance.get("pipeline_sha256") != original["pipeline_sha256"]
            or provenance.get("source_manifest_sha256") != original["source_manifest_sha256"]):
        raise ValueError("Audited cohort provenance disagrees with its pose index")
    indices, videos = set(), set()
    for item, source in zip(samples, original["samples"]):
        if (_identity(item) != _identity(source) or item["split"] != cohort["split"]
                or item.get("status") not in {"ok", "reused"}
                or source.get("status") not in {"ok", "reused"}):
            raise ValueError("Geometry cohort identity/order/split/failure mismatch")
        if type(item["index"]) is not int or item["index"] in indices or item["video_sha256"] in videos:
            raise ValueError("Geometry cohort contains duplicate or invalid sample identities")
        indices.add(item["index"])
        videos.add(item["video_sha256"])
        for key in ("pose_path", "cache_manifest_path"):
            if _path(path.parent, item[key]) != _path(path.parent, source[key]):
                raise ValueError("Audited pose artifact path differs from source index")
        if item.get("video_metadata") != source.get("video_metadata"):
            raise ValueError("Audited video frame metadata differs from source index")
        require_sha(item["pose_sha256"], "raw pose SHA256")
        require_sha(item["video_sha256"], "video SHA256")
    spec = geometry_spec()
    return {"version": CACHE_VERSION, "geometry_spec": spec,
            "geometry_spec_sha256": canonical_hash(spec), "source_sha256": source_hashes(),
            "pose_index_sha256": cohort["sha256"], "pipeline_sha256": original["pipeline_sha256"],
            "source_manifest_sha256": original["source_manifest_sha256"],
            "split": cohort["split"], "samples": len(samples),
            "complete_official_split": provenance["complete_official_split"]}


def _raw_descriptor(cohort: dict, item: dict, *, verify: bool) -> dict:
    base = Path(cohort["path"]).parent
    raw_path = _path(base, item["pose_path"])
    metadata_path = _path(base, item["cache_manifest_path"])
    if metadata_path.name != "manifest.json" or raw_path != metadata_path.parent / "raw_pose.npz":
        raise ValueError("Raw pose path differs from its per-video metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = item["video_metadata"]["decoded_frames"]
    if (metadata.get("identity") != item["identity"] or metadata.get("status") != "ok"
            or metadata.get("pipeline_sha256") != cohort["provenance"]["pipeline_sha256"]
            or metadata.get("video_metadata", {}).get("decoded_frames") != frames
            or metadata["artifacts"]["raw_pose.npz"]["sha256"] != item["pose_sha256"]):
        raise ValueError("Raw pose metadata identity/hash/frame count mismatch")
    if type(frames) is not int or frames < 1:
        raise ValueError("Raw pose needs a positive decoded frame count")
    before = _stamp(raw_path)
    if before[2] != metadata["artifacts"]["raw_pose.npz"]["bytes"]:
        raise ValueError("Raw pose size differs from original metadata")
    if verify:
        verify_file(raw_path, item["pose_sha256"], "geometry source raw pose")
    if _stamp(raw_path) != before:
        raise ValueError("Raw pose changed during verification")
    return {"raw_path": str(raw_path), "raw_stamp": before, "frames": frames}


def _validate_array(array: np.ndarray, frames: int) -> None:
    if array.shape != (frames, GEOMETRY_DIM) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Geometry must be finite float32 full-frame T x 28")
    relations = array.reshape(frames, -1, 4)
    masks = relations[..., 3]
    if not np.isin(masks, [0, 1]).all() or np.any(relations[..., :3][masks == 0] != 0):
        raise ValueError("Geometry masks/zero missing relations are invalid")
    if np.any(relations[..., 2] < 0) or not np.allclose(
            np.linalg.norm(relations[..., :2].astype(np.float64), axis=-1), relations[..., 2], rtol=2e-6, atol=1e-6):
        raise ValueError("Geometry distances disagree with image-xy offsets")


def _load_array(path: Path, digest: str, frames: int, size: int) -> np.ndarray:
    before = _stamp(path)
    if before[2] != size:
        raise ValueError("Geometry cache size mismatch")
    verify_file(path, digest, "full-frame geometry")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"geometry", "frame_ids"}:
            raise ValueError("Geometry NPZ has an unexpected array inventory")
        array, frame_ids = archive["geometry"], archive["frame_ids"]
    if frame_ids.dtype != np.int64 or not np.array_equal(frame_ids, np.arange(frames)):
        raise ValueError("Geometry frame IDs must be the contiguous original frames")
    _validate_array(array, frames)
    if _stamp(path) != before:
        raise ValueError("Geometry cache changed during reading")
    return array


def _sample_contract(item: dict, contract: dict, frames: int) -> dict:
    return {"version": CACHE_VERSION, "identity": _identity(item), "frames": frames,
            "pipeline_sha256": contract["pipeline_sha256"],
            "geometry_spec_sha256": contract["geometry_spec_sha256"],
            "source_sha256": contract["source_sha256"]}


def _audit_entry(directory: Path, item: dict, contract: dict, raw: dict) -> dict:
    if directory.is_symlink():
        raise ValueError("Geometry cache directories cannot be symlinks")
    metadata_path = directory / "manifest.json"
    if metadata_path.is_symlink():
        raise ValueError("Geometry metadata cannot be a symlink")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = _sample_contract(item, contract, raw["frames"])
    if metadata.get("status") != "complete" or metadata.get("contract") != expected:
        raise ValueError("Geometry metadata spec/source/raw/identity mismatch; preserve existing cache")
    artifact = metadata["artifact"]
    if artifact.get("name") != "geometry.npz":
        raise ValueError("Unexpected geometry artifact name")
    path = directory / "geometry.npz"
    array = _load_array(path, artifact["sha256"], raw["frames"], artifact["bytes"])
    masks = array[:, 3::4]
    stats = {"valid_relations": int(masks.sum()), "total_relations": int(masks.size),
             "frames_with_geometry": int(masks.any(axis=1).sum())}
    if metadata.get("statistics") != stats:
        raise ValueError("Geometry statistics differ from actual cached features")
    return {**_identity(item), "frames": raw["frames"], "geometry_path": str(path.resolve()),
            "geometry_sha256": artifact["sha256"], "geometry_bytes": artifact["bytes"],
            "metadata_path": str(metadata_path.resolve()), "metadata_sha256": sha256(metadata_path),
            "statistics": stats}


def build_geometry_cache(cohort: dict, output_dir: Path) -> dict:
    """Build/reuse every sample; a failed item never disappears from the cohort.

    Existing entries are immutable and are strictly audited before reuse. A
    partial build can resume into the same root when its contract is unchanged.
    The complete index is committed only after every sample succeeds.
    """
    started = time.perf_counter()
    contract = _contract(cohort)
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError("Geometry output cannot be a symlink")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract_path = output / "cache_contract.json"
        if contract_path.exists():
            if contract_path.is_symlink() or json.loads(contract_path.read_text()) != contract:
                raise ValueError("Existing geometry root has different spec/source/cohort; choose a new root")
        else:
            if any(p.name != ".build.lock" for p in output.iterdir()):
                raise ValueError("Geometry output is nonempty without a matching cache contract")
            atomic_json(contract_path, contract)
        index_path = output / "cache_manifest.json"
        if index_path.exists():
            result = audit_geometry_index(index_path, cohort)
            result["build"] = {"new": 0, "reused": len(cohort["samples"]), "seconds": time.perf_counter() - started}
            return result
        entries, reused = [], 0
        for number, item in enumerate(cohort["samples"], 1):
            raw = _raw_descriptor(cohort, item, verify=True)
            sample_contract = _sample_contract(item, contract, raw["frames"])
            key = canonical_hash(sample_contract)
            directory = output / "samples" / key[:2] / key
            if directory.exists():
                reused += 1
            else:
                directory.parent.mkdir(parents=True, exist_ok=True)
                temporary = directory.with_name(".staging-" + key + "-" + uuid.uuid4().hex)
                temporary.mkdir()
                try:
                    # Keep ALL raw arrays, especially body_visibility (the old
                    # runtime's convenience loader deliberately drops it).
                    with np.load(raw["raw_path"], allow_pickle=False) as archive:
                        original = {key: archive[key] for key in archive.files}
                    if _stamp(Path(raw["raw_path"])) != raw["raw_stamp"]:
                        raise ValueError("Raw pose changed after its SHA256 audit")
                    frames = raw["frames"]
                    if len(original["body_pose"]) != frames:
                        raise ValueError("Raw pose frames differ from original metadata")
                    geometry = build_geometry_features(original, np.arange(frames, dtype=np.int64))
                    _validate_array(geometry, frames)
                    path = temporary / "geometry.npz"
                    with path.open("wb") as stream:
                        np.savez_compressed(stream, geometry=geometry, frame_ids=np.arange(frames, dtype=np.int64))
                        stream.flush()
                        os.fsync(stream.fileno())
                    masks = geometry[:, 3::4]
                    atomic_json(temporary / "manifest.json", {
                        "status": "complete", "contract": sample_contract,
                        "artifact": {"name": "geometry.npz", "sha256": sha256(path), "bytes": path.stat().st_size},
                        "statistics": {"valid_relations": int(masks.sum()), "total_relations": int(masks.size),
                                       "frames_with_geometry": int(masks.any(axis=1).sum())}})
                    os.rename(temporary, directory)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            entries.append(_audit_entry(directory, item, contract, raw))
            if number % 500 == 0:
                print(f"Prepared geometry {number}/{len(cohort['samples'])}", flush=True)
        # Catch a source manifest/code change before declaring a complete cache.
        if _contract(cohort) != contract:
            raise ValueError("Geometry source or spec changed during cache construction")
        index = {"schema_version": 1, "status": "complete", "contract": contract,
                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "samples": entries, "summary": _summary(entries),
                 "build": {"new": len(entries) - reused, "reused": reused,
                           "seconds": time.perf_counter() - started}}
        atomic_json(index_path, index)
    # Entries were just hashed/decoded, so avoid a second pass over all raw NPZs.
    return {"path": str(index_path), "sha256": sha256(index_path),
            "geometry_spec_sha256": contract["geometry_spec_sha256"],
            "source_sha256": contract["source_sha256"], "contract": contract,
            "summary": index["summary"], "build": index["build"]}


def _summary(entries: list[dict]) -> dict:
    return {"samples": len(entries), "frames": sum(x["frames"] for x in entries),
            "bytes": sum(x["geometry_bytes"] for x in entries),
            **{key: sum(x["statistics"][key] for x in entries)
               for key in ("valid_relations", "total_relations", "frames_with_geometry")}}


def audit_geometry_index(path: Path, cohort: dict) -> dict:
    """Bind a complete geometry index to an already-audited train/val cohort.

    Called once at startup, this checks every raw SHA again and every geometry
    payload SHA/schema. Later reads rehash only the small geometry NPZ; a raw
    stat change is rejected without decompressing face/body arrays each epoch.
    """
    contract = _contract(cohort)
    path = _path(Path.cwd(), str(path))
    digest = sha256(path)
    index = json.loads(path.read_text(encoding="utf-8"))
    if (index.get("schema_version") != 1 or index.get("status") != "complete"
            or index.get("contract") != contract or len(index.get("samples", [])) != len(cohort["samples"])):
        raise ValueError("Geometry index spec/source/cohort/denominator mismatch")
    root_contract = path.parent / "cache_contract.json"
    if root_contract.is_symlink() or json.loads(root_contract.read_text()) != contract:
        raise ValueError("Geometry root contract does not match the audited index")
    by_index, checked = {}, []
    for number, (entry, item) in enumerate(zip(index["samples"], cohort["samples"]), 1):
        if _identity(entry) != _identity(item):
            raise ValueError("Geometry sample identity/order differs from the audited pose cohort")
        raw = _raw_descriptor(cohort, item, verify=True)
        key = canonical_hash(_sample_contract(item, contract, raw["frames"]))
        directory = path.parent / "samples" / key[:2] / key
        actual = _audit_entry(directory, item, contract, raw)
        if actual != entry:
            raise ValueError("Geometry index and per-video artifact disagree")
        checked.append(actual)
        by_index[item["index"]] = {**actual, **raw}
        if number % 500 == 0:
            print(f"Audited geometry {number}/{len(cohort['samples'])}", flush=True)
    if index.get("summary") != _summary(checked) or sha256(path) != digest or _contract(cohort) != contract:
        raise ValueError("Geometry summary, source, or index changed during audit")
    return {"path": str(path), "sha256": digest, "geometry_spec_sha256": contract["geometry_spec_sha256"],
            "source_sha256": contract["source_sha256"], "contract": contract,
            "summary": index["summary"], "by_index": by_index}


def read_geometry(item: dict, source_frame_indices, geometry_index: dict) -> np.ndarray:
    """Return S x 28 aligned with prepare_features' exact source indices.

    Duplicates are retained and -1 becomes all zeros; there is no independent
    geometry resampling and the scale remains the full video's fixed statistic.
    """
    entry = geometry_index["by_index"].get(item["index"])
    if entry is None or _identity(entry) != _identity(item):
        raise ValueError("Geometry reader received a different sample identity")
    if _stamp(Path(entry["raw_path"])) != entry["raw_stamp"]:
        raise ValueError("Raw pose changed after geometry startup audit")
    indices = np.asarray(source_frame_indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or np.any(indices < -1) or np.any(indices >= entry["frames"])):
        raise ValueError("Geometry source indices must be integers in [-1, original frame count)")
    geometry = _load_array(Path(entry["geometry_path"]), entry["geometry_sha256"],
                           entry["frames"], entry["geometry_bytes"])
    aligned = np.zeros((len(indices), GEOMETRY_DIM), dtype=np.float32)
    real = indices >= 0
    aligned[real] = geometry[indices[real]]
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=WORKSPACE / "assets/checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    import torch
    from _poet_runtime import load_bundle
    from finetune_baseline import audit_cohort
    torch.set_num_threads(1)
    bundle = load_bundle(args.checkpoint_dir, torch.device("cpu"))
    cohort = audit_cohort(args.cache_manifest, bundle, args.engineering_smoke, "train")
    if not args.engineering_smoke and not cohort["provenance"]["complete_official_split"]:
        raise ValueError("Formal geometry preparation requires a complete official split")
    result = build_geometry_cache(cohort, args.output_dir)
    print(json.dumps({key: result[key] for key in ("path", "sha256", "summary", "build")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
