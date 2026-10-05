#!/usr/bin/env python3
"""Make a local reference package from already extracted classical features.

This tool does not download, extract poses, train a model, or publish data. Input
rows are verified privately; only numeric arrays, a glossary and aggregate
provenance leave the source directory. ASL Citizen redistribution restrictions
still apply to the generated package. See docs/DATA_PROVENANCE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


FEATURE_DIM = 1020
FEATURE_SPEC = "classical-pose-motion-v1"
SOURCE_RUN_ID = "20261002T160743Z-classical-full-features-3470975b"
HEX256 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_artifact(source: Path, name: str, info: dict) -> str:
    require(info.get("path") == name, f"Unexpected artifact path for {name}")
    path = source / name
    require(path.is_file(), f"Missing input artifact: {name}")
    require(path.stat().st_size == info.get("size_bytes"), f"Size mismatch: {name}")
    actual = sha256(path)
    require(actual == info.get("sha256"), f"SHA256 mismatch: {name}")
    return actual


def load_verified_source(source: Path) -> tuple[dict, dict, dict]:
    metadata = load_json(source / "dataset.json")
    require(metadata.get("schema_version") == 1, "Unsupported source schema")
    require(metadata.get("status") == "completed", "Source extraction did not complete")
    require(metadata.get("test_used") is False, "Only a source with test_used=false is accepted")
    spec = metadata.get("feature_spec", {})
    require(spec.get("version") == FEATURE_SPEC and spec.get("dimension") == FEATURE_DIM,
            "Incompatible classical feature definition")
    labels = metadata.get("id_to_label")
    require(isinstance(labels, list) and bool(labels) and all(isinstance(x, str) and x for x in labels),
            "Source glossary must be a nonempty string list")
    require(len(set(labels)) == len(labels), "Duplicate source glossary entries")
    data, hashes = {}, {}
    for split in ("train", "val"):
        info = metadata[split]
        for suffix in ("features.npy", "labels.npy", "rows.json"):
            name = f"{split}_{suffix}"
            hashes[name] = verify_artifact(source, name, info["artifacts"][name])
        features = np.load(source / f"{split}_features.npy", mmap_mode="r", allow_pickle=False)
        target = np.load(source / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        rows = load_json(source / f"{split}_rows.json")
        n = info["count"]
        require(features.dtype == np.dtype("float32") and features.shape == (n, FEATURE_DIM),
                f"Invalid {split} feature shape/dtype")
        require(target.dtype == np.dtype("int64") and target.shape == (n,),
                f"Invalid {split} label shape/dtype")
        require(np.isfinite(features).all(), f"Non-finite {split} features")
        require(isinstance(rows, list) and len(rows) == n, f"Invalid {split} row index")
        require(n > 0 and target.min() >= 0 and target.max() < len(labels),
                f"Out-of-range {split} class IDs")
        lookup, videos, signers = {}, set(), set()
        for index, record in enumerate(rows):
            require(record.get("index") == index, f"Misordered {split} row index")
            require(record.get("split") == split, f"Incorrect source split at {split} row {index}")
            require(record.get("gloss") == labels[int(target[index])],
                    f"Gloss/array-label mismatch at {split} row {index}")
            digest = record.get("video_sha256", "")
            require(isinstance(digest, str) and HEX256.fullmatch(digest) is not None,
                    f"Invalid video digest at {split} row {index}")
            video_file = PurePosixPath(record["source_path"]).name
            require(video_file not in lookup and digest not in videos,
                    f"Duplicate video in source {split}")
            signer = record.get("signer_id")
            require(isinstance(signer, str) and bool(signer), f"Missing source signer at {split} row {index}")
            lookup[video_file] = index
            videos.add(digest)
            signers.add(signer)
        data[split] = {"features": features, "labels": target, "rows": rows,
                       "lookup": lookup, "videos": videos, "signers": signers}
    require(not (set(data["train"]["lookup"]) & set(data["val"]["lookup"])),
            "Source video filenames cross train/validation")
    require(not (data["train"]["videos"] & data["val"]["videos"]),
            "Source video hashes cross train/validation")
    require(not (data["train"]["signers"] & data["val"]["signers"]),
            "Source signers cross train/validation")
    return metadata, data, hashes


def build_package(source: Path, selection_path: Path, output: Path,
                  profile: str = "mini", source_run_id: str = SOURCE_RUN_ID) -> dict:
    require(profile in ("mini", "full"), "Profile must be mini or full")
    require(not output.exists(), "Output already exists; choose a new directory")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", source_run_id) is not None,
            "source_run_id must be an identifier, not a path")
    metadata, data, input_hashes = load_verified_source(source)
    selection = load_json(selection_path)
    glossary = selection.get("selected_glosses")
    require(isinstance(glossary, list) and bool(glossary) and all(isinstance(g, str) for g in glossary),
            "selected_glosses must be a nonempty ordered string list")
    require(len(set(glossary)) == len(glossary), "Duplicate selected glosses")
    source_label_to_id = {g: i for i, g in enumerate(metadata["id_to_label"])}
    require(all(g in source_label_to_id for g in glossary), "Selected gloss absent from source")
    local_label_to_id = {g: i for i, g in enumerate(glossary)}
    original_ids = [source_label_to_id[g] for g in glossary]
    selected_arrays, selected_counts = {}, {}
    for split in ("train", "val"):
        samples = selection.get("splits", {}).get(split)
        require(isinstance(samples, list) and bool(samples), f"Missing selected {split} rows")
        indices, local_labels, seen = [], [], set()
        for record in samples:
            video_file = record.get("video_file")
            require(isinstance(video_file, str) and video_file == PurePosixPath(video_file).name,
                    "Selected video_file must be a basename")
            require(video_file in data[split]["lookup"], f"Selected video is absent from source {split}")
            index = data[split]["lookup"][video_file]
            require(index not in seen, f"Duplicate selected {split} video")
            seen.add(index)
            original = data[split]["rows"][index]
            require(record.get("gloss") == original["gloss"], f"Selected gloss mismatch in {split}")
            require(record.get("video_sha256") == original["video_sha256"],
                    f"Selected video SHA256 mismatch in {split}")
            require(record["gloss"] in local_label_to_id, "Selected row is outside selected_glosses")
            indices.append(index)
            local_labels.append(local_label_to_id[record["gloss"]])
        require(set(local_labels) == set(range(len(glossary))), f"Selected {split} does not cover all selected classes")
        if profile == "full":
            require(len(glossary) == len(metadata["id_to_label"]) and len(indices) == len(data[split]["rows"]),
                    "Full profile must retain all source classes and rows")
        selected_arrays[f"{split}_features.npy"] = np.asarray(data[split]["features"][indices], dtype=np.float32)
        selected_arrays[f"{split}_labels.npy"] = np.asarray(local_labels, dtype=np.int64)
        selected_counts[split] = {"count": len(indices)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-team-data-", dir=output.parent) as temporary:
        target = Path(temporary) / "package"
        target.mkdir()
        file_info = {}
        for filename, array in selected_arrays.items():
            path = target / filename
            np.save(path, array, allow_pickle=False)
            file_info[filename] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                                   "shape": list(array.shape), "dtype": str(array.dtype)}
        label_path = target / "labels.json"
        label_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        file_info["labels.json"] = {"sha256": sha256(label_path), "bytes": label_path.stat().st_size}
        manifest = {"schema_version": 1, "profile": profile, "dataset": "ASL Citizen",
                    "feature_spec": FEATURE_SPEC, "feature_dim": FEATURE_DIM,
                    "source_run_id": source_run_id,
                    "source_sha256": {n: h for n, h in input_hashes.items() if n.endswith(".npy")},
                    "source_dataset_sha256": sha256(source / "dataset.json"),
                    "source_row_index_sha256": {s: input_hashes[f"{s}_rows.json"] for s in ("train", "val")},
                    "selection_manifest_sha256": sha256(selection_path),
                    "exporter_sha256": sha256(Path(__file__)),
                    "selected_original_class_ids": original_ids,
                    "original_class_id_space": "source dataset.json id_to_label",
                    "splits": selected_counts, "files": file_info, "test_used": False,
                    "subset": {"name": f"reference-mini-{len(glossary)}" if profile == "mini" else "full",
                               "selection": "exact selected-manifest rows and glossary order"},
                    "source_validation": {"array_hashes_verified": True, "row_labels_verified": True,
                                          "train_val_video_disjoint": True, "train_val_signer_disjoint": True},
                    "raw_extractor": "historical feature run; differs from portable team preparation",
                    "distribution": "local-use-only; no redistribution permission established"}
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        target.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("mini", "full"), default="mini")
    parser.add_argument("--source-run-id", default=SOURCE_RUN_ID)
    args = parser.parse_args()
    try:
        result = build_package(args.source_features, args.selected_manifest, args.output,
                               args.profile, args.source_run_id)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Data export failed: {exc}\n")
    print(json.dumps({"profile": result["profile"], "classes": len(result["selected_original_class_ids"]),
                      "splits": result["splits"], "manifest_sha256": sha256(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
