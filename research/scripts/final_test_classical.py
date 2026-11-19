#!/usr/bin/env python3
"""Frozen classical systems (S01-S12) and classical-on-CLS systems (S27-S29).

Authorized by docs/environment/final-test-preregistration-20261012.json. No model
is refit. Descriptors are recomputed from the verified raw pose cache with the
unchanged classical_features.extract_features; the canonical variant applies the
frozen classical_handedness signer rule to the cohort's own unlabeled descriptors.
Every frozen model is loaded by SHA256 with its stored train scaler and predicts
with the same entry point and settings as its historical run.

  --cohort val   validation replay (gate G3_classical): descriptors must equal the
                 frozen validation features bit for bit, the canonical matrix the
                 handedness run's matrix, signer decisions the recorded ones, and
                 every model's top-5 must equal its historical predictions.
  --cohort test  refused unless the validation gate passed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = WORKSPACE / "docs/environment/final-test-preregistration-20261012.json"
PREREGISTRATION_SHA256 = "e241b1bbb6a7de21187d31a914a3308321132d2f10ed3e0952fbd652b2664286"
V2_PIPELINE = "ceca202ce01a6327c7433b1a29a7ba5ea482f6b8335a87d7732938463dd1941b"
COUNTS = {"val": 10304, "test": 32941}
VAL = {
    "index": "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json",
    "original_features": "runs/20261002T160743Z-classical-full-features-3470975b/features/val_features.npy",
    "canonical_features": "runs/20261010T065305Z-handedness-features-81bcfbae/features/val_features.npy",
    "decisions": "runs/20261010T065305Z-handedness-features-81bcfbae/features/handedness_decisions.json",
    "official_cls": "runs/20261011T071432Z-cls-representation-dataset-73a92db2/features/val_features.npy",
}
GROUPS = ("classical_original_descriptor_original_config", "classical_canonical_original_config",
          "classical_canonical_tuned", "classical_original_descriptor_tuned", "cls_classical")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_samples(path: Path, split: str) -> list[dict]:
    index = json.loads(path.read_text(encoding="utf-8"))
    require(index.get("status") == "completed" and index.get("pipeline_sha256") == V2_PIPELINE, f"{path} is not a completed v2 cache")
    samples = sorted(index["samples"], key=lambda item: item["index"])
    require([s["index"] for s in samples] == list(range(COUNTS[split])) and all(s["split"] == split for s in samples),
            f"{path} must hold the complete {split} cohort in index order")
    require(all(s["status"] in {"ok", "reused"} for s in samples), "failed cache rows must be handled explicitly")
    for item in samples:
        value = Path(item["pose_path"])
        item["pose_path"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return samples


def descriptor(task: tuple[str, str]) -> np.ndarray:
    from classical_features import extract_features
    pose_path, expected = task
    if sha256_file(Path(pose_path)) != expected:
        raise ValueError(f"raw pose SHA256 changed: {pose_path}")
    with np.load(pose_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    return extract_features(raw)


def predict(model, scaler, X: np.ndarray, tuned: bool, device: str) -> dict:
    from classical_models import predict_topk, transform_features
    Q = transform_features(scaler, X)
    if tuned:
        from run_classical_tuning import predict_point
        return predict_point(model, Q, device)
    return predict_topk(model, Q, top_k=5, device=device if model.method == "knn" else "cpu",
                        batch_size=128, reference_chunk_size=8192)


def historical_predictions(group: str, method: str, model_path: str) -> Path:
    directory = Path(model_path).parent
    return WORKSPACE / directory / "predictions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-index", type=Path, help="test v2 pose cache index (test cohort)")
    parser.add_argument("--official-cls", type=Path, help="official_logits.npz of the test official inference (test cohort)")
    parser.add_argument("--val-gate", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    require(sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256, "final-test preregistration changed")
    prereg = json.loads(PREREGISTRATION.read_text())
    from classical_handedness import canonicalize_descriptors, signer_decisions
    from classical_models import load_model
    if args.cohort == "val":
        index_path = WORKSPACE / VAL["index"]
        cls = np.load(WORKSPACE / VAL["official_cls"])
    else:
        require(args.original_index and args.official_cls and args.val_gate, "test needs --original-index, --official-cls, --val-gate")
        gate = json.loads(args.val_gate.read_text())
        require(gate.get("gate") == "G3_classical" and gate.get("passed") is True, "validation replay gate did not pass")
        index_path = args.original_index
        cls = np.load(args.official_cls)["embeddings"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    samples = load_samples(index_path, args.cohort)
    require(cls.shape == (len(samples), 128) and cls.dtype == np.float32, "official CLS matrix shape differs from the cohort")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(descriptor, [(s["pose_path"], s["pose_sha256"]) for s in samples], chunksize=32))
    X_original = np.stack(rows).astype(np.float32)
    signers = [s["signer_id"] for s in samples]
    decisions = signer_decisions(X_original, signers)
    X_canonical = canonicalize_descriptors(X_original, signers, decisions)
    np.save(output / "original_features.npy", X_original)
    np.save(output / "canonical_features.npy", X_canonical)
    (output / "handedness_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    checks = {}
    if args.cohort == "val":
        checks["original_descriptors_bit_identical"] = X_original.tobytes() == np.load(WORKSPACE / VAL["original_features"]).tobytes()
        checks["canonical_descriptors_bit_identical"] = X_canonical.tobytes() == np.load(WORKSPACE / VAL["canonical_features"]).tobytes()
        recorded = json.loads((WORKSPACE / VAL["decisions"]).read_text())["decisions"]["val"]
        checks["signer_decisions_equal"] = recorded == decisions
    results = {}
    for group in GROUPS:
        for method, entry in prereg["frozen_artifacts"][group].items():
            name = f"{group}__{method}"
            model, scaler = load_model(WORKSPACE / entry["path"], expected_sha256=entry["sha256"])
            require(scaler is not None, f"{name} has no stored train scaler")
            X = cls if group == "cls_classical" else (X_canonical if "canonical" in group else X_original)
            prediction = predict(model, scaler, X, tuned=group.endswith("_tuned"), device=args.device)
            records = []
            for i, sample in enumerate(samples):
                valid = prediction["valid_mask"][i]
                ids = prediction["topk_indices"][i][valid].astype(int).tolist()
                scores = prediction["topk_scores"][i][valid].astype(float).tolist()
                require(bool(ids), f"{name}: no valid class at row {i}")
                records.append({**{k: sample[k] for k in ("index", "gloss", "signer_id", "split", "video_sha256")},
                                "status": "ok", "top5_ids": ids, "top5_scores": scores, "score_kind": prediction["score_kind"]})
            with (output / f"predictions_{name}.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if args.cohort == "val":
                history = [json.loads(line) for line in historical_predictions(group, method, entry["path"]).read_text().splitlines()]
                same_ids = len(history) == len(records) and all(h["top5_ids"] == r["top5_ids"] for h, r in zip(history, records))
                same_scores = len(history) == len(records) and all(
                    np.allclose(h["top5_scores"], r["top5_scores"], rtol=1e-10, atol=1e-10) for h, r in zip(history, records))
                checks[name] = {"top5_ids_exact": same_ids, "scores_within_1e-10": same_scores}
            results[name] = {"rows": len(records), "model_sha256": entry["sha256"]}
            print(f"{name}: done", flush=True)
    summary = {"cohort": args.cohort, "systems": results, "seconds": time.perf_counter() - started, "completed_at_utc": now(),
               "preregistration_sha256": PREREGISTRATION_SHA256, "index": str(index_path), "index_sha256": sha256_file(index_path),
               "mirrored_signers": sorted(k for k, v in decisions.items() if v["mirror"])}
    if args.cohort == "val":
        flat = [v for v in checks.values() if isinstance(v, bool)] + [x for v in checks.values() if isinstance(v, dict) for x in v.values()]
        gate = {"gate": "G3_classical", "passed": all(flat), "checks": checks, "verified_at_utc": now()}
        (output / "gate_G3_classical.json").write_text(json.dumps(gate, indent=2) + "\n")
        summary["gate"] = {"passed": gate["passed"]}
        print(json.dumps({"gate": "G3_classical", "passed": gate["passed"]}), flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if args.cohort == "test" or summary["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
