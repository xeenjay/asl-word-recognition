#!/usr/bin/env python3
"""Fixed retrieval reranking (R) for the six frozen B/G checkpoints (S16, S18, ... S26).

Authorized by docs/environment/final-test-preregistration-20261012.json. Each
checkpoint's own 40,154-clip train bank is loaded and validated by the unchanged
rerank_factorial_retrieval.load_split; build_memory, memory_scores and
rank_candidates come unchanged from rerank_retrieval_pilot (beta=10, signer-
balanced kernels, top-5 only, 0.75 model + 0.25 memory). Queries are the
CLS/top-5 outputs of final_test_poet.py --family finetuned. No query label is used.

  --cohort val   gate G3_retrieval: fixed-blend rankings must equal the historical
                 factorial rerank predictions for all six models.
  --cohort test  refused unless the validation gate passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import rerank_retrieval_pilot as core
from rerank_factorial_retrieval import load_split

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = WORKSPACE / "docs/environment/final-test-preregistration-20261012.json"
PREREGISTRATION_SHA256 = "e241b1bbb6a7de21187d31a914a3308321132d2f10ed3e0952fbd652b2664286"
FACTORIAL = WORKSPACE / "docs/environment/factorial-retrieval-preregistration-20260926.json"
FACTORIAL_SHA256 = "8ff28628da3dc07374aa06ddf5d2d612ad8a5052c21485885ba42f81367244fc"
HISTORICAL_RERANK = WORKSPACE / "runs/20260926T224853Z-factorial-retrieval-rerank-94b64b9e/rerank/models"
COUNTS = {"val": 10304, "test": 32941}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", choices=("val", "test"), required=True)
    parser.add_argument("--queries-dir", type=Path, required=True, help="inference dir of final_test_poet.py --family finetuned")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-gate", type=Path)
    args = parser.parse_args()
    require(sha256_file(PREREGISTRATION) == PREREGISTRATION_SHA256, "final-test preregistration changed")
    require(sha256_file(FACTORIAL) == FACTORIAL_SHA256, "factorial preregistration changed")
    if args.cohort == "test":
        require(args.val_gate is not None, "test reranking requires the passed validation gate")
        gate = json.loads(args.val_gate.read_text())
        require(gate.get("gate") == "G3_retrieval" and gate.get("passed") is True, "validation replay gate did not pass")
    prereg = json.loads(PREREGISTRATION.read_text())
    factorial = json.loads(FACTORIAL.read_text())
    banks = {m["model_id"]: m for m in prereg["frozen_artifacts"]["poet_finetuned"]}
    queries = args.queries_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks, summary = {}, {}
    for entry in factorial["models"]:
        model_id = entry["model_id"]
        bank = banks[model_id]
        bank_manifest = WORKSPACE / Path(bank["train_bank_path"]).parent / "manifest.json"
        train = load_split(bank_manifest, bank["train_bank_manifest_sha256"], "train", entry, factorial, FACTORIAL_SHA256)
        with np.load(queries / f"{model_id}_features.npz") as archive:
            embeddings, candidates, logits = (archive[k].copy() for k in ("embeddings", "top5_class_ids", "top5_logits"))
        rows = [json.loads(line) for line in (queries / f"predictions_{model_id}.jsonl").read_text().splitlines()]
        require(len(rows) == COUNTS[args.cohort] and [r["index"] for r in rows] == list(range(len(rows)))
                and all(r["split"] == args.cohort for r in rows), "query rows are not the complete cohort in index order")
        require(not ({r["video_sha256"] for r in rows} & train["video_sha256s"]), "query videos overlap the train bank")
        require(np.array_equal(candidates, np.array([r["top5_ids"] for r in rows], dtype=np.int64)), "query top-5 differs from its predictions file")
        memory = core.build_memory(train["arrays"]["embeddings"], train["rows"], train["class_ids"])
        scores = core.memory_scores(memory, embeddings, candidates, chunk_size=256)
        ranked = core.rank_candidates(candidates, logits, scores)
        ranking = ranked["rankings"]["fixed_blend"]
        blend = np.take_along_axis(ranked["probabilities"]["fixed_blend"], ranked["positions"]["fixed_blend"], axis=1)
        with (output / f"predictions_{model_id}_R.jsonl").open("w", encoding="utf-8") as stream:
            for i, row in enumerate(rows):
                record = {k: row[k] for k in ("index", "gloss", "signer_id", "split", "video_sha256")}
                record.update(status="ok", top5_ids=ranking[i].tolist(), top5_scores=blend[i].tolist(),
                              original_top5_ids=candidates[i].tolist(), memory_kernel_scores=scores[i].tolist())
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary[model_id] = {"rows": len(rows), "changed_top1": int((ranking[:, 0] != candidates[:, 0]).sum()),
                             "train_bank_manifest_sha256": bank["train_bank_manifest_sha256"]}
        if args.cohort == "val":
            history = [json.loads(line) for line in (HISTORICAL_RERANK / model_id / "fixed_blend" / "predictions.jsonl").read_text().splitlines()]
            checks[model_id] = {
                "rankings_exact": len(history) == len(rows) and all(h["ranked_class_ids"] == ranking[i].tolist() for i, h in enumerate(history)),
                "memory_scores_exact": len(history) == len(rows) and all(h["memory_kernel_scores"] == scores[i].tolist() for i, h in enumerate(history)),
            }
        print(f"{model_id}: reranked {len(rows)} (top-1 changed {summary[model_id]['changed_top1']})", flush=True)
    result = {"cohort": args.cohort, "models": summary, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "preregistration_sha256": PREREGISTRATION_SHA256, "queries_dir": str(queries)}
    if args.cohort == "val":
        gate = {"gate": "G3_retrieval", "passed": all(v for c in checks.values() for v in c.values()), "checks": checks}
        (output / "gate_G3_retrieval.json").write_text(json.dumps(gate, indent=2) + "\n")
        result["gate"] = {"passed": gate["passed"]}
        print(json.dumps({"gate": "G3_retrieval", "passed": gate["passed"]}), flush=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if args.cohort == "test" or result["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
