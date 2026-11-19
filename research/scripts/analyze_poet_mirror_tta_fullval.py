#!/usr/bin/env python3
"""Independent analysis of the full-validation PoET mirror TTA evaluation.

Recomputes every gate, outcome and the preregistered verdict from saved
artifacts: logits, per-clip predictions, gate files and the historical official
predictions. It shares no code with the extraction wrapper or the evaluator
(own label-free sample rule, own float64 top-k, own statistics). Fixed by
docs/environment/poet-mirror-tta-fullval-preregistration-20261011.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION_ID = "poet-mirror-tta-fullval-v1"
SOURCE_MANIFEST_SHA256 = "b59fa41ee85955a310d665a3af09a67274a6ddb48b4d18cae1faaa06c6fb7a4f"
HISTORICAL_PREDICTIONS_SHA256 = "8d7e6fe2d347ef7a23ef5b8b59a0534602109d83148fcbedeb81545651aaaca5"
PRIOR_MIRRORED_SHA256 = "086103f9d96f60a8ca1d485ab661c30a3dd042542470da11774d41a869eb8f73"
MIRROR_PIPELINE_SHA256 = "8af5b135fdc1ed6defc8666dfaffe0450caada1d57ab6f1f4b9a894818fb2ed9"
REUSED_INDEX_SHA256 = "651582c4b11c04c6c3ad523cbc1566a9a64522babce00e862430a68c987a9d7b"
CLIPS, REUSED_CLIPS, CLASSES = 10304, 2806, 2731
HISTORICAL_TOP1_CORRECT = 8340
IDENTITY_SALT = "poet-mirror-tta-fullval-identity-20261011:"
REUSE_SALT = "poet-mirror-tta-fullval-reuse-20261011:"
PER_SIGNER = 4
LEFT_SIGNERS = ("P12", "P5")
MIN_GAIN_PP, ALPHA = 0.50, 0.05
G4_GATE = 0.85
PROBABILITY_TOLERANCE, TTA_TOLERANCE = 1e-5, 1e-9
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261011
# Known 2026-10-10 Top-1 counts (original, mirrored, tta) that the prior subset must reproduce.
PRIOR_COUNTS = {"P12_P5": (981, 990, 1006), "P21_P26_P39_sample": (1248, 1253, 1268)}
ARMS = ("original", "mirrored", "tta")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mcnemar_exact(fixed: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = fixed + regressed
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, regressed) + 1)) / 2 ** n)


def first_per_signer(rows: list[tuple[int, str, str]], salt: str, count: int = PER_SIGNER) -> list[int]:
    """rows: (index, signer, video_sha256). min(count, n) smallest SHA256(salt + sha) per signer."""
    by_signer: dict[str, list[tuple[str, int]]] = {}
    for index, signer, digest in rows:
        by_signer.setdefault(signer, []).append((hashlib.sha256((salt + digest).encode("ascii")).hexdigest(), index))
    return sorted(index for keyed in by_signer.values() for _, index in sorted(keyed)[:count])


def stable_top5(values):
    """Float64 top-5 by descending value; a stable sort keeps the lower class ID first on ties."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, axis=1, kind="stable")[:, :5]
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return order, np.take_along_axis(probabilities, order, axis=1)


def has_tie(row_values) -> bool:
    """True when two adjacent values among the six largest are exactly equal."""
    import numpy as np
    top = np.sort(np.asarray(row_values))[::-1][:6]
    return bool(np.any(top[:-1] == top[1:]))


def paired(base: list[bool], arm: list[bool], base_top5: list[bool], arm_top5: list[bool],
           base_pred: list[int], arm_pred: list[int]) -> dict:
    n = len(base)
    fixed = sum((not b) and a for b, a in zip(base, arm))
    regressed = sum(b and (not a) for b, a in zip(base, arm))
    return {"clips": n, "base_top1_correct": sum(base), "arm_top1_correct": sum(arm),
            "base_top1": sum(base) / n, "arm_top1": sum(arm) / n, "delta_top1_pp": 100 * (sum(arm) - sum(base)) / n,
            "base_top5": sum(base_top5) / n, "arm_top5": sum(arm_top5) / n,
            "delta_top5_pp": 100 * (sum(arm_top5) - sum(base_top5)) / n,
            "fixed": fixed, "regressed": regressed, "net": fixed - regressed,
            "mcnemar_exact_p": mcnemar_exact(fixed, regressed),
            "top1_prediction_agreement": sum(b == a for b, a in zip(base_pred, arm_pred)) / n}


def verdict(full: dict, fresh: dict) -> dict:
    gain, p = full["delta_top1_pp"], full["mcnemar_exact_p"]
    fresh_ok = fresh["delta_top1_pp"] > 0 and fresh["mcnemar_exact_p"] < ALPHA
    if gain >= MIN_GAIN_PP and p < ALPHA and fresh_ok:
        return {"verdict": "supported", "pattern": None, "reasons": []}
    if gain > 0 and p < ALPHA:
        reasons = ([f"full-validation gain {gain:.4f} pp < {MIN_GAIN_PP} pp"] if gain < MIN_GAIN_PP else []) + \
                  ([] if fresh_ok else ["fresh-clip gain not positive with p < 0.05"])
        return {"verdict": "partially_supported", "pattern": None, "reasons": reasons}
    return {"verdict": "not_supported", "pattern": "harmful" if gain < 0 and p < ALPHA else "no_detectable_gain",
            "reasons": [f"full-validation delta {gain:.4f} pp, exact McNemar p {p:.4g}"]}


def bootstrap(differences: dict[str, list[int]], seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Signer-stratified clip bootstrap of the pooled Top-1 difference (percentage points)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    total = sum(len(v) for v in differences.values())
    sums = np.zeros(resamples)
    for signer in sorted(differences):
        values = np.asarray(differences[signer], dtype=np.float64)
        sums += values[rng.integers(0, len(values), size=(resamples, len(values)))].sum(axis=1)
    estimates = 100 * sums / total
    return {"resamples": resamples, "seed": seed, "point_pp": 100 * sum(sum(v) for v in differences.values()) / total,
            "ci95_pp": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]}


def macro(labels: list[int], correct: list[bool]) -> float:
    per_class: dict[int, list[bool]] = {}
    for label, value in zip(labels, correct):
        per_class.setdefault(label, []).append(value)
    return sum(sum(v) / len(v) for v in per_class.values()) / len(per_class)


def compare(records: dict[int, dict], expected: dict[int, dict], indices: list[int]) -> dict:
    mismatches, largest = 0, 0.0
    for index in indices:
        a, b = records[index]["top5"], expected[index]["top5"]
        largest = max(largest, max(abs(x["probability"] - y["probability"]) for x, y in zip(a, b)))
        mismatches += [x["class_id"] for x in a] != [y["class_id"] for y in b]
    return {"clips": len(indices), "top5_id_mismatches": mismatches, "max_abs_probability_difference": largest,
            "passed": mismatches == 0 and largest <= PROBABILITY_TOLERANCE}


def analyze(args: argparse.Namespace) -> dict:
    import numpy as np
    problems: list[str] = []
    gates: dict[str, dict] = {}

    def require(condition: bool, message: str) -> bool:
        if not condition:
            problems.append(message)
        return condition

    preregistration_sha = sha256(args.preregistration)
    require(preregistration_sha == args.preregistration_sha256, "preregistration SHA256 differs from the launch record")
    require(read_json(args.preregistration).get("id") == PREREGISTRATION_ID, "unexpected preregistration id")
    require(sha256(args.source_manifest) == SOURCE_MANIFEST_SHA256, "source manifest SHA256 differs")
    require(sha256(args.historical_predictions) == HISTORICAL_PREDICTIONS_SHA256, "historical predictions SHA256 differs")
    require(sha256(args.prior_mirrored_predictions) == PRIOR_MIRRORED_SHA256, "2026-10-10 mirrored predictions SHA256 differs")
    samples = read_json(args.source_manifest)["samples"]
    historical = {row["index"]: row for row in read_jsonl(args.historical_predictions)}
    prior_mirrored = {row["index"]: row for row in read_jsonl(args.prior_mirrored_predictions)}
    indices = list(range(CLIPS))
    require(len(samples) == CLIPS and sorted(historical) == indices, "manifest/historical do not cover 10,304 clips")
    for index in indices:
        s, h = samples[index], historical[index]
        require(s["index"] == index and s["split"] in {"val", "validation"} and h["status"] == "ok"
                and s["gloss"] == h["gloss"] and s["signer_id"] == h["signer_id"] and s["sha256"] == h["video_sha256"],
                f"label/signer/video mismatch at {index}")
    labels = [historical[i]["class_id"] for i in indices]
    signers = [samples[i]["signer_id"] for i in indices]

    # Upstream gates, re-derived where possible.
    identity_gate = read_json(args.identity_gate)
    expected_identity = first_per_signer([(s["index"], s["signer_id"], s["sha256"]) for s in samples], IDENTITY_SALT)
    gates["G1"] = {"status": "passed" if require(identity_gate.get("status") == "passed" and identity_gate.get("indices") == expected_identity
                                                 and identity_gate.get("identical") == len(expected_identity) == 22,
                                                 "G1 identity gate not passed for the recomputed 22-clip sample") else "failed",
                   "clips": len(expected_identity), "identical": identity_gate.get("identical")}
    combined_path = args.mirror_dir / "fullval_cache_manifest.json"
    combined = read_json(combined_path)
    rows = {row["index"]: row for row in combined["samples"]}
    reused = sorted(i for i, row in rows.items() if row.get("mirror_source", "").startswith("reused:"))
    fresh = sorted(set(indices) - set(reused))
    require(sorted(rows) == indices and combined.get("pipeline_sha256") == MIRROR_PIPELINE_SHA256
            and combined.get("composition", {}).get("reused_index_sha256") == REUSED_INDEX_SHA256
            and all(row["status"] == "ok" for row in rows.values()), "combined index is not the preregistered complete mirror index")
    require(len(reused) == REUSED_CLIPS and reused == sorted(prior_mirrored), "reused rows differ from the 2026-10-10 clips")
    reuse_gate = read_json(args.reuse_gate)
    expected_reuse = first_per_signer([(i, rows[i]["signer_id"], rows[i]["video_sha256"]) for i in reused], REUSE_SALT)
    gates["R1"] = {"status": "passed" if require(reuse_gate.get("status") == "passed" and reuse_gate.get("reused_index_sha256") == REUSED_INDEX_SHA256
                                                 and reuse_gate.get("pipeline_sha256") == MIRROR_PIPELINE_SHA256
                                                 and reuse_gate.get("audit", {}).get("verified_rows") == REUSED_CLIPS
                                                 and reuse_gate.get("audit", {}).get("problem_count") == 0
                                                 and reuse_gate.get("determinism", {}).get("indices") == expected_reuse
                                                 and reuse_gate.get("determinism", {}).get("identical") == len(expected_reuse) == 20,
                                                 "R1 reuse gate not passed") else "failed",
                   "verified_rows": reuse_gate.get("audit", {}).get("verified_rows"),
                   "determinism_identical": reuse_gate.get("determinism", {}).get("identical")}
    g5 = read_json(args.mirror_dir / "extraction_gate.json")
    gates["G5"] = {"status": "passed" if require(g5.get("status") == "passed" and g5.get("selected") == g5.get("ok") == len(fresh) == CLIPS - REUSED_CLIPS,
                                                 "G5 extraction gate not passed") else "failed",
                   "selected": g5.get("selected"), "ok": g5.get("ok"), "wall_seconds": g5.get("wall_seconds")}
    g4 = read_json(args.mirror_dir / "flip_sanity.json")
    recount = {}
    for group in ("left_dominant_signers", "right_dominant_signers"):
        clear = [c for c in g4["clips"] if c["group"] == group and c["original_motion_ratio"] <= 0.5]
        recount[group] = sum(c["mirrored_left_motion"] == c["original_right_motion"] for c in clear) / len(clear) if clear else None
    require(sorted(c["index"] for c in g4["clips"]) == indices, "G4 does not cover every clip")
    gates["G4"] = {"status": "passed" if require(g4.get("status") == "passed" and all(v is not None and v >= G4_GATE for v in recount.values()),
                                                 "G4 flip check not passed on recount") else "failed",
                   "clear_motion_consistency": recount, "amendment1_value_met": g4.get("amendment1_value_met")}

    # Inference artifacts.
    summary = read_json(args.inference_dir / "summary.json")
    require(summary.get("status") == "completed", "inference run did not complete")
    for name, digest in summary.get("artifact_sha256", {}).items():
        require(sha256(args.inference_dir / name) == digest, f"inference artifact {name} SHA256 differs from its summary")
    with np.load(args.inference_dir / "logits.npz", allow_pickle=False) as archive:
        saved_indices, original, mirrored = archive["indices"], archive["original"], archive["mirrored"]
    require(saved_indices.tolist() == indices and original.shape == mirrored.shape == (CLIPS, CLASSES)
            and original.dtype == mirrored.dtype == np.float32 and np.isfinite(original).all() and np.isfinite(mirrored).all(),
            "saved logits have an unexpected layout")
    predictions = {arm: {row["index"]: row for row in read_jsonl(args.inference_dir / f"predictions_{arm}.jsonl")} for arm in ARMS}
    for arm in ARMS:
        require(sorted(predictions[arm]) == indices and all(predictions[arm][i]["class_id"] == labels[i] for i in indices),
                f"{arm} predictions do not cover all clips with the historical labels")
    tta = 0.5 * (original.astype(np.float64) + mirrored.astype(np.float64))
    own = {"original": stable_top5(original), "mirrored": stable_top5(mirrored), "tta": stable_top5(tta)}
    values_by_arm = {"original": original, "mirrored": mirrored, "tta": tta}
    consistency = {}
    for arm in ARMS:
        ids, probabilities = own[arm]
        mismatched, ties = [], []
        largest = 0.0
        for position in indices:
            theirs = predictions[arm][position]["top5"]
            same = ids[position].tolist() == [p["class_id"] for p in theirs]
            largest = max(largest, max(abs(float(a) - b["probability"]) for a, b in zip(probabilities[position], theirs)))
            if not same:
                (ties if has_tie(values_by_arm[arm][position]) else mismatched).append(position)
        tolerance = TTA_TOLERANCE if arm == "tta" else PROBABILITY_TOLERANCE
        consistency[arm] = {"clips": CLIPS, "non_tie_top5_mismatches": mismatched[:20], "non_tie_mismatch_count": len(mismatched),
                            "exact_tie_mismatch_count": len(ties), "max_abs_probability_difference": largest, "tolerance": tolerance,
                            "passed": not mismatched and (arm != "tta" or not ties) and largest <= tolerance}
        require(consistency[arm]["passed"], f"{arm}: own top-5 recomputation disagrees with the evaluator")
    g2 = compare(predictions["original"], historical, indices)
    g2b = compare(predictions["mirrored"], prior_mirrored, reused)
    gates["G2"] = {"status": "passed" if require(g2["passed"] and summary["gates"]["G2"]["status"] == "passed", "G2 not passed") else "failed", **g2}
    gates["G2b"] = {"status": "passed" if require(g2b["passed"] and summary["gates"]["G2b"]["status"] == "passed", "G2b not passed") else "failed", **g2b}
    frozen = read_json(args.frozen_eval_dir / "summary.json")
    frozen_predictions = {row["index"]: row for row in read_jsonl(args.frozen_eval_dir / "predictions.jsonl")}
    g3 = compare(predictions["mirrored"], frozen_predictions, indices) if sorted(frozen_predictions) == indices else {"passed": False}
    gates["G3"] = {"status": "passed" if require(g3["passed"] and frozen.get("status") == "completed" and frozen["metrics"]["failed"] == 0
                                                 and frozen["data"]["complete_official_split"] is True
                                                 and frozen["data"]["pipeline_sha256"] == MIRROR_PIPELINE_SHA256
                                                 and frozen["data"]["cache_manifest_sha256"] == sha256(combined_path),
                                                 "G3 frozen evaluator cross-check not passed") else "failed", **g3}

    # Correctness. Original = historical official predictions (identical to the recomputed arm by G2).
    pred = {"original": [historical[i]["top5"][0]["class_id"] for i in indices],
            "mirrored": own["mirrored"][0][:, 0].tolist(), "tta": own["tta"][0][:, 0].tolist()}
    top5_ids = {"original": [[p["class_id"] for p in historical[i]["top5"]] for i in indices],
                "mirrored": own["mirrored"][0].tolist(), "tta": own["tta"][0].tolist()}
    top1 = {arm: [pred[arm][i] == labels[i] for i in indices] for arm in ARMS}
    top5 = {arm: [labels[i] in top5_ids[arm][i] for i in indices] for arm in ARMS}
    require(sum(top1["original"]) == HISTORICAL_TOP1_CORRECT, "original arm does not reproduce 8,340 correct")

    def compare_arm(arm: str, subset: list[int]) -> dict:
        pick = lambda values: [values[i] for i in subset]  # noqa: E731
        return paired(pick(top1["original"]), pick(top1[arm]), pick(top5["original"]), pick(top5[arm]),
                      pick(pred["original"]), pick(pred[arm]))

    left = [i for i in indices if signers[i] in LEFT_SIGNERS]
    right = [i for i in indices if signers[i] not in LEFT_SIGNERS]
    subsets = {"full": indices, "fresh_7498": fresh, "prior_2806": reused, "left_dominant_signers": left, "right_dominant_signers": right}
    results = {name: {arm: compare_arm(arm, subset) for arm in ("tta", "mirrored")} for name, subset in subsets.items()}
    prior_left = [i for i in reused if signers[i] in LEFT_SIGNERS]
    prior_right = [i for i in reused if signers[i] not in LEFT_SIGNERS]
    reproduced = {"P12_P5": tuple(sum(top1[arm][i] for i in prior_left) for arm in ARMS),
                  "P21_P26_P39_sample": tuple(sum(top1[arm][i] for i in prior_right) for arm in ARMS)}
    gates["prior_subset_reproduction"] = {"status": "passed" if require(reproduced == PRIOR_COUNTS, "prior 2,806 subset does not reproduce the 2026-10-10 counts") else "failed",
                                          "expected": PRIOR_COUNTS, "observed": reproduced}
    per_signer = {}
    for signer in sorted(set(signers), key=lambda s: (-signers.count(s), s)):
        subset = [i for i in indices if signers[i] == signer]
        per_signer[signer] = {"clips": len(subset), **{arm: compare_arm(arm, subset) for arm in ("tta", "mirrored")},
                              "interpreted": len(subset) >= 100}
    arms = {arm: {"top1_correct": sum(top1[arm]), "top1": sum(top1[arm]) / CLIPS, "top5": sum(top5[arm]) / CLIPS,
                  "macro_top1_observed_classes": macro(labels, top1[arm]), "macro_top5_observed_classes": macro(labels, top5[arm])}
            for arm in ARMS}
    arms["observed_classes"] = len(set(labels))
    differences = lambda subset: {s: [int(top1["tta"][i]) - int(top1["original"][i]) for i in subset if signers[i] == s]  # noqa: E731
                                  for s in sorted({signers[i] for i in subset})}
    intervals = {"full": bootstrap(differences(indices), BOOTSTRAP_SEED), "fresh_7498": bootstrap(differences(fresh), BOOTSTRAP_SEED)}
    agreement = {"original_vs_mirrored_top1": sum(a == b for a, b in zip(pred["original"], pred["mirrored"])) / CLIPS,
                 "tta_equals_original_or_mirrored": sum(t in (a, b) for t, a, b in zip(pred["tta"], pred["original"], pred["mirrored"])) / CLIPS,
                 "tta_equals_neither": sum(t not in (a, b) for t, a, b in zip(pred["tta"], pred["original"], pred["mirrored"]))}
    all_passed = not problems and all(g["status"] == "passed" for g in gates.values())
    decision = verdict(results["full"]["tta"], results["fresh_7498"]["tta"]) if all_passed else {
        "verdict": "not_interpretable", "pattern": None, "reasons": ["an engineering gate failed; see problems"]}
    return {"schema_version": 1, "preregistration_id": PREREGISTRATION_ID, "preregistration": str(args.preregistration),
            "preregistration_sha256": preregistration_sha, "created_at_utc": now(), "test_used": False,
            "training_or_parameter_update": False, "gates": gates, "consistency": consistency,
            "problems": problems, "all_gates_passed": all_passed,
            "decision_rule": {"supported": f"full delta >= +{MIN_GAIN_PP} pp and p < {ALPHA} and fresh-7,498 delta > 0 with p < {ALPHA}",
                              "partially_supported": f"full delta > 0 and p < {ALPHA}, other conditions not all met",
                              "not_supported": "otherwise (harmful if delta < 0 and p < 0.05, else no_detectable_gain)"},
            "decision": decision, "primary": results["full"]["tta"], "subsets": results, "per_signer": per_signer,
            "arms": arms, "bootstrap": intervals, "agreement": agreement,
            "cost": {"new_flipped_clips": g5.get("selected"), "extraction_wall_seconds": g5.get("wall_seconds"),
                     "extraction_clips_per_wall_second": g5.get("clips_per_wall_second"),
                     "note": "TTA needs a second MediaPipe pass per video; 16 workers x 2 logical CPUs"},
            "inputs": {"source_manifest_sha256": SOURCE_MANIFEST_SHA256, "historical_predictions_sha256": HISTORICAL_PREDICTIONS_SHA256,
                       "prior_mirrored_predictions_sha256": PRIOR_MIRRORED_SHA256, "combined_index": str(combined_path),
                       "combined_index_sha256": sha256(combined_path), "inference_dir": str(args.inference_dir),
                       "frozen_eval_dir": str(args.frozen_eval_dir), "identity_gate": str(args.identity_gate),
                       "reuse_gate": str(args.reuse_gate)},
            "analysis_code_sha256": sha256(Path(__file__)),
            "caveats": ["Validation only; the official checkpoint was selected on this split and the hypothesis came from 2,806 of these clips.",
                        "Six validation signers (P13 has 2 clips); McNemar and the bootstrap treat clips as independent.",
                        "Any gain is a two-view ensemble of re-extraction plus orientation, not a handedness correction.",
                        "Not adopted by this experiment; test use needs a separate frozen decision and flipped extraction of test videos."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=WORKSPACE / "docs/environment/poet-mirror-tta-fullval-preregistration-20261011.json")
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=WORKSPACE / "data/asl_citizen/val_all_manifest.json")
    parser.add_argument("--historical-predictions", type=Path, default=WORKSPACE / "runs/20260923T050910Z-official-val-v2-evaluation-019ecd54/evaluation/predictions.jsonl")
    parser.add_argument("--prior-mirrored-predictions", type=Path, default=WORKSPACE / "runs/20261010T154553Z-poet-handedness-inference-cd7e0157/inference/predictions_mirrored.jsonl")
    parser.add_argument("--identity-gate", type=Path, required=True)
    parser.add_argument("--reuse-gate", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True, help="extraction directory of the mirror run")
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--frozen-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = analyze(args)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    primary = result["primary"]
    print(json.dumps({"all_gates_passed": result["all_gates_passed"], "decision": result["decision"]["verdict"],
                      "pattern": result["decision"]["pattern"], "top1_original": primary["base_top1"], "top1_tta": primary["arm_top1"],
                      "delta_pp": primary["delta_top1_pp"], "fixed": primary["fixed"], "regressed": primary["regressed"],
                      "p": primary["mcnemar_exact_p"], "problems": len(result["problems"])}), flush=True)
    return 0 if result["all_gates_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
