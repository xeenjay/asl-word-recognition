#!/usr/bin/env python3
"""Independent analysis of the PoET handedness (mirror) check.

Recomputes every gate, metric and the preregistered verdict from saved
artifacts only: per-clip predictions, logits, raw pose caches and the
historical official predictions. It shares no code with the evaluator; the
classical descriptor functions are imported only for the label-free G4/D2
indicators. Fixed by docs/environment/poet-handedness-preregistration-20261010.json.
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
PREREGISTRATION_ID = "poet-handedness-mirror-check-v1"
SOURCE_MANIFEST_SHA256 = "b59fa41ee85955a310d665a3af09a67274a6ddb48b4d18cae1faaa06c6fb7a4f"
HISTORICAL_PREDICTIONS_SHA256 = "8d7e6fe2d347ef7a23ef5b8b59a0534602109d83148fcbedeb81545651aaaca5"
COHORT_SIGNERS = {"left_dominant": ("P12", "P5"), "control": ("P21", "P26", "P39")}
CONTROL_PER_SIGNER, CONTROL_SALT = 500, "poet-handedness-control-20261010:"
IDENTITY_SIGNERS, IDENTITY_PER_SIGNER, IDENTITY_SALT = ("P12", "P5", "P21", "P26", "P39"), 4, "poet-handedness-identity-20261010:"
MATERIAL_PP = 2.0
ALPHA = 0.05
PROBABILITY_TOLERANCE = 1e-5
MOTION_GATE = 0.75  # preregistered all-clip G4 value; reported only after amendment 1
CLEAR_MOTION_RATIO, CLEAR_GATE = 0.5, 0.90  # amendment 1 binding G4
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261010


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mcnemar_exact(fixed: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = fixed + regressed
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(fixed, regressed) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def recompute_cohorts(samples: list[dict]) -> dict[str, list[int]]:
    """Independent re-implementation of the preregistered label-free cohort rule."""
    by_signer: dict[str, list[tuple[str, int]]] = {}
    for position, sample in enumerate(samples):
        assert sample["index"] == position and sample["split"] in {"val", "validation"}
        by_signer.setdefault(sample["signer_id"], []).append((sample["sha256"], position))

    def first(signer: str, salt: str, count: int) -> list[int]:
        keyed = sorted((hashlib.sha256((salt + digest).encode("ascii")).hexdigest(), position)
                       for digest, position in by_signer[signer])
        return [position for _, position in keyed[:count]]

    return {"left_dominant": sorted(p for s in COHORT_SIGNERS["left_dominant"] for _, p in by_signer[s]),
            "control": sorted(p for s in COHORT_SIGNERS["control"] for p in first(s, CONTROL_SALT, CONTROL_PER_SIGNER)),
            "identity": sorted(p for s in IDENTITY_SIGNERS for p in first(s, IDENTITY_SALT, IDENTITY_PER_SIGNER))}


def correctness(record: dict) -> tuple[bool, bool, int]:
    ids = [p["class_id"] for p in record["top5"]]
    return ids[0] == record["class_id"], record["class_id"] in ids, ids[0]


def paired(base_rows: list[dict], arm_rows: list[dict]) -> dict:
    n = len(base_rows)
    base_top1 = sum(r["top1"] for r in base_rows)
    arm_top1 = sum(r["top1"] for r in arm_rows)
    fixed = sum((not b["top1"]) and a["top1"] for b, a in zip(base_rows, arm_rows))
    regressed = sum(b["top1"] and (not a["top1"]) for b, a in zip(base_rows, arm_rows))
    return {"clips": n, "base_top1_correct": base_top1, "arm_top1_correct": arm_top1,
            "base_top1": base_top1 / n, "arm_top1": arm_top1 / n,
            "delta_top1_pp": 100 * (arm_top1 - base_top1) / n,
            "base_top5": sum(r["top5"] for r in base_rows) / n, "arm_top5": sum(r["top5"] for r in arm_rows) / n,
            "delta_top5_pp": 100 * (sum(r["top5"] for r in arm_rows) - sum(r["top5"] for r in base_rows)) / n,
            "fixed": fixed, "regressed": regressed, "net": fixed - regressed,
            "mcnemar_exact_p": mcnemar_exact(fixed, regressed),
            "top1_prediction_agreement": sum(b["pred"] == a["pred"] for b, a in zip(base_rows, arm_rows)) / n}


def bootstrap(diffs_by_signer: dict[str, list[dict]], rng, resamples: int):
    """Signer-stratified clip bootstrap of a pooled mean difference (percentage points)."""
    import numpy as np
    total = sum(len(v) for v in diffs_by_signer.values())
    sums = np.zeros(resamples)
    for signer in sorted(diffs_by_signer):
        values = np.asarray(diffs_by_signer[signer], dtype=np.float64)
        draws = rng.integers(0, len(values), size=(resamples, len(values)))
        sums += values[draws].sum(axis=1)
    return 100 * sums / total


def load_arrays(path: str) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def extraction_diagnostics(original: dict, mirrored: dict) -> dict:
    """Compare flipped-video extraction with the coordinate mirror of the original."""
    import numpy as np
    from classical_handedness import pose_permutation
    perm = pose_permutation()
    result = {}
    body_both = original["body_pose_present"] & mirrored["body_pose_present"]
    if body_both.any():
        expected = original["body_pose"][body_both][:, perm, :2].copy()
        expected[..., 0] = 1.0 - expected[..., 0]
        result["body_median_xy_distance"] = float(np.median(np.linalg.norm(mirrored["body_pose"][body_both][..., :2] - expected, axis=-1)))
    deviations = []
    for new_side, old_side in (("left_hand", "right_hand"), ("right_hand", "left_hand")):
        both = mirrored[new_side + "_present"] & original[old_side + "_present"]
        if both.any():
            expected = original[old_side][both][..., :2].copy()
            expected[..., 0] = 1.0 - expected[..., 0]
            deviations.append(np.linalg.norm(mirrored[new_side][both][..., :2] - expected, axis=-1).ravel())
    if deviations:
        result["hand_median_xy_distance"] = float(np.median(np.concatenate(deviations)))
    frames = len(original["right_hand_present"])
    result["presence_swap_agreement"] = float(np.mean((mirrored["left_hand_present"] == original["right_hand_present"])
                                                      & (mirrored["right_hand_present"] == original["left_hand_present"])))
    result["frames"] = frames
    for side in ("left_hand", "right_hand"):
        result[f"original_{side}_frames"] = int(original[side + "_present"].sum())
        result[f"mirrored_{side}_frames"] = int(mirrored[side + "_present"].sum())
    result["original_any_hand_frames"] = int((original["left_hand_present"] | original["right_hand_present"]).sum())
    result["mirrored_any_hand_frames"] = int((mirrored["left_hand_present"] | mirrored["right_hand_present"]).sum())
    return result


def indicators(raw: dict) -> dict:
    import numpy as np
    from classical_features import extract_features
    from classical_handedness import COVERAGE_LEFT, COVERAGE_RIGHT, MOTION_LEFT, MOTION_RIGHT
    x = extract_features(raw).astype(np.float64)
    lm, rm = float(x[list(MOTION_LEFT)].sum()), float(x[list(MOTION_RIGHT)].sum())
    return {"left_coverage": bool(x[COVERAGE_LEFT] > x[COVERAGE_RIGHT]), "right_coverage": bool(x[COVERAGE_RIGHT] > x[COVERAGE_LEFT]),
            "left_motion": lm > rm, "right_motion": rm > lm, "motion_ratio": min(lm, rm) / max(lm, rm) if max(lm, rm) > 0 else 1.0}


def verdict(delta_l: float, p_l: float, delta_r: float, p_r: float) -> dict:
    """Preregistered rule: H1 (left-dominant signers gain when mirrored) is necessary."""
    h1 = delta_l >= MATERIAL_PP and p_l < ALPHA
    h2 = delta_r <= -MATERIAL_PP and p_r < ALPHA
    if h1:
        label, pattern = ("supported", "left_dominant_signers_hurt") if h2 else ("partially_supported", "left_gain_only")
    else:
        label = "not_supported"
        if delta_l <= -MATERIAL_PP and delta_r <= -MATERIAL_PP:
            pattern = "mirror_penalty"
        elif abs(delta_l) < MATERIAL_PP and abs(delta_r) < MATERIAL_PP:
            pattern = "approximately_invariant"
        elif h2:
            pattern = "control_loss_only"
        else:
            pattern = "mixed"
    return {"H1": h1, "H2": h2, "verdict": label, "pattern": pattern}


def run(args: argparse.Namespace) -> int:
    import numpy as np
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if prereg.get("id") != PREREGISTRATION_ID:
        raise ValueError("Unexpected preregistration document")
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if amendment.get("id") != PREREGISTRATION_ID + "-amendment-1" or amendment["amends"]["sha256"] != sha256(args.preregistration):
        raise ValueError("Amendment does not belong to this preregistration")
    smoke = json.loads(args.smoke_reassessment.read_text(encoding="utf-8"))
    problems = []
    if smoke.get("status") != "passed":
        problems.append("smoke re-assessment under amendment 1 did not pass")

    # Cohorts: independent recomputation of the label-free rule.
    if sha256(args.source_manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Validation source manifest SHA256 differs")
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))["samples"]
    cohorts = recompute_cohorts(source)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    for name in ("left_dominant", "control", "identity"):
        if sorted(row["index"] for row in selection["cohorts"][name]) != cohorts[name]:
            problems.append(f"selection.json {name} cohort differs from the independent recomputation")
    cohort_of = {i: name for name in ("left_dominant", "control") for i in cohorts[name]}
    indices = sorted(cohort_of)

    # Predictions of the evaluator, checked against history and the frozen evaluator.
    if sha256(args.historical_predictions) != HISTORICAL_PREDICTIONS_SHA256:
        raise ValueError("Historical predictions SHA256 differs")
    historical = {row["index"]: row for row in read_jsonl(args.historical_predictions)}
    if len(historical) != 10304 or any(row["status"] != "ok" for row in historical.values()):
        raise ValueError("Historical official predictions must cover all 10,304 validation clips")
    arms = {}
    for arm in ("original", "mirrored", "tta"):
        rows = read_jsonl(args.inference_dir / f"predictions_{arm}.jsonl")
        if [r["index"] for r in rows] != indices:
            raise ValueError(f"{arm} predictions do not cover exactly the preregistered clips in order")
        converted = []
        for r in rows:
            past = historical[r["index"]]
            if r["class_id"] != past["class_id"] or r["signer_id"] != past["signer_id"] or r["gloss"] != past["gloss"]:
                raise ValueError(f"Label/signer mismatch at {r['index']}")
            top1, top5, pred = correctness(r)
            converted.append({"index": r["index"], "signer": r["signer_id"], "cohort": cohort_of[r["index"]],
                              "top1": top1, "top5": top5, "pred": pred, "ids": [p["class_id"] for p in r["top5"]],
                              "probs": [p["probability"] for p in r["top5"]]})
        arms[arm] = converted

    def compare(rows_a, rows_b, tolerance):
        bad, worst = [], 0.0
        for a, b in zip(rows_a, rows_b):
            ids_b = [p["class_id"] for p in b["top5"]] if "top5" in b else b["ids"]
            probs_b = [p["probability"] for p in b["top5"]] if "top5" in b else b["probs"]
            difference = max(abs(x - y) for x, y in zip(a["probs"], probs_b))
            worst = max(worst, difference)
            if a["ids"] != ids_b or difference > tolerance:
                bad.append(a["index"])
        return bad, worst

    g2_bad, g2_worst = compare(arms["original"], [historical[i] for i in indices], PROBABILITY_TOLERANCE)
    frozen_summary = json.loads((args.frozen_eval_dir / "summary.json").read_text(encoding="utf-8"))
    frozen = read_jsonl(args.frozen_eval_dir / "predictions.jsonl")
    g3_bad, g3_worst = (None, None)
    if [r["index"] for r in frozen] != indices or frozen_summary.get("status") != "completed" or any(r["status"] != "ok" for r in frozen):
        problems.append("frozen evaluator output does not cover the mirrored clips without failure")
    else:
        g3_bad, g3_worst = compare(arms["mirrored"], frozen, PROBABILITY_TOLERANCE)

    # TTA and top-5 lists recomputed from the saved logits (float64).
    with np.load(args.inference_dir / "logits.npz", allow_pickle=False) as archive:
        logit_indices, original_logits, mirrored_logits = archive["indices"], archive["original"], archive["mirrored"]
    if logit_indices.tolist() != indices:
        raise ValueError("Logit rows do not match the preregistered clips")
    tta = 0.5 * (original_logits.astype(np.float64) + mirrored_logits.astype(np.float64))
    order = np.argsort(-tta, axis=1, kind="stable")[:, :5]
    tta_mismatch = [i for i, ids, row in zip(indices, order.tolist(), arms["tta"]) if ids != row["ids"]]
    logit_top1_mismatch = {}
    for arm, values in (("original", original_logits), ("mirrored", mirrored_logits)):
        best = np.argmax(values, axis=1)
        # A different argmax is accepted only for an exact float32 tie (topk tie order is implementation-defined).
        logit_top1_mismatch[arm] = [i for row_number, (i, row) in enumerate(zip(indices, arms[arm]))
                                    if best[row_number] != row["pred"]
                                    and values[row_number, best[row_number]] != values[row_number, row["pred"]]]

    # Label-free engineering evidence: G1 re-compared, G4 recomputed, G5 checked.
    identity_gate = json.loads(args.identity_gate.read_text(encoding="utf-8"))
    reference = {item["index"]: item for item in json.loads(args.reference_index.read_text(encoding="utf-8"))["samples"]}
    identity_index = {item["index"]: item for item in json.loads(args.identity_index.read_text(encoding="utf-8"))["samples"]}
    g1_rows = []
    for index in cohorts["identity"]:
        new, old = identity_index[index], reference[index]
        same = new["status"] == "ok"
        for key in ("pose_path", "features_path"):
            if same:
                a, b = load_arrays(new[key]), load_arrays(old[key])
                same = set(a) == set(b) and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in b)
        g1_rows.append({"index": index, "identical": bool(same)})
    g1_pass = identity_gate.get("status") == "passed" and all(row["identical"] for row in g1_rows) and len(g1_rows) == 20
    mirror_index = json.loads(args.mirror_index.read_text(encoding="utf-8"))
    mirror_samples = {item["index"]: item for item in mirror_index["samples"]}
    g5_pass = (mirror_index.get("status") == "completed" and sorted(mirror_samples) == indices
               and all(item["status"] == "ok" for item in mirror_samples.values())
               and mirror_index["pipeline"]["extractor"]["flip"] is True)

    clip_diagnostics, indicator_rows = [], []
    for index in indices:
        original_raw, mirrored_raw = load_arrays(reference[index]["pose_path"]), load_arrays(mirror_samples[index]["pose_path"])
        diag = extraction_diagnostics(original_raw, mirrored_raw)
        diag.update(index=index, signer=reference[index]["signer_id"], cohort=cohort_of[index])
        clip_diagnostics.append(diag)
        o, m = indicators(original_raw), indicators(mirrored_raw)
        indicator_rows.append({"index": index, "signer": reference[index]["signer_id"], "cohort": cohort_of[index],
                               "original": o, "mirrored": m,
                               "motion_consistent": m["left_motion"] == o["right_motion"],
                               "coverage_consistent": m["left_coverage"] == o["right_coverage"]})

    def indicator_summary(rows):
        n = len(rows)
        clear = [r for r in rows if r["original"]["motion_ratio"] <= CLEAR_MOTION_RATIO]
        return {"clips": n, "clear_clips": len(clear),
                "clear_motion_consistency": sum(r["motion_consistent"] for r in clear) / len(clear) if clear else None,
                "motion_consistency": sum(r["motion_consistent"] for r in rows) / n,
                "coverage_consistency": sum(r["coverage_consistent"] for r in rows) / n,
                "original_left_coverage_fraction": sum(r["original"]["left_coverage"] for r in rows) / n,
                "original_left_motion_fraction": sum(r["original"]["left_motion"] for r in rows) / n,
                "mirrored_left_coverage_fraction": sum(r["mirrored"]["left_coverage"] for r in rows) / n,
                "mirrored_left_motion_fraction": sum(r["mirrored"]["left_motion"] for r in rows) / n}

    g4 = {c: indicator_summary([r for r in indicator_rows if r["cohort"] == c]) for c in ("left_dominant", "control")}
    g4_pass = all(v["clear_motion_consistency"] is not None and v["clear_motion_consistency"] >= CLEAR_GATE for v in g4.values())
    g4_v1_met = all(v["motion_consistency"] >= MOTION_GATE for v in g4.values())
    flip_sanity = json.loads(args.flip_sanity.read_text(encoding="utf-8"))
    for c in g4:
        for key in ("motion_consistency", "clear_motion_consistency", "clear_clips"):
            if abs(flip_sanity["by_cohort"][c][key] - g4[c][key]) > 1e-12:
                problems.append(f"G4 recomputation of {key} differs from flip_sanity.json for {c}")
    d2 = {s: indicator_summary([r for r in indicator_rows if r["signer"] == s]) for s in sorted({r["signer"] for r in indicator_rows})}

    def diag_summary(rows):
        def median(key):
            values = [r[key] for r in rows if key in r]
            return float(np.median(values)) if values else None
        frames = sum(r["frames"] for r in rows)
        return {"clips": len(rows), "frames": frames,
                "median_clip_body_xy_distance": median("body_median_xy_distance"), "median_clip_hand_xy_distance": median("hand_median_xy_distance"),
                "frame_weighted_presence_swap_agreement": sum(r["presence_swap_agreement"] * r["frames"] for r in rows) / frames,
                "original_any_hand_frame_fraction": sum(r["original_any_hand_frames"] for r in rows) / frames,
                "mirrored_any_hand_frame_fraction": sum(r["mirrored_any_hand_frames"] for r in rows) / frames,
                "original_left_hand_frame_fraction": sum(r["original_left_hand_frames"] for r in rows) / frames,
                "original_right_hand_frame_fraction": sum(r["original_right_hand_frames"] for r in rows) / frames,
                "mirrored_left_hand_frame_fraction": sum(r["mirrored_left_hand_frames"] for r in rows) / frames,
                "mirrored_right_hand_frame_fraction": sum(r["mirrored_right_hand_frames"] for r in rows) / frames}

    d1 = {"by_cohort": {c: diag_summary([r for r in clip_diagnostics if r["cohort"] == c]) for c in ("left_dominant", "control")},
          "by_signer": {s: diag_summary([r for r in clip_diagnostics if r["signer"] == s]) for s in sorted({r["signer"] for r in clip_diagnostics})}}

    gates = {"G1_identity_reextraction": {"passed": g1_pass, "clips": len(g1_rows), "identical": sum(r["identical"] for r in g1_rows)},
             "G2_original_reproduces_history": {"passed": not g2_bad, "mismatches": g2_bad, "max_abs_probability_difference": g2_worst},
             "G3_frozen_evaluator_agrees": {"passed": g3_bad == [], "mismatches": g3_bad, "max_abs_probability_difference": g3_worst},
             "G4_flip_sanity": {"passed": g4_pass, "rule": f"amendment 1: clear-motion clips (min/max <= {CLEAR_MOTION_RATIO}) consistency >= {CLEAR_GATE} per cohort",
                                "preregistered_v1_all_clip_rule_met": g4_v1_met, "by_cohort": g4},
             "G5_mirror_extraction_complete": {"passed": g5_pass, "clips": len(mirror_samples)},
             "tta_recomputed_from_logits": {"passed": not tta_mismatch, "mismatches": tta_mismatch},
             "top1_recomputed_from_logits": {"passed": not any(logit_top1_mismatch.values()), "mismatches": logit_top1_mismatch}}
    all_gates = all(g["passed"] for g in gates.values()) and not problems

    # Metrics.
    def subset(arm, cohort=None, signer=None):
        return [r for r in arms[arm] if (cohort is None or r["cohort"] == cohort) and (signer is None or r["signer"] == signer)]

    results = {"cohorts": {}, "signers": {}}
    for cohort in ("left_dominant", "control"):
        results["cohorts"][cohort] = {f"{arm}_vs_original": paired(subset("original", cohort), subset(arm, cohort))
                                      for arm in ("mirrored", "tta")}
        for signer in COHORT_SIGNERS[cohort]:
            results["signers"][signer] = {"cohort": cohort, **{f"{arm}_vs_original": paired(subset("original", signer=signer), subset(arm, signer=signer))
                                                               for arm in ("mirrored", "tta")}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {}
    for arm in ("mirrored", "tta"):
        draws = {}
        for cohort in ("left_dominant", "control"):
            diffs = {s: [int(a["top1"]) - int(b["top1"]) for b, a in zip(subset("original", signer=s), subset(arm, signer=s))]
                     for s in COHORT_SIGNERS[cohort]}
            draws[cohort] = bootstrap(diffs, rng, BOOTSTRAP_RESAMPLES)
        draws["difference_in_differences"] = draws["left_dominant"] - draws["control"]
        boot[arm] = {k: {"ci95_pp": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in draws.items()}
    point_did = {arm: results["cohorts"]["left_dominant"][f"{arm}_vs_original"]["delta_top1_pp"]
                 - results["cohorts"]["control"][f"{arm}_vs_original"]["delta_top1_pp"] for arm in ("mirrored", "tta")}

    # S2: query-only canonicalization of the frozen-rule signers on the full validation set.
    mirrored_by_index = {r["index"]: r for r in arms["mirrored"]}
    full_top1 = full_top5 = base_top1 = base_top5 = 0
    for index, past in historical.items():
        ids = [p["class_id"] for p in past["top5"]]
        base_top1 += ids[0] == past["class_id"]
        base_top5 += past["class_id"] in ids
        if cohort_of.get(index) == "left_dominant":
            row = mirrored_by_index[index]
            full_top1 += row["top1"]
            full_top5 += row["top5"]
        else:
            full_top1 += ids[0] == past["class_id"]
            full_top5 += past["class_id"] in ids
    s2 = {"clips": len(historical), "historical_top1": base_top1 / len(historical), "historical_top5": base_top5 / len(historical),
          "canonicalized_top1": full_top1 / len(historical), "canonicalized_top5": full_top5 / len(historical),
          "delta_top1_pp": 100 * (full_top1 - base_top1) / len(historical),
          "definition": "historical official predictions for the 8,998 unmirrored clips; mirrored-input predictions for the 1,306 P12/P5 clips"}

    left = results["cohorts"]["left_dominant"]["mirrored_vs_original"]
    control = results["cohorts"]["control"]["mirrored_vs_original"]
    decision = verdict(left["delta_top1_pp"], left["mcnemar_exact_p"], control["delta_top1_pp"], control["mcnemar_exact_p"])
    if not all_gates:
        decision = {"verdict": "not_interpretable_gate_failure", "pattern": None, "H1": None, "H2": None}

    analysis = {"schema_version": 1, "preregistration_id": PREREGISTRATION_ID,
                "preregistration_sha256": sha256(args.preregistration),
                "amendment": {"path": str(args.amendment), "sha256": sha256(args.amendment), "id": amendment["id"]},
                "smoke_reassessment": {"status": smoke["status"], "original_criterion_status": smoke["original_criterion_status"]},
                "created_at_utc": now(),
                "test_used": False, "training_or_parameter_update": False,
                "gates": gates, "problems": problems, "all_gates_passed": all_gates,
                "decision_rule": {"H1": f"left_dominant mirrored-minus-original Top-1 >= +{MATERIAL_PP} pp and exact McNemar p < {ALPHA}",
                                  "H2": f"control mirrored-minus-original Top-1 <= -{MATERIAL_PP} pp and exact McNemar p < {ALPHA}"},
                "decision": decision, "results": results,
                "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                              "scheme": "clips resampled with replacement within each signer; cohorts independent",
                              "point_difference_in_differences_pp": point_did, **boot},
                "secondary_full_validation_canonicalization": s2,
                "diagnostics": {"D1_extraction": d1, "D2_label_free_indicators_by_signer": d2},
                "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
                    ("selection", args.selection), ("historical_predictions", args.historical_predictions),
                    ("mirror_index", args.mirror_index), ("identity_index", args.identity_index),
                    ("identity_gate", args.identity_gate), ("flip_sanity", args.flip_sanity),
                    ("logits", args.inference_dir / "logits.npz"),
                    ("predictions_original", args.inference_dir / "predictions_original.jsonl"),
                    ("predictions_mirrored", args.inference_dir / "predictions_mirrored.jsonl"),
                    ("predictions_tta", args.inference_dir / "predictions_tta.jsonl"),
                    ("frozen_predictions", args.frozen_eval_dir / "predictions.jsonl"))},
                "analysis_code_sha256": sha256(Path(__file__)),
                "caveats": ["Hypothesis is post hoc from the same validation data; validation results show mechanism, not independent confirmation.",
                            "The left-dominant cohort contains only two signers; McNemar and the clip bootstrap treat clips as independent.",
                            "Mirrored input is a re-extraction from flipped frames; MediaPipe is not exactly mirror-equivariant (see D1).",
                            "TTA is a secondary descriptive arm on these clips only, not a full-validation method result."]}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_gates_passed": all_gates, "problems": problems, "decision": decision,
                      "left_dominant": {k: left[k] for k in ("base_top1", "arm_top1", "delta_top1_pp", "fixed", "regressed", "mcnemar_exact_p")},
                      "control": {k: control[k] for k in ("base_top1", "arm_top1", "delta_top1_pp", "fixed", "regressed", "mcnemar_exact_p")}},
                     indent=1), flush=True)
    return 0 if all_gates else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=WORKSPACE / "docs/environment/poet-handedness-preregistration-20261010.json")
    parser.add_argument("--amendment", type=Path, default=WORKSPACE / "docs/environment/poet-handedness-preregistration-amendment1-20261010.json")
    parser.add_argument("--smoke-reassessment", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=WORKSPACE / "data/asl_citizen/val_all_manifest.json")
    parser.add_argument("--reference-index", type=Path, default=WORKSPACE / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json")
    parser.add_argument("--historical-predictions", type=Path, default=WORKSPACE / "runs/20260923T050910Z-official-val-v2-evaluation-019ecd54/evaluation/predictions.jsonl")
    parser.add_argument("--identity-index", type=Path, required=True)
    parser.add_argument("--identity-gate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--mirror-index", type=Path, required=True)
    parser.add_argument("--flip-sanity", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--frozen-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        code = run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    raise SystemExit(code)


if __name__ == "__main__":
    main()
