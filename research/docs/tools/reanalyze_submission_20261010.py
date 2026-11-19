#!/usr/bin/env python3
"""Read-only re-derivation of the numbers used in the 2026-10-10 submission revision.

No model is loaded or executed, no test data is read and no network is used. Inputs are
the locally mirrored, SHA-pinned evidence of completed validation runs plus the PoET paper.
Output: docs/environment/submission-reanalysis-20261010.json (or the path given as argv[1]).
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # poet-workspace
COURSE = ROOT.parents[1]                            # course root
N = 10304

PINNED = {
    "factorial_analysis": ("runs/20260926T224941Z-factorial-retrieval-analysis-06b2704e/analysis.json",
                           "7d919e7017852e419e5992aef91abdcfa5c49320722d74c3a7671c864205d1a5"),
    "paired_seed_analysis": ("docs/environment/paired-seed-analysis-20260926.json",
                             "bdc9beb03dc18827a16c689fc0895dd99cbb3c0ea6abf1c46b109d317d56fb00"),
    "retrieval_pilot_analysis": ("runs/20260926T203914Z-retrieval-pilot-completion-analysis-a0123bf1/analysis.json",
                                 "cc9896a4417668525fbb0f985386d9f56ba7b3b0d9bb7f596fa934c237132bc7"),
    "handedness_analysis": ("runs/20261010T084703Z-handedness-analysis-5acb02a8/analysis/analysis.json",
                            "d16574abd6fa1835090388757259506203ace2bddad7d4efb55e3858a58afdfd"),
    "handedness_preregistration": ("docs/environment/handedness-preregistration-20261010.json",
                                   "c1077c17148b2bb853c8677d165d3153e200c4727de51847138758ca5f5e7ff1"),
    "knn_predictions": ("runs/20261002T161254Z-classical-formal-three-methods-dac0c02f/experiment/knn/predictions.jsonl",
                        "c22ac4645ceb32bc4fe1657f2e48a05da6406d4e8e8a80d590c302c6a5e353d2"),
    # Inference-only PoET flip check completed by a sibling session on 2026-10-10 (docs/POET_HANDEDNESS_CHECK.md).
    "poet_handedness_analysis": ("runs/20261010T154740Z-poet-handedness-analysis-4ebed291/analysis/analysis.json",
                                 "32cb04c1763afdc17d858d90112c0b6b5d866f2182f488c36a48ebc4403a18ff"),
}
UNPINNED = {
    "comparison_demo_analysis": "docs/comparison-demo/analysis.json",
    "retrieval_preregistration": "docs/environment/retrieval-preregistration-20260926.json",
}
PAPER = COURSE / "project/papers/PoET-VISAPP-2026.pdf"
EXTENDED_GLOBS = ("runs/20260927T0*-extended-baseline-seed*/training/metrics_history.json",
                  "runs/20260927T0*-extended-geometry-seed*/training/metrics_history.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str):
    rel, expected = PINNED[name]
    path = ROOT / rel
    got = sha(path)
    assert got == expected, (name, got)
    return path, got


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test: binomial(b+c, 1/2) on the discordant pairs."""
    n = b + c
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def t975_df2() -> float:
    # Student t with 2 df has CDF 1/2 + t / (2 sqrt(2 + t^2)); solve F(t) = 0.975 in closed form.
    q = 0.95
    return math.sqrt(2 * q * q / (1 - q * q))


def seed_interval(values_pp: list[float]) -> dict:
    assert len(values_pp) == 3
    mean = statistics.mean(values_pp)
    sd = statistics.stdev(values_pp)
    half = t975_df2() * sd / math.sqrt(3)
    return {"per_seed_pp": values_pp, "mean_pp": mean, "sample_sd_pp": sd, "t_0975_df2": t975_df2(),
            "ci95_low_pp": mean - half, "ci95_high_pp": mean + half}


def macro_best_epoch(epochs: list[dict], last_epoch: int) -> dict:
    """Rule used by the training code: highest full-validation macro Top-1, epoch 0 included, ties keep earlier."""
    best = None
    for e in epochs:
        if e["epoch"] > last_epoch:
            continue
        key = e["metrics"]["per_class_full_vocabulary"]["top1"]
        if best is None or key > best["metrics"]["per_class_full_vocabulary"]["top1"]:
            best = e
    return {"epoch": best["epoch"], "top1_correct": best["metrics"]["top1_correct"]}


def main(out: Path) -> dict:
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "read-only re-derivation from saved validation evidence; no model execution, no test data",
              "test_used": False, "inputs": {}}

    # 1. Selection-free view of the extended B/G training histories (epochs 0-20, same seeds).
    histories = {}
    for pattern in EXTENDED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            run = path.parents[1].name
            method = "B" if "-extended-baseline-" in run else "G"
            seed = int(run.split("seed")[1].split("-")[0])
            data = json.loads(path.read_text())
            epochs = sorted(data["epochs"], key=lambda e: e["epoch"])
            assert [e["epoch"] for e in epochs] == list(range(21)), run
            for e in epochs:
                m = e["metrics"]
                assert e["split"] == "val" and m["attempted"] == N and m["failed"] == 0, run
            histories[(method, seed)] = epochs
            result["inputs"][str(path.relative_to(ROOT))] = sha(path)
    seeds = sorted({s for _, s in histories})
    assert seeds == [20260922, 20260923, 20260924] and len(histories) == 6
    correct = {k: [e["metrics"]["top1_correct"] for e in v] for k, v in histories.items()}
    for s in seeds:
        assert correct[("B", s)][0] == correct[("G", s)][0] == 8340, "epoch 0 must equal the official checkpoint"
    b_all = [correct[("B", s)][e] for s in seeds for e in range(1, 21)]
    g_all = [correct[("G", s)][e] for s in seeds for e in range(1, 21)]
    diffs = [correct[("G", s)][e] - correct[("B", s)][e] for s in seeds for e in range(1, 21)]
    diffs10 = [correct[("G", s)][e] - correct[("B", s)][e] for s in seeds for e in range(1, 11)]
    b_sd_by_seed = {s: statistics.stdev(correct[("B", s)][1:21]) for s in seeds}
    g_sd_by_seed = {s: statistics.stdev(correct[("G", s)][1:21]) for s in seeds}
    pooled_b_sd = math.sqrt(statistics.mean(v * v for v in b_sd_by_seed.values()))
    best10 = {f"{m}_{s}": macro_best_epoch(histories[(m, s)], 10) for m in "BG" for s in seeds}
    best20 = {f"{m}_{s}": macro_best_epoch(histories[(m, s)], 20) for m in "BG" for s in seeds}
    # The 10-epoch checkpoints used in the paper's ablation (Table 3) must be reproduced by the rule.
    expected10 = {"B_20260922": (2, 8369), "B_20260923": (10, 8362), "B_20260924": (10, 8365),
                  "G_20260922": (10, 8383), "G_20260923": (10, 8401), "G_20260924": (10, 8401)}
    for key, (epoch, count) in expected10.items():
        assert (best10[key]["epoch"], best10[key]["top1_correct"]) == (epoch, count), (key, best10[key])
    best10_b = statistics.mean(best10[f"B_{s}"]["top1_correct"] for s in seeds)
    best10_g = statistics.mean(best10[f"G_{s}"]["top1_correct"] for s in seeds)
    result["selection_free_training_curves"] = {
        "runs": "extended 20-epoch B/G runs; epochs 1-10 continue the original 10-epoch histories",
        "official_checkpoint_top1_correct": 8340,
        "B_mean_top1_correct_epochs_1_20": statistics.mean(b_all),
        "B_mean_top1_percent_epochs_1_20": 100 * statistics.mean(b_all) / N,
        "G_mean_top1_correct_epochs_1_20": statistics.mean(g_all),
        "G_mean_top1_percent_epochs_1_20": 100 * statistics.mean(g_all) / N,
        "B_epoch_to_epoch_sd_clips_by_seed": b_sd_by_seed,
        "B_epoch_to_epoch_sd_clips_pooled": pooled_b_sd,
        "G_epoch_to_epoch_sd_clips_by_seed": g_sd_by_seed,
        "G_minus_B_same_seed_same_epoch_1_20": {
            "comparisons": len(diffs), "G_higher": sum(d > 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs), "G_lower": sum(d < 0 for d in diffs),
            "mean_clips": statistics.mean(diffs), "mean_pp": 100 * statistics.mean(diffs) / N,
            "min_clips": min(diffs), "max_clips": max(diffs)},
        "G_minus_B_same_seed_same_epoch_1_10": {
            "comparisons": len(diffs10), "G_higher": sum(d > 0 for d in diffs10),
            "ties": sum(d == 0 for d in diffs10), "G_lower": sum(d < 0 for d in diffs10),
            "mean_clips": statistics.mean(diffs10), "mean_pp": 100 * statistics.mean(diffs10) / N},
        "best_epoch_rule": "max full-validation macro Top-1 including epoch 0; ties keep the earlier epoch",
        "best_epoch_within_10": best10, "best_epoch_within_20": best20,
        "best10_B_mean_top1_correct": best10_b, "best10_G_mean_top1_correct": best10_g,
        "best10_G_minus_B_pp": 100 * (best10_g - best10_b) / N,
        "best10_B_minus_selection_free_B_pp": 100 * (best10_b - statistics.mean(b_all)) / N,
        "per_epoch_top1_correct": {f"{m}_{s}": correct[(m, s)] for m in "BG" for s in seeds},
    }

    # 2. Paired exact McNemar per seed and seed-level t intervals (n = 3).
    fpath, fsha = load("factorial_analysis")
    result["inputs"][PINNED["factorial_analysis"][0]] = fsha
    factorial = json.loads(fpath.read_text())["factorial"]
    names = {"G_vs_B": "G_vs_B", "G_plus_R_vs_B": "G_plus_R_vs_B", "G_plus_R_vs_G": "G_plus_R_vs_G",
             "B_plus_R_vs_B": "B_plus_R_vs_B", "G_plus_R_vs_B_plus_R": "G_plus_R_vs_B_plus_R"}
    paired = {}
    for label, key in names.items():
        rows = []
        for s in seeds:
            c = factorial["by_seed"][str(s)]["comparisons"][key]
            assert c["samples"] == N and c["net_correct"] == c["fixed_count"] - c["regressed_count"]
            rows.append({"seed": s, "fixed": c["fixed_count"], "regressed": c["regressed_count"],
                         "net": c["net_correct"], "delta_pp": c["delta_top1_percentage_points"],
                         "exact_mcnemar_p": mcnemar_exact(c["fixed_count"], c["regressed_count"])})
        paired[label] = {"per_seed": rows, "seed_level": seed_interval([r["delta_pp"] for r in rows])}
    ppath, psha = load("paired_seed_analysis")
    result["inputs"][PINNED["paired_seed_analysis"][0]] = psha
    for seed_row, row in zip(json.loads(ppath.read_text())["paired_seeds"], paired["G_vs_B"]["per_seed"]):
        assert seed_row["seed"] == row["seed"]
        assert (seed_row["paired"]["incorrect_to_correct"], seed_row["paired"]["correct_to_incorrect"]) == \
               (row["fixed"], row["regressed"]), "independent paired-seed analysis disagrees"
    result["paired_statistics"] = paired
    result["paired_statistics_caveats"] = [
        "McNemar treats the 10,304 clips as independent, but they come from only six validation signers.",
        "The same validation split selected every checkpoint and is reused across seeds; p-values are descriptive.",
        "Seed intervals use n = 3 (t with 2 df) and describe seed-to-seed variation only."]

    # 3. Retrieval pilot on the official checkpoint (no epoch selection) and its preregistered gate.
    rpath, rsha = load("retrieval_pilot_analysis")
    result["inputs"][PINNED["retrieval_pilot_analysis"][0]] = rsha
    pilot = json.loads(rpath.read_text())
    blend = pilot["paired"]["fixed_blend"]
    prereg = json.loads((ROOT / UNPINNED["retrieval_preregistration"]).read_text())
    result["inputs"][UNPINNED["retrieval_preregistration"]] = sha(ROOT / UNPINNED["retrieval_preregistration"])
    gate = prereg["continuation_signal"]["minimum_net_correct"]
    assert pilot["continuation_signal"]["minimum_net_correct"] == gate == 52
    result["retrieval_pilot_official_checkpoint"] = {
        "fixed": blend["fixed"], "regressed": blend["regressed"], "net": blend["net_correct"],
        "delta_pp": 100 * blend["net_correct"] / N, "exact_mcnemar_p": mcnemar_exact(blend["fixed"], blend["regressed"]),
        "preregistered_minimum_net": gate, "gate_passed": pilot["continuation_signal"]["pass"],
        "macro_gain_pp": pilot["continuation_signal"]["macro_gain_percentage_points"]}

    # 4. Seed used for cases and the demo.
    gains = {r["seed"]: r["net"] for r in paired["G_plus_R_vs_B"]["per_seed"]}
    result["case_and_demo_seed"] = {"seed": 20260923, "G_plus_R_minus_B_net_by_seed": gains,
                                    "is_largest_gain": max(gains, key=gains.get) == 20260923}

    # 5. Four-system table and classical handedness experiment.
    demo_path = ROOT / UNPINNED["comparison_demo_analysis"]
    result["inputs"][UNPINNED["comparison_demo_analysis"]] = sha(demo_path)
    demo = json.loads(demo_path.read_text())
    assert demo["count"] == N and not demo["test_used"]
    table2 = {m["id"]: {"top1_correct": m["top1_correct"], "top1": 100 * m["top1"], "top5": 100 * m["top5"],
                        "macro_top1": 100 * m["macro_top1"]} for m in demo["methods"]}
    hpath, hsha = load("handedness_analysis")
    result["inputs"][PINNED["handedness_analysis"][0]] = hsha
    _, prsha = load("handedness_preregistration")
    result["inputs"][PINNED["handedness_preregistration"][0]] = prsha
    hand = json.loads(hpath.read_text())
    assert hand["status"] == "verified" and not hand["test_used"] and hand["validation_rows"] == N
    assert hand["preregistered_evaluation"]["verdict"] == "supported"
    mirrored = {split: sorted((p for p, d in hand["decisions"][split].items() if d["mirror"]),
                              key=lambda p: -hand["decisions"][split][p]["clips"]) for split in ("train", "val")}
    handed = {}
    for method in ("knn", "linear_svm", "random_forest"):
        arms = hand["results"][method]
        base = arms["baseline"]["overall"]
        assert base["top1_correct"] == table2[method]["top1_correct"], "baseline mismatch with Table 2"
        handed[method] = {}
        for arm in ("baseline", "arm0_query_only", "armA_canonical_refit"):
            a = arms[arm]
            entry = {"overall_top1_correct": a["overall"]["top1_correct"], "overall_top1": 100 * a["overall"]["top1"],
                     "overall_top5": 100 * a["overall"]["top5"],
                     "overall_macro_top1": 100 * a["overall"]["macro_top1_observed_classes"],
                     "mirrored_signers_top1": 100 * a["mirrored_signers"]["top1"],
                     "unmirrored_signers_top1": 100 * a["unmirrored_signers"]["top1"],
                     "per_signer_top1": {p: 100 * v["top1"] for p, v in a["per_signer"].items()}}
            if arm != "baseline":
                pv = a["paired_vs_baseline"]["overall"]
                entry["paired_vs_baseline"] = pv
                fixed = pv.get("fixed", pv.get("fixed_count"))
                regressed = pv.get("regressed", pv.get("regressed_count"))
                entry["recomputed_exact_mcnemar_p"] = mcnemar_exact(fixed, regressed)
            handed[method][arm] = entry
    result["table2"] = table2
    result["handedness"] = {"mirrored_signers": mirrored,
                            "mirrored_clips": {split: sum(hand["decisions"][split][p]["clips"] for p in mirrored[split])
                                               for split in ("train", "val")},
                            "verdict": hand["preregistered_evaluation"]["verdict"],
                            "hypotheses": hand["preregistered_evaluation"], "methods": handed}

    # 6. kNN Top-5 returns only voted classes.
    kpath, ksha = load("knn_predictions")
    result["inputs"][PINNED["knn_predictions"][0]] = ksha
    counts = {}
    for line in kpath.read_text().splitlines():
        record = json.loads(line)
        counts[len(record["top5_ids"])] = counts.get(len(record["top5_ids"]), 0) + 1
    assert sum(counts.values()) == N
    result["knn_top5_class_counts"] = {"distribution": dict(sorted(counts.items())),
                                       "fewer_than_five": sum(v for k, v in counts.items() if k < 5)}

    # 6b. Official PoET on horizontally flipped videos (inference only; validation subsets).
    qpath, qsha = load("poet_handedness_analysis")
    result["inputs"][PINNED["poet_handedness_analysis"][0]] = qsha
    flip = json.loads(qpath.read_text())
    assert flip["all_gates_passed"] and not flip["test_used"] and not flip["training_or_parameter_update"]
    cohorts = {}
    for name in ("left_dominant", "control"):
        m = flip["results"]["cohorts"][name]["mirrored_vs_original"]
        assert m["net"] == m["fixed"] - m["regressed"]
        cohorts[name] = {k: m[k] for k in ("clips", "base_top1_correct", "arm_top1_correct", "fixed", "regressed", "net")}
        cohorts[name].update({"original_top1": 100 * m["base_top1"], "flipped_top1": 100 * m["arm_top1"],
                              "delta_pp": m["delta_top1_pp"], "exact_mcnemar_p": mcnemar_exact(m["fixed"], m["regressed"]),
                              "reported_p": m["mcnemar_exact_p"]})
        assert abs(cohorts[name]["exact_mcnemar_p"] - m["mcnemar_exact_p"]) < 1e-12
    p5 = flip["results"]["signers"]["P5"]["mirrored_vs_original"]
    result["poet_flip_check"] = {"verdict": flip["decision"]["verdict"], "pattern": flip["decision"]["pattern"],
                                 "cohorts": cohorts, "P5_original_top1": 100 * p5["base_top1"],
                                 "P5_flipped_top1": 100 * p5["arm_top1"],
                                 "difference_in_differences_pp": flip["bootstrap"]["point_difference_in_differences_pp"]["mirrored"],
                                 "difference_in_differences_ci95_pp": flip["bootstrap"]["mirrored"]["difference_in_differences"]["ci95_pp"]}

    # 7. PoET paper statements used in the disclosure.
    from pypdf import PdfReader
    text = " ".join((p.extract_text() or "") for p in PdfReader(str(PAPER)).pages)
    flat = " ".join(text.split())
    needles = {"optuna": "hyperparameter optimization with Optuna",
               "early_stopping": "Early stopping with a patience of 20 based on per-class accuracy",
               "selection": "The model with the highest validation per-class accuracy was selected",
               "test_metric": "Performance was evaluated on the test set",
               "horizontal_flip": "horizontal flip with a proba",
               "table6_row": "PoET (ours) 773k 73.54 92.62 73.52 92.63"}
    found = {k: (v in flat) for k, v in needles.items()}
    assert all(found.values()), found
    result["inputs"][str(PAPER.relative_to(COURSE))] = sha(PAPER)
    result["poet_paper"] = {"section": "3.2-3.3 and Table 6", "statements_found": found,
                            "asl_citizen_test_pi_top1": 73.54, "asl_citizen_test_pc_top1": 73.52,
                            "asl_citizen_test_pi_top5": 92.62, "asl_citizen_test_pc_top5": 92.63,
                            "parameters": "773k", "our_validation_top1_same_checkpoint": table2["poet_official_v2"]["top1"],
                            "validation_minus_paper_test_pp": table2["poet_official_v2"]["top1"] - 73.54}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/environment/submission-reanalysis-20261010.json"
    r = main(target)
    sf = r["selection_free_training_curves"]
    print("B mean epochs1-20:", round(sf["B_mean_top1_correct_epochs_1_20"], 2), "SD by seed:",
          {k: round(v, 1) for k, v in sf["B_epoch_to_epoch_sd_clips_by_seed"].items()})
    print("G-B same epoch 1-20:", sf["G_minus_B_same_seed_same_epoch_1_20"])
    print("best10 G-B pp:", round(sf["best10_G_minus_B_pp"], 4), "best10 B optimism pp:",
          round(sf["best10_B_minus_selection_free_B_pp"], 4))
    for label, v in r["paired_statistics"].items():
        print(label, [(x["fixed"], x["regressed"], f"{x['exact_mcnemar_p']:.2g}") for x in v["per_seed"]],
              "CI", round(v["seed_level"]["ci95_low_pp"], 3), round(v["seed_level"]["ci95_high_pp"], 3))
    print("pilot:", r["retrieval_pilot_official_checkpoint"])
    print("knn:", r["knn_top5_class_counts"])
    print("handedness mirrored:", r["handedness"]["mirrored_signers"], r["handedness"]["mirrored_clips"])
    print("paper:", r["poet_paper"]["statements_found"], round(r["poet_paper"]["validation_minus_paper_test_pp"], 4))
