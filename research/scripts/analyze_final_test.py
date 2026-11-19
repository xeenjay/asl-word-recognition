#!/usr/bin/env python3
"""Independent final-test analysis (standard library only; gate G4).

Reads only saved prediction files of the 29 preregistered systems and the test
cohort identities, recomputes every metric from scratch, and evaluates the
preregistered comparisons. No model, array library or training code is used.
Labels come from the cohort's official glosses mapped through the pinned official
vocabulary (plain-dict pickles opened with a restricted unpickler).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from math import comb, sqrt
from pathlib import Path
import pickle
import random
import statistics

WORKSPACE = Path(__file__).resolve().parents[1]
PREREGISTRATION = WORKSPACE / "docs/environment/final-test-preregistration-20261012.json"
PREREGISTRATION_SHA256 = "e241b1bbb6a7de21187d31a914a3308321132d2f10ed3e0952fbd652b2664286"
LABEL_MAP_SHA256 = "a2d0f112d9c26cdd75fe9beaa10b54747df3793f4334e95539d70f2dde9e86dd"
TEST_COUNT = 32941
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10000, 20261012
SEEDS = (20260922, 20260923, 20260924)
METHODS = ("knn", "linear_svm", "random_forest")


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"unexpected global {module}.{name}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def mcnemar(fixed: int, regressed: int) -> float:
    n = fixed + regressed
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, regressed) + 1)) / 2 ** n)


def system_files(dirs: dict) -> dict[str, Path]:
    files = {"S13": dirs["official"] / "predictions_official_original.jsonl",
             "S14": dirs["official"] / "predictions_official_tta.jsonl"}
    groups = ("classical_original_descriptor_original_config", "classical_canonical_original_config",
              "classical_canonical_tuned", "classical_original_descriptor_tuned")
    number = 1
    for group in groups:
        for method in METHODS:
            files[f"S{number:02d}"] = dirs["classical"] / f"predictions_{group}__{method}.jsonl"
            number += 1
    number = 15
    for seed in SEEDS:
        for method in ("baseline", "geometry"):
            model_id = f"{method}_seed{seed}"
            files[f"S{number:02d}"] = dirs["finetuned"] / f"predictions_{model_id}.jsonl"
            files[f"S{number + 1:02d}"] = dirs["retrieval"] / f"predictions_{model_id}_R.jsonl"
            number += 2
    for i, method in enumerate(METHODS):
        files[f"S{27 + i:02d}"] = dirs["classical"] / f"predictions_cls_classical__{method}.jsonl"
    return files


def load_outcomes(path: Path, label_id: dict, reference: list[dict] | None) -> tuple[list[tuple[bool, bool]], list[dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(len(rows) == TEST_COUNT, f"{path.name}: {len(rows)} rows, expected {TEST_COUNT}")
    outcomes = []
    for i, row in enumerate(rows):
        require(row["index"] == i and row["split"] == "test" and row["status"] == "ok", f"{path.name}: bad row {i}")
        if reference is not None:
            require(row["video_sha256"] == reference[i]["video_sha256"] and row["gloss"] == reference[i]["gloss"]
                    and row["signer_id"] == reference[i]["signer_id"], f"{path.name}: identity differs at {i}")
        target = label_id[row["gloss"]]
        ids = row["top5_ids"]
        require(1 <= len(ids) <= 5 and len(set(ids)) == len(ids), f"{path.name}: invalid top-5 at {i}")
        outcomes.append((ids[0] == target, target in ids))
    return outcomes, rows


def summarize(outcomes, rows) -> dict:
    n = len(outcomes)
    per_class, per_signer = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for (top1, _), row in zip(outcomes, rows):
        per_class[row["gloss"]][0] += 1
        per_class[row["gloss"]][1] += top1
        per_signer[row["signer_id"]][0] += 1
        per_signer[row["signer_id"]][1] += top1
    return {"count": n, "top1_correct": sum(o[0] for o in outcomes), "top1": sum(o[0] for o in outcomes) / n,
            "top5": sum(o[1] for o in outcomes) / n, "macro_top1": sum(c / t for t, c in per_class.values()) / len(per_class),
            "classes": len(per_class), "per_signer": {s: {"count": t, "top1": c / t} for s, (t, c) in sorted(per_signer.items())}}


def paired(base, other) -> dict:
    fixed = sum((not b[0]) and o[0] for b, o in zip(base, other))
    regressed = sum(b[0] and (not o[0]) for b, o in zip(base, other))
    return {"fixed": fixed, "regressed": regressed, "net": fixed - regressed,
            "delta_pp": 100 * (fixed - regressed) / len(base), "mcnemar_exact_p": mcnemar(fixed, regressed)}


def signer_bootstrap(per_signer_correct: dict[str, dict[str, int]], counts: dict[str, int], contrast) -> dict:
    """Resample signers with replacement; contrast maps per-system accuracy -> difference (pp)."""
    signers = sorted(counts)
    rng = random.Random(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = [rng.choice(signers) for _ in signers]
        total = sum(counts[s] for s in chosen)
        accuracy = {k: sum(v[s] for s in chosen) / total for k, v in per_signer_correct.items()}
        values.append(contrast(accuracy))
    values.sort()
    return {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "test signer",
            "ci95_pp": [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "finetuned", "retrieval", "classical"):
        parser.add_argument(f"--{name}-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(sha256(PREREGISTRATION) == PREREGISTRATION_SHA256, "preregistration changed")
    prereg = json.loads(PREREGISTRATION.read_text())
    label_path = WORKSPACE / "assets/checkpoints/d_Label2LId.pkl"
    require(sha256(label_path) == LABEL_MAP_SHA256, "label map changed")
    label_id = PlainUnpickler(io.BytesIO(label_path.read_bytes())).load()
    require(len(label_id) == 2731, "official vocabulary must have 2731 labels")
    dirs = {"official": args.official_dir, "finetuned": args.finetuned_dir, "retrieval": args.retrieval_dir, "classical": args.classical_dir}
    files = system_files({k: v.resolve() for k, v in dirs.items()})
    require(sorted(files) == [s["id"] for s in prereg["systems"]], "system list differs from the preregistration")
    outcomes, metrics, reference, inputs = {}, {}, None, {}
    for sid in sorted(files):
        result, rows = load_outcomes(files[sid], label_id, reference)
        reference = reference or rows
        outcomes[sid] = result
        metrics[sid] = summarize(result, rows)
        inputs[sid] = {"path": str(files[sid]), "sha256": sha256(files[sid])}
    names = {s["id"]: s["name"] for s in prereg["systems"]}
    counts = defaultdict(int)
    for row in reference:
        counts[row["signer_id"]] += 1
    signer_correct = {sid: defaultdict(int) for sid in outcomes}
    for sid, result in outcomes.items():
        for (top1, _), row in zip(result, reference):
            signer_correct[sid][row["signer_id"]] += top1
    comparisons = {}
    # P2: PoET ablation over seeds (S15.. = B, B+R, G, G+R per seed).
    seed_ids = {seed: {"B": f"S{15 + 4 * i:02d}", "B+R": f"S{16 + 4 * i:02d}", "G": f"S{17 + 4 * i:02d}", "G+R": f"S{18 + 4 * i:02d}"}
                for i, seed in enumerate(SEEDS)}
    ablation = {}
    for arm in ("B", "B+R", "G", "G+R"):
        values = [100 * metrics[seed_ids[s][arm]]["top1"] for s in SEEDS]
        ablation[arm] = {"per_seed_top1_pct": values, "mean_pct": statistics.mean(values), "sd_pct": statistics.stdev(values)}
    for a, b in (("G+R", "B"), ("G", "B"), ("B+R", "B"), ("G+R", "G"), ("G+R", "B+R")):
        per_seed = {s: paired(outcomes[seed_ids[s][b]], outcomes[seed_ids[s][a]]) for s in SEEDS}
        diffs = [per_seed[s]["delta_pp"] for s in SEEDS]
        mean = statistics.mean(diffs)
        half = 4.302653 * statistics.stdev(diffs) / sqrt(3)
        boot = signer_bootstrap({k: signer_correct[k] for s in SEEDS for k in (seed_ids[s][a], seed_ids[s][b])}, counts,
                                lambda acc, a=a, b=b: 100 * statistics.mean(acc[seed_ids[s][a]] - acc[seed_ids[s][b]] for s in SEEDS))
        comparisons[f"P2 {a} vs {b}"] = {"per_seed": per_seed, "mean_delta_pp": mean, "seed_t_ci95_pp": [mean - half, mean + half],
                                         "signer_bootstrap": boot}
    # P3: classical handedness and tuning.
    for i, method in enumerate(METHODS):
        for label, base, other in (("canonical vs original descriptor (original config)", f"S{1 + i:02d}", f"S{4 + i:02d}"),
                                   ("tuned vs original config (canonical)", f"S{4 + i:02d}", f"S{7 + i:02d}"),
                                   ("tuned vs original config (original descriptor)", f"S{1 + i:02d}", f"S{10 + i:02d}")):
            comparisons[f"P3 {method} {label}"] = {"base": base, "other": other, **paired(outcomes[base], outcomes[other])}
    # P4: TTA.
    comparisons["P4 official TTA vs official"] = {**paired(outcomes["S13"], outcomes["S14"]),
        "signer_bootstrap": signer_bootstrap({k: signer_correct[k] for k in ("S13", "S14")}, counts, lambda acc: 100 * (acc["S14"] - acc["S13"]))}
    # P5: representation vs classifier (canonical descriptor, original config, as on validation).
    head = 100 * metrics["S13"]["top1"]
    decomposition = {}
    for i, method in enumerate(METHODS):
        handcrafted, cls = 100 * metrics[f"S{4 + i:02d}"]["top1"], 100 * metrics[f"S{27 + i:02d}"]["top1"]
        gap = head - handcrafted
        decomposition[method] = {"handcrafted_pct": handcrafted, "cls_pct": cls, "official_head_pct": head, "total_gap_pp": gap,
                                 "representation_pp": cls - handcrafted, "classifier_pp": head - cls,
                                 "representation_share": (cls - handcrafted) / gap if gap else None}
    analysis = {"schema_version": 1, "status": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "preregistration_sha256": PREREGISTRATION_SHA256, "test_count": TEST_COUNT,
                "signers": dict(sorted(counts.items())), "metrics": metrics, "names": names, "ablation": ablation,
                "comparisons": comparisons, "representation_decomposition": decomposition,
                "validation_reference": prereg["validation_reference_results"], "inputs": inputs}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    lines = ["| ID | System | Top-1 | Top-5 | Macro Top-1 |", "| --- | --- | ---: | ---: | ---: |"]
    for sid in sorted(metrics):
        m = metrics[sid]
        lines.append(f"| {sid} | {names[sid]} | {100 * m['top1']:.4f}% | {100 * m['top5']:.4f}% | {100 * m['macro_top1']:.4f}% |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({sid: round(100 * metrics[sid]["top1"], 4) for sid in sorted(metrics)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
