#!/usr/bin/env python3
"""Rebuild report figures from audited validation artifacts (no model inference)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEMO = ROOT / "docs/comparison-demo"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/asl-submission-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

DATA = json.loads((DEMO / "analysis.json").read_text())
FRAMES = json.loads((DEMO / "frame-provenance.json").read_text())
# 2026-10-10: post hoc handedness-canonical classical results (Arm A refit), read-only analysis mirror.
HAND_PATH = ROOT / "runs/20261010T084703Z-handedness-analysis-5acb02a8/analysis/analysis.json"
HAND_SHA256 = "d16574abd6fa1835090388757259506203ace2bddad7d4efb55e3858a58afdfd"
HAND = json.loads(HAND_PATH.read_text())
METHODS = [m["id"] for m in DATA["methods"]]
COLORS = ["#4989A6", "#007D79", "#BB874A", "#364D86"]
TEXT = "#172B40"
GOOD = "#007A67"
BAD = "#A74843"
EVIDENCE = {"source_analysis_sha256": hashlib.sha256((DEMO / "analysis.json").read_bytes()).hexdigest(),
            "handedness_analysis": {"path": str(HAND_PATH.relative_to(ROOT)), "sha256": HAND_SHA256},
            "count": DATA["count"], "split": DATA["split"], "test_used": False,
            "new_model_inference": False, "checks": [], "figures": [], "media": []}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.labelcolor": TEXT, "text.color": TEXT,
                     "axes.edgecolor": "#A8B1BE", "xtick.color": TEXT,
                     "ytick.color": TEXT, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "svg.hashsalt": "asl-submission-20261004",
                     "savefig.facecolor": "white"})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem):
    for extension in ("png", "pdf", "svg"):
        out = HERE / (stem + "." + extension)
        meta = {"CreationDate": None, "ModDate": None} if extension == "pdf" else ({"Date": None} if extension == "svg" else None)
        fig.savefig(out, dpi=250, bbox_inches="tight", pad_inches=0.12, metadata=meta)
        EVIDENCE["figures"].append({"file": out.name, "sha256": sha(out), "bytes": out.stat().st_size})
    plt.close(fig)


def polish(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E3E7EC", linewidth=0.6)
    ax.tick_params(length=0)


def check_data():
    assert DATA["count"] == 10304 and DATA["split"] == "val" and not DATA["test_used"]
    for m in DATA["methods"]:
        assert m["count"] == DATA["count"] and m["failed"] == 0
        for name in ("top1", "top5"):
            assert np.isclose(m[name], m[name + "_correct"] / m["count"], atol=1e-12)
    for p in DATA["pairwise"]:
        assert sum(p[k] for k in ("both_correct", "only_a_correct", "only_b_correct", "both_incorrect")) == DATA["count"]
    for entry in DATA["factorial"]["summary"]:
        values = np.array([p[entry["id"]]["top1"] for p in DATA["factorial"]["per_seed"]])
        assert np.isclose(np.mean(values), entry["mean_top1"], atol=1e-12)
        assert np.isclose(np.std(values, ddof=1), entry["std_top1"], atol=1e-12)
    for c in DATA["cases"]:
        for name, top in c["top5"].items():
            assert (top[0]["gloss"] == c["gloss"]) == c["correct"][name]
    for c in DATA["factorial"]["cases"]:
        for name in ("B", "G_plus_R"):
            assert (c[name]["prediction"] == c["gloss"]) == c[name]["correct"]
    assert sha(HAND_PATH) == HAND_SHA256
    assert HAND["status"] == "verified" and not HAND["test_used"] and HAND["validation_rows"] == DATA["count"]
    for m in DATA["methods"]:
        if m["id"] in HAND["results"]:
            base = HAND["results"][m["id"]]["baseline"]["overall"]
            canon = HAND["results"][m["id"]]["armA_canonical_refit"]["overall"]
            assert base["top1_correct"] == m["top1_correct"] and base["top5_correct"] == m["top5_correct"]
            assert canon["count"] == DATA["count"]
            for name in ("top1", "top5"):
                assert np.isclose(canon[name], canon[name + "_correct"] / canon["count"], atol=1e-12)
    EVIDENCE["checks"] += ["Handedness analysis SHA pinned; its baseline arm equals the main-table classical counts",
                           "Canonical (Arm A) Top-1/Top-5 ratios agree with integer counts"]
    EVIDENCE["checks"] += ["Full validation denominator and no test use confirmed",
                           "All main-table Top-1/Top-5 ratios agree with integer counts",
                           "All pairwise partitions sum to 10,304",
                           "Ablation mean/sample-SD recomputed from all three seeds",
                           "Illustrative case predictions agree with correctness flags"]


def method_comparison():
    # Revised 2026-10-10: original descriptor first (headline), post hoc handedness-canonical refit second.
    names = {"knn": "kNN", "linear_svm": "Linear SVM", "random_forest": "Bounded RF"}
    rows = []
    for m in DATA["methods"]:
        if m["id"] in names:
            rows.append((names[m["id"]] + "\noriginal", m["top1"] * 100, m["top5"] * 100, False, m["id"] == "knn"))
            c = HAND["results"][m["id"]]["armA_canonical_refit"]["overall"]
            rows.append((names[m["id"]] + "\ncanonical", c["top1"] * 100, c["top5"] * 100, True, m["id"] == "knn"))
        else:
            rows.append(("Official\nPoET\u2020", m["top1"] * 100, m["top5"] * 100, False, False))
    fig, ax = plt.subplots(figsize=(7.3, 3.9))
    x = np.arange(len(rows))
    for i, (label, one, five, canonical, knn) in enumerate(rows):
        style = {"hatch": "////", "edgecolor": "white", "linewidth": 0} if canonical else {}
        ax.bar(i - 0.19, one, width=0.36, color="#244B79", **style)
        ax.bar(i + 0.19, five, width=0.36, color="#73B4BB", **style)
        ax.text(i - 0.19, one + 1.2, f"{one:.2f}", ha="center", va="bottom", fontsize=6.9, rotation=90)
        ax.text(i + 0.19, five + 1.2, f"{five:.2f}" + ("*" if knn else ""), ha="center", va="bottom", fontsize=6.9, rotation=90)
    ax.set(xticks=x, xticklabels=[r[0] for r in rows], ylim=(0, 116), ylabel="Validation accuracy (%)",
           yticks=np.arange(0, 101, 20))
    ax.tick_params(axis="x", labelsize=8.2)
    ax.set_title("Four complete systems on the same validation split", loc="left", pad=24, fontsize=12)
    ax.text(0, 1.035, "10,304 clips  |  2,731 classes  |  0 processing failures", transform=ax.transAxes, fontsize=9, color="#536175")
    handles = [Patch(color="#244B79", label="Top-1"), Patch(color="#73B4BB", label="Top-5"),
               Patch(facecolor="#9AA7B5", hatch="////", edgecolor="white", label="Canonical descriptor (post hoc fix)")]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=3, fontsize=8.2, bbox_to_anchor=(0, -0.2))
    fig.text(0.012, 0.012, "* kNN returns only voted classes; its Top-5 is not comparable.   \u2020 Selected on this validation split by its authors; "
             "published test Top-1: 73.54%.", fontsize=7.2, color="#536175")
    polish(ax)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.27)
    save(fig, "fig01_method_comparison")


def paired_ablation():
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.2, 3.65), gridspec_kw={"width_ratios": [1.35, 1]})
    ids = [m["id"] for m in DATA["factorial"]["summary"]]
    colors = ["#A1B9CC", "#85ADB0", "#CBB396"]
    for s, color in zip(DATA["factorial"]["per_seed"], colors):
        left.plot(range(4), [s[i]["top1"] * 100 for i in ids], marker="o", markersize=4,
                  color=color, linewidth=1, label=str(s["seed"]))
    means = [m["mean_top1"] * 100 for m in DATA["factorial"]["summary"]]
    stds = [m["std_top1"] * 100 for m in DATA["factorial"]["summary"]]
    left.errorbar(np.arange(4) + 0.06, means, yerr=stds, fmt="D", color="#193955", capsize=4,
                   markersize=5, linewidth=1.7, label="Mean ± sample SD")
    for x, val in enumerate(means):
        left.text(x + 0.02, 82.025, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    left.set(xticks=range(4), xticklabels=["B", "B+R", "G", "G+R"], ylim=(81, 82.1),
              ylabel="Validation Top-1 (%)", yticks=np.arange(81, 82.01, 0.2))
    left.set_title("(a) Budget-matched PoET ablation", loc="left", fontsize=11, pad=12)
    left.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper left", bbox_to_anchor=(-0.06, -0.18))
    polish(left)
    ps = DATA["factorial"]["per_seed"]
    delta = [p["delta_pp"] for p in ps]
    right.barh(range(3), delta, color="#007D79", height=0.47)
    for i, (d, p) in enumerate(zip(delta, ps)):
        right.text(d + 0.015, i, f"+{d:.3f}\n(+{p['net_correct']} clips)", va="center", fontsize=9)
    right.text(0.02, -0.72, f"Mean: +{DATA['factorial']['delta_pp']:.3f} pp", fontsize=10, fontweight="bold")
    right.set(yticks=range(3), yticklabels=[str(p["seed"]) for p in ps], xlim=(0, 0.95), ylim=(2.55, -1.05),
               xlabel="G+R minus B (percentage points)", xticks=np.arange(0, 1, 0.2))
    right.set_title("(b) Paired improvements", loc="left", fontsize=11, pad=12)
    polish(right, "x")
    fig.subplots_adjust(left=0.085, right=0.99, wspace=0.48, bottom=0.27, top=0.85)
    save(fig, "fig02_paired_ablation")


def error_complementarity():
    fig, ax = plt.subplots(figsize=(7.3, 3.7))
    pairs = [p for p in DATA["pairwise"] if p["method_b"] == "poet_official_v2"]
    keys = ["both_correct", "only_b_correct", "only_a_correct", "both_incorrect"]
    labels = ["Both correct", "Only official PoET correct", "Only classical method correct", "Both incorrect"]
    colors = ["#86B7B5", "#496C93", "#E0B365", "#CED3D9"]
    starts = np.zeros(3)
    for key, label, color in zip(keys, labels, colors):
        counts = [p[key] for p in pairs]
        ax.barh(range(3), counts, left=starts, color=color, label=label, height=0.5, edgecolor="white", linewidth=0.6)
        for i, (st, n) in enumerate(zip(starts, counts)):
            if key != "only_a_correct":
                ax.text(st + n / 2, i, f"{n:,}", ha="center", va="center", color="white" if key == "only_b_correct" else TEXT, fontsize=9)
            else:
                ax.annotate(f"{n}", (st + n / 2, i - 0.25), xytext=(st + n / 2, i - 0.47),
                            ha="center", va="center", fontsize=9, color="#865516",
                            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#865516"})
        starts += counts
    ax.set(yticks=range(3), yticklabels=["kNN vs official PoET", "SVM vs official PoET", "RF vs official PoET"],
           xlim=(0, 10304), xticks=[0, 2000, 4000, 6000, 8000, 10304], xlabel="Validation clips (total 10,304)", ylim=(2.65, -0.9))
    ax.set_xticklabels(["0", "2,000", "4,000", "6,000", "8,000", "10,304"])
    ax.set_title("Error overlap reveals complementary correct predictions", loc="left", fontsize=11.4, pad=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left", bbox_to_anchor=(-0.28, -0.22))
    polish(ax, "x")
    fig.subplots_adjust(left=0.26, right=0.98, top=0.84, bottom=0.3)
    save(fig, "fig03_error_complementarity")


def existing_frames(index):
    case = next(c for c in DATA["cases"] if c["index"] == index)
    p = FRAMES[case["id"]]
    assert p["source_sha256"] == case["video_sha256"]
    result = []
    for frame in p["frames"]:
        path = DEMO / frame["file"]
        assert sha(path) == frame["sha256"]
        result.append((path, frame["actual_seconds"]))
        EVIDENCE["media"].append({"file": str(path.relative_to(ROOT)), "sha256": sha(path),
                                 "index": index, "actual_seconds": frame["actual_seconds"],
                                 "video_sha256": case["video_sha256"], "source": "existing audited gallery frame"})
    return case, result


def new_fixed_frames(index=59):
    import cv2
    manifest = json.loads((ROOT / "runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json").read_text())
    sample = manifest["samples"][index]
    case = next(c for c in DATA["factorial"]["cases"] if c["index"] == index)
    video = ROOT / "data/asl_citizen" / sample["identity"]["path"]
    assert sample["index"] == index and sample["gloss"] == case["gloss"]
    assert sha(video) == case["video_sha256"] == sample["video_sha256"]
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert len(frames) == sample["video_metadata"]["decoded_frames"]
    dest = HERE / "source-frames"
    dest.mkdir(exist_ok=True)
    result = []
    for fraction in (0.25, 0.5, 0.75):
        number = round((len(frames) - 1) * fraction)
        path = dest / f"val-{index:05d}-{int(fraction * 100)}.png"
        assert cv2.imwrite(str(path), frames[number])
        timestamp = number / fps
        result.append((path, timestamp))
        EVIDENCE["media"].append({"file": str(path.relative_to(ROOT)), "sha256": sha(path),
                                 "index": index, "frame_index_zero_based": number,
                                 "timestamp_from_frame_index_and_reported_fps": timestamp,
                                 "video_sha256": sha(video), "video_file": str(video.relative_to(ROOT)),
                                 "source": f"Sequential OpenCV {cv2.__version__} decode, round((N-1)*fraction), no crop or resize"})
    return case, result


def qualitative_cases():
    c1, f1 = existing_frames(11)
    c2, f2 = existing_frames(1)
    c3, f3 = new_fixed_frames(59)
    c4, f4 = existing_frames(0)
    c4_pair = next(c for c in DATA["factorial"]["cases"] if c["index"] == 0)
    rows = [("(a) All four systems correct", c1, f1, False),
            ("(b) All four systems incorrect", c2, f2, False),
            ("(c) G+R corrects matched B", c3, f3, True),
            ("(d) G+R regresses from matched B", c4_pair, f4, True)]
    fig = plt.figure(figsize=(8.1, 8.65))
    gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 1.72], hspace=0.43, wspace=0.03,
                          left=0.025, right=0.995, top=0.94, bottom=0.055)
    short = {"knn": "kNN", "linear_svm": "SVM", "random_forest": "RF", "poet_official_v2": "Official PoET"}
    for row, (title, case, frames, paired) in enumerate(rows):
        for col, (path, sec) in enumerate(frames):
            ax = fig.add_subplot(gs[row, col])
            with Image.open(path) as im:
                ax.imshow(np.asarray(im))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.set_xlabel(f"{[25,50,75][col]}%  |  {sec:.2f} s", fontsize=8.2, labelpad=3)
            if col == 0:
                ax.text(0, 1.22, title, transform=ax.transAxes, fontsize=10.3, fontweight="bold", va="bottom")
                ax.text(0, 1.06, f"val-{case['index']:05d}  |  Target: {case['gloss']}", transform=ax.transAxes, fontsize=9, va="bottom")
        tx = fig.add_subplot(gs[row, 3]);tx.axis("off")
        if paired:
            tx.text(0.055, 0.92, "Matched seed 20260923", fontsize=9, fontweight="bold", transform=tx.transAxes)
            for y, name, key in [(0.67, "B", "B"), (0.40, "G+R", "G_plus_R")]:
                value = case[key]
                tx.text(0.055, y, f"{name}: {value['prediction']}", color=GOOD if value["correct"] else BAD,
                        fontsize=9.8, fontweight="bold", transform=tx.transAxes)
                tx.text(0.055, y - 0.14, "Correct" if value["correct"] else "Incorrect", fontsize=8.3,
                        color=GOOD if value["correct"] else BAD, transform=tx.transAxes)
        else:
            tx.text(0.055, 0.93, "Saved Top-1 predictions", fontsize=9, fontweight="bold", transform=tx.transAxes)
            for y, name in zip((0.7, 0.48, 0.26, 0.04), METHODS):
                prediction = case["top5"][name][0]["gloss"]
                tx.text(0.055, y, f"{short[name]}: {prediction}", color=GOOD if case["correct"][name] else BAD,
                        fontsize=9.1, transform=tx.transAxes)
    fig.text(0.025, 0.014, "Illustrative validation examples selected by correctness; not a random sample. B is fine-tuned PoET, not the official checkpoint.", fontsize=8.2)
    save(fig, "fig04_qualitative_cases")


def detection_groups():
    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    groups = [g for g in DATA["groups"] if g["dimension"] == "hand_detection"]
    for mid, name, color, marker in zip(METHODS, ["kNN", "Linear SVM", "Bounded RF", "Official PoET"], COLORS, ["o", "s", "^", "D"]):
        ax.plot(range(4), [g["metrics"][mid]["top1"] * 100 for g in groups], marker=marker, color=color, label=name, lw=1.5, markersize=5)
    labels = ["0%\n(n = 7)", "(0%, 25%)\n(n = 2,895)", "[25%, 50%)\n(n = 6,256)", "≥ 50%\n(n = 1,146)"]
    ax.set(xticks=range(4), xticklabels=labels, ylim=(-3, 100), yticks=range(0, 101, 20),
           ylabel="Validation Top-1 (%)", xlabel="Mean of left- and right-hand detection fractions")
    ax.set_title("Descriptive accuracy by detected-hand fraction", loc="left", fontsize=12, pad=16)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.02))
    polish(ax)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.24, top=0.8)
    save(fig, "fig05_hand_detection_groups")


if __name__ == "__main__":
    HERE.mkdir(exist_ok=True, parents=True)
    check_data()
    method_comparison()
    paired_ablation()
    error_complementarity()
    qualitative_cases()
    detection_groups()
    EVIDENCE["checks"].append("All reused frame hashes match gallery provenance; new KICK2 video hash matches selected val59")
    EVIDENCE["matplotlib_version"] = matplotlib.__version__
    EVIDENCE["numpy_version"] = np.__version__
    (HERE / "figure-verification.json").write_text(json.dumps(EVIDENCE, indent=2) + "\n")
    print(json.dumps({"status": "verified", "figure_files": len(EVIDENCE["figures"]),
                      "media_frames": len(EVIDENCE["media"]), "checks": EVIDENCE["checks"]}, indent=2))
