# Research code behind the report

This folder holds the remaining code that produced numbers in the final report and slides. Together with the top-level `scripts/`, every experiment the report cites now has its code in this repository, except the hand-RGB branch (see below). The section-by-section index is [docs/REPORT_CODE_MAP.md](../docs/REPORT_CODE_MAP.md).

## What this is, and what it is not

- **Verbatim copies.** Each file was copied unchanged from the research workspace on November 19, 2026. [MANIFEST.json](MANIFEST.json) lists the source path and SHA-256 of every file; the hashes equal the workspace files at copy time. Nothing was edited, so the code is exactly what ran.
- **Not runnable as is.** These scripts expect the full research layout: complete train/validation/test videos and pose caches, the official PoET source and weights, run directories and preregistration files. None of those are in Git. Reading the code is the intended use.
- **Layout.** `research/` mirrors the root of the research workspace (`scripts/`, `docs/tools/`, `docs/comparison-demo/`, `docs/submission-20261004/figures/`), so relative paths inside the files mean the same thing they did there. Modules shared with the top-level `scripts/` were not duplicated; to import them, put both folders on the path, for example `PYTHONPATH=scripts:research/scripts:research/docs/tools`.

## Contents

| Folder | What is in it |
| --- | --- |
| `scripts/` | Final test (download, pose extraction, PoET, retrieval, classical, analysis), mirror TTA, PoET flip check, classical handedness experiment, bounded tuning, representation swap, geometry ablation, contrastive trial, demo server and gallery builder |
| `docs/tools/` | Analysis scripts that computed reported statistics: selection effects and paired tests (`reanalyze_submission_20261010.py`), factorial retrieval, paired seeds, retrieval pilot, 20-epoch extension, geometry ablation, contrastive trial |
| `docs/comparison-demo/` | The demo web page (`index.html`, `app.js`, `style.css`) without its data and media |
| `docs/submission-20261004/figures/` | `build_figures.py`, which drew report Figures 1–3 |

Some files are here only because an analysis script imports helper functions from them: `docs/tools/run_rgb_pilot.py`, `run_paired_seed_lane.py`, `run_extended_training_lane.py`, `run_contrastive_lane.py`, and `scripts/fetch_asl_sample.py`, `fetch_asl_dataset.py`. The RGB pilot itself is not part of the report.

## Left out on purpose

- **Data, weights and media**: videos, pose and feature caches, the official PoET checkpoint, the demo's `data.js`, `analysis.json`, video clips and frames. The ASL Citizen license restricts redistribution.
- **Hand-RGB temporal branch** (report Section 4.2, slide 8, one sentence each): its code stays in the research workspace.
- **Launch, queue and verification tooling** that only scheduled runs or checked them, unless an analysis script imports it.
