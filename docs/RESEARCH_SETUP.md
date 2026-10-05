# Advanced research tools and asset setup

These instructions concern the original research-tool interfaces. For the new
editable classical team workflow, use [Setup](SETUP.md), `run.py`, and `students/`.
Do not use a portable complexity-1 package as if it were the historical PoET
cache. The old snapshot verification section below describes the initial export;
current team-workflow checks are in [Status](STATUS.md).

## 2. Optional PoET runtime and assets

The original runtime used Python 3.10, torch 2.6.0, MediaPipe 0.10.21 and
opencv-contrib-python 4.11.0.86. `requirements-poet.txt` lists the optional pinned
runtime. Install an official torch wheel appropriate to your OS and, if desired,
CUDA configuration; dependency installation alone does not install model assets.

```bash
python -m pip install -r requirements-poet.txt
python scripts/fetch_vendor.py
```

Before real inference, obtain the following directly from their upstream sources
and follow their usage terms. See the repository's upstream/data documentation.

| Local destination | Required content |
|---|---|
| `vendor/poet/` | Fixed official PoET source, obtained with `fetch_vendor.py`. |
| `assets/checkpoints/` | `ASL_Citizen_PoET.pth`, `d_LId2Label.pkl`, `d_Label2LId.pkl`. |
| `assets/mediapipe/pose_landmark_heavy.tflite` | Official MediaPipe heavy pose asset. |
| Your chosen local video path | A valid, presegmented ASL word clip you are permitted to process. |

The adapter expects the official checkpoint SHA256
`5511f1270bd252c586705da8233f7ca1caf0b079a8ddf498b5bb0606f5e6d5c0`.
The two vocabulary hashes are declared in `evaluate_pose_cache.py`.
The heavy pose asset source and hash are in `install_mediapipe_asset.py`; that
script installs only the already-downloaded, hash-checked asset into this
repository's `.venv`, and refuses to alter other Python installations.

```bash
python scripts/install_mediapipe_asset.py
python scripts/infer_video.py --video path/to/word.mp4 --checkpoint-dir assets/checkpoints --device cpu --output-dir outputs/word-01
```

Replace `path/to/word.mp4` with your own file; this is a conditional invocation,
not a bundled sample. It writes raw pose arrays, the exact model input and Top-5
predictions. CUDA may be selected explicitly as `--device cuda:0`. CPU inference
is supported, but this package makes no real-time performance claim.

## 3. Full-data classical and PoET experiments

These tools are retained for understanding and subsequent controlled work; this
repository is not a complete historical experiment replay bundle. They require
the official train/validation data and compatible locally generated manifests,
cache arrays, source fingerprints and, for PoET, checkpoint files. Do not replace
audit hashes merely to make incompatible caches or checkpoints load.

`prepare_poses.py` consumes a complete input video manifest and creates a new
pose-cache index. Its manifest contract is documented in `select_samples`:
`status="complete"`, `selection.count`, and samples with relative video path,
SHA256, gloss, signer_id and split. Formal evaluation additionally verifies the
official split CSV bytes, video membership, vocabulary and signer separation.
The original data download/orchestration environment and its manifests are not
included. Running the code on a fresh checkout therefore requires data
preparation, not just replacing one path in a command.

After creating and validating complete local train/validation pose caches, the
classical preparation entry point accepts explicit files and their verified hashes:

```bash
python scripts/prepare_classical_dataset.py --train-cache-manifest path/to/train/cache_manifest.json --train-cache-sha256 TRAIN_SHA256 --val-cache-manifest path/to/val/cache_manifest.json --val-cache-sha256 VAL_SHA256 --checkpoint-dir assets/checkpoints --workers 2 --output-dir outputs/classical-features-01
python scripts/run_classical_baselines.py --feature-dir outputs/classical-features-01 --output-dir outputs/classical-models-01 --device cpu
python scripts/infer_classical.py --experiment-dir outputs/classical-models-01 --pose-path path/to/raw_pose.npz --method linear_svm --device cpu
```

These are templates: substitute the real cache paths and SHA256 strings first.
`run_classical_baselines.py` uses the full formal recipe and starts separate
classifiers, including up to 8 OVR workers and 8 forest workers; it is not the
small laptop quickstart. Inspect the resource settings before running it.
`infer_classical.py` needs its complete saved experiment metadata and scaler,
not just a standalone joblib file or the synthetic example output.

The unified PoET training command below likewise assumes complete verified caches
and separately acquired official weights; it is not run by setup:

```bash
python scripts/finetune_methods.py --method baseline --train-cache-manifest path/to/train/cache_manifest.json --val-cache-manifest path/to/val/cache_manifest.json --checkpoint-dir assets/checkpoints --epochs 10 --device cuda:0 --output-dir outputs/baseline-01
```

For geometry, first use `prepare_geometry_cache.py` on each compatible pose
cache. Pass `--method geometry --module-lr 3e-4` and the two corresponding
`--train-geometry-manifest` / `--val-geometry-manifest` arguments to the unified
trainer. The retained default base learning rate is `3e-5`. Use
`evaluate_methods.py --help` to evaluate a newly saved method checkpoint; method
checkpoints are not interchangeable with the raw official checkpoint format.

Retrieval feature extraction and factorial scripts retain strict protocol checks
and need correctly bound model/cache/preregistration files. They are reference
implementation entry points, not one-command reproduction of the reported
three-seed results. For a self-contained understanding of the math, read and run
`tests/test_retrieval_rerank.py`; its fixture is explicitly synthetic.

The current formal code accepts train/validation workflows. A final-test pipeline
is not supplied or implied by the existence of an official test split.

## Verification performed for this package

The synthetic CPU example was run locally and its model reload checks passed.
The selected tests cover descriptor invariance and missing observations,
train-only scaling, class alignment, ranking/ties, serialization, geometry
alignment and signer-balanced retrieval arithmetic. Optional model/GPU tests
report their skips explicitly. The four selected modules ran 101 tests with
88 passing and 13 optional skips; the added portable-runner check also passed
(102 checks total: 89 passed, 13 skipped, no remaining failures).

Local package checks used Python 3.10.18, NumPy 2.2.6, SciPy 1.15.2,
scikit-learn 1.7.1, joblib 1.5.1 and threadpoolctl 3.6.0 from an existing
environment. This is a portability check, not verification of a fresh install
of every pinned dependency. No new real-video inference, model training, full
validation rerun or test-set evaluation was performed to create this package.
