# Code map

[Home](../README.md) · [Setup](SETUP.md) · [Methods](METHODS.md) · [Student interface](TEAM_GUIDE.md)

The main team workflow separates one-time real-data preparation from classifier implementation. You can write a classical method without editing video extraction, PoET, or historical orchestration.

## Main files

| File or directory | Purpose |
| --- | --- |
| `scripts/prepare_team_data.py` | Obtain selected official data after license acceptance, extract local landmarks/features, record provenance |
| `run.py` | Choose data/method/implementation/config, fit common preprocessing, evaluate, and record outputs |
| `team/runner.py` | Data/config validation, train-only scaling, student dispatch, metrics, reload checks, and provenance |
| `team/reference.py` | Thin adapter to the shared reference classifiers, with delegation tracking |
| `students/knn.py` | Editable kNN interface; initially a labeled reference delegate |
| `students/linear_svm.py` | Editable SVM interface |
| `students/random_forest.py` | Editable Random Forest interface |
| `configs/<method>.json` | Method name and algorithm parameters |
| `scripts/classical_models.py` | Existing reference models, ranking, scaler utilities, and serialization |
| `scripts/classical_features.py` | Raw body/hand coordinates → a 1,020-dimensional clip descriptor |
| `data/classical-mini/` | Locally prepared real mini arrays/labels/manifest; excluded from Git |
| `outputs/<run-name>/` | Metrics, per-class results, predictions, model/scaler, and provenance; use a new directory per run |
| `requirements-data.txt` | Classical dependencies plus official-data download and MediaPipe/OpenCV preparation dependencies |
| `requirements-classical.txt` | CPU fitting/evaluation after features are available |

The student contract is `fit(X_train, y_train, num_classes, config)` and `predict_topk(model, X, top_k)`. The runner has already standardized X using train-only statistics. [Team guide](TEAM_GUIDE.md) gives the required candidate-ID/score/mask return format.

An unchanged student module delegates to reference, and run provenance records that fact. `--implementation student` alone does not imply an independent algorithm.

JSON configs wrap parameters under `parameters`; the student `fit` function receives only the resolved parameter dictionary. The runner's JSON accuracy values are 0–1 proportions, while the CLI prints percentages.

## Data flow

```text
official ASL Citizen source + user's license acceptance
  -> deterministic train/validation mini clip selection
  -> local video download / reuse
  -> MediaPipe landmarks + missing-detection handling
  -> 1,020-dimensional clip features, labels, provenance
  -> train-only StandardScaler
  -> reference or students/<method>.py
  -> ranked labels + scores + validity masks
  -> common metrics and saved run records
```

Default mini: sorted gloss order's first 10 labels, filename-hash selection within each class/split, up to 6 train and 2 validation clips per label (smaller groups keep all available clips). Local class IDs are 0–9. Full preparation is separately requested and retains the complete 2,731-label train/validation task. See [Data provenance](DATA_PROVENANCE.md) for the exact source contract.

The portable extractor uses MediaPipe complexity 1 and all decoded frames; historical extraction used complexity 2 and a recorded protocol. Shared descriptor formulas do not guarantee identical feature values or exact old-score reproduction.

The classifier starts from cached features; changing its code does not download video again. The descriptor includes temporal-pyramid means, standard deviations, signed/absolute motion, and coverage statistics. Missing observations use masks. Read [Methods](METHODS.md) for the dimension breakdown.

## Advanced research source

These files explain the completed research. They are not prerequisites for implementing a mini classical classifier.

| Area | Main files in `scripts/` | Separate requirements |
| --- | --- | --- |
| Official PoET inference | `infer_video.py`, `_poet_runtime.py` | Upstream source, official weights, MediaPipe assets, a permitted video |
| Research pose preparation/audit | `prepare_poses.py`, `evaluate_pose_cache.py` | Verified video/pose manifests and payloads |
| Original full classical pipeline | `prepare_classical_dataset.py`, `run_classical_baselines.py`, `verify_classical_results.py`, `infer_classical.py` | Compatible caches and historical-style records; distinct from the teaching `run.py` |
| Relative geometry | `poet_geometry.py`, `poet_methods.py`, `prepare_geometry_cache.py` | Geometry/pose assets with identical selected frame indices |
| Neural training/evaluation | `finetune_methods.py`, `evaluate_methods.py`, `method_runtime.py`, `method_optimizer.py` | Protocol, optional dependencies, weights, complete caches/manifests |
| Learned-embedding retrieval | `extract_retrieval_features.py`, `extract_factorial_retrieval_features.py`, `rerank_retrieval_pilot.py`, `rerank_factorial_retrieval.py` | Each model's own train CLS bank and bound evaluation records |
| Attribute direction | `build_asllex_targets.py`, `poet_methods.py` | Independently obtained ASL-LEX data; historical auxiliary experiment |
| Upstream source | `fetch_vendor.py` | Pinned source retrieval/check; vendor stays outside Git |
| Local command records | `run_logged.py` | Wrap an explicit local command; separate from the teaching runner's own records |
| Synthetic mechanics | `quickstart_classical.py` | Invented trajectories; no dataset download and no ASL recognition claim |

Use [Third-party sources](THIRD_PARTY.md) and [Experiment protocol](EXPERIMENT_PROTOCOL.md) for advanced work. The original full research runner has different input contracts and parallelism from the one-worker teaching runner.

## Tests and common mistakes

`tests/` covers numerical behavior and contracts. Some fixtures are synthetic; optional PoET/GPU tests may need assets. A skip is not successful model inference. Actual new-workflow acceptance is recorded in [Status](STATUS.md); archived checks do not establish that newly changed code passed.

- Mini scores and full-vocabulary scores describe different tasks.
- Reported archived project scores are validation, not final test.
- PoET's learned-embedding retrieval is different from handcrafted-feature kNN.
- Geometry uses shared image x/y, not recovered metric 3D.
- An upstream docstring mentions 194 values, but the pinned PoET input implementation produces 188 coordinates (94 x/y landmarks).
- The original web demo depends on omitted artifacts and is not supplied as a portable app.

MediaPipe, PoET, and scikit-learn retain upstream credit. Custom adaptation/evaluation code used substantial agent assistance. Distinguish shared reference infrastructure from each member's actual implementation, experiments, and analysis.
