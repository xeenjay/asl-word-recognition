# Code reading map

Start with the small CPU example. The repository contains the project's own
core code, not a copy of the original experiment directories. Large data,
weights, caches, and the upstream PoET source must be obtained separately.

## Suggested reading order

| Step | Read | What to understand |
|---|---|---|
| 1 | `scripts/quickstart_classical.py` | An executable synthetic example of the feature → scaler → classifier → reload flow. It is not an ASL benchmark. |
| 2 | `scripts/classical_features.py` | Raw body and hand coordinates become one fixed 1,020-dimensional vector. Missing observations have masks and coverage values. |
| 3 | `scripts/classical_models.py` | A train-only scaler, standard cosine kNN, scikit-learn LinearSVC, bounded RandomForest, Top-k ranking, and serialization. |
| 4 | `scripts/infer_video.py`, `scripts/_poet_runtime.py` | Official landmark extraction, normalization, temporal sampling, fixed vocabulary and strict checkpoint loading. Requires optional assets. |
| 5 | `scripts/poet_geometry.py`, `scripts/poet_methods.py` | Original-image 2D relationships form a 28-dimensional frame vector; a 28→32→128 residual MLP augments the PoET frame embedding. |
| 6 | `scripts/rerank_retrieval_pilot.py` | `build_memory`, `memory_scores`, `rank_candidates`: train-only, signer-balanced similarity scores rerank the existing five candidates. |
| 7 | `scripts/finetune_methods.py`, `scripts/evaluate_methods.py` | Paired sampling, weighted word loss, optional geometry/attribute loss, model selection, strict epoch-boundary recovery and independent evaluation. |

## Data flow

```text
one presegmented word video
  -> MediaPipe raw landmarks + presence masks
     -> 1,020-dimensional handcrafted descriptor
        -> train-only StandardScaler -> kNN / Linear SVM / Random Forest
     -> official normalization and temporal preparation: 128 x 188
        -> official PoET -> word logits and 128-dimensional CLS representation
     -> shared image x/y + identical sampled frame indices
        -> 28-dimensional geometry -> residual MLP -> geometry PoET
           -> training CLS memory -> rerank its original Top-5 candidates
```

The traditional classifiers share the handcrafted descriptor. PoET uses another
representation and pretrained weights, so their comparison is a system
comparison, not an experiment changing only the classifier. The geometry and
retrieval comparisons use matched PoET baselines.

Retrieval uses a cosine kernel `exp(-10 * (1 - cosine))`, averaging examples
within each signer and then signers within each class. It mixes normalized
memory scores with the original Top-5 softmax at weights 0.25 and 0.75. It cannot
introduce a class outside those five candidates or improve Top-5 membership.
This learned-embedding reranker is separate from the handcrafted-feature kNN.

## Supporting modules

| Area | Files | Status in this learning repository |
|---|---|---|
| Pose preparation and audit | `prepare_poses.py`, `evaluate_pose_cache.py`, `install_mediapipe_asset.py` | Retained core logic; requires independently acquired official assets and verified input manifests. Formal scripts retain train/validation restrictions. |
| Classical full-data flow | `prepare_classical_dataset.py`, `run_classical_baselines.py`, `verify_classical_results.py`, `infer_classical.py` | Explicit cache arguments replace historical run locations. Complete input cache manifests, arrays and trained artifacts are not included. |
| Geometry cache | `prepare_geometry_cache.py` | Stores full-frame geometry and checks that main-branch sampling selects the same frames. |
| Training contracts | `finetune_baseline.py`, `method_runtime.py`, `method_optimizer.py` | Complete local import dependencies of the unified trainer; no private launch or queue infrastructure. |
| Attribute experiment | `build_asllex_targets.py`, `method_runtime.py`, `poet_methods.py` | Historical auxiliary training: handshape/location/movement heads. Requires an independently acquired ASL-LEX table. Not required for the geometry+retrieval method. |
| Retrieval experiments | `extract_retrieval_features.py`, `extract_factorial_retrieval_features.py`, `rerank_retrieval_pilot.py`, `rerank_factorial_retrieval.py` | Core feature extraction and fixed comparisons. Full factorial replay requires new correctly bound preregistration, model/cache manifests and reference predictions; these are not bundled. |
| Local evidence | `run_logged.py` | Runs one explicit local command and records configuration, code hashes, logs and exit state. Does not access a remote server. |
| Upstream source | `fetch_vendor.py` | Downloads or checks the pinned upstream source. Vendor source is excluded from version control. |

## What was reused and what the project added

MediaPipe provides pretrained landmark detection. The official PoET code,
weights, normalization and Transformer are reused. SVM and RF use scikit-learn;
kNN search and voting are an implementation of a standard algorithm.

Project additions include the common handcrafted video descriptor, dataset and
vocabulary adaptation, relative-geometry residual branch, train-only
signer-balanced reranking, controlled comparisons, missing-observation handling,
and reproducible evaluation. The repository does not claim invention of kNN,
SVM, RF, or the original PoET model.

## Changes made for this repository

- `quickstart_classical.py` is a new, small synthetic teaching example. It fits a
  smaller RF than the formal experiment and makes no recognition-accuracy claim.
- `prepare_classical_dataset.py` accepts train/validation cache paths and SHA256s
  explicitly instead of referencing historical private run directories. The
  full-cohort, split-separation and identity checks remain.
- `run_classical_baselines.py` defaults to CPU; all other classifier settings are
  retained. Its full-data runner still uses multiple worker processes and should
  only be started after reading its resource settings.
- `method_runtime.py` drops an unused default pointing at a historical attribute
  target file. Attribute training still requires an explicit verified target file.
- `run_logged.py` no longer records a machine hostname or uses a private path to
  decide its location. Its source snapshot allows absent optional vendor code,
  so it can wrap the classical-only example. It still records the command and
  paths used in each local run; generated runs are excluded from the repository.
- The four selected test modules retain algorithm and contract tests. PoET tests
  additionally check whether optional vendor source is installed. Historical
  preregistration-file tests use a small explicitly labeled synthetic fixture;
  it is not a real experiment authorization or historical evidence.
- `tests/test_portable_runner.py` additionally checks a successful and a failed
  local command without optional vendor assets, including preserved exit status.

No private SSH setup, data-transfer/recovery scripts, experiment dispatch tools,
frozen experiment directories, models, datasets or original video material are
part of this code package. The old five-method web service was bound to specific
local experiment artifacts and is not a portable application in this package.

## Known reading traps

- The upstream extraction docstring says 194 features; its actual implementation
  produces 188 (94 x/y points), which our adapter checks.
- The geometry uses shared original image x/y, not the independently normalized
  PoET input or a shared metric 3D coordinate system.
- A synthetic test pass, cache extraction success, or saved checkpoint is not a
  validation accuracy result. Reported project results are validation results;
  no final test evaluation is provided here.
