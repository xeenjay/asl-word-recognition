# Data provenance and local preparation

## Dataset terms and distribution boundary

The source is [Microsoft's ASL Citizen dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/). The [official description](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/) documents version 1.0, 2,731 labels, and the signer-disjoint 40,154 / 10,304 / 32,941 train / validation / test split. Read the [official dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) before obtaining or using it. These pages were checked on October 4, 2026.

The Microsoft Research License Terms restrict use to non-commercial, non-revenue-generating research. Section 1(c) prohibits distribution of the data or modifications to it; section 2(e) further restricts sharing or transferring the materials to third parties. Section 5 permits publication of research results without including a material or substantial portion of the dataset. The license does not state a private-GitHub exception. These provisions do not establish permission to redistribute this project's extracted feature arrays. Their exact application to a proposed team sharing arrangement requires clarification from the rights holder; private visibility alone is not treated as permission. [Source: dataset license, sections 1, 2 and 5](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/).

Accordingly, this repository distributes preparation code and aggregate results, **not a real mini dataset or a full feature release asset**. Each user obtains the official data under its terms and prepares local inputs. No real feature package has been uploaded as part of this update. A local export used by the original researcher for checking remains in an ignored staging directory. Removing filenames and signer fields minimizes unnecessary metadata; it does not resolve redistribution rights.

## Portable local preparation

The team's preparation entry point is `scripts/prepare_team_data.py`. It selects clips without consulting predictions or accuracy and obtains their bytes from the official download. It can reuse a user's already obtained videos and split CSVs through `--source-dir`. This avoids redistributing another person's dataset copy. Follow [Setup](SETUP.md) for the actual command and required dependencies.

The default learning profile uses the first 10 glosses in sorted full-vocabulary order. Within each class and official split, filenames are ordered by a deterministic hash; the first 6 train clips and first 2 validation clips are selected. This gives a small, deterministic 60-train / 20-validation exercise. Its accuracy must not be compared with full 2,731-class results. The full profile retains all official train and validation clips. Test is not part of either profile.

Preparation preserves official train/validation identity and checks signer separation. Labels are remapped into a package-local contiguous range; `labels.json` defines the exact local-ID order. `selected_original_class_ids` must be interpreted using the manifest's `original_class_id_space`. The historical PoET vocabulary order is not alphabetical, so its original IDs cannot be compared numerically with a newly sorted vocabulary without matching gloss strings.

The portable extractor uses MediaPipe Holistic with `static_image_mode=True`, `model_complexity=1`, all frames, and no horizontal flip, then applies the shared 1,020-dimensional `classical-pose-motion-v1` descriptor. Its raw landmark extraction differs from the historical full experiment. The descriptor name records the same mathematical feature layout; it does not imply identical raw coordinates, hashes, or recognition results. Existing reported full-validation scores are not guaranteed to reproduce from this new extractor.

## Package contract

The team runner consumes these six files from a locally prepared data directory:

| File | Content |
|---|---|
| `train_features.npy` | Finite `float32`, shape `(N_train, 1020)` |
| `train_labels.npy` | `int64`, shape `(N_train,)`, package-local class IDs |
| `val_features.npy` | Finite `float32`, shape `(N_val, 1020)` |
| `val_labels.npy` | `int64`, shape `(N_val,)`, package-local class IDs |
| `labels.json` | Gloss strings in local class-ID order |
| `manifest.json` | Schema version, profile, dataset/feature definition, source hashes, selected class IDs, split counts, file hashes/sizes/shapes/dtypes, and `test_used: false` |

No scaler is fitted into these files. Each classifier experiment fits its scaler on package train features only. These six runner inputs contain no video filenames, signer identities, private paths, original frames, or trained models. The same local output directory also retains `selected-manifest.json` and `preparation.json` as preparation/provenance artifacts. The selection manifest contains filenames for local identity checks; the whole `data/` directory, including those extra files, is ignored by Git. The runner verifies its input-file consistency and provenance declarations; it does not independently recheck original signer identities from inputs that omit them.

## Historical reference export for local checking

`scripts/build_team_data.py` is a separate local tool for an owner of the historical feature cache. It does not fetch data or regenerate features. It reads completed source `dataset.json`, four arrays, and the two private row-index files; verifies their recorded sizes and SHA256 values; checks labels against rows and the source glossary; verifies train/validation video and signer separation; and selects the exact requested rows by filename and video hash. It exports only arrays, glossary, and aggregate provenance.

Use a new output directory. This template assumes the caller already lawfully holds the original features and a compatible selection manifest:

```bash
python scripts/build_team_data.py --source-features path/to/original-features --selected-manifest path/to/selection.json --output .team-staging/reference-mini
```

The selection file has `selected_glosses` in output-local-ID order and `splits.train` / `splits.val` lists. Each selected row contains `video_file` (basename), `gloss`, and `video_sha256`. Additional selection/source metadata is permitted. The file is used privately and its hash is recorded; its row contents are not copied into the package.

The reference exporter uses the source glossary's actual class-ID order, records that ID space, and preserves the requested sample order. For the portable mini, this reference is stored separately from `data/classical-mini`; it must never overwrite newly extracted portable features. Class labels, shapes, and sample/split provenance can be checked across the two preparations. Their raw feature values are not expected to be equal because the extractors differ.

The completed historical source is run `20261002T160743Z-classical-full-features-3470975b`, containing all 40,154 train and 10,304 validation descriptors. Its `dataset.json` SHA256 is `3a9ef620bb5b3593375f831ada5b191fe66b9b1bafefe9aba9ff4e075daca83f`. Source array hashes are:

| Source array | SHA256 |
|---|---|
| `train_features.npy` | `557f6a747e5620793984a4f6b42dcdad917d5603229d997adc272671e6aaecdb` |
| `train_labels.npy` | `8ed2dd9d674e26d2981a80dbc058e422910a7b42220077d761829324ed8db25e` |
| `val_features.npy` | `dfb159eb01da2949b0ccc19e50d90baf5202d6e286e82439be62fdc26a552653` |
| `val_labels.npy` | `4ba86035c72d9a9b423b4153a2fa46f3357fa4e420d6c165653d8df0d654e332` |

The local copy was verified against that completed metadata, including both official vocabulary-file hashes. A local 60-train / 20-validation reference was then exported for the portable mini's selected clips. All 80 video identities, glosses and split assignments matched, and exported feature rows exactly matched the corresponding historical source rows. No comparison claiming equality to the new portable extractor was made. This is a source-integrity check, not a new model evaluation or a fresh audit of every original pose. Eight focused exporter tests cover selected-order/remapping correctness and rejection of tampered arrays, mismatched identities, overlapping signers, misaligned labels, duplicate rows, misleading full profiles, and output overwrites.

## Comparison boundaries

Use one prepared package consistently across kNN, SVM, and Random Forest for a feature-controlled comparison. Record whether results came from the portable extractor or the historical reference. Always report class count, train/validation counts, selection policy, package manifest hash, and split. Do not relabel a small local learning run as the original full experiment or as a test result. See [Experiment protocol](EXPERIMENT_PROTOCOL.md) and [Results](RESULTS.md).
