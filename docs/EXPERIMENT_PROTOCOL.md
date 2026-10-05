# Experiment protocol

This document separates the protocol used for the existing results from requirements for any future team experiment. Reading this repository does not launch training, download data, or evaluate test.

## Team development workflow

`prepare_team_data.py` creates each user's own local inputs from the official
source or their existing personal copy. Read and accept the dataset's terms
before downloading. Real videos, derived arrays, and local selection records
stay out of Git and release assets; see [Data provenance](DATA_PROVENANCE.md).

The default mini exercise contains ten fixed glosses, 60 training clips and 20
validation clips from their official signer-disjoint partitions. Class selection
and within-class filename-hash selection do not consult model predictions. The
portable MediaPipe extractor uses complexity 1, whereas the historical research
extractor used complexity 2. The 1,020-dimensional descriptor formula is shared;
the raw landmarks and resulting scores need not match. Neither mini accuracy nor
a new full run from this extractor is a reproduction of the historical table by
default. Always preserve the recorded extractor, vocabulary and data hashes.

Use `run.py` for one method at a time. `--implementation reference` uses the
shared reference code. `--implementation student` uses that method's file under
`students/`. The initial student files explicitly delegate to the reference;
they become a student's contribution only through actual understanding and
documented work. Custom-code errors stop the run instead of silently switching
to the reference. The runner fits standardization on training data only and
keeps validation labels outside the student's fit function.

Create a new output directory for each attempt. Results record the requested and
effective implementation, method settings, data and code hashes, environment,
full validation denominator, timings, predictions, and a model reload check.
Record changes and AI assistance in the team contribution log. Start with a
small, explainable comparison; running the full profile is an explicit separate
choice with a much larger preparation and training cost.

## Task, data, and split discipline

The task is isolated-word classification on ASL Citizen: one presegmented clip maps to one of 2,731 word labels. It does not include continuous sentence translation, temporal word-boundary detection, or a claim of real-time camera operation.

| Official split | Clips | Use |
|---|---:|---|
| Train | 40,154 | Fit models, fit preprocessing statistics, and build retrieval memory |
| Validation | 10,304 | Development, hyperparameter choices, and checkpoint selection |
| Test | 32,941 | Reserved final evaluation after the team freezes its methods |

Use the official signer-disjoint splits. Keep every frame, crop, augmentation, and cached feature from a video in that video's split. Do not create frame-level random train/validation splits. Standardization, class weights, feature normalization statistics, and retrieval reference banks must be fitted from training data only.

Retain all selected inputs, including clips with missing hands. Report the attempted count, successful count, failures, reasons for exclusion if any, and the complete denominator. A load, extraction, or inference failure counts as incorrect rather than silently disappearing. The existing full-validation evaluations cover all 10,304 clips and all 2,731 classes with zero processing failures. Successful processing does not imply accurate hand detection.

## Existing classical comparison

The three reference classifiers use the same 1,020-dimensional pose/motion descriptor: temporal-pyramid means over 1/2/4 segments, global standard deviation, signed and absolute motion, and coverage indicators. StandardScaler is fitted only on train.

| Method | Fixed reference configuration |
|---|---|
| kNN | Cosine distance, k=5, uniform vote, full train reference bank |
| Linear SVM | scikit-learn LinearSVC, one-vs-rest over 2,731 classes, C=0.01 |
| Random Forest | scikit-learn, 64 trees, at most 256 leaves per tree |

These are bounded reference configurations, not exhaustive hyperparameter searches. PoET consumes a different sequence representation and uses publicly trained weights. Therefore, the four-system comparison does not isolate classifier architecture. For a new classifier-only study, hold features, splits, preprocessing, and selection budget constant. For a new representation study, explicitly identify which components change.

MediaPipe is also a pretrained front end. Record external pretrained models and any additional supervision instead of describing the whole pipeline as trained from scratch.

## Existing matched PoET protocol

Baseline B and geometry G start from the same official checkpoint and use matched vocabulary, train/validation cohorts, pose preprocessing, temporal sampling/augmentation, trainable base layers, base learning rate, optimizer-step budget, and validation rules. G adds a relative-geometry residual branch before the Transformer.

| Setting | Value |
|---|---|
| Paired seeds | 20260922, 20260923, 20260924 |
| Training budget per model | 10 epochs, 3,140 optimizer steps, 401,540 sample visits |
| Batch size | 128 |
| Base learning rate | 3e-5 |
| Geometry-module learning rate | 3e-4 |
| Checkpoint selection | Highest full-validation macro Top-1 over epochs0–10; earliest epoch wins a tie |
| Selected checkpoint | B seed22: epoch2; the other five B/G models: epoch10 |

Do not compare a newly trained improvement only against untouched official weights and attribute all gain to the new module. The matched fine-tuned B control is necessary. Fixed-checkpoint off/mismatch interventions answer whether the trained model uses the branch; they do not replace a separately trained equal-capacity control.

For fixed retrieval R, each selected model creates its own 40,154-example training memory using that model's final CLS embedding. Use the fixed signer-balanced retrieval calculation, beta=10, and 0.75 base / 0.25 memory blend over the original model's Top-5 candidates. No new training or hyperparameter scan was added in this factorial trial. Report all B, B+R, G, and G+R cells. The predefined retention condition required G+R to strictly exceed both G and B+R in instance and macro Top-1 for every seed, with complete evaluation and independent numerical verification. Passing that condition is not a statistical significance test.

## Separate exploratory protocols

- **20-epoch continuation:** resume each B/G model from its own epoch10 last checkpoint, preserving optimizer and random-generator state; add 10 epochs, for 6,280 total steps and 803,080 visits. Select best over the full epoch0–20 history. Retrieval was not rerun in this experiment.
- **Cross-signer SupCon:** compare against CE with the same cross-signer sampler, seed20260926, and actual temporal augmentations. Both receive 3,140 steps and 401,920 sample visits. Repeated sampling means visits are not the number of unique videos.
- **ASL-LEX auxiliary loss:** retain the word-classification loss for every video and mask missing public attributes. Disclose the additional dictionary supervision; do not provide the target word or its correct attributes at inference.
- **Ordinary RGB pilot:** use the fixed 256-class cohort (3,774 train / 971 validation) and report that restricted denominator explicitly. Its scores cannot be substituted for full-vocabulary performance.
- **Hand-domain RGB:** use fixed pose seed20260923 and RGB seed20260927; freeze pose/DINO and train Pool and GRU heads for 30 epochs each. Select by RGB-only validation macro Top-1 over epochs0–30, earliest tie; apply the fixed 0.75/0.25 fusion. Report both heads and the corresponding pose controls. This trial contains neither retrieval nor TTA.

Do not pool these protocols as if their controls, budgets, vocabularies, or seeds matched. Preserve weak and negative results, including predefined continuation gates that did not pass.

## Metrics and interpretation

Report per-instance Top-1/Top-5, per-class macro Top-1/Top-5, class coverage, correct counts, failed counts, and the actual evaluation denominator. Every row must identify the split and selected checkpoint. Macro metrics weight classes equally; instance metrics weight clips equally. Report a percentage-point difference for a difference between percentages.

For paired predictions, report corrected clips, regressed clips, and their net difference. Count a clip as corrected only if the reference was wrong and the compared method is right. A changed prediction can remain wrong. Error-overlap oracle unions are diagnostic upper bounds, not implemented ensembles.

Report the mean and sample standard deviation across seeds (`ddof=1`) for repeated experiments. The same validation clips reused across seeds are not independent new observations. Seed20260922 participated in early method development, and all three seeds use validation for model selection. Do not call the existing small gains statistically significant or independently confirmed test generalization.

Classification scores have different meanings across methods: neighbor votes, SVM margins, RF class probabilities, and PoET softmax probabilities. They have not been calibrated to a common confidence scale.

## Evidence for future runs

Use a new run identifier and output directory for each experiment. Preserve earlier successful, failed, and interrupted runs. Before starting, save the method/configuration, intended budget, selection rule, seeds, code revision/hash, dependency versions, pretrained asset hashes, split and vocabulary hashes, and train/validation sample counts. Save the actual command and only non-secret environment information.

During and after a run, retain logs, per-epoch metrics, model and optimizer checkpoints, random-generator state when resumption is claimed, actual optimizer-step counts, elapsed time, exit status, and output checksums. Reload the selected checkpoint and verify predictions/metrics before treating it as a deliverable. A zero exit code alone does not establish a valid experiment result.

Measure model-only latency separately from end-to-end video decoding and pose extraction. Record batch size, sequence length, warmup, timing method, parameter count, trainable parameters, and memory usage. Existing timing fragments were collected for engineering purposes and are not a common end-to-end speed benchmark.

Raw run records can contain private paths and participant-level data. Share only approved artifacts; the committed result summaries use an explicit field allowlist and source hashes rather than copying complete logs or per-video records.

## Final test handoff

Before evaluating test, the team must record its final methods, checkpoints, seeds, preprocessing, fusion settings, metrics, and failure policy. Perform final evaluation without using test errors or scores to select models, tune thresholds, choose epochs, or select seeds. If an evaluation implementation error is discovered, record the cause and apply a justified common correction consistently; do not use it as an opportunity for test-driven method changes.

Until that evaluation is completed, all current numerical findings must remain labeled **validation**. Team members must be able to explain the methods and results they present, document their actual contributions, and retain a record of AI assistance consistent with course requirements.
