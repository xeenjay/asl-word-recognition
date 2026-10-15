# Team workflow: run, implement, compare

[Home](../README.md) · [Setup](SETUP.md) · [Methods](METHODS.md) · [Results](RESULTS.md)

Start with a reference on a small real dataset, understand its behavior, then write the student version yourself. Each of three members owns one method — Member 1 kNN, Member 2 Linear SVM, Member 3 random forest — with a step-by-step guide in [docs/roles](roles/README.md). The team is Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen; the team decides who takes which role.

## The normal workflow

1. Clone or download the private repository and install `requirements-data.txt`.
2. Read the official data terms and, if you accept, run the mini preparation described in [Setup](SETUP.md). It obtains up to 80 official clips and extracts local numeric features.
3. Run your method with `--implementation reference`. Read its metrics and class-level outputs.
4. Read [Methods](METHODS.md), `configs/<method>.json`, and the reference in `scripts/classical_models.py`.
5. Fill in the TODOs in `students/<method>.py` (your guide in [docs/roles](roles/README.md) says which), check them with your self-check test, then run with `--implementation student` and a new output directory.
6. Compare the same data/configuration, explain discrepancies, and record the work you actually completed.

Once features exist, method runs do not download videos or repeat landmark extraction. Until your TODOs are done, `--implementation student` stops with `NotImplementedError`; use the reference to learn in the meantime. The run record states whether a student run called the reference. Using a library may be appropriate, but credit its implementation and identify your adaptation and analysis.

## Student interface

Each module exposes:

```python
fit(X_train, y_train, num_classes, config)
predict_topk(model, X, top_k)
```

`fit` returns your model or model state. X has 1,020 columns. **The runner has already fitted StandardScaler on train and transformed both splits.** Do not fit another scaler on validation.

Labels use the data pack's local order, from 0 to `num_classes - 1`. Use the supplied class count instead of hard-coding 10 or 2,731.

The `config` passed to `fit` is the resolved algorithm-parameter dictionary, not the outer JSON object with `method`/`parameters`. If you change the implementation, set `IMPLEMENTATION_STATUS = "student_modified"` and explain the actual changes in `IMPLEMENTATION_NOTE`. Calling reference functions remains visible in the recorded provenance even if that declaration changes.

`predict_topk` returns a dictionary:

| Key | Contract |
| --- | --- |
| `topk_indices` | Integer N×K array; K is at most the requested `top_k` |
| `topk_scores` | Finite numeric array of the same shape |
| `valid_mask` | Boolean array of the same shape |
| `score_kind` | Optional description of the scores |

Valid IDs must be within the local vocabulary and cannot repeat within one row. The default request is Top-5. If fewer candidates are supported, return fewer columns or pad with `-1` and `valid_mask=False`. Do not invent zero-score candidate labels. A row with no valid prediction remains in the denominator and counts as wrong.

The runner owns common preprocessing, checks, metrics, and output records. The student module owns fitting and prediction. Do not change shared evaluation simply to make a result look better.

## Compare implementations before new experiments

Keep the same feature directory and config. Compare row counts, label order, Top-1/Top-5/macro metrics, per-class changes, and a few disagreeing predictions. Check whether tie rules, candidate counts, or algorithm settings changed, and whether the student still calls reference.

A score difference may indicate a bug or an intentional method difference. Explain which it is. Passing the self-check tests shows the code meets its contract; your notes should still explain how it works and why any difference from the reference appears.

JSON accuracy fields use proportions from 0 to 1; terminal output uses percentages. Additional helper files you import are not automatically included in the runner's selected-source hash list, so record their commits/files in your contribution evidence too.

The default mini contains at most 60 training and 20 validation clips across 10 classes; inspect its manifest for actual counts. It is a small pipeline check and programming exercise. Its accuracy is not a robust estimate and cannot be ranked against the report's full 2,731-class results.

## Agree on a bounded investigation

After understanding the reference, agree on a question and finite run budget. Possible investigations include neighbor-count sensitivity or a bounded forest-complexity change. Record the data, fixed settings, changed factor, metric, and budget before running.

Full preparation and training are explicit larger tasks, not automatic next steps. Full video payload is approximately 31 GB and preprocessing takes additional time. Full SVM has 2,731 binary fits; the teaching runner uses one worker and limits numerical threads. Do not estimate laptop cost from archived parallel runs.

The portable extractor uses MediaPipe complexity 1, while historical features used complexity 2. Matching descriptor dimensions does not guarantee identical input values or old scores; identify which feature preparation produced each result.

Keep the official split assignments and train-only fitting rules. Final test remains reserved until methods and evaluation are frozen; do not tune using test errors.

## Record actual contributions

| Field | Include |
| --- | --- |
| Date and member | Who did the work and when |
| Actual activity | Implementation, reproduction, experiment, analysis, review, writing, or presentation |
| Evidence | Commit/file, data identity, config, output directory, or report section |
| Findings and limits | What the evidence supports and what remains unresolved |
| Assistance | Reference/library code, collaborators, AI assistance, and what you checked or changed |

The initial research was organized by Xinjie with substantial agent assistance. Supplied reference code is shared infrastructure, not evidence of a teammate's completed work. Read the [AI-use record](AI_USE_AND_CONTRIBUTIONS.md), and update it from actual activity. Each member should be able to explain their submitted contribution.

Preserve previous and failed outputs and negative findings. Keep dataset payloads, derived data, generated models, credentials, and private machine details out of Git. Accessing the repository does not submit coursework or launch experiments automatically.
