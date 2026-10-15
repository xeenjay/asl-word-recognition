# Member 1 — k-nearest neighbours (kNN)

[All roles](README.md) · Code: `students/knn.py` · Self-check: `python -m unittest tests.test_member1_knn -v` · Notes: [member1-notes.md](member1-notes.md)

**Goal.** Write cosine kNN yourself in NumPy, show that it gives exactly the same predictions as the reference, then use it to answer a real project question: *how much of PoET's advantage comes from its learned representation rather than its classifier?*

## A. Learn

- kNN and distance-based classification: CS231n notes, ["Image Classification: Data-driven Approach, k-Nearest Neighbor"](https://cs231n.github.io/classification/) (sections on nearest neighbour and kNN, hyperparameters, validation sets).
- [scikit-learn user guide, Nearest Neighbors classification](https://scikit-learn.org/stable/modules/neighbors.html#classification) (uniform vs distance weights).
- Our features: [Methods](../METHODS.md), section on the 1,020-D descriptor. Each video becomes one fixed-length vector of pose/motion statistics.
- How the runner calls you: [Team guide](../TEAM_GUIDE.md), "Student interface".

Concept questions — answer in your notes, in your own words:

1. After L2 normalization, why is a dot product the same as cosine similarity? What information does cosine similarity ignore, and why could that help when comparing different signers?
2. Why must the StandardScaler be fitted on training data only? What would go wrong if it were fitted on train + validation?
3. What happens to the predictions when k = 1? When k = the number of training clips?
4. Why does the reference sometimes return fewer than 5 candidates instead of filling the list with classes that got no votes? How would padding affect Top-5?
5. kNN has almost no training cost. Where does the cost go instead? Estimate the number of multiply-adds needed to compare 10,304 validation clips with 40,154 training clips of 1,020 features, and the memory to store the training matrix in float32.
6. (After Part D) The same kNN is run on two different feature sets. Explain the size of the gap.

<!-- BEGIN SOLUTION member1 -->
**Answer notes (released after the milestone).**
1. With ‖a‖ = ‖b‖ = 1, a·b = ‖a‖‖b‖cos θ = cos θ. Cosine ignores vector length: after standardization the length mostly says how unusual a clip is overall (e.g. large motion or a far-from-average pose), while the direction says which features are high or low together. Comparing patterns rather than magnitudes reduces the effect of signer-specific speed or amplitude.
2. Fitting on validation leaks validation statistics into the features, so the evaluation no longer simulates unseen signers and becomes optimistic; at deployment time future data are not available for fitting either.
3. k = 1 copies the label of the single most similar clip (low bias, high variance, sensitive to noise). k = N makes every query see the whole training set, so every query gets the same class ranking — the class frequencies.
4. Classes without votes have no evidence. Filling the list with arbitrary class IDs would make Top-5 partly luck and not comparable across methods. With k = 5 at most five classes can receive votes, fewer when neighbours agree.
5. Prediction is where the work is: 10,304 × 40,154 × 1,020 ≈ 4.2 × 10^11 multiply-adds, and the training matrix needs 40,154 × 1,020 × 4 bytes ≈ 164 MB that must be kept in memory.
6. The CLS vector is learned by PoET to separate the 2,731 signs; the hand-crafted descriptor summarises pose and motion with fixed statistics and loses much of the temporal order and hand-shape detail. A nearest-neighbour classifier can only be as good as its distance, so changing the representation changes almost everything (see Part D numbers).
<!-- END SOLUTION -->

## B. Implement

Fill in the five functions in `students/knn.py`; `fit` and `predict_topk` are already written and call them. Run the self-check after each TODO.

| TODO | Function | Hint |
| --- | --- | --- |
| 1 | `l2_normalize_rows` | Norm of each row in float64; divide only rows whose norm is not zero; return float32. |
| 2 | `cosine_similarity` | Both inputs are unit rows, so one matrix product gives all pairwise cosines. Check the shape `[num_queries, num_references]`. |
| 3 | `nearest_neighbors` | Sort each row from most to least similar. A *stable* sort keeps the smaller training index first when similarities tie. `np.take_along_axis` collects the matching similarities. |
| 4 | `vote` | Walk through the neighbour labels once: count votes and remember the rank where each class first appeared. Sort the classes by (−votes, first rank, class ID). |
| 5 | `similarity_vote` (extension) | Same structure as `vote`, but add max(similarity, 0) instead of 1, divide by the total weight, drop classes whose total is 0, and fall back to `vote` when every weight is 0. |

Why do the tie rules matter? With exact rules, two correct implementations give **identical** output, so any difference from the reference points to a bug instead of "randomness". The self-check compares your predictions with the reference on 25 random problems (including exact ties and an all-zero query) and, if prepared, on the real mini data.

Afterwards, update `IMPLEMENTATION_NOTE` in `students/knn.py` with what you actually wrote.

## C. Compare with the reference

```bash
python run.py --method knn --implementation reference --output outputs/knn-ref
python run.py --method knn --implementation student --output outputs/knn-mine
```

Compare `metrics.json` and `per_class.json`. They should be identical; if not, find the first validation clip whose prediction differs and explain why. Also compare `run.json → timings_seconds` and note in which part of the code the time goes.

## D. Investigate

**D1 — mini experiment.** Create one config per setting, for example `configs/knn-k10-similarity.json`:

```json
{"method": "knn", "parameters": {"n_neighbors": 10, "vote_weighting": "similarity"}}
```

Run your implementation for k ∈ {1, 3, 5, 10, 20} × vote_weighting ∈ {uniform, similarity} (10 runs; `--config` selects the file; the reference does not support `vote_weighting`). Record Top-1, Top-5 and macro Top-1 in a table. Remember that the mini has only 6 training clips per word and 20 validation clips (one clip = 5 percentage points), so look for patterns, not winners. Why does k = 20 behave differently when each word has only 6 training examples?

**D2 — full data (Xinjie runs it).** Before you see the numbers, write down your predictions in your notes:

1. Full validation Top-1 of your kNN with k = 5 uniform, and with k = 10 similarity weighting.
2. Top-1 when exactly the same kNN is applied to PoET's 128-D CLS embedding (the vector PoET computes right before its final linear layer) instead of the 1,020-D hand-crafted descriptor.

Then interpret the real numbers: what does the comparison in (2) say about *why* PoET is so much better than the classical methods?

<!-- BEGIN SOLUTION results-2026-10 -->
**Full-data results (2,731 words; 10,304 validation clips, 32,941 test clips).** Descriptor rows use the left/right-normalized descriptor introduced in October 2026.

| System | Validation Top-1 | Test Top-1 |
| --- | ---: | ---: |
| kNN k=5 uniform, original descriptor | 18.68% | 15.52% |
| kNN k=5 uniform, left/right-normalized descriptor | 20.23% | 16.15% |
| kNN k=10 similarity, left/right-normalized (selected on validation) | 20.53% | 16.34% |
| kNN k=5 uniform on PoET 128-D CLS | 79.16% | 71.94% |
| Official PoET (its own linear classifier) | 80.94% | 73.56% |

kNN is insensitive to k and weighting (all 10 validation settings lie between 19.7% and 20.5%; the selected setting is not significantly better, McNemar p = 0.25). With the same classifier, switching the features from hand-crafted to PoET CLS raises test Top-1 from 16.15% to 71.94%: about 97% of the gap to PoET (57.4 points) is explained by the representation, and only 1.6 points by PoET's own classifier.
<!-- END SOLUTION -->

## E. Write

Report subsection (about 300–500 words) for "kNN baseline and representation analysis":

- method: cosine similarity, voting rule, tie rules, why the scaler is fitted on train only;
- verification: exact agreement with the reference (your Part C);
- results: the full-data table, with validation and test kept clearly apart;
- analysis: the representation-vs-classifier comparison and what it implies.

Add one table or figure and 1–2 slides. Keep the draft in your notes until the team merges the report.
