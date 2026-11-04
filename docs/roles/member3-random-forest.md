# Member 3 — Decision trees and random forest

[All roles](README.md) · Code: `students/random_forest.py` · Self-check: `python -m unittest tests.test_member3_random_forest -v` · Notes: [member3-notes.md](member3-notes.md)

**Goal.** Build a CART decision tree (Gini impurity, best split, recursive growth, prediction) and a bootstrap random forest in NumPy, compare it with scikit-learn's forest, and explain the project's most surprising classical result: why the original forest was so weak on 2,731 words and what fixed it.

## A. Learn

- *An Introduction to Statistical Learning* (free PDF at [statlearning.com](https://www.statlearning.com/)), Chapter 8: decision trees, bagging, random forests.
- [scikit-learn user guide, Decision Trees](https://scikit-learn.org/stable/modules/tree.html) (the "Mathematical formulation" section: Gini, splits) and [Random forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests).
- Our features: [Methods](../METHODS.md), the 1,020-D descriptor.
- `forest_memory_upper_bound` in `scripts/classical_models.py` (read its docstring).

Concept questions — answer in your notes:

1. Compute the Gini impurity of class counts [3, 1] and [2, 2]. Why is a split judged by the *weighted* impurity of its two children?
2. Why does each tree use a bootstrap sample and a random subset of features at each split? What would happen if every tree saw the same rows and all features?
3. Every node stores a probability for all classes. Estimate the memory for 64 trees with 256 leaves each and 2,731 classes in float64 (a binary tree with L leaves has 2L − 1 nodes).
4. A tree with at most 256 leaves can output at most 256 different leaf distributions. What does that imply for a problem with 2,731 classes and about 15 training clips per class?
5. Forest "probabilities" are averages of leaf frequencies. Why are they not calibrated probabilities, and why does the ranking return only classes with probability > 0?
6. Name two differences between your tree and scikit-learn's that can change accuracy.

<!-- BEGIN SOLUTION member3 -->
**Answer notes (released after the milestone).**
1. [3, 1]: 1 − (0.75² + 0.25²) = 0.375. [2, 2]: 1 − (0.5² + 0.5²) = 0.5. A split sends different numbers of rows to each side; weighting by n_left/n and n_right/n measures the expected impurity of a randomly chosen training row after the split, so a tiny pure child cannot hide a large impure one.
2. Single deep trees have high variance. Bootstrapping and random feature subsets make the trees different, so their errors are less correlated and averaging reduces variance. With identical data and all features, every tree would be (almost) the same and the forest would be no better than one tree.
3. Nodes = 2 × 256 − 1 = 511 per tree; 64 × 511 × 2,731 × 8 bytes ≈ 714 MB (the project's bound adds 128 bytes per node: about 0.7 GB). With 4,096 leaves it is 16 times more: the tuned model file was 11.5 GB.
4. Each tree can separate at most 256 groups, far fewer than 2,731 words, so most words never own a leaf in a given tree. The original forest indeed used exactly 256 leaves and depth 12 in all 64 trees — the leaf budget, not the data, was the bottleneck. Raising the budget to 4,096 leaves (no depth limit) doubled validation accuracy.
5. Leaf frequencies come from bootstrap samples and very small leaves, and averaging trees does not produce calibrated probabilities; scores are only used for ranking. Classes with probability 0 never reached any leaf the clip fell into, so including them would add arbitrary candidates to Top-5.
6. Examples: depth-first growth limited by max_depth instead of a best-first max_leaf_nodes budget; when no sampled feature gives a valid split, scikit-learn keeps drawing features while ours makes a leaf; different random-number streams for bootstrap and feature choice; tie handling between equally good splits.
<!-- END SOLUTION -->

## B. Implement

Fill in `students/random_forest.py`; `resolve_max_features`, `forest_predict_proba` and `predict_topk` are written for you.

| TODO | Function | Hint |
| --- | --- | --- |
| 1 | `gini_impurity` | Turn counts into proportions along the last axis; guard the all-zero row. Works for one row or a matrix of rows. |
| 2 | `best_split` | Per feature: sort the rows, build one-hot labels, `np.cumsum` gives the left-side class counts for every cut; right = parent − left. Allow only cuts between different values with enough rows on both sides. Keep the best decrease (strictly better replaces). |
| 3 | `build_tree` | A recursive helper `grow(rows, depth)` that appends a node to Python lists and returns its index. Decide "leaf or split" exactly as the docstring lists, then recurse left (`<= threshold`) and right. Convert the lists to arrays at the end. |
| 4 | `tree_predict_proba` | Start every row at node 0; in a loop, move all rows that are not at a leaf one level down; stop when all rows are at leaves. |
| 5 | `fit` | One `rng`; for each tree draw a bootstrap sample with `rng.integers(0, N, N)` and grow a tree on it. |

The self-check compares your root split with scikit-learn's decision tree on random data, checks the depth and leaf-size limits, and checks that the forest is deterministic for a fixed `random_state` yet has different trees.

Afterwards, update `IMPLEMENTATION_NOTE` with what you actually wrote.

## C. Compare with the reference

```bash
python run.py --method random_forest --implementation reference --output outputs/rf-ref
python run.py --method random_forest --implementation student --output outputs/rf-mine
```

The two forests use different random numbers, so do not expect identical predictions. Run your forest with at least five different `random_state` values (one config file each) and report the mean and range of Top-1/Top-5 next to the reference. Is the difference larger than the seed-to-seed variation? Also compare fit and predict times.

## D. Investigate

**D1 — mini experiment.** For both your forest and the reference, vary one setting at a time: n_estimators ∈ {8, 32, 64, 128}, max_depth ∈ {4, 12, 30}, max_features ∈ {"sqrt", 100} (the reference needs an integer max_depth; your forest also accepts `null` for "no limit"). Record Top-1/Top-5 and time. With 20 validation clips, look for trends across seeds rather than single winners.

**D2 — leaf budget.** Using concept questions 3–4, argue which setting most limits the original full-data forest (64 trees, depth 12, at most 256 leaves, 2,731 classes), and predict what happens with more leaves.

**D3 — full data (Xinjie runs it).** Write your predictions first (validation Top-1 of the original forest and of a forest with 4,096 leaves and no depth limit), then interpret the real numbers.

Unlike kNN and SVM, your own forest is not run on the full data: recursive NumPy trees are far too slow for 40,154 clips × 2,731 classes. The full-data forests are scikit-learn's. The original setting runs through `run.py --implementation reference`. The 4,096-leaf forest with no depth limit is outside the teaching runner's reference limits (integer `max_depth`, bounded tree storage), so it comes from the research pipeline instead. Your forest is checked on mini (Parts B–D1), and your full-data contribution is the analysis.

<!-- BEGIN SOLUTION results-2026-10 -->
**Full-data results (2,731 words; 10,304 validation clips, 32,941 test clips).** The tuned forest is scikit-learn's with 64 trees, at most 4,096 leaves and no depth limit.

| System | Validation Top-1 | Test Top-1 |
| --- | ---: | ---: |
| Original forest (64 trees, depth 12, 256 leaves), original descriptor | 16.31% | 12.54% |
| Original forest, left/right-normalized descriptor | 17.94% | 14.84% |
| 64 trees, 4,096 leaves, no depth limit, normalized (selected on validation) | **35.55%** | **28.22%** |
| Same tuned forest, original descriptor (secondary check) | 32.39% | 24.84% |
| Random forest on PoET 128-D CLS (original settings) | 25.84% | 20.93% |

- In the original forest every one of the 64 trees used exactly 256 leaves and depth 12: the leaf budget was binding. With 4,096 leaves accuracy roughly doubled (+13.4 test points, McNemar p < 1e-300), and all 64 trees still filled the budget, so even this is not the forest's ceiling.
- The 4,096-leaf model needs about 37 GB of memory to train and is 11.5 GB on disk — the price of storing a 2,731-class distribution in every node.
- Even on PoET's CLS embedding the original forest reaches only 20.93% on test, while kNN reaches 71.94% on the same features: a forest with a small leaf budget cannot use a good representation either.
<!-- END SOLUTION -->

## E. Write

Report subsection (about 300–500 words) for "Random forest baseline and the leaf-budget effect":

- tree growth, the Gini criterion and bagging, and how your implementation was verified;
- the mini comparison with the reference across seeds;
- the full-data table (validation and test kept apart) and the leaf-budget explanation;
- the memory cost as a practical limitation.

Add one table or figure and 1–2 slides.
