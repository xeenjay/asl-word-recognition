# Member 2 — Linear SVM and left/right-hand normalization

[All roles](README.md) · Code: `students/linear_svm.py`, `scripts/classical_handedness.py` · Self-check: `python -m unittest tests.test_member2_linear_svm tests.test_member2_handedness -v` · Notes: [member2-notes.md](member2-notes.md)

**Goal.** Derive and implement the one-vs-rest linear SVM objective and its gradient, train it with an off-the-shelf optimizer, and show that your weights match scikit-learn's `LinearSVC`. Then implement the label-free rule that mirrors left-hand-dominant signers, and explain why it helps the classical features.

## A. Learn

- Linear classifiers and losses: CS231n notes, ["Linear classification: Support Vector Machine / Softmax"](https://cs231n.github.io/linear-classify/) (the multiclass SVM loss, regularization, bias trick).
- Optimization and gradient checking: CS231n ["Optimization"](https://cs231n.github.io/optimization-1/) (analytic vs numerical gradient).
- [scikit-learn user guide, SVM classification](https://scikit-learn.org/stable/modules/svm.html#classification) and the `LinearSVC` parameters `C`, `loss="squared_hinge"`, `penalty="l2"`.
- Our features: [Methods](../METHODS.md), the 1,020-D descriptor; the hand-normalization module docstring in `scripts/classical_handedness.py`.

Concept questions — answer in your notes:

1. Write the objective for one class and explain each term. What does a large C do, and a small C?
2. Derive the gradient of the objective with respect to W (show the steps). Why is the squared hinge differentiable everywhere while the plain hinge max(0, 1 − m) is not?
3. One-vs-rest trains 2,731 separate classifiers. Are their margins on the same scale? Why can we still rank classes by margin?
4. Our bias is regularized (it is just the last row of W), like `LinearSVC`. What would change if it were not?
5. Why does mirroring the clips of left-hand-dominant signers help the hand-crafted descriptor so much more than PoET? Why is the decision made per signer and without labels, and why is that acceptable for validation/test signers?
6. C is chosen by validation accuracy. Why is the validation score of the chosen C optimistic, and what protects the final test number?

<!-- BEGIN SOLUTION member2 -->
**Answer notes (released after the milestone).**
1. ½‖w_c‖² keeps weights small (wide margin); C·Σ max(0, 1 − y·m)² penalizes clips on the wrong side or inside the margin. Large C trusts the training data more (smaller training error, risk of overfitting, slower convergence); small C regularizes harder (simpler boundary, risk of underfitting). The project's original C = 0.01 was too strong: C = 0.03 improved validation Top-1 by 8.6 points.
2. With m = Y ∘ (X̃W) and s = max(0, 1 − m): ∂(½‖W‖²)/∂W = W and ∂(C Σ s²)/∂W = C Σ 2s · ∂s/∂W = −2C X̃ᵀ(Y ∘ s), since ∂s/∂m = −1 where s > 0 and 0 elsewhere. s² has slope 0 on both sides of m = 1, so it is smooth there; the plain hinge has a kink at m = 1.
3. No — each binary classifier is trained separately, so margins are not calibrated probabilities. Ranking still works because every classifier is trained on the same features with the same loss and regularization, so margins are roughly comparable; that is also why Top-5 is a fair metric while "probabilities" are not reported.
4. Without regularization the bias could grow freely; for rare classes (one positive versus 40,000 negatives) the bias would drift strongly negative. The optimum and therefore the weights would differ from `LinearSVC`, which regularizes the bias through `intercept_scaling`.
5. The descriptor stores left-hand and right-hand features in fixed positions, so a left-dominant signer's clip looks like a different sign to a classifier trained mostly on right-dominant signers. Mirroring puts the dominant hand in the same slots. PoET was trained with this variation and reads the sequence with attention, so it is far less sensitive. The rule uses only unlabeled descriptor statistics of the signer's own clips, so no label or score leaks; every clip of a signer gets the same orientation, which matches how handedness works.
6. The chosen C is the maximum over several validation scores, so part of its advantage is luck on this validation set (selection bias). The test split was used once, after all choices were frozen, so it gives the unbiased number.
<!-- END SOLUTION -->

## B. Implement

Fill in `students/linear_svm.py`; `fit`, `decision_function` and `predict_topk` are written for you.

| TODO | Function | Hint |
| --- | --- | --- |
| 1 | `one_vs_rest_targets` | Start from a matrix of −1 and set one +1 per row. |
| 2 | `squared_hinge_objective` | Follow the formula in the docstring with matrix operations only (no loops). Derive the gradient on paper first (concept question 2). The test `Todo2Objective.test_gradient_matches_finite_differences` compares your gradient with finite differences. |
| 3 | `train_weights` | `scipy.optimize.minimize` needs a function of a 1-D vector: reshape inside, `ravel()` the gradient on the way out. Start from zeros. Return the weights and a small info dict (`converged`, `iterations`). |
| 4 | `top_k_from_scores` | A stable sort of the negated scores; return at most `min(top_k, num_classes)` columns. |

Optional challenge: write plain gradient descent with step size 1/L, where L = 1 + 2C·σ_max(X̃)² bounds the curvature, and compare how many iterations it needs with L-BFGS.

Then the extension in `scripts/classical_handedness.py` (the mirror maps are provided):

| TODO | Function | Hint |
| --- | --- | --- |
| H1 | `clip_indicators` | Two boolean vectors from fixed descriptor columns: hand coverage (`COVERAGE_LEFT` vs `COVERAGE_RIGHT`) and wrist motion (`MOTION_LEFT` vs `MOTION_RIGHT`, summed). Ties are not left-dominant. |
| H2 | `signer_decisions` | Group rows by signer; the fractions are the means of the two indicators; mirror only if both fractions are strictly above the threshold. |
| H3 | `canonicalize_descriptors` | Copy the input, then replace only the rows of mirrored signers with `mirror_descriptors(...)`. |

Afterwards, update `IMPLEMENTATION_NOTE` with what you actually wrote.

## C. Compare with the reference

```bash
python run.py --method linear_svm --implementation reference --output outputs/svm-ref
python run.py --method linear_svm --implementation student --output outputs/svm-mine
```

Both minimize the same objective, so the predictions should agree; the self-check compares the weights (relative difference below 0.1% on a random problem) and the top-5 on the mini data. In your notes compare the run times and the number of optimizer iterations (`model.joblib → info`), and explain what "converged" means for each optimizer.

## D. Investigate

**D1 — C on mini.** Create configs such as `configs/linear_svm-C0.03.json` (`{"method": "linear_svm", "parameters": {"C": 0.03}}`) and run your implementation for C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 1}. Record validation Top-1/Top-5, the final objective and the iterations. With only 20 validation clips the differences are noisy; describe the trend in the objective and the iterations as C grows.

**D2 — hand normalization.** Finish H1–H3 and explain the rule in your notes with one concrete example from the tests.

**D3 — full data (Xinjie runs it).** Write your predictions first: full validation Top-1 for (a) the original setup (C = 0.01, original descriptor), (b) C = 0.01 with left/right-normalized descriptors, (c) the C selected on validation with normalized descriptors. Which signers should gain most from (b)? Then interpret the real numbers, including the per-signer breakdown.

<!-- BEGIN SOLUTION results-2026-10 -->
**Full-data results (2,731 words; 10,304 validation clips, 32,941 test clips).**

| System | Validation Top-1 | Test Top-1 |
| --- | ---: | ---: |
| Linear SVM C=0.01, original descriptor | 38.76% | 30.93% |
| C=0.01, left/right-normalized descriptor | 43.87% | 35.76% |
| C=0.03 (selected on validation), normalized | **52.47%** | **42.52%** |
| C=0.03, original descriptor (secondary check) | 47.03% | 37.32% |
| Linear SVM on PoET 128-D CLS, C=0.01 | 69.91% | 64.11% |
| Official PoET | 80.94% | 73.56% |

- The rule mirrored 2 of 6 validation signers and 3 of 11 test signers (P6, P17, P49). On test, normalization raised SVM Top-1 by 13.8 points on those three signers but only 1.5 points on the other eight — the gain comes almost entirely from the left-dominant signers.
- C grid on validation (normalized descriptor): 0.001 → 0.1; C=0.03 was best. At C=0.1, 280 of the 2,731 binary problems had not converged after 2,000 iterations.
- Normalization (+4.84) and tuning (+6.76) are both significant on test (McNemar p < 1e-130). Even the best classical system remains about 31 points below PoET; with the same classifier on PoET's CLS embedding, about 75% of that gap is due to the representation.
<!-- END SOLUTION -->

## E. Write

Report subsection (about 300–500 words) for "Linear SVM baseline, hand normalization and tuning":

- the objective, the optimizer and the verification against `LinearSVC`;
- the hand-normalization rule and why it is label-free;
- the full-data table (validation and test kept apart) and the per-signer finding;
- one honest limitation (for example, selection on validation).

Add one table or figure and 1–2 slides.
