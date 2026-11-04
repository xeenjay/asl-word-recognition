# Methods: from one sign video to a label

[Repository home](../README.md) · [Code map](CODE_MAP.md) · [Results](RESULTS.md)

This guide explains the implemented reference methods and the questions their comparison can answer. It describes the completed research configuration; it is not a promise that every workflow runs from this checkout without external assets. Use [Setup](SETUP.md) for executable instructions.

## 1. The prediction task and shared data

An input is a presegmented ASL Citizen clip containing one labeled sign. The prediction is a class from the fixed 2,731-label vocabulary. The system does not identify word boundaries, translate a sentence, or validate that an arbitrary upload belongs to the vocabulary.

The official splits contain 40,154 training, 10,304 validation, and 32,941 test clips, with no signer shared between splits. Training supplies model fitting, dataset-level statistics, and retrieval references. Validation supports development and checkpoint selection. Test was evaluated once, on October 12, 2026, after the compared systems were frozen under a preregistered protocol; it was never used for development or tuning (see [Results](RESULTS.md)). Unless labeled as test, the numbers in this guide are validation results.

MediaPipe extracts landmarks from the video. Raw detections support both classical and neural processing, but the final representations differ. Presence masks distinguish missing detections from valid coordinates equal to zero. The seven validation clips with no detected hands remain in all reported denominators. Producing a valid cache record does not imply that all landmarks are accurate or the sign is recognized correctly.

## 2. A fixed 1,020-dimensional clip descriptor

The classical methods receive one vector per video. The vector is constructed without using PoET's weights.

Each frame provides 98 coordinate values:

| Coordinate group | Dimensions | Meaning |
| --- | ---: | --- |
| Hand coordinates relative to their wrists | 80 | Hand shape after removing wrist translation |
| Seven body landmarks | 14 | Normalized body configuration |
| Both wrists relative to shoulder center | 4 | Hand locations in a shared body frame |

Hand shape is scaled using a per-video palm-size statistic; body and wrist locations use shoulder width. These are normalized image-plane coordinates, not metric 3D positions. Validity rules exclude missing groups from their corresponding summaries and prevent motion differences across missing observations.

The descriptor combines:

| Summary | Dimensions |
| --- | ---: |
| Mean coordinates over the whole clip, its two halves, and four quarters | 7 × 98 = 686 |
| Whole-clip coordinate standard deviations | 98 |
| Mean signed adjacent-frame change | 98 |
| Mean absolute adjacent-frame change | 98 |
| Group and adjacent-pair coverage statistics | 40 |
| **Total** | **1,020** |

The temporal pyramid keeps some coarse ordering: a hand position in the first quarter contributes somewhere different from the same position in the last quarter. It still loses detailed trajectories and fine timing. Motion uses normalized clip time and is not physical velocity.

A StandardScaler is fitted on training descriptors only. Its stored parameters are reused for validation and new clips. This dataset-level fitting is distinct from deterministic normalization within an individual video, which is also applied at inference.

## 3. Three classical reference classifiers

### Cosine kNN

The standardized descriptors are L2-normalized. Similarity between query q and training reference z is their dot product q·z. The five nearest training examples vote equally for their labels.

The implementation specifies tie handling: equal neighbor similarities are ordered by training-row index; vote ties are resolved by the nearest supporting neighbor, then class ID. It returns only labels with positive votes. Five neighbors can represent fewer than five distinct labels, so a kNN Top-5 list may be shorter than five.

kNN stores a training bank rather than learning network parameters. Most computation occurs when comparing a query against that bank. The project implements standard search and voting with NumPy/PyTorch. The classical descriptor bank is different from the learned-embedding bank used by PoET retrieval below.

Useful study question: which distinctions in a sign can be lost when similarity is computed over a pooled descriptor?

### Linear SVM

The SVM learns a linear decision function for each label. A one-versus-rest classifier treats one label as positive and all other labels as negative; prediction takes the largest decision value across 2,731 classifiers.

The completed reference run uses scikit-learn `LinearSVC` inside `OneVsRestClassifier`, with C = 0.01, L2 regularization, squared-hinge loss, `dual=False`, tolerance 0.001, and at most 2,000 iterations. All binary classifiers converged in that run. Their decision margins are not probabilities.

Useful study question: how can learned class boundaries outperform nearest-example matching while using exactly the same input vector?

### Bounded Random Forest

The forest combines randomized decision trees and averages their class probability estimates. The reference configuration uses 64 trees, maximum depth 12, at most 256 leaves per tree, at least two samples per leaf, and the square root of the feature count as the split-feature budget.

The vocabulary makes unconstrained trees expensive. These bounds control complexity but also restrict capacity: every fitted tree reached 256 leaves. The observed score therefore describes this bounded configuration, not the best possible Random Forest after extensive tuning.

Useful study question: what can a tree express that a linear SVM cannot, and how might an aggressive leaf budget offset that flexibility?

All three methods use one bounded reference configuration. Their relative scores do not establish a universal ranking of the algorithms or an exhaustive hyperparameter search.

## 4. PoET and a relative-geometry branch

The deployed official PoET model processes 128-frame sequences containing 188 coordinate values per frame. Its frame embedding has width 128. The pinned architecture uses two Transformer layers, four attention heads, a CLS representation, and a linear classifier. Its normalization and temporal length adjustment follow the recorded implementation.

The Transformer already models time, and PoET already includes position information such as wrist locations. Our hypothesis is narrower: normalizing articulators separately can weaken certain relations between the hands, face, and body.

The geometry branch computes seven relations in the shared image coordinates:

- Each wrist to the nose.
- Each wrist to the mouth center.
- Each wrist to the shoulder center.
- One wrist-to-wrist relation.

For each relation, target-minus-source displacement is divided by the video's median valid shoulder width. The x displacement, y displacement, Euclidean length, and presence bit give four features, totaling 28 per frame. Invalid relations are zeroed; without a valid scale, the video receives zero geometry. These quantities do not prove physical contact or recover reliable depth.

A 28→32→128 MLP with ReLU adds 5,152 parameters. Its final layer is initialized to zero. The residual is inserted before the Transformer:

`modified frame embedding = original frame embedding + valid-frame mask × geometry MLP`

The geometry and original sequence use exactly the same selected frame indices. The branch requires no true class label or target attribute at inference.

## 5. The matched PoET experiment

The official checkpoint in the four-system comparison is not the same model as the fine-tuned baseline B in this experiment.

For seeds 20260922, 20260923, and 20260924, each B/G pair starts from the same official checkpoint and uses all training videos for 10 epochs, batch size 128, and 3,140 optimizer steps. The original network remains trainable. AdamW uses weight decay 0.01, a fixed learning rate of 0.00003 for the original network, and 0.0003 for the geometry branch. Runs use FP32 and the same recorded temporal sampling and class-weighted cross-entropy, with no added spatial augmentation.

The class weight is `w_c = 1 - n_c / N`, where `n_c` is the training count for class c and N is the total training count. The best checkpoint is selected by complete-validation macro Top-1, including epoch zero, with earlier checkpoints retained on ties.

The first seed informed earlier development. The additional seeds provide repeatability evidence under the tested protocol, not a new held-out dataset. There is no equal-parameter control branch using only the original features, which limits claims that any gain comes uniquely from geometric information rather than added capacity.

## 6. Retrieval refinement R

Each B or G checkpoint encodes the training clips into its own 128-dimensional CLS-embedding bank. Query and reference vectors are L2-normalized. Their similarity kernel is:

`K(q, z) = exp(-10 × (1 - q·z))`

Within each candidate class, kernel values are averaged first within each training signer, then across that class's signers. This reduces the influence of signers with more examples.

Only the model's **original five candidate classes** are considered. Their logits are converted to a conditional softmax distribution, and the retrieval scores are normalized across the same candidates. The final score is:

`0.75 × model probability + 0.25 × retrieval probability`

The rule is fixed and uses no query label. It can change the first prediction but cannot add a missing class to the candidate set. Therefore Top-5 membership is unchanged for the corresponding B or G model. The factorial comparison reuses the six trained checkpoints and adds no parameter updates.

| Variant | Geometry branch | Retrieval reranking |
| --- | --- | --- |
| B | No | No |
| B+R | No | Yes |
| G | Yes | No |
| G+R | Yes | Yes |

See [Results](RESULTS.md) for the paired gains, errors corrected, and previously correct predictions lost. A gain smaller than the sum of two separate gains is consistent with overlapping benefits; it is not evidence of super-additive synergy.

## 7. Evaluation and interpretation

Top-1 measures the fraction of clips classified correctly. Top-5 checks whether the label appears among up to five returned candidates. Macro Top-1 computes accuracy within each label and averages across all 2,731 labels, giving classes equal weight.

Keep these distinctions when discussing results:

- The traditional systems share a descriptor, but PoET uses a learned sequence representation and pretrained initialization. Their comparison is about complete systems.
- B/G comparisons match more experimental factors; validation reuse and earlier development still limit generalization claims.
- Reloading a saved model and reproducing predictions verifies engineering reproducibility, not new statistical generalization.
- Three-seed error bars in the report are sample standard deviations, not confidence intervals.
- Classical votes, SVM margins, forest estimates, and softmax-derived scores are not interchangeable calibrated confidence values.
- Cached-model timing excludes video decoding and landmark extraction. A fair end-to-end cost comparison must account for those stages separately.

## 8. Other experiments and demo boundaries

The project retains unsuccessful or weak directions, including attribute auxiliary supervision, cross-signer supervised contrastive training, extended training, and hand-RGB experiments. More complexity did not reliably improve the main result. The written report explains these observations; do not omit them when describing how the current candidate was chosen.

The demo shares one landmark extraction across kNN, SVM, Random Forest, official PoET, and a fixed seed-20260923 G+R model. It distinguishes saved labeled examples from fresh inference. An uploaded video without a known label has predictions, not an accuracy score. The system has no validated unknown-sign rejection mechanism and makes no established end-to-end streaming claim.

## Sources and implementation credit

- [ASL Citizen paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf)
- [PoET paper](https://www.scitepress.org/PublishedPapers/2026/142375/) and [official source](https://gitlab.lisn.upsaclay.fr/mtals/publications/poet)
- [MediaPipe Holistic](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/)
- [scikit-learn LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [one-versus-rest](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html), and [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

The project used substantial agent assistance for adapters, feature code, experiment infrastructure, verification, and explanations. Team contributions should identify actual work and review, while retaining credit for upstream algorithms and software.
