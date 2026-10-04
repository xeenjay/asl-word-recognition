# Comparing Classical Pose Descriptors and a Lightweight Transformer for Isolated ASL Recognition

> Team learning copy of the October 4 working report. Original video-frame Figure 4 is omitted from Git; the dataset videos and gallery remain separate assets. All results below refer to the original experiments, not a fresh run of this export.

**EN.601.661 Computer Vision · October 4, 2026**
**Prepared by Xinjie Huang with agent assistance**
Project team: Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen

**Version scope.** This report documents completed experiments on the official ASL Citizen training and validation splits. It contains no final-test results. The contribution statement describes the preparation of this version; the team's final individual contributions have not yet been consolidated.

## 1. Introduction

Recognizing a sign from a video is useful for a dictionary interface: a user demonstrates an unfamiliar sign and receives candidate entries. The task studied here is isolated sign language recognition. Each input is a presegmented American Sign Language (ASL) clip containing one labeled sign, and the output is one of 2,731 dataset labels. A label such as UNDERSTAND is a class name, not a generated English sentence. Identifying boundaries in continuous signing, modeling sentence structure, and translating conversations are outside this project's scope.

ASL Citizen provides a practical setting for this question. Desai et al. introduced a community-sourced dataset with 83,399 consented videos from 52 signers, recorded in varied environments [1]. Its official splits separate signers, making recognition of previously unseen people a central challenge. The dataset paper motivates sign-to-dictionary retrieval and evaluates both appearance-based and pose-based systems. PoET, proposed by Lascar et al., uses normalized keypoint sequences and a compact Transformer encoder for isolated recognition [2]. Its public implementation and trained ASL Citizen weights make it possible to study adaptation around an existing recognizer without collecting or annotating new videos [3].

This project asks three questions. First, how well do simple classifiers perform when given the same fixed pose-and-motion descriptor? Second, how do these systems compare with a pretrained sequence model on the same validation videos? Third, can a small relative-geometry branch and a fixed training-example retrieval rule improve a budget-matched PoET baseline? These questions distinguish an educational system comparison from a controlled study of modifications to one model.

The completed work includes a shared data and evaluation pipeline; k-nearest neighbors (kNN), linear support vector machine (SVM), and bounded random forest baselines; deployment of the official PoET checkpoint; matched fine-tuning and geometry/retrieval ablations; and a video demonstration with quantitative and qualitative error analysis. The contribution is the adaptation, implementation, controlled comparison, and interpretation of these components. It does not include inventing kNN, SVM, random forests, MediaPipe, or PoET. All reported project scores use validation data, which has also supported development and checkpoint selection.

## 2. Methods

### 2.1 Data, preprocessing, and evaluation

We retain the official ASL Citizen vocabulary and signer-disjoint partitions [1]. Table 1 gives the exact split sizes from the supplied metadata. The complete training and validation videos and pose caches were prepared. The test metadata is available, but the test videos have not been prepared or evaluated in this version. No new manual annotations were created, and no difficult validation examples were removed.

**Table 1. Official ASL Citizen partitions and their use in this report.**

| Split | Videos | Signers | Use |
| --- | ---: | ---: | --- |
| Training | 40,154 | 35 | Fit classifiers, fine-tune PoET, build retrieval banks |
| Validation | 10,304 | 6 | Development, checkpoint selection, and all reported results |
| Test | 32,941 | 11 | Reserved; no results reported |
| Total | 83,399 | 52 | 2,731 labels |

MediaPipe supplies pretrained body, face, and hand landmark extraction [4]. The same stored raw detections support the classical and neural pipelines, although their representations differ. Presence masks distinguish absent landmarks from valid zero coordinates. Extraction success means that the pipeline produced a usable record; it does not mean that the word was recognized correctly or that every landmark is accurate. Seven validation clips have no hand detections throughout the video; they remain in the denominator.

Top-1 is the fraction of videos whose first prediction matches the label. Top-5 is the fraction whose label appears among the returned candidates, up to five. Macro Top-1 first computes accuracy within each label and then averages over all 2,731 labels, giving classes equal weight. Thus, instance accuracy measures an average clip and macro accuracy measures an average class. Predictions, class order, split membership, and video identities are checked before comparisons. A separate reload evaluation verifies saved model outputs; this checks reproducibility, not independent statistical generalization.

### 2.2 A shared fixed descriptor and three classical classifiers

The classical systems summarize a whole clip rather than classifying a single extracted photograph. Each frame contributes 98 normalized coordinate values: 80 from the two hands relative to their wrists, 14 from seven body landmarks, and four from wrist positions relative to the shoulder center. Hand shape is scaled by a per-video palm-size statistic; body and wrist locations use shoulder width. The descriptor uses image-plane coordinates, not metric three-dimensional measurements. Explicit validity rules exclude missing groups from their corresponding averages and prevent motion differences across missing observations.

Temporal means over one whole-clip segment, two halves, and four quarters contribute 7 × 98 = 686 dimensions. Whole-clip standard deviations add 98; mean signed and absolute adjacent-frame changes add 98 each. Group and adjacent-pair coverage statistics add 40, totaling 1,020 dimensions. Motion is expressed relative to normalized clip time, so it is not physical velocity. The temporal pyramid retains coarse ordering but discards detailed trajectories. StandardScaler is fitted only on the 40,154 training descriptors and reused unchanged for validation and new clips. No PoET weights are used to construct these descriptors.

The kNN baseline follows the standard nearest-neighbor principle [5]. After training-set standardization, descriptors are L2-normalized. For a query q and training descriptor z, cosine similarity is q·z. The five nearest training examples cast equal votes; exactly tied neighbor similarities are ordered by the original training row index. Vote ties are resolved by the nearest supporting neighbor and then class ID. Only classes receiving votes are returned, so Top-5 may contain fewer than five distinct labels. The project implements search and voting with NumPy/PyTorch; kNN has no gradient-training stage but retains the training bank and incurs query-time search cost.

The linear SVM fits one-versus-rest classifiers, one per label, and predicts the largest decision value. In the margin-based classification family [6], it learns a linear boundary over the supplied descriptor rather than matching raw video frames. We use scikit-learn LinearSVC with C = 0.01, L2 regularization, squared-hinge loss, dual = False, tolerance 0.001, and at most 2,000 iterations. OneVsRestClassifier manages the 2,731 binary problems. All classifiers converged in the completed run. Their margins are not probabilities.

Random forests combine randomized decision trees [7]. Our scikit-learn model uses 64 trees, maximum depth 12, at most 256 leaves per tree, at least two samples per leaf, and the square root of the feature count as the split-feature budget. Predictions average tree class probabilities. These bounds make the large vocabulary manageable but limit capacity: every fitted tree reached 256 leaves. SVM and forest use established library implementations [8]; project code supplies the video representation, adapters, serialization, and common evaluation. Each classical method uses one bounded configuration, so the comparison does not estimate its best achievable performance after extensive tuning.

### 2.3 PoET and the relative-geometry branch

The deployed official PoET checkpoint uses sequences of 128 frames with 188 coordinate values per frame. The pinned implementation has hidden width 128, two Transformer layers, four attention heads, and a CLS representation followed by a linear classifier [2,3]. Its preprocessing includes articulator-specific normalization and the original temporal length adjustment. Training and evaluation sampling follow their respective recorded modes. We distinguish the unmodified official checkpoint, used in the system comparison, from B, our additionally fine-tuned matched baseline.

Relative geometry addresses information that can be weakened by normalizing articulators separately. From the shared image coordinates, we compute seven relations: each wrist to the nose, mouth center, and shoulder center, plus one wrist-to-wrist relation. For each relation, the target-minus-source displacement is divided by the video's median valid shoulder width. Its x displacement, y displacement, Euclidean length, and presence bit form four values, giving 28 per frame. Invalid relations are zeroed; a video without a valid scale receives zero geometry. These features do not encode physical contact or reliable 3D depth.

The geometry branch is a 28→32→128 MLP with ReLU and 5,152 parameters. Its last layer starts at zero. A simple description of its insertion is:

**modified frame embedding = original frame embedding + valid-frame mask × geometry MLP(28-D relations).**

The residual enters before the Transformer, allowing geometry to affect subsequent temporal processing. Geometry uses exactly the frame indices selected for the main sequence. This alignment prevents a hand position from one moment being attached to another moment. No target label or true class attribute is needed at inference.

For matched B/G comparisons, seeds 20260922, 20260923, and 20260924 each start from the same official checkpoint. Every run uses all training videos for 10 epochs, batch size 128, and 3,140 optimizer steps. AdamW uses weight decay 0.01 and a fixed learning rate of 0.00003 for the original network; the new geometry branch uses 0.0003. All original layers remain trainable. Runs use FP32 and the same recorded temporal-sampling and class-weighted cross-entropy rules, with class weight w_c = 1 - n_c / N (n_c is the training count for class c, N the total), without added spatial augmentation. The best checkpoint is selected by complete-validation macro Top-1, including epoch zero and retaining the earlier checkpoint on ties. The first seed informed earlier development; the later seeds therefore provide additional repeatability evidence, not a new held-out dataset.

### 2.4 Fixed retrieval and the demonstration

R is a postprocessing rule that uses each checkpoint's own 128-dimensional CLS embeddings. It is distinct from classical kNN on hand-engineered descriptors. Each B or G checkpoint encodes all training examples to create its own labeled bank. For an L2-normalized query q and reference z, the kernel is **K(q,z) = exp[−10 × (1 − q·z)]**. Within a candidate class, kernel values are first averaged within each training signer and then averaged across that class's signers. This limits domination by a signer with more references.

Only the model's original five candidate classes are reranked. The five logits are converted to a conditional softmax distribution, and the five retrieval scores are normalized across those candidates. The final score is **0.75 × model probability + 0.25 × retrieval probability**. This fixed rule changes ordering without adding candidates, so Top-5 membership and hit rate remain identical to the corresponding B or G model. The query's true label is never used. The factorial comparison reuses six existing checkpoints and performs no further parameter updates.

The accompanying browser demonstration accepts a presegmented video, extracts landmarks once, and runs kNN, SVM, forest, official PoET, and one fixed G+R checkpoint. Saved validation examples are separated from fresh inference, and uploaded clips without labels do not receive an accuracy claim. Displayed scores have method-specific meanings: votes, margins, tree estimates, and softmax-derived values are not interchangeable calibrated confidence measures. The demo is an offline clip recognizer; no end-to-end streaming performance claim is made.

## 3. Results

### 3.1 Complete validation comparison

Table 2 uses all 10,304 validation videos, with all classes represented and zero processing failures. SVM is the strongest fixed-descriptor method, exceeding kNN by 20.0795 percentage points. The official PoET checkpoint correctly recognizes 8,340 videos, versus 3,994 for SVM. Macro and instance rankings agree. The large PoET gap is a system-level observation: learned temporal features, landmark selection, preprocessing, training history, and development effort differ alongside the classifier.

**Table 2. Four systems on the same complete validation cohort. All accuracy entries are percentages.**

| System | Top-1 | Top-5 | Macro Top-1 | Top-1 correct |
| --- | ---: | ---: | ---: | ---: |
| Cosine kNN | 18.6821 | 36.1025 | 19.1649 | 1,925 |
| Linear SVM | 38.7616 | 63.1308 | 39.5262 | 3,994 |
| Bounded random forest | 16.3141 | 38.2182 | 16.7851 | 1,681 |
| Official PoET checkpoint | 80.9394 | 96.3800 | 81.1950 | 8,340 |

![Figure 1. Four complete systems on all 10,304 validation clips. The classical methods share fixed descriptors; official PoET uses its released sequence model. Differences in representation and training history prevent attributing the gap solely to the classifier.](figures/fig01_method_comparison.png)

### 3.2 Matched geometry and retrieval ablation

Table 3 reports the 2×2 ablation separately from Table 2. B is fine-tuned PoET, not the original official weights. G+R improves mean Top-1 by 0.5499 percentage points over B, 0.2620 over G, and 0.2135 over B+R. It exceeds each single modification in both instance and macro accuracy for every observed seed. The paired G+R minus B differences are +0.4367, +0.6793, and +0.5338 points, corresponding to 45, 70, and 55 additional correct clips.

**Table 3. Three matched seeds; mean ± sample standard deviation, in percentage units. The same validation videos are reused across seeds.**

| Method | Top-1 | Macro Top-1 | Mean Top-1 difference from B |
| --- | ---: | ---: | ---: |
| B: fine-tuned PoET | 81.1853 ± 0.0341 | 81.4711 ± 0.0578 | — |
| B+R: retrieval | 81.5217 ± 0.0700 | 81.7976 ± 0.1061 | +0.3364 pp |
| G: geometry | 81.4732 ± 0.1009 | 81.7451 ± 0.1112 | +0.2879 pp |
| G+R: geometry and retrieval | 81.7352 ± 0.0889 | 81.9937 ± 0.0791 | +0.5499 pp |

The improvement contains both fixes and regressions. For seed 20260923, G+R corrects 111 B errors but changes 41 previously correct outputs into errors, producing the net gain of 70. The mean interaction, (G+R−G)−(B+R−B), is −0.0744 points: the gains partially overlap rather than combine super-additively. Separately, disabling the geometry residual in one fixed trained checkpoint reduced Top-1 from 81.3568% to 80.7842%. This shows reliance on that input but is not a substitute for the matched B comparison, because disabling a learned branch changes its operating conditions.

![Figure 2. Matched PoET ablation across three seeds. Error bars show sample standard deviation, not a confidence interval; the restricted vertical axis makes small changes visible. B differs from the official checkpoint in Figure 1. Mean G+R improvement is 0.550 percentage points on reused validation data.](figures/fig02_paired_ablation.png)

### 3.3 Error overlap and illustrative clips

Although weaker overall, SVM correctly recognizes 357 clips missed by official PoET. Conversely, PoET recognizes 4,703 clips missed by SVM. Across all four systems, 461 clips are correct for every method and 1,428 are wrong for every method. Such overlap distinguishes shared hard cases from model-specific errors. The hypothetical SVM-or-PoET accuracy of 84.4041% requires an oracle that knows which prediction is correct; it is not a deployable fusion result.

![Figure 3. Correctness overlap with official PoET. The classical-only segments are potential complementarity, not achieved ensemble gains. Counts for different classical systems overlap and cannot be added together. Each row covers all 10,304 validation clips.](figures/fig03_error_complementarity.png)

**Table 4. Examples from the complete validation records. O = official PoET. B and G+R use fixed seed 20260923. Index is the project validation index.**

| Index / target | Observed predictions | Interpretation supported by the record |
| --- | --- | --- |
| 11 / UNDERSTAND | All four systems: UNDERSTAND | Shared success; visible blur did not prevent recognition |
| 17 / FINS | SVM: FINS; O: SWEARIN | Classical rescue; FINS is O's second candidate |
| 10 / SPECIAL1 | O: CHIMNEY; all four wrong | Shared failure; target absent from O's Top-5 |
| 2312 / INTUITIVE | O: ACCESS; all four wrong | Cache records zero hand detections throughout |
| 59 / KICK2 | B: BRIDGE1; G+R: KICK2 | A correction relative to the matched baseline |
| 0 / SUPPORT1 | B: SUPPORT1; G+R: CABINET | A regression relative to the matched baseline |

The gallery contains 16 deterministically selected examples and 48 frames sampled at 25%, 50%, and 75% of each clip. Selection covers shared successes, shared failures, classical rescues, PoET-only successes, and absent hand detections. These are teaching examples, not a random accuracy sample. Frame review records visible positions, blur, and framing separately from hypotheses; still images alone do not establish ASL semantics or why a prediction failed.

Performance also varies across observational slices. Excluding the signer with only two clips, official PoET signer-specific Top-1 ranges from 69.95% to 85.67%. For mean hand-detection fractions below 25% but above zero, PoET achieves 78.10%; at 50% or above it achieves 84.64%. This fraction is the sum of left- and right-hand detection counts divided by twice the decoded frame count, not the proportion with either hand visible. One-handed signs, idle frames, signer identity, and class mix confound these associations. All seven clips with no hand detections are misclassified, but correcting all seven could improve total accuracy by at most 0.0679 points.

*Figure 4. Real validation frames at approximately 25%, 50% and 75% of duration. UNDERSTAND is a shared success; PARADE is a shared failure. In the matched seed20260923 comparison, G+R fixes KICK2 but regresses on SUPPORT1. Panels a-b use official PoET; panels c-d use matched fine-tuned B and G+R. Labels are provided by the dataset. These deliberately selected examples do not establish a causal explanation or an accuracy estimate.*

Original image panels are not included in this code repository.

## 4. Discussion

### 4.1 What the comparisons establish

Under a shared representation, the SVM result suggests that learned discriminative boundaries use the fixed summaries more effectively than our cosine distance or bounded forest. It does not prove that linear SVM is intrinsically superior on every pose representation. The forest reached its leaf budget, and coarse temporal pooling can merge distinct trajectories. Feature audits found finite values, aligned labels, correct training-only standardization, and complete rows. Thus, low scores cannot simply be dismissed as missing input files, but the audits do not establish that every feature design choice is optimal.

PoET provides a much stronger reference system. The matched ablation supports a narrower contribution: explicit image-plane relations and a fixed retrieval rule produce modest, directionally consistent validation gains within the tested budget. The absence of an equal-parameter branch using only the original inputs limits claims that the benefit comes uniquely from new geometry rather than extra capacity. Likewise, standard deviations over three runs measure seed variation under this experiment, not uncertainty over every future signer or dataset.

### 4.2 Relation to prior work and unsuccessful directions

Our work follows ASL Citizen's isolated dictionary-retrieval setting [1] and adapts the existing PoET recognizer [2,3]. The dataset paper reports roughly 63% recognition accuracy and 91% recall at ten on held-out test signers [1]. Those published test results cannot be ranked directly against our validation figures. Nor does reusing an official checkpoint reproduce the paper's entire training search, all datasets, or its online user study. The present contribution is a documented comparison and a small adaptation of an established system, not a new state-of-the-art claim.

More elaborate additions were not automatically better. Attribute auxiliary supervision yielded only four additional correct clips initially and two after changing the new-head learning rate, relative to its matched baseline. A cross-signer supervised-contrastive experiment obtained 81.0850% against its matched classification control at 81.1141%. Extending matched B/G training to 20 epochs reduced the mean geometry advantage from 0.2879 to 0.2588 points. A later hand-RGB temporal fusion reached 81.8517% for one seed, only two clips above same-seed G+R at 81.8323%, while lowering the underlying G Top-5 from 96.5450% to 96.1277%. These results motivate retaining the simpler geometry/retrieval candidate rather than presenting complexity as evidence of progress.

### 4.3 Limitations and next evaluation

Repeated development on validation is the main limitation. Checkpoint selection, design revisions, and exploratory alternatives can favor this particular cohort; the first seed also informed tuning. A positive three-seed mean therefore does not establish statistical significance or an improvement on the reserved test split. The planned final evaluation should first freeze checkpoint choices, feature definitions, retrieval weights, and failure accounting, then evaluate the official test videos once under that protocol. Test errors must not become a new tuning set.

Generalization beyond ASL Citizen is also unmeasured. The vocabulary is closed, word boundaries are supplied, and arbitrary uploads may differ in framing, signing style, or label coverage. The current system always returns candidates rather than a validated unknown-sign decision. A useful future study would evaluate calibrated rejection and new signers under an agreed protocol. More targeted engineering questions include less conservative body masks and robust motion summaries, but they require new controlled comparisons. We have not completed a cost benchmark for every method under common hardware and measurement conditions, so isolated timing observations are not used to rank speed. Finally, mutual error rescue suggests potential complementarity; learning a selector would require separate development data and cannot be justified by choosing the correct model retrospectively.

## 5. Conclusion

This project produces a reproducible comparison of four isolated ASL recognition systems and a controlled adaptation of PoET. On the complete ASL Citizen validation split, the official PoET checkpoint reaches 80.9394% Top-1, while the best fixed-descriptor baseline, linear SVM, reaches 38.7616%. Relative geometry and fixed retrieval increase matched fine-tuned PoET from 81.1853% to 81.7352% on average over three seeds. The gain is small, includes regressions, and remains a validation finding. Complete denominators, negative experiments, error overlap, and a working video interface provide evidence beyond a single favorable accuracy number.

**Contribution statement for this version.** Xinjie Huang initiated and directed the implementation and experiment program, selected the project scope, and prepared this report with coding-agent and writing assistance. Agent-assisted work covered data preparation, model adapters, training and evaluation infrastructure, result verification, the demonstration, and analysis drafts under his supervision. Xinjie remains responsible for reviewing and explaining the submitted material. Duncan Stadler, Leyun Kim, and Ming Chen are confirmed project members, but their completed individual contributions have not yet been consolidated in this version. No baseline is attributed to them merely because a reference implementation exists. The final team contribution statement must record their actual work before submission; the project retains experiment and AI-use records for review.

## References

[1] A. Desai, L. Berger, F. O. Minakov, V. Milan, C. Singh, K. Pumphrey, R. E. Ladner, H. Daumé III, A. X. Lu, N. Caselli, and D. Bragg. **ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition.** Advances in Neural Information Processing Systems 36, Datasets and Benchmarks, 2023. [Paper](https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf).

[2] J. Lascar, J. Françoise, M. Gouiffès, A. Braffort, and D. Fabre. **PoET: Lightweight Pose Encoder Transformer for Online Sign Language Recognition.** Proceedings of VISAPP 2026, vol. 1, pp. 19–28, 2026. [DOI: 10.5220/0014237500004084](https://doi.org/10.5220/0014237500004084).

[3] J. Lascar et al. **PoET: Official Implementation and Pretrained Weights.** Project repository, revision dca74c30d9d604cb3b92de8fc73ce91484644259, 2026. [Repository](https://gitlab.lisn.upsaclay.fr/mtals/publications/poet).

[4] I. Grishchenko and V. Bazarevsky. **MediaPipe Holistic—Simultaneous Face, Hand and Pose Prediction, on Device.** Google Research, December 10, 2020. [Technical description](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/).

[5] T. M. Cover and P. E. Hart. **Nearest Neighbor Pattern Classification.** IEEE Transactions on Information Theory, 13(1), pp. 21–27, 1967. [DOI: 10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964).

[6] C. Cortes and V. Vapnik. **Support-Vector Networks.** Machine Learning, 20, pp. 273–297, 1995. [DOI: 10.1007/BF00994018](https://doi.org/10.1007/BF00994018).

[7] L. Breiman. **Random Forests.** Machine Learning, 45, pp. 5–32, 2001. [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

[8] F. Pedregosa et al. **Scikit-learn: Machine Learning in Python.** Journal of Machine Learning Research, 12, pp. 2825–2830, 2011. [Paper](https://www.jmlr.org/papers/v12/pedregosa11a.html).
