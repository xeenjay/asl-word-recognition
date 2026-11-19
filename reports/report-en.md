# Comparing Classical Pose Descriptors and a Lightweight Transformer for Isolated ASL Recognition

> Team copy of the final report draft. First written October 4, 2026; this version (November 15, 2026) matches the PDF shared with the team and includes the one-shot final test of October 12 and the follow-up experiments. Original video-frame Figure 4 is omitted from Git; the dataset videos and gallery remain separate assets. All results below refer to the original experiments, not a fresh run of this export.

**EN.601.661 Computer Vision · Final project report · Version of November 15, 2026**  
Team: Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen

**Scope.** Development used the official ASL Citizen training and validation splits. The reserved test split was evaluated once, on October 12, 2026, for 29 systems fixed in advance. Validation and test results are always reported separately.

## 1. Introduction

Recognizing a sign from a video is useful for a dictionary interface: a user demonstrates an unfamiliar sign and receives candidate entries. The task studied here is isolated sign language recognition. Each input is a presegmented American Sign Language (ASL) clip containing one labeled sign, and the output is one of 2,731 dataset labels. A label such as UNDERSTAND is a class name, not a generated English sentence. Identifying boundaries in continuous signing, modeling sentence structure, and translating conversations are outside this project's scope.

ASL Citizen provides a practical setting for this question. Desai et al. introduced a community-sourced dataset with 83,399 consented videos from 52 signers, recorded in varied environments [1]. Its official splits separate signers, making recognition of previously unseen people a central challenge. The dataset paper motivates sign-to-dictionary retrieval and evaluates both appearance-based and pose-based systems. PoET, proposed by Lascar et al., uses normalized keypoint sequences and a compact Transformer encoder for isolated recognition [2]. Its public implementation and trained ASL Citizen weights make it possible to study adaptation around an existing recognizer without collecting or annotating new videos [3].

This project asks three questions. First, how well do simple classifiers perform when given the same fixed pose-and-motion descriptor? Second, how do these systems compare with a pretrained sequence model on the same validation videos? Third, can a small relative-geometry branch and a fixed training-example retrieval rule improve a budget-matched PoET baseline? These questions distinguish an educational system comparison from a controlled study of modifications to one model. After development, a single evaluation on the reserved test split checks which validation findings hold for unseen signers. Follow-up experiments then ask why the classical baselines are weak: how much is due to left/right-hand variation between signers and to the bounded original settings, and how much to the descriptor itself.

The completed work includes a shared data and evaluation pipeline; k-nearest neighbors (kNN), linear support vector machine (SVM), and bounded random forest baselines; a label-free handedness canonicalization of their shared descriptor, a bounded tuning study, and a representation-swap experiment; deployment of the official PoET checkpoint; matched fine-tuning and geometry/retrieval ablations; mirror test-time augmentation for the official checkpoint; a one-shot final test evaluation; and a video demonstration with quantitative and qualitative error analysis. The contribution is the adaptation, implementation, controlled comparison, and interpretation of these components. It does not include inventing kNN, SVM, random forests, MediaPipe, or PoET. Validation data supported our development and checkpoint selection, and the authors of PoET also selected the official checkpoint on it (Section 2.3), so validation scores are optimistic. Test results come from the single October 12 evaluation and are labeled as such.

## 2. Methods

### 2.1 Data, preprocessing, and evaluation

We retain the official ASL Citizen vocabulary and signer-disjoint partitions [1]. Table 1 gives the exact split sizes from the supplied metadata. The complete training and validation videos and pose caches were prepared for development. The test videos were used only for the one-shot final evaluation on October 12, 2026. Its protocol was frozen before any test data were used: 29 systems were fixed in advance and all are reported, and nothing was selected or tuned on test. No new manual annotations were created, and no difficult validation examples were removed.

**Table 1. Official ASL Citizen partitions and their use in this report.**

| Split | Videos | Signers | Use |
| --- | ---: | ---: | --- |
| Training | 40,154 | 35 | Fit classifiers, fine-tune PoET, build retrieval banks |
| Validation | 10,304 | 6 | Development, checkpoint and hyperparameter selection; validation results |
| Test | 32,941 | 11 | One-shot final evaluation (October 12, 2026) |
| Total | 83,399 | 52 | 2,731 labels |

MediaPipe supplies pretrained body, face, and hand landmark extraction [4]. The same stored raw detections support the classical and neural pipelines, although their representations differ. Presence masks distinguish absent landmarks from valid zero coordinates. Extraction success means that the pipeline produced a usable record; it does not mean that the word was recognized correctly or that every landmark is accurate. Seven validation clips have no hand detections throughout the video; they remain in the denominator.

Top-1 is the fraction of videos whose first prediction matches the label. Top-5 is the fraction whose label appears among the returned candidates, up to five. Macro Top-1 first computes accuracy within each label and then averages over all 2,731 labels, giving classes equal weight. Thus, instance accuracy measures an average clip and macro accuracy measures an average class. Predictions, class order, split membership, and video identities are checked before comparisons. A separate reload evaluation verifies saved model outputs; this checks reproducibility, not independent statistical generalization.

When two systems are scored on the same clips, we count fixes (clips only the second system recognizes) and regressions (the reverse) and apply an exact two-sided McNemar test to these discordant clips. For the three-seed PoET ablation we also report 95% t-intervals of the seed-level differences (n = 3, two degrees of freedom). Both summaries are descriptive. McNemar treats clips as independent, although the validation clips come from only six signers, and the same validation split selected the checkpoints being compared. With three seeds, the intervals are wide and describe seed-to-seed variation only. For test comparisons we also report a bootstrap 95% interval that resamples the 11 test signers.

### 2.2 A shared fixed descriptor and three classical classifiers

The classical systems summarize a whole clip rather than classifying a single extracted photograph. Each frame contributes 98 normalized coordinate values: 80 from the two hands relative to their wrists, 14 from seven body landmarks, and four from wrist positions relative to the shoulder center. Hand shape is scaled by a per-video palm-size statistic; body and wrist locations use shoulder width. The descriptor uses image-plane coordinates, not metric three-dimensional measurements. Explicit validity rules exclude missing groups from their corresponding averages and prevent motion differences across missing observations.

Temporal means over one whole-clip segment, two halves, and four quarters contribute 7 × 98 = 686 dimensions. Whole-clip standard deviations add 98; mean signed and absolute adjacent-frame changes add 98 each. Group and adjacent-pair coverage statistics add 40, totaling 1,020 dimensions. Motion is expressed relative to normalized clip time, so it is not physical velocity. The temporal pyramid retains coarse ordering but discards detailed trajectories. StandardScaler is fitted only on the 40,154 training descriptors and reused unchanged for validation and new clips. No PoET weights are used to construct these descriptors.

The kNN baseline follows the standard nearest-neighbor principle [5]. After training-set standardization, descriptors are L2-normalized. For a query q and training descriptor z, cosine similarity is q·z. The five nearest training examples cast equal votes; exactly tied neighbor similarities are ordered by the original training row index. Vote ties are resolved by the nearest supporting neighbor and then class ID. Only classes receiving votes are returned, so Top-5 may contain fewer than five distinct labels; with the original descriptor this happens for 3,203 of the 10,304 validation queries. kNN Top-5 is therefore not comparable with the Top-5 of the other systems. The project implements search and voting with NumPy/PyTorch; kNN has no gradient-training stage but retains the training bank and incurs query-time search cost.

The linear SVM fits one-versus-rest classifiers, one per label, and predicts the largest decision value. In the margin-based classification family [6], it learns a linear boundary over the supplied descriptor rather than matching raw video frames. We use scikit-learn LinearSVC with C = 0.01, L2 regularization, squared-hinge loss, dual = False, tolerance 0.001, and at most 2,000 iterations. OneVsRestClassifier manages the 2,731 binary problems. All classifiers converged in the completed runs. Their margins are not probabilities.

Random forests combine randomized decision trees [7]. Our scikit-learn model uses 64 trees, maximum depth 12, at most 256 leaves per tree, at least two samples per leaf, and the square root of the feature count as the split-feature budget. Predictions average tree class probabilities. These bounds make the large vocabulary manageable but limit capacity: every fitted tree reached 256 leaves. SVM and forest use established library implementations [8]; project code supplies the video representation, adapters, serialization, and common evaluation. Each classical method uses one bounded configuration fixed in advance, without a validation search, so the main comparison does not estimate its best achievable performance. Section 2.5 describes a later bounded search.

**Handedness canonicalization.** The descriptor stores the left and right hands in fixed slots and applies no rotation or handedness normalization. A left-dominant signer's dominant hand therefore fills the slots that right-dominant signers use for the non-dominant hand. We found this omission after the first validation comparison, from the per-signer validation results (Section 3.2). Before computing any mirrored result, we wrote down a label-free rule. Within each split, all clips of a signer are mirrored if two indicators read from the original descriptor both favor the left hand in more than half of that signer's clips: whole-clip detection coverage, and wrist motion relative to the shoulder center. Mirroring maps x to 1 − x for detected landmarks, swaps the hand arrays and presence masks, and swaps symmetric body landmarks; for this descriptor it is exactly a fixed signed permutation of the 1,020 values. The rule mirrored four training signers (3,099 clips) and two validation signers, P12 and P5 (1,306 clips). It uses signer identity and each signer's own unlabeled clips, so it is a signer-level adaptation rather than a per-clip feature; a deployed system could instead ask users for their dominant hand. We evaluated it in two ways, with all hyperparameters unchanged. Query-only mirroring keeps the original models and mirrors only the validation clips of the selected signers. Canonical refitting applies the rule to training and validation data, refits the scaler on training data, and retrains all three classifiers. For the test evaluation the same rule was frozen in advance and applied to each test signer's unlabeled clips; it mirrored three of the 11 test signers.

### 2.3 PoET and the relative-geometry branch

The deployed official PoET checkpoint uses sequences of 128 frames with 188 coordinate values per frame. The pinned implementation has hidden width 128, two Transformer layers, four attention heads, and a CLS representation followed by a linear classifier [2,3]. Its preprocessing includes articulator-specific normalization and the original temporal length adjustment. Training and evaluation sampling follow their respective recorded modes. We distinguish the unmodified official checkpoint, used in the system comparison, from B, our additionally fine-tuned matched baseline.

The official checkpoint was selected on the validation data that this report uses. Its authors tuned dropout, learning rate, weight decay, and batch size with Optuna, applied early stopping on validation per-class accuracy, and kept the model with the highest validation per-class accuracy [2, Section 3.3]; the released code takes its validation set from the official ASL Citizen validation file [3]. For the same 773k-parameter model, the paper reports 73.54% per-instance and 73.52% per-class Top-1 on the test split [2, Table 6], whereas we measure 80.94% on validation. The PoET validation scores in this report are therefore optimistic. The four-system validation comparison also favors PoET, because the classical configurations were not selected on validation, apart from the post hoc handedness fix. Our own test measurement of the same weights is 73.56% (Section 3.5).

Relative geometry addresses information that can be weakened by normalizing articulators separately. From the shared image coordinates, we compute seven relations: each wrist to the nose, mouth center, and shoulder center, plus one wrist-to-wrist relation. For each relation, the target-minus-source displacement is divided by the video's median valid shoulder width. Its x displacement, y displacement, Euclidean length, and presence bit form four values, giving 28 per frame. Invalid relations are zeroed; a video without a valid scale receives zero geometry. These features do not encode physical contact or reliable 3D depth.

The geometry branch is a 28→32→128 MLP with ReLU and 5,152 parameters. Its last layer starts at zero. A simple description of its insertion is:

**modified frame embedding = original frame embedding + valid-frame mask × geometry MLP(28-D relations).**

The residual enters before the Transformer, allowing geometry to affect subsequent temporal processing. Geometry uses exactly the frame indices selected for the main sequence. This alignment prevents a hand position from one moment being attached to another moment. No target label or true class attribute is needed at inference.

For matched B/G comparisons, seeds 20260922, 20260923, and 20260924 each start from the same official checkpoint. Every run uses all training videos for 10 epochs, batch size 128, and 3,140 optimizer steps. AdamW uses weight decay 0.01 and a fixed learning rate of 0.00003 for the original network; the new geometry branch uses 0.0003. All original layers remain trainable. Runs use FP32 and the same recorded temporal-sampling and class-weighted cross-entropy rules, with class weight w_c = 1 - n_c / N (n_c is the training count for class c, N the total), without added spatial augmentation. The best checkpoint is selected by complete-validation macro Top-1, including epoch zero and retaining the earlier checkpoint on ties. Because epoch zero is the official checkpoint and every epoch is scored on the same clips, this rule takes the best of eleven noisy validation measurements; Section 3.3 estimates the resulting optimism. The first seed informed earlier development; the later seeds therefore provide additional repeatability evidence, not a new held-out dataset.

### 2.4 Fixed retrieval, mirror test-time augmentation, and the demonstration

R is a postprocessing rule that uses each checkpoint's own 128-dimensional CLS embeddings. It is distinct from classical kNN on hand-engineered descriptors. Each B or G checkpoint encodes all training examples to create its own labeled bank. For an L2-normalized query q and reference z, the kernel is **K(q,z) = exp[−10 × (1 − q·z)]**. Within a candidate class, kernel values are first averaged within each training signer and then averaged across that class's signers. This limits domination by a signer with more references.

Only the model's original five candidate classes are reranked. The five logits are converted to a conditional softmax distribution, and the five retrieval scores are normalized across those candidates. The final score is **0.75 × model probability + 0.25 × retrieval probability**. This fixed rule changes ordering without adding candidates, so Top-5 membership and hit rate remain identical to the corresponding B or G model. The query's true label is never used. The factorial comparison reuses six existing checkpoints and performs no further parameter updates.

The kernel scale and blend weights were fixed in a preregistration written before any retrieval score was computed, and they were not tuned afterwards. The first retrieval run, a pilot on the official checkpoint, had a preregistered continuation gate: at least 52 additional correct clips (0.5 points) and higher macro accuracy. It gained 48 clips, so the gate was not met. The 2×2 study in Section 3.3 was a separately planned follow-up that reused the same fixed rule.

Mirror test-time augmentation (TTA) is a second training-free rule, applied only to the official checkpoint. The model is run on the original clip and on its horizontal mirror, with landmarks re-extracted from the flipped video, and the two logit vectors are averaged. It uses no label. On validation it raised Top-1 from 80.94% to 82.25%, and it was one of the systems fixed before the test evaluation. It was not combined with B, G, or R.

The accompanying browser demonstration accepts a presegmented video, extracts landmarks once, and runs kNN, SVM, forest, official PoET, and one fixed G+R checkpoint (seed 20260923; Section 3.4 explains this choice). Its classical models use the original descriptor and settings; the canonical and tuned classical models and mirror TTA have not been deployed in the demo. Saved validation examples are separated from fresh inference, and uploaded clips without labels do not receive an accuracy claim. Displayed scores have method-specific meanings: votes, margins, tree estimates, and softmax-derived values are not interchangeable calibrated confidence measures. The demo is an offline clip recognizer; no end-to-end streaming performance claim is made.

### 2.5 Classical follow-ups and the final test protocol

Two further experiments revisit the classical baselines. *Bounded tuning.* A preregistered 20-point grid over the three methods was evaluated on validation with the canonical descriptor, and one setting per method was selected by validation macro Top-1: kNN with k = 10 and similarity-weighted votes, SVM with C = 0.03, and a 64-tree forest with at most 4,096 leaves and no depth limit. The same settings were also run on the original descriptor. *Representation swap.* The three classifiers, with their original settings, are also trained on PoET's 128-dimensional CLS embedding, the input of PoET's own linear classification head. Holding the classifier fixed while changing the input separates the effect of the representation from that of the classifier. On validation this comparison favors the CLS embedding, because the encoder was trained on the training clips and its checkpoint was selected on the validation split.

For the final evaluation, a protocol written before any test video was downloaded fixed 29 systems and their model files: the three classical methods in four variants (original or canonical descriptor, original or tuned settings), the three classifiers on the CLS embedding, the official checkpoint with and without mirror TTA, and B, B+R, G, and G+R for each of the three seeds. Each system was evaluated once on all 32,941 test clips, with zero processing failures. All 29 are reported; nothing was selected, tuned, or re-evaluated on test.

## 3. Results

### 3.1 Complete validation comparison

Table 2 uses all 10,304 validation videos, with all classes represented and zero processing failures. Each classical method appears twice: with the original descriptor, and with the handedness-canonical descriptor of Section 3.2. With either descriptor, SVM is the strongest fixed-descriptor method, at 38.76% (original) or 43.87% (canonical) Top-1. The official PoET checkpoint correctly recognizes 8,340 videos, versus 3,994 and 4,520 for the two SVM variants. Macro and instance rankings agree. The large PoET gap is a system-level observation: learned temporal features, landmark selection, preprocessing, training history, and development effort differ alongside the classifier. PoET's validation score is also optimistic: the same weights reach 73.56% on test (Section 3.5), 7.38 points lower.

**Table 2. Four systems on the same complete validation cohort. Classical rows show the original descriptor and the post hoc handedness-canonical descriptor (Section 3.2). All accuracy entries are percentages.**

| System | Input | Top-1 | Top-5 | Macro Top-1 | Top-1 correct |
| --- | --- | ---: | ---: | ---: | ---: |
| Cosine kNN | Original descriptor | 18.6821 | 36.1025* | 19.1649 | 1,925 |
| | Canonical descriptor | 20.2252 | 38.4026* | 20.6033 | 2,084 |
| Linear SVM | Original descriptor | 38.7616 | 63.1308 | 39.5262 | 3,994 |
| | Canonical descriptor | 43.8665 | 70.4290 | 44.4713 | 4,520 |
| Bounded random forest | Original descriptor | 16.3141 | 38.2182 | 16.7851 | 1,681 |
| | Canonical descriptor | 17.9445 | 42.2360 | 18.2119 | 1,849 |
| Official PoET checkpoint† | Pose sequence | 80.9394 | 96.3800 | 81.1950 | 8,340 |

*kNN returns only classes that received votes (fewer than five labels for 3,203 queries with the original descriptor), so its Top-5 is not comparable with the other Top-5 values. †Selected on this validation split by its authors; the paper reports 73.54% per-instance Top-1 on the test split for the same weights [2].

![Figure 1. Four complete systems on all 10,304 validation clips; each classical method is shown with the original and the post hoc handedness-canonical descriptor. Official PoET was selected on this validation split by its authors (published test Top-1: 73.54%). Differences in representation and training history prevent attributing the gap solely to the classifier.](figures/fig01_method_comparison.png)

### 3.2 Handedness: a diagnosed and fixed classical failure mode

The per-signer validation results of the first comparison showed that all three classical methods nearly failed on two of the six validation signers. Linear SVM recognized 15.06% of clips from P12 and 11.66% from P5, against 39.98–45.28% for the three large right-handed signers, while official PoET still reached 79.63% and 69.95% on those two signers. Two label-free indicators showed that P12 and P5 are the only validation signers whose left hand, in most of their clips, is both detected more often and moves more than the right hand, and the descriptor has no handedness normalization. Before computing any mirrored result, we recorded the canonicalization rule of Section 2.2, two hypotheses, and a verdict rule. H1 states that each method improves on the mirrored signers; H2 states that no method loses overall accuracy. The verdict is "supported" if H1 holds for all three methods and SVM improves overall. The budget was one feature build, one query-only evaluation, and one refit, with no hyperparameter search.

**Table 3. Handedness canonicalization on all 10,304 validation clips (Top-1, %). Mirrored signers are P12 and P5 (1,306 clips); the other 8,998 clips keep their original inputs.**

| Method | Variant | Overall | Mirrored signers | Other signers | Net vs original (fixes / regressions) |
| --- | --- | ---: | ---: | ---: | ---: |
| kNN | Original | 18.6821 | 8.9587 | 20.0934 | — |
| | Query-only mirroring | 19.6914 | 16.9219 | 20.0934 | +104 (166 / 62) |
| | Canonical refit | 20.2252 | 18.0704 | 20.5379 | +159 (346 / 187) |
| Linear SVM | Original | 38.7616 | 13.4763 | 42.4317 | — |
| | Query-only mirroring | 42.0516 | 39.4334 | 42.4317 | +339 (377 / 38) |
| | Canonical refit | 43.8665 | 40.8882 | 44.2987 | +526 (815 / 289) |
| Bounded random forest | Original | 16.3141 | 5.1302 | 17.9373 | — |
| | Query-only mirroring | 17.8960 | 17.6110 | 17.9373 | +163 (187 / 24) |
| | Canonical refit | 17.9445 | 18.3002 | 17.8929 | +168 (908 / 740) |

The preregistered verdict was "supported": H1 and H2 held for all three methods, and SVM improved overall. Query-only mirroring changes no model and leaves all 8,998 other predictions identical. Yet it raises SVM accuracy on the two mirrored signers from 13.48% to 39.43%, close to the 42.43% of the other signers. Most of the collapse was therefore a left-right mismatch rather than poor video quality or harder vocabulary. Canonical refitting also helps right-handed signers, because four left-dominant training signers now contribute consistently oriented examples. SVM rises by 1.87 points on the other signers (421 fixes, 253 regressions) and kNN by 0.44 points, while the bounded forest is essentially unchanged there (701 fixes, 705 regressions). Per signer, SVM accuracy rises from 15.06% to 47.06% on P12 and from 11.66% to 33.83% on P5. P5 remains the weakest signer; its left-coverage indicator is only 0.57, and we have not examined its clips individually. Every comparison in Table 3 has exact McNemar p < 1e-4, but most of the gain comes from two signers, so the per-signer pattern is the more informative summary.

### 3.3 Matched geometry and retrieval ablation

Table 4 reports the 2×2 ablation separately from Table 2. B is fine-tuned PoET, not the original official weights. G+R improves mean Top-1 by 0.5499 percentage points over B, 0.2620 over G, and 0.2135 over B+R. It exceeds each single modification in both instance and macro accuracy for every observed seed. The paired G+R minus B differences are +0.4367, +0.6793, and +0.5338 points, corresponding to 45, 70, and 55 additional correct clips.

**Table 4. Three matched seeds; mean ± sample standard deviation, in percentage units. The bracketed range is a 95% t-interval over the three seed-level differences. McNemar p-values compare each variant with B on the same clips for seeds 20260922, 20260923, and 20260924. The same validation videos are reused across seeds.**

| Method | Top-1 | Macro Top-1 | Mean Top-1 difference from B [95% interval] | Exact McNemar p vs B, by seed |
| --- | ---: | ---: | ---: | ---: |
| B: fine-tuned PoET | 81.1853 ± 0.0341 | 81.4711 ± 0.0578 | — | — |
| B+R: retrieval | 81.5217 ± 0.0700 | 81.7976 ± 0.1061 | +0.3364 [+0.24, +0.43] | 3.8e-5; 6.4e-4; 1.1e-4 |
| G: geometry | 81.4732 ± 0.1009 | 81.7451 ± 0.1112 | +0.2879 [−0.04, +0.62] | 0.46; 6.2e-4; 2.0e-3 |
| G+R: geometry and retrieval | 81.7352 ± 0.0889 | 81.9937 ± 0.0791 | +0.5499 [+0.25, +0.85] | 0.014; 1.2e-8; 7.7e-6 |

The improvement contains both fixes and regressions. For seed 20260923, G+R corrects 111 B errors but changes 41 previously correct outputs into errors, producing the net gain of 70. Against G alone, G+R has exact McNemar p = 1.9e-4, 1.2e-4, and 0.025 across the three seeds. The mean interaction, (G+R−G)−(B+R−B), is −0.0744 points: the gains partially overlap rather than combine super-additively. Separately, disabling the geometry residual in one fixed trained checkpoint reduced Top-1 from 81.3568% to 80.7842%. This shows reliance on that input but is not a substitute for the matched B comparison, because disabling a learned branch changes its operating conditions.

**Selection effects.** Each B and G entry uses the epoch with the highest validation macro Top-1, and the same clips then score that epoch. The 20-epoch extension runs (Section 4.2) recorded validation accuracy after every epoch for the same seeds, which gives a selection-free view. Averaged over epochs 1–20 and the three seeds, B recognizes 8,340.1 of 10,304 clips, the same as the official checkpoint (8,340); within a seed, accuracy varies by about 15 clips (standard deviation) from epoch to epoch. The 0.25-point advantage of B over the official checkpoint therefore reflects best-epoch selection, not fine-tuning. Geometry holds up better. At the same seed and epoch, G exceeds B in 58 of 60 comparisons (one tie, one loss), by 19.9 clips (0.19 points) on average, compared with 0.29 points after best-epoch selection. Retrieval needs no epoch selection. Applied to the official checkpoint with its fixed rule, it added 48 clips (+0.47 points; 67 fixes, 19 regressions; exact McNemar p = 1.9e-7), although this missed the pilot's preregistered gate of 52 clips.

Read together, retrieval is the most consistent component: it improves the unselected official checkpoint and every fine-tuned checkpoint, and its seed interval excludes zero. Geometry gives a small but consistent gain whose seed interval includes zero and whose selection-free size is about 0.19 points. The combination adds about 0.5 points over B. These statistics remain descriptive (Section 2.1): the clips come from six signers, and the validation split selected every checkpoint.

![Figure 2. Matched PoET ablation across three seeds. Error bars show sample standard deviation, not a confidence interval; the restricted vertical axis makes small changes visible. B differs from the official checkpoint in Figure 1, and B and G use validation-selected epochs, so selection-free gains are smaller (Section 3.3). Mean G+R improvement is 0.550 percentage points on reused validation data.](figures/fig02_paired_ablation.png)

### 3.4 Error overlap and illustrative clips

Although weaker overall, SVM with the original descriptor correctly recognizes 357 clips missed by official PoET. Conversely, PoET recognizes 4,703 clips missed by SVM. Across the four systems, with classical predictions from the original descriptor, 461 clips are correct for every method and 1,428 are wrong for every method. Such overlap distinguishes shared hard cases from model-specific errors. The hypothetical SVM-or-PoET accuracy of 84.4041% requires an oracle that knows which prediction is correct; it is not a deployable fusion result. This overlap analysis uses the original-descriptor predictions saved for the demo and has not been repeated for the canonical descriptor.

![Figure 3. Correctness overlap with official PoET, using original-descriptor classical predictions. The classical-only segments are potential complementarity, not achieved ensemble gains. Counts for different classical systems overlap and cannot be added together. Each row covers all 10,304 validation clips.](figures/fig03_error_complementarity.png)

**Table 5. Examples from the complete validation records. O = official PoET. B and G+R use seed 20260923, the seed with the largest G+R gain (+70 clips, versus +45 and +55), so its examples show the most favorable seed rather than a typical one. Index is the project validation index. Classical predictions use the original descriptor.**

| Index / target | Observed predictions | Interpretation supported by the record |
| --- | --- | --- |
| 11 / UNDERSTAND | All four systems: UNDERSTAND | Shared success; visible blur did not prevent recognition |
| 17 / FINS | SVM: FINS; O: SWEARIN | Classical rescue; FINS is O's second candidate |
| 10 / SPECIAL1 | O: CHIMNEY; all four wrong | Shared failure; target absent from O's Top-5 |
| 2312 / INTUITIVE | O: ACCESS; all four wrong | Cache records zero hand detections throughout |
| 59 / KICK2 | B: BRIDGE1; G+R: KICK2 | A correction relative to the matched baseline |
| 0 / SUPPORT1 | B: SUPPORT1; G+R: CABINET | A regression relative to the matched baseline |

The gallery contains 16 deterministically selected examples and 48 frames sampled at 25%, 50%, and 75% of each clip. Selection covers shared successes, shared failures, classical rescues, PoET-only successes, and absent hand detections. These are teaching examples, not a random accuracy sample. Frame review records visible positions, blur, and framing separately from hypotheses; still images alone do not establish ASL semantics or why a prediction failed. The seed for the B and G+R examples and the demo was chosen after the factorial results were known.

Performance also varies across observational slices. Excluding the signer with only two clips, official PoET signer-specific Top-1 ranges from 69.95% to 85.67%; the lowest value belongs to P5, one of the two left-dominant validation signers, but flipping P5's videos leaves official PoET at 71.10% (Section 4.1). For mean hand-detection fractions below 25% but above zero, PoET achieves 78.10%; at 50% or above it achieves 84.64%. This fraction is the sum of left- and right-hand detection counts divided by twice the decoded frame count, not the proportion with either hand visible. One-handed signs, idle frames, signer identity, and class mix confound these associations. All seven clips with no hand detections are misclassified, but correcting all seven could improve total accuracy by at most 0.0679 points.

*Figure 4. Real validation frames at approximately 25%, 50% and 75% of duration. UNDERSTAND is a shared success; PARADE is a shared failure. In the matched seed-20260923 comparison (the seed with the largest G+R gain), G+R fixes KICK2 but regresses on SUPPORT1. Panels a-b use official PoET; panels c-d use matched fine-tuned B and G+R. Labels are provided by the dataset. These deliberately selected examples do not establish a causal explanation or an accuracy estimate.*

Original image panels are not included in this code repository.

### 3.5 One-shot final test (October 12, 2026)

Table 6 gives the test results for the four original systems, mirror TTA, and the matched PoET ablation. Each test score comes from a single evaluation of a system fixed in advance; validation scores are repeated for comparison. The official checkpoint reaches 73.56% test Top-1, within 0.02 points of the 73.54% reported for PoET [2]. This agreement checks our data and pose pipeline on the test signers.

**Table 6. Validation and one-shot test results, in percent. B to G+R are means over the three matched seeds; "—" means not reported. *kNN returns fewer than five labels for many queries, so its Top-5 is not comparable with the others.**

| System | Validation Top-1 | Test Top-1 | Test Top-5 | Test Macro Top-1 |
| --- | ---: | ---: | ---: | ---: |
| Cosine kNN, k = 5 | 18.68 | 15.52 | 30.67* | 15.61 |
| Linear SVM, C = 0.01 | 38.76 | 30.93 | 52.23 | 31.06 |
| Bounded random forest | 16.31 | 12.54 | 30.10 | 12.56 |
| Official PoET checkpoint | 80.94 | 73.56 | 92.60 | 73.54 |
| Official PoET + mirror TTA | 82.25 | 74.72 | 93.34 | 74.72 |
| B: fine-tuned PoET | 81.19 | 73.45 | 92.47 | — |
| B+R | 81.52 | 73.87 | 92.47 | — |
| G | 81.47 | 73.76 | 92.61 | — |
| G+R | 81.74 | 74.18 | 92.61 | — |

Every system scores lower on test than on validation. The PoET systems and the SVM lose 7.4–7.8 points; kNN and the bounded forest, starting lower, lose 3.2–3.8 points. Part of the drop is expected, because validation was reused for development and selection, and the test signers are different people. The ranking of the four original systems is unchanged. Per-signer accuracy varies widely on test: official PoET ranges from 59.07% to 85.81% across the 11 signers.

The matched gain holds on test. G+R improves on B by 0.74 points (three-seed t interval [+0.54, +0.93]; bootstrap over the 11 test signers [+0.58, +0.89]; exact McNemar p below 1e-13 for every seed), and the G, R, and G+R gains over B are positive for all three seeds (means +0.31, +0.42, and +0.74). As on validation, R leaves Top-5 unchanged. B itself is about 0.1 points below the official checkpoint on test, although it was about 0.25 points above it on validation. Fine-tuning alone, with checkpoints selected on validation, therefore gives no test gain over the official weights.

Mirror TTA adds 1.17 points to the official checkpoint (968 fixes, 583 regressions; signer bootstrap [+0.82, +1.53]) and helps all 11 test signers. Official PoET with TTA has the highest test Top-1 in Table 6 (74.72%), above G+R (74.18%). The two start from different baselines, the official weights versus fine-tuned B, and TTA was not combined with G+R, so this does not show that one modification is better than the other. Because the test split has now been used, a combined system could only be developed and reported on validation.

### 3.6 Classical follow-ups on validation and test

Table 7 reports the classical variants of Sections 2.2 and 2.5. The handedness rule mirrored 2 of the 6 validation signers and 3 of the 11 test signers.

**Table 7. Classical follow-ups, Top-1 in percent (kNN / SVM / RF). "Canonical" is the handedness-canonical descriptor; the tuned settings were selected on validation.**

| System | Validation | Test |
| --- | ---: | ---: |
| Original descriptor and settings | 18.68 / 38.76 / 16.31 | 15.52 / 30.93 / 12.54 |
| Original settings, canonical descriptor | 20.23 / 43.87 / 17.94 | 16.15 / 35.76 / 14.84 |
| Tuned settings, canonical descriptor | 20.53 / 52.47 / 35.55 | 16.34 / 42.52 / 28.22 |
| Tuned settings, original descriptor | 18.97 / 47.03 / 32.39 | 15.52 / 37.32 / 24.84 |
| Original settings on PoET's 128-D CLS embedding | 79.16 / 69.91 / 25.84 | 71.94 / 64.11 / 20.93 |

Canonicalization helps the SVM most. On test it adds 4.84 points: 13.8 points on the three mirrored signers and only 1.5 points on the other eight. Tuning C, which was best at 0.03 in a validation grid from 0.001 to 0.1, adds a further 6.76 points, to 42.52%. Both steps are significant on test (McNemar p < 1e-130). In the validation grid at C = 0.1, 280 of the 2,731 binary problems had not converged after 2,000 iterations.

The forest gains 13.4 test points from the larger leaf budget (McNemar p < 1e-300). In the original forest, every one of the 64 trees used exactly 256 leaves and reached depth 12. With 4,096 leaves every tree again fills its budget, so this is not the forest's ceiling. The tuned forest is costly: its fit was admitted only with 37 GB of free memory (peak use was not measured), and the model is 11.5 GB on disk, because every node stores a distribution over all 2,731 classes.

kNN barely changes. All ten validation settings lie between 19.7% and 20.5%, and the selected setting is not significantly better (McNemar p = 0.25 on validation and 0.16 on test).

The best classical system, the tuned SVM on the canonical descriptor, remains 31 points below the official checkpoint on test. With each classifier and its original settings held fixed, replacing the canonical hand-crafted descriptor by PoET's CLS embedding closes about 75% of the SVM's gap to the official checkpoint and about 97% of kNN's. kNN on the CLS embedding reaches 71.94%, only 1.6 points below PoET's own linear head. The original forest gains much less from the better input (20.93%), consistent with its small leaf budget.

## 4. Discussion

### 4.1 What the comparisons establish

Under a shared representation, the SVM result suggests that learned discriminative boundaries use the fixed summaries more effectively than our cosine distance or bounded forest. It does not prove that linear SVM is intrinsically superior on every pose representation. The forest reached its leaf budget, and coarse temporal pooling can merge distinct trajectories. Feature audits found finite values, aligned labels, correct training-only standardization, and complete rows. Thus, low scores cannot simply be dismissed as missing input files. The audits did not catch a design omission, however: the descriptor had no handedness normalization.

The handedness result is a diagnosed and fixed failure mode. A per-signer breakdown exposed it, a label-free mechanism explained it, and a preregistered test confirmed the mechanism: mirroring only the queries of two left-dominant signers restored most of their accuracy without changing any model. Four points bound the result. First, the hypothesis came post hoc from the same validation table, so the validation gain confirms the mechanism rather than providing an independent estimate of its size. Second, the rule uses signer identity and each signer's unlabeled clips. The same frozen rule was applied to the test signers, where it added 4.84 points for the SVM, 13.8 points on the three mirrored signers (Section 3.6); this is the independent estimate. Third, the PoET gap remains large (43.87% against 80.94% validation Top-1), so the fix corrects an underestimate of the classical baselines rather than changing the main comparison. Fourth, PoET was not modified. A separate preregistered, inference-only check re-extracted landmarks from horizontally flipped videos and found the official checkpoint approximately invariant: Top-1 on the two left-dominant signers moved from 75.11% to 75.80% (61 fixes, 52 regressions; exact McNemar p = 0.45), and on 1,500 right-handed control clips from 83.20% to 83.53%. The paper reports random horizontal flips during training [2]; our B and G fine-tuning did not use them and was not checked. The handedness defect is therefore specific to the fixed descriptor, and P5's lower PoET accuracy (71.10% after flipping) is not explained by handedness.

The tuning and representation follow-ups sharpen this picture. Part of the classical gap to PoET came from avoidable choices: fixed left/right slots in the descriptor, an over-regularized SVM, and a binding leaf budget. Correcting them raises the best classical test score from 30.93% to 42.52%. Most of the remaining gap is due to the representation. With the classifier fixed, PoET's learned embedding does most of the work, and a nearest-neighbor vote on it comes within 1.6 points of PoET's own classifier. The SVM's lead over kNN on the hand-crafted descriptor therefore says more about how each classifier copes with a weak representation than about the classifiers in general.

PoET provides a much stronger reference system. The matched ablation supports a narrower contribution. On validation, retrieval was the most robust component, geometry gave about 0.2 points once epoch selection was removed, and the fine-tuned baseline B was not better than the official checkpoint on average. The reserved test split agrees with each of these readings: B is about 0.1 points below the official weights, B+R and G exceed B by 0.42 and 0.31 points, and G+R exceeds B by 0.74 points, positive for every seed, with a signer-level bootstrap interval above zero. The absence of an equal-parameter branch using only the original inputs still limits claims that the geometry benefit comes uniquely from new geometry rather than extra capacity. The test intervals describe these three seeds and 11 test signers, not every future signer or dataset.

Two further test observations temper this result. Fine-tuning alone (B) did not improve on the official weights, so the matched gain is measured against a baseline that is itself no better than the released model. A training-free mirror average also added more to the official checkpoint (+1.17 points) than geometry and retrieval added to B (+0.74). The simplest change was the most effective on test, although the two were never combined or compared from the same starting point.

### 4.2 Relation to prior work and unsuccessful directions

Our work follows ASL Citizen's isolated dictionary-retrieval setting [1] and adapts the existing PoET recognizer [2,3]. On the official test split, our official-checkpoint Top-1 of 73.56% matches the 73.54% reported for PoET [2] to within 0.02 points. The dataset paper reports roughly 63% recognition accuracy and 91% recall at ten on the same held-out test signers [1]. The PoET-based systems here exceed that figure largely because of PoET itself, not because of our modifications. Nor does reusing an official checkpoint reproduce the paper's entire training search, all datasets, or its online user study. The present contribution is a documented comparison and a small adaptation of an established system, not a new state-of-the-art claim.

More elaborate additions were not automatically better. Attribute auxiliary supervision yielded only four additional correct clips initially and two after changing the new-head learning rate, relative to its matched baseline. A cross-signer supervised-contrastive experiment obtained 81.0850% against its matched classification control at 81.1141%. Extending matched B/G training to 20 epochs reduced the mean geometry advantage from 0.2879 to 0.2588 points. A later hand-RGB temporal fusion reached 81.8517% for one seed, only two clips above same-seed G+R at 81.8323%, while lowering the underlying G Top-5 from 96.5450% to 96.1277%. These results motivate retaining the simpler geometry/retrieval candidate rather than presenting complexity as evidence of progress. None of these exploratory variants was evaluated on test.

### 4.3 Limitations and further work

Repeated development on validation was the main risk. Checkpoint selection, design revisions, and exploratory alternatives can favor the validation cohort, the first seed also informed tuning, the authors of PoET selected the official checkpoint on this split, and the handedness hypothesis came from its per-signer table. The one-shot test evaluation shows the size of this effect: the PoET systems and the SVM lose 7–8 points from validation to test, while the matched G+R gain persists. The test split has now been used once and will not be used again. Any further change, such as combining mirror TTA with G+R, can only be developed and reported on validation, and must be labeled as a validation result. Three seeds and 11 test signers remain a small basis for uncertainty estimates.

The classical follow-ups add two limitations. Handedness canonicalization decides per signer from that signer's unlabeled clips, so it assumes a known signer with several clips. It does not leak labels, but it is a signer-level adaptation, and its use on a single uploaded clip was not evaluated. The tuned settings were selected on validation; the single test evaluation confirms their gains, but it does not show that the selected values are optimal. The tuned forest's memory needs also make it impractical without a large machine.

Generalization beyond ASL Citizen is also unmeasured. The vocabulary is closed, word boundaries are supplied, and arbitrary uploads may differ in framing, signing style, or label coverage. The current system always returns candidates rather than a validated unknown-sign decision. A useful future study would evaluate calibrated rejection and new signers under an agreed protocol. More targeted engineering questions include less conservative body masks and robust motion summaries, but they require new controlled comparisons. We have not completed a cost benchmark for every method under common hardware and measurement conditions, so isolated timing observations are not used to rank speed. Finally, mutual error rescue suggests potential complementarity; learning a selector would require separate development data and cannot be justified by choosing the correct model retrospectively.

## 5. Conclusion

This project produces a reproducible comparison of four isolated ASL recognition systems and a controlled adaptation of PoET. On the complete validation split, the official PoET checkpoint reaches 80.94% Top-1 and the best original fixed-descriptor baseline, the linear SVM, reaches 38.76%; in the one-shot test evaluation they reach 73.56% and 30.93%. A label-free handedness canonicalization and bounded tuning raise the best classical test score to 42.52%, still 31 points below PoET. Swapping only the input to PoET's embedding closes most of the gap for kNN and the SVM, so the representation, not the classifier, is the main difference between the classical systems and PoET. Relative geometry and fixed retrieval raise matched fine-tuned PoET from 81.19% to 81.74% on validation and from 73.45% to 74.18% on test, averaged over three seeds. The gain is small, positive for every seed, and includes regressions. A training-free mirror average of the official checkpoint gives the best test Top-1, 74.72%. Complete denominators, negative experiments, error overlap, and a working video interface provide evidence beyond a single favorable accuracy number.

**Contributions.** The team jointly discussed and defined the project scope and the overall approach. Each member is responsible for one method: Xinjie Huang for PoET, including the geometry, retrieval, and test-time-augmentation experiments, the shared data and evaluation pipeline, and the demonstration; Leyun Kim for kNN; and Duncan Stadler and Ming Chen for the linear SVM and the random forest [TO CONFIRM: which of the two takes which method].

## References

[1] A. Desai, L. Berger, F. O. Minakov, V. Milan, C. Singh, K. Pumphrey, R. E. Ladner, H. Daumé III, A. X. Lu, N. Caselli, and D. Bragg. **ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition.** Advances in Neural Information Processing Systems 36, Datasets and Benchmarks, 2023. [Paper](https://papers.nips.cc/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf).

[2] J. Lascar, J. Françoise, M. Gouiffès, A. Braffort, and D. Fabre. **PoET: Lightweight Pose Encoder Transformer for Online Sign Language Recognition.** Proceedings of VISAPP 2026, vol. 1, pp. 19–28, 2026. [DOI: 10.5220/0014237500004084](https://doi.org/10.5220/0014237500004084).

[3] J. Lascar et al. **PoET: Official Implementation and Pretrained Weights.** Project repository, revision dca74c30d9d604cb3b92de8fc73ce91484644259, 2026. [Repository](https://gitlab.lisn.upsaclay.fr/mtals/publications/poet).

[4] I. Grishchenko and V. Bazarevsky. **MediaPipe Holistic—Simultaneous Face, Hand and Pose Prediction, on Device.** Google Research, December 10, 2020. [Technical description](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/).

[5] T. M. Cover and P. E. Hart. **Nearest Neighbor Pattern Classification.** IEEE Transactions on Information Theory, 13(1), pp. 21–27, 1967. [DOI: 10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964).

[6] C. Cortes and V. Vapnik. **Support-Vector Networks.** Machine Learning, 20, pp. 273–297, 1995. [DOI: 10.1007/BF00994018](https://doi.org/10.1007/BF00994018).

[7] L. Breiman. **Random Forests.** Machine Learning, 45, pp. 5–32, 2001. [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

[8] F. Pedregosa et al. **Scikit-learn: Machine Learning in Python.** Journal of Machine Learning Research, 12, pp. 2825–2830, 2011. [Paper](https://www.jmlr.org/papers/v12/pedregosa11a.html).
