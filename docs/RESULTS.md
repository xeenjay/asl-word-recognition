# Results and how to read them

These are existing **development-validation results**, exported on October 4, 2026. No training, inference, or test evaluation was performed to create this repository summary. Final test results are not available.

The task is isolated ASL word recognition: one presegmented video is assigned one of 2,731 labels. The official signer-disjoint split contains 40,154 training clips, 10,304 validation clips, and 32,941 reserved test clips. All full-vocabulary results below retain the complete validation denominator, including clips with missing hand detections. Processing failures were zero.

## Four-system comparison

| System | Top-1 | Top-5 | Macro Top-1 | Top-1 correct |
|---|---:|---:|---:|---:|
| Cosine kNN, k=5 | 18.6821% | 36.1025% | 19.1649% | 1,925 / 10,304 |
| Linear SVM | 38.7616% | 63.1308% | 39.5262% | 3,994 / 10,304 |
| Restricted Random Forest | 16.3141% | 38.2182% | 16.7851% | 1,681 / 10,304 |
| Official pretrained PoET | 80.9394% | 96.3800% | 81.1950% | 8,340 / 10,304 |

The classical methods share a 1,020-dimensional pose/motion descriptor and training-only standardization. PoET uses a different sequential representation and an official pretrained checkpoint. This is a **system comparison**: representation, pretraining, training procedure, and development budget differ. The score gap cannot be attributed solely to the classifier. The bounded classical configurations are reference implementations, not upper bounds on these algorithms.

Top-1 and Top-5 weight videos equally. Macro Top-1 weights word classes equally. All 2,731 classes occur in validation. A kNN prediction can list fewer than five classes if fewer than five classes receive votes.

Source: [system_comparison.csv](../results/system_comparison.csv).

## Matched PoET comparison: the main improvement result

Here **B** means PoET fine-tuned within this project, **G** adds the relative-geometry branch, and **R** is fixed training-memory reranking. B is not the untouched official checkpoint in the previous table.

Each B/G model has the same 10-epoch training budget. The three paired seeds are 20260922, 20260923, and 20260924. Checkpoints are selected using full-validation macro Top-1, including epoch 0; ties keep the earlier checkpoint. Retrieval adds no optimizer updates.

| Method | Mean Top-1 | Sample SD across seeds | Mean Macro Top-1 | Mean Top-5 |
|---|---:|---:|---:|---:|
| B | 81.1853% | 0.0341 pp | 81.4711% | 96.4092% |
| B+R | 81.5217% | 0.0700 pp | 81.7976% | 96.4092% |
| G | 81.4732% | 0.1009 pp | 81.7451% | 96.5289% |
| G+R | **81.7352%** | 0.0889 pp | **81.9937%** | 96.5289% |

| Seed | B correct | B+R correct | G correct | G+R correct | G+R minus B |
|---|---:|---:|---:|---:|---:|
| 20260922 | 8,369 | 8,406 | 8,383 | 8,414 | +45 |
| 20260923 | 8,362 | 8,392 | 8,401 | 8,432 | +70 |
| 20260924 | 8,365 | 8,402 | 8,401 | 8,420 | +55 |

G+R improves mean Top-1 by **0.5499 percentage points** over B, 0.2620 over G, and 0.2135 over B+R. It exceeds both single additions in instance and macro Top-1 for all three seeds, satisfying the predefined combination-retention rule. The mean interaction, `(G+R−G)−(B+R−B)`, is **−0.0744 pp**: the benefits partly overlap, so this is not evidence of super-additivity.

R only reorders the original model's Top-5 candidate set; it cannot rescue a word outside that set and does not change Top-5 accuracy.

The improvement is small and remains a development result. Seed 20260922 participated in early method and learning-rate development. All three seeds use the same validation clips for model selection. Their SD describes seed variation; it is not a confidence interval, a significance test, or evidence from three independent datasets. Fixed-checkpoint geometry interventions support use of aligned geometry, but a separately trained equal-capacity old-input branch control has not established how much benefit comes from geometry information versus added capacity.

Sources: [per-seed results](../results/poet_factorial_per_seed.csv), [aggregate results](../results/poet_factorial_summary.csv).

## Explorations, including weak and negative results

These experiments have separate controls and budgets. They should not be merged into a single leaderboard or added to the main improvement.

| Experiment | Matched observation | Interpretation |
|---|---|---|
| ASL-LEX attribute auxiliary loss | Seed22 B: 81.2209%; attributes: 81.2403%, net +2 clips | Negligible classification gain in this configuration; auxiliary-loss improvement is not equivalent to word-recognition improvement. |
| Cross-signer supervised contrastive loss | Same-sampler CE: 81.1141%; SupCon: 81.0850%, net −3 clips | Essentially tied in this single-seed trial. Use its same-sampler CE control, not the main B average. |
| Ordinary RGB pilot, 256 classes | Pose probe: 96.2925%; fusion: 74.9743%, net −207 / 971 clips | Negative result. Its restricted vocabulary and 971-clip denominator are not comparable to the 2,731-class table. |
| Fixed retrieval on official PoET | 80.9394% → 81.4053%, net +48 clips | Positive but below the predefined continuation threshold of +52. The threshold was not lowered after observing results. |
| Continue B/G from epoch10 to total20 | B20: 81.2144%; G20: 81.4732%; paired advantage +0.2588 pp | Extra training did not improve the selected G checkpoints. The old G+R result is still a 10-epoch-model result. |
| Hand-domain RGB + GRU with G | Same pose seed23 G: 81.5314%; G+RGB GRU: 81.8517% | Only 2 additional correct clips over the same-seed G+R result of 81.8323%; G Top-5 decreases from 96.5450% to 96.1277%. |

The 20-epoch comparison selects the best checkpoint within the full budget, including inherited epochs 0–10. All three G models retain their epoch10 checkpoint; only B seed24 improves, by 9 correct clips. These are selected-best scores, not scores at the last epoch. Continued training was completed even though the best G checkpoint did not change.

The hand-RGB trial freezes the pose models and the external hand-domain DINO encoder. Its Pool and GRU heads each train for 30 epochs, with RGB seed20260927 and pose reference seed20260923. RGB-only Pool reaches 30.4639%; GRU reaches 43.5171%. The heads have different parameter counts, so that gap is not a pure temporal-modeling ablation. Fusion uses a fixed 0.75 pose / 0.25 RGB probability mixture, without retrieval or TTA. The single-seed fusion should be compared with the corresponding single-seed G+R result, not only the three-seed average. External hand-domain pretraining is additional information, and exact video-level non-overlap with ASL Citizen has not been independently established.

Sources: [exploratory_results.csv](../results/exploratory_results.csv), [extended_training.csv](../results/extended_training.csv).

## Error overlap

SVM correctly identifies 357 clips missed by official PoET. Conversely, PoET correctly identifies 4,703 clips missed by SVM. Their either-correct count is 8,697 / 10,304 = **84.4041%**. This requires an oracle to select the correct answer; it is **not achieved ensemble accuracy**.

kNN vote shares, SVM decision margins, RF class probabilities, and PoET softmax probabilities are not calibrated to a common scale. Selecting the highest score across models is not a validated fusion method.

Source: [error_overlap.csv](../results/error_overlap.csv), with six aggregate method-pair comparisons and no per-video or per-signer records.

## Evidence and remaining work

[summary.json](../results/summary.json) contains the numerical tables, selected settings, and a source registry. Each CSV's `source_id` resolves to an internal run ID, input artifact name, and SHA256. These hashes identify the exact source evidence used for the export; they do not make this small repository a complete raw experiment archive. Full checkpoints, videos, per-video predictions, and private infrastructure logs are excluded from these summaries.

The export checks that correct counts reproduce Top-1/Top-5 percentages, that each aggregate overlap table sums to its denominator, and that no private absolute paths or per-video identifiers were copied. It does not rerun models or repeat the earlier full evidence audits.

Final test evaluation, a common end-to-end cost benchmark, team review, and each member's actual contribution still need to be completed or recorded. Existing code and results prepared in advance do not establish that another team member has already performed that work. See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) before extending the comparison.
