# Report → code map

Where to find the code behind each part of the final report (November 15, 2026 version) and the slides. `scripts/…` is the top-level folder; `research/…` holds verbatim copies of the remaining research code (see [research/README.md](../research/README.md)). None of it runs without the full data, weights and run records, which are not in Git.

| Report | What | Code |
| --- | --- | --- |
| 2.1, Table 1 | Pose extraction for train/validation | `scripts/prepare_poses.py`, `scripts/evaluate_pose_cache.py`; full download `research/scripts/fetch_asl_dataset.py` |
| 2.2, 3.1, Table 2 (original rows), Figure 1 | 1,020-D descriptor; kNN, Linear SVM, bounded forest | `scripts/classical_features.py`, `scripts/classical_models.py`, `scripts/prepare_classical_dataset.py`, `scripts/run_classical_baselines.py`, `scripts/verify_classical_results.py` |
| 2.2, 3.2, Table 2 (canonical rows), Table 3 | Handedness canonicalization | `scripts/classical_handedness.py`; `research/scripts/prepare_handedness_dataset.py`, `evaluate_handedness_query.py`, `analyze_handedness.py` |
| 2.3 | Official PoET checkpoint and preprocessing | `scripts/fetch_vendor.py`, `scripts/_poet_runtime.py`, `scripts/infer_video.py` |
| 2.3, 3.3, Table 4, Figure 2 | Geometry branch; matched B/G fine-tuning | `scripts/poet_geometry.py`, `scripts/poet_methods.py`, `scripts/prepare_geometry_cache.py`, `scripts/finetune_methods.py`, `scripts/evaluate_methods.py`, `scripts/method_optimizer.py`, `scripts/method_runtime.py` |
| 2.4, 3.3, Table 4 | Fixed retrieval rerank (pilot and 2×2) | `scripts/extract_retrieval_features.py`, `scripts/rerank_retrieval_pilot.py`, `scripts/extract_factorial_retrieval_features.py`, `scripts/rerank_factorial_retrieval.py`; analysis `research/docs/tools/analyze_retrieval_pilot.py`, `analyze_factorial_retrieval.py`, `analyze_paired_seeds.py` |
| 3.3 | Seed intervals, McNemar tests, selection effects | `research/docs/tools/reanalyze_submission_20261010.py`, `research/docs/tools/analyze_extended_training.py` |
| 3.3 | Disabling the geometry residual (81.36% → 80.78%) | `research/scripts/evaluate_geometry_ablation.py`, `research/docs/tools/analyze_geometry_ablation.py` |
| 2.4 | Mirror TTA on validation | `research/scripts/prepare_poet_mirror_cache.py`, `prepare_poet_mirror_fullval.py`, `evaluate_poet_mirror.py`, `evaluate_poet_mirror_fullval.py`, `analyze_poet_mirror_tta_fullval.py` |
| 4.1 | PoET flip check on left-dominant signers | `research/scripts/prepare_poet_mirror_cache.py`, `evaluate_poet_mirror.py`, `analyze_poet_handedness.py` |
| 2.5, 3.6, Table 7 | Bounded tuning | `research/scripts/run_classical_tuning.py` |
| 2.5, 3.6, Table 7 | Classifiers on PoET's CLS embedding | `research/scripts/prepare_cls_representation_dataset.py`, `replay_cls_official_head.py`, `analyze_cls_representation.py` |
| 2.5, 3.5, Tables 6–7, slides 4–6 | One-shot final test (29 systems) | `research/scripts/fetch_asl_test.py`, `prepare_test_poses.py`, `final_test_poet.py`, `final_test_retrieval.py`, `final_test_classical.py`, `analyze_final_test.py` |
| 3.4, Table 5, Figure 3, Figure 4 cases, slide 7 | Error overlap and selected cases | `research/scripts/build_comparison_gallery.py` |
| 2.4, slide 9 | Video demo | `research/scripts/serve_comparison_demo.py`, `research/scripts/infer_comparison.py`, `research/docs/comparison-demo/`; presentation video `demo/make_demo_video.py` |
| Figures 1–3 | Report figures | `research/docs/submission-20261004/figures/build_figures.py` |
| 4.2, slide 8 | Attribute auxiliary loss | `scripts/build_asllex_targets.py`, `scripts/poet_methods.py`, `scripts/finetune_methods.py` |
| 4.2, slide 8 | Cross-signer contrastive loss | `research/scripts/finetune_contrastive.py`, `contrastive_components.py`, `evaluate_contrastive.py`, `research/docs/tools/analyze_contrastive_trial.py` |
| 4.2 | 20-epoch extension | `scripts/finetune_methods.py` (resume from epoch 10), `research/docs/tools/analyze_extended_training.py` |
| 4.2, slide 8 | Hand-RGB temporal branch | Not included; code stays in the research workspace |

The members' from-scratch versions in `students/` are teaching re-implementations. The reported classical numbers come from `scripts/classical_models.py`.
