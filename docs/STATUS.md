# Current status

[Home](../README.md) · [Setup](SETUP.md) · [Data provenance](DATA_PROVENANCE.md) · [Research results](RESULTS.md)

Snapshot: **October 4, 2026**, with an update on **October 15, 2026**: the student files are now exercises (Member 1 kNN, Member 2 Linear SVM and hand normalization, Member 3 random forest) with self-check tests and guides in [docs/roles](roles/README.md), and the final test was evaluated on October 12 ([Results](RESULTS.md)). The rest of this page describes the October 4 snapshot unless marked otherwise.

The current collaboration task is a real-data classical reference → student implementation → saved-results comparison workflow. It is separate from the archived full research experiments.

## Team workflow and assets

| Component | Scope |
| --- | --- |
| Local data preparation | `prepare_team_data.py` obtains official metadata/selected clips and extracts features after explicit dataset-license acceptance |
| Default mini | 10 sorted-gloss labels; deterministic selection of up to 6 train + 2 validation clips per label; at most 80 real clips |
| Method runner | `run.py` selects reference/student kNN, Linear SVM, or Random Forest |
| Student code | `students/<method>.py`; since Oct 15 these are from-scratch exercises with TODOs (see [roles](roles/README.md)); on Oct 4 they were templates delegating to reference |
| Configuration | `configs/<method>.json`; common preprocessing is fitted on train only |
| Data distribution | No videos or derived feature packs in Git/Release; each user obtains selected inputs from the official source |
| Full preparation | Explicit `--profile full`; complete train/validation only, approximately 31 GB of video plus processing outputs/overhead |
| Acceptance | Fresh environment, official download components, real 80-clip extraction, six classifier runs, and cache reuse verified; scope below |

Cloning or downloading the ZIP supplies code, not the data. After dependency installation and the user's own acceptance of the dataset terms, mini preparation is the small real-data entry point. Later classifier runs reuse local arrays and do not need to redownload videos. This flow needs no PoET weights, vendor source, or GPU.

The mini is a small pipeline check and programming exercise. With at most 60 train and 20 validation examples, its accuracy is not a reliable full-benchmark estimate. It must not be compared with the historical 2,731-class scores.

The portable feature extractor uses MediaPipe complexity 1 and all decoded frames. Historical full experiments used complexity 2 and their recorded preprocessing. The descriptor remains 1,020-dimensional, but newly prepared features and scores are not guaranteed to match historical ones, even on the full split.

## Workflow verification

The new workflow was checked on **macOS arm64 with Python 3.10.18**. A fresh repository `.venv` installed `requirements-data.txt` successfully, and `pip check` found no dependency conflicts. The [verification record](WORKFLOW_VERIFICATION.json) retains the evidence.

| Check | Actual result |
| --- | --- |
| Automated suite | 126 tests: 113 passed; 13 optional GPU/PoET checks skipped |
| Official HTTP Range download | Metadata and one training plus one validation video downloaded: 13,622,614 bytes total; hashes matched the existing copies |
| Real-data extraction | 80 already available local videos freshly processed into 60 train / 20 validation rows across 10 classes |
| Reference/student runs | All three methods × both implementations completed: six runs; saved models/scalers reproduced predictions after reload |
| Default student provenance | Each unchanged template was explicitly marked as delegating to reference and matched its reference metrics |
| Repeat preparation | Rerunning preparation verified and reused the cache |
| Independent implementation contract | A custom nearest-centroid unit test passed without reference delegation; this is a software check, not a member contribution |

This verifies the download components and the real-data extraction/classifier workflow separately. **It does not constitute a fresh-machine download of all 80 videos**: the extraction check reused 80 existing local video files. The new full profile was not run, Linux/Windows have not been verified on those platforms, and skipped optional checks do not establish PoET/GPU inference. This workflow verification performed no test evaluation; the one-shot final test came later, on October 12 ([Results](RESULTS.md)).

## Archived research already completed

The research workspace has complete 40,154 train and 10,304 validation videos/pose caches, three classical systems, official PoET inference, matched fine-tuning, geometry/retrieval experiments, repeated seeds, and documented weak/negative experiments. Error analysis and an original five-output video demo were also completed there.

Full-vocabulary validation Top-1 results are kNN **18.6821%**, SVM **38.7616%**, bounded RF **16.3141%**, and official PoET **80.9394%**. Matched three-seed fine-tuned B is **81.1853%**, and G+R is **81.7352%**, approximately **+0.55 percentage points**. These are validation findings after repeated development. Final-test results (October 12) are in [Results](RESULTS.md).

The classical-versus-PoET comparison changes representation and training history alongside model type. PoET's matched ablation is a separate experiment. Neither should be conflated with the new mini runs.

## Remaining project work

- ~~Freeze final methods and evaluate the 32,941-video test split~~ — done on October 12, 2026 under a preregistered one-shot protocol; see [Results](RESULTS.md). No further test evaluation is planned.
- Measure costs under common conditions, separating fitting/prediction from video/pose processing. Historical scattered timings are not a controlled speed ranking.
- Complete and record each member's actual investigation, analysis, writing, and review. Supplied reference code does not assign credit.
- Review/update final materials together. The separate individual paper review is not completed by the report's reference list.

Extra full-data training and test evaluation are not launched by cloning, setup, or reading this repository.

## Materials and access

The [English report](../reports/report-en.md), [Chinese review copy](../reports/report-zh.md), and three statistical figures explain the completed research. Real-video cases are described in text rather than distributing participant frames. [AI-use records](AI_USE_AND_CONTRIBUTIONS.md) explain initial assistance and known contributions.

Original proposal/report/midpoint/presentation PDFs, PPTX, speaker notes, and the video gallery remain outside this learning repository. Working materials are not evidence of final submission, grading, or full-team review.

Each collaborator needs their own authorized GitHub account. A signed-in ZIP download works for local execution. Data terms still apply to a private repository. Raw media, derived feature payloads, trained weights, full pose caches, vendor source, credentials, and private infrastructure details are omitted from Git.

Complete PoET reproduction needs separate assets and an experiment protocol. It is not an automatic extension of the classical mini workflow.
