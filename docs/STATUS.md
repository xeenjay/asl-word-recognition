# Project status

[Home](../README.md) · [Results](RESULTS.md) · [Setup](SETUP.md) · [Team guide](TEAM_GUIDE.md)

Snapshot: **October 4, 2026**. This document describes the completed research and its remaining boundaries. The portable learning repository has a narrower set of files than the original research workspace. Use [Setup](SETUP.md) for the checks supported by this checkout; a completed original experiment is not proof that all its external assets are present here.

## Confirmed scope

- Isolated ASL Citizen recognition: one trimmed sign clip → one of 2,731 labels.
- Comparison of kNN, Linear SVM, bounded Random Forest, and PoET, plus controlled PoET geometry/retrieval ablations.
- No new manual labels. Continuous sentence translation is outside scope.
- Four confirmed members: Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen. Other members' exact method ownership and completed contributions remain to be consolidated.

## Completed in the research workspace

| Area | Completed evidence and output |
| --- | --- |
| Data | All 40,154 training and 10,304 validation videos and their pose records prepared; official split identities retained |
| Classical systems | Shared 1,020-D features; kNN, SVM, bounded RF fitting/bank creation; full validation predictions, reload checks, and metric verification |
| PoET | Official checkpoint deployment, matched fine-tuning, relative geometry, retrieval, three-seed component comparison |
| Broader investigation | Attribute, cross-signer, extended-training, and RGB findings retained, including negative and small-gain results |
| Analysis | Complete validation comparison; paired ablations; error overlap and descriptive slices; representative successes and failures |
| Demonstration | One clip extraction shared across the three classical methods, official PoET, and fixed-seed G+R; saved examples distinguished from fresh inference |
| Written materials | Revised full proposal, report, progress snapshot, slides, speaker notes, and AI/contribution record |

The original analysis gallery contains 16 deterministically selected videos and 48 frames. They are illustrative cases, not a randomly sampled accuracy estimate, and those videos and frames showing people are not distributed here. The original demo's functional checks do not establish that its service is currently running or available from this repository.

## Result boundaries

All current project scores use **validation**, not test. The four-system Top-1 results are kNN 18.6821%, SVM 38.7616%, bounded RF 16.3141%, and official PoET 80.9394%. They share the complete 10,304-clip cohort but differ in representation and training history.

The matched three-seed PoET mean is 81.1853% for fine-tuned B and 81.7352% for G+R, approximately +0.55 percentage points. The gain includes regressions and comes after repeated validation development. It is not evidence of statistical significance, a final-test improvement, or a new state of the art.

## Remaining work

| Item | Current boundary | Completion evidence needed |
| --- | --- | --- |
| Final test | Official metadata exists; 32,941 test videos have not been prepared or evaluated | Freeze the methods, selected checkpoints, parameters, and failure policy before preparing and evaluating the test split |
| Cost comparison | Scattered execution records exist; no common-hardware, common-protocol benchmark for every method | Separate fitting, cached prediction, storage, and video/pose preprocessing costs under a documented measurement protocol |
| Team contributions | Shared reference implementations exist; actual individual work is not fully recorded | Each member's real code, experiments, analysis, writing, review, and presentation contributions linked to evidence |
| Team review | Initial materials were organized by Xinjie with agent assistance | Members review and can explain the final version; contribution and AI-use statements updated from actual activity |
| Individual paper review | Separate individual assignment; not replaced by report references | Each student's own completed review |
| Final materials | Current reports/slides are complete working drafts based on validation | Update after agreed final evaluation and team review, then follow the actual course submission requirements |

These tasks are not automatically started by publishing or cloning this repository. Additional training and final-test evaluation need a concrete agreed plan.

## Prepared material status

This repository includes [English report text](../reports/report-en.md), a [Chinese review copy](../reports/report-zh.md), three statistical figures, and the [AI-use/contribution record](AI_USE_AND_CONTRIBUTIONS.md). The original fourth report figure showed frames from real videos; the repository version describes those cases in text instead of redistributing the images. Reports follow Introduction, Methods, Results, Discussion, and Conclusion and retain the numerical findings, limitations, and references.

The original research workspace also has a one-page full proposal, a ten-page English report and nine-page Chinese review PDF, a two-page midpoint snapshot, nine-slide English PDF/PPTX, and speaker notes. Those PDF/PPTX deliverables and the original gallery are not included in this learning repository. Their existence is recorded here to explain overall project progress, not to promise downloadable files in this checkout.

The report's contribution statement describes this version's preparation rather than inventing completed teammate work. The original midpoint snapshot describes October 4 progress and must be updated before representing a later milestone.

The pre-proposal has a previously recorded submission. The other prepared documents are working materials, not evidence of final submission, grading, or instructor approval. Current project information should be read from this snapshot and its evidence rather than older planning language retained in historical documents.

## What is intentionally absent from Git

- Dataset video payloads and frame/pose caches.
- Original report images containing people, the video gallery, and PDF/PPTX deliverables.
- Trained checkpoints and full model or retrieval-bank payloads.
- Third-party vendor source and local environments.
- Credentials, machine connection details, and unrelated private course records.

Some included source files need these omitted assets, their exact revisions, and recorded manifests to run. Follow [Setup](SETUP.md) and the [third-party source record](THIRD_PARTY.md); do not interpret the source export as a turnkey training or inference distribution. The [code map](CODE_MAP.md) identifies the available implementation and tests, while [Results](RESULTS.md) explains the exported evidence.
