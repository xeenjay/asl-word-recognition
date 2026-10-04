# ASL Word Recognition: Classical Methods and PoET

[中文说明](README.zh-CN.md) · [Setup](docs/SETUP.md) · [Code map](docs/CODE_MAP.md) · [Methods](docs/METHODS.md) · [Results](docs/RESULTS.md) · [Team guide](docs/TEAM_GUIDE.md)

This private project repository helps our team understand, reproduce, and discuss a comparison of classical and deep learning approaches to isolated American Sign Language (ASL) recognition. The input is a **trimmed video containing one sign**; the output is a word label from the ASL Citizen vocabulary. A predicted label is not an English sentence. Continuous sign segmentation and sentence translation are outside the current scope.

The project has three questions:

1. How do k-nearest neighbors (kNN), Linear SVM, and a bounded Random Forest compare on the same fixed pose-and-motion features?
2. How do those systems compare with PoET, a pretrained lightweight pose Transformer?
3. Within PoET, do explicit relative geometry and a fixed retrieval rule improve a matched baseline?

## Start here

If you are new to the project, read this page, then [Methods](docs/METHODS.md) and [Team guide](docs/TEAM_GUIDE.md). Read the [English report](reports/report-en.md) for the full analysis, or the [Chinese report](reports/report-zh.md) as a review aid. This repository includes the report text and statistical figures; original videos, frames showing people, PDF/PPTX deliverables, and a ready-to-run demo are not included.

For implementation work, use the [code map](docs/CODE_MAP.md), then follow [Setup](docs/SETUP.md) for the supported environment and available checks. The [results guide](docs/RESULTS.md) explains the exported evidence. [Status](docs/STATUS.md) separates completed work from remaining tasks.

**Cloning this repository does not install a complete inference system.** Dataset videos, pose caches, trained weights, and third-party vendor source are not included in Git. Some scripts originated in the full research workspace and need those assets and their recorded metadata. Read the setup instructions before running a workflow; do not assume every script is a standalone command.

## Dataset and common protocol

We use [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/), with 2,731 sign labels and official partitions that separate signers:

| Split | Videos | Signers | Current use |
| --- | ---: | ---: | --- |
| Training | 40,154 | 35 | Fit models, estimate feature statistics, build retrieval banks |
| Validation | 10,304 | 6 | Development, checkpoint selection, and every project score below |
| Test | 32,941 | 11 | Reserved; no final-test results yet |

Provided labels and automatically extracted MediaPipe landmarks avoid new manual annotation. Dataset-level standardization is fitted on training data only. Missing detections remain visible in the records, and difficult clips are not removed from evaluation denominators. Obtain data from its official source under the dataset terms; this repository is not a dataset mirror.

## Methods at a glance

| Method | What it receives | What it learns or stores |
| --- | --- | --- |
| Cosine kNN | A 1,020-dimensional pose/motion descriptor per clip | Stores standardized training descriptors; five neighbors vote |
| Linear SVM | The same fixed descriptor | Fits 2,731 one-versus-rest linear decision functions |
| Bounded Random Forest | The same fixed descriptor | Fits 64 trees with explicit complexity limits |
| Official PoET | A sequence of normalized landmarks | Uses the released pretrained sequence model |
| PoET + geometry + retrieval | Landmark sequence plus relative-geometry features | Adds a small geometry branch and reranks original Top-5 candidates using training examples |

The last row is a PoET variant, not a fifth independent method assigned to a fifth person. The demo can display all five rows. The controlled PoET study separately compares fine-tuned baseline **B**, geometry **G**, retrieval **B+R**, and their combination **G+R**.

SVM and Random Forest use scikit-learn implementations. The project implements standard cosine kNN search and voting with NumPy/PyTorch. PoET and MediaPipe are existing methods. Our work is in data and feature adaptation, the geometry branch, retrieval integration, controlled experiments, evaluation, and analysis; it does not claim to invent these base algorithms.

## Current validation results

The four-system comparison covers the same **10,304 validation clips**:

| System | Top-1 (%) | Top-5 (%) | Macro Top-1 (%) |
| --- | ---: | ---: | ---: |
| Cosine kNN | 18.6821 | 36.1025 | 19.1649 |
| Linear SVM | 38.7616 | 63.1308 | 39.5262 |
| Bounded Random Forest | 16.3141 | 38.2182 | 16.7851 |
| Official PoET checkpoint | 80.9394 | 96.3800 | 81.1950 |

PoET's matched, three-seed study gives these mean Top-1 results:

| Variant | Mean Top-1 (%) |
| --- | ---: |
| B: fine-tuned PoET | 81.1853 |
| B+R: baseline with retrieval | 81.5217 |
| G: geometry branch | 81.4732 |
| G+R: geometry with retrieval | 81.7352 |

G+R improves on B by approximately **0.55 percentage points**, averaging the three paired runs. It also introduces some new errors. B is an additionally fine-tuned baseline and is different from the official checkpoint in the first table.

The classical-versus-PoET table compares **complete systems**: their input representations, pretrained initialization, and training histories differ. It does not isolate classifier architecture. The PoET ablation controls more of these factors, but validation has been reused for development, so its small gain is not a test result or a claim of statistical significance. Published paper test scores must not be ranked directly against these validation scores.

## What is ready, and what remains

Completed in the research workspace: train/validation preparation, the three classical reference systems, official PoET inference, matched geometry/retrieval experiments, several documented negative experiments, error analysis, and a five-output video demo. The written material includes an English report, review translations, proposal, progress report, presentation, speaker notes, and an AI-use record.

Still pending: final-test preparation and evaluation after freezing the protocol; a cost benchmark under common measurement conditions; teammates' actual method investigations and contribution records; and team review of the final materials. The individual paper review is a separate assignment and is not completed by the team's report references.

The written materials and demo listed above describe work completed in the original research workspace. This repository exports selected code, the full report text, and statistical analysis for learning; it does not distribute the complete original workspace. The original demo is a clip recognizer, and real-time streaming performance has not been established. See [Status](docs/STATUS.md) for this snapshot's scope.

## Working together

The confirmed team is Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen. Xinjie organized the initial pipeline and PoET investigation with substantial coding-agent and writing assistance. The classical implementations are shared starting points; they are not evidence that a particular teammate has already completed a method.

Choose and record responsibilities together. A useful contribution can be a reproducible method investigation, a controlled comparison, a careful error analysis, or an evidence-backed report section. Each member should understand and be able to explain the material they contribute. Use the lightweight contribution-record format in [Team guide](docs/TEAM_GUIDE.md), including AI assistance where used, and read the [existing AI-use record](docs/AI_USE_AND_CONTRIBUTIONS.md).

Do not commit dataset payloads, model weights, credentials, machine-specific connection details, or private course records. Experiments and final-test evaluation require an agreed, explicit plan; reading this repository does not start or authorize them.

## Sources

[Third-party sources](docs/THIRD_PARTY.md) records upstream provenance and external source requirements. Follow [Setup](docs/SETUP.md) for the pinned vendor retrieval procedure; vendor source remains outside Git.

- [ASL Citizen dataset](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-description/) and [dataset paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f29cf8f8b4996a4a453ef366cf496354-Paper-Datasets_and_Benchmarks.pdf)
- [PoET paper](https://www.scitepress.org/PublishedPapers/2026/142375/) and [official implementation](https://gitlab.lisn.upsaclay.fr/mtals/publications/poet)
- [MediaPipe Holistic](https://research.google/blog/mediapipe-holistic-simultaneous-face-hand-and-pose-prediction-on-device/)
- [scikit-learn](https://scikit-learn.org/stable/)

Snapshot: October 4, 2026. The completed results and the portable repository's setup status are separate: consult the setup and status documents before interpreting a successful research run as a locally reproducible workflow.
