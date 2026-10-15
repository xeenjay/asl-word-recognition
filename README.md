# ASL Word Recognition — Team Workspace

[中文](README.zh-CN.md) · [Setup](docs/SETUP.md) · [Methods](docs/METHODS.md) · [Code map](docs/CODE_MAP.md) · [Results](docs/RESULTS.md) · [Team guide](docs/TEAM_GUIDE.md)

Use this repository to **run a reference method on real ASL data, write your method, and compare the results**. The task is one trimmed sign video → one word label. Continuous sentence translation is outside the project scope.

The three classical methods — **kNN**, **Linear SVM**, and **Random Forest** — share 1,020-dimensional pose/motion features. Each of three team members writes one of them from scratch in `students/`, guided by TODOs, hints, and self-check tests.

**Start here: [Member tasks](docs/roles/README.md)** — what each role learns, writes, tests, investigates, and reports, with a suggested schedule.

<!-- BEGIN SOLUTION instructor -->
> **Answer-key repository (Xinjie only).** This is `huiliyi-XJ/asl-word-recognition`, which contains the answers. Members use the generated teaching repository `asl-word-recognition-team`; see [teaching/README.md](teaching/README.md). Do not invite members here.
<!-- END SOLUTION -->

## Quick start

Accept the private repository invitation and authenticate GitHub with your own account. You can clone it or sign in and choose **Code → Download ZIP**. A ZIP copy runs locally too.

```bash
git clone https://github.com/huiliyi-XJ/asl-word-recognition-team.git
cd asl-word-recognition-team
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
```

Use Python 3.10 or 3.11. See [Setup](docs/SETUP.md) for environment details.

Read the [ASL Citizen dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) yourself. If you accept its terms, prepare a small real dataset:

```bash
python scripts/prepare_team_data.py --accept-dataset-license
```

This downloads metadata and **up to 80 selected clips directly from the official source**, then extracts local numeric features: 10 labels, up to 6 training clips and 2 validation clips per label (all available clips are kept if a group is smaller). The labels are the first 10 in sorted gloss order; clips within each split/class are selected deterministically by a filename hash. Training and validation keep their official split assignments.

No dataset or feature pack is redistributed through Git or GitHub Releases. The first preparation needs network access and CPU processing. Later method runs reuse `data/classical-mini/`; **each member does not need to download the approximately 31 GB full train/validation videos to get started**. PoET weights, vendor source, and a GPU are unnecessary for this workflow.

Run the reference:

```bash
python run.py --method knn --implementation reference --data data/classical-mini --output outputs/knn-reference
```

Fill in the TODOs of your role's file (for example `students/knn.py`, see [Member tasks](docs/roles/README.md)) and check them:

```bash
python -m unittest tests.test_member1_knn -v
```

Then run your version through the same runner:

```bash
python run.py --method knn --implementation student --data data/classical-mini --output outputs/knn-student
```

Use `linear_svm` or `random_forest` instead of `knn` for another method and its corresponding student file. Defaults come from `configs/<method>.json`; `--config` can select another saved configuration. Keep the data and settings identical when comparing implementations.

Every run needs a **new output directory**. Compare `metrics.json`, `per_class.json`, and saved predictions. `run.json` records provenance, including any calls to the reference. Until your TODOs are filled in, a student run stops with `NotImplementedError`.

## Where to work

| Next step | File or guide |
| --- | --- |
| Your role and tasks | [Member tasks](docs/roles/README.md) and your guide; notes in `docs/roles/memberN-notes.md` |
| Write a method | `students/knn.py`, `students/linear_svm.py`, `students/random_forest.py` (Member 2 also `scripts/classical_handedness.py`) |
| Check your code | `python -m unittest tests.test_member1_knn -v` (or `test_member2_linear_svm`, `test_member2_handedness`, `test_member3_random_forest`) |
| Inspect settings | `configs/<method>.json` |
| Understand the fit/predict interface | [Team guide](docs/TEAM_GUIDE.md) |
| Find the runner and reference implementation | [Code map](docs/CODE_MAP.md) |
| Understand features and algorithms | [Methods](docs/METHODS.md) |
| Read the research | [English report](reports/report-en.md), [Chinese report](reports/report-zh.md), [Results](docs/RESULTS.md) |
| Check completion and acceptance status | [Status](docs/STATUS.md) |

The runner fits the scaler on training rows only and gives standardized features to the classifier. Do not fit preprocessing on validation or remove difficult examples to improve accuracy.

## Mini practice and historical results

The mini is a **small pipeline check and implementation exercise**, not a reliable full-benchmark estimate. Its 10-class scores cannot be compared with the historical 2,731-class scores below.

| Historical system | Full validation Top-1 (%) |
| --- | ---: |
| Cosine kNN | 18.6821 |
| Linear SVM | 38.7616 |
| Bounded Random Forest | 16.3141 |
| Official PoET checkpoint | 80.9394 |

The historical study used 40,154 training and 10,304 validation clips. PoET has a different sequence representation and pretrained initialization, so this is a comparison of complete systems. A separate matched three-seed PoET study found fine-tuned B **81.1853%** → G+R **81.7352%**, approximately **+0.55 percentage points**. B differs from the official checkpoint above. These are validation findings; the one-shot final test of October 12, 2026 (32,941 videos) is summarized in [Results](docs/RESULTS.md).

The portable extractor uses MediaPipe model complexity 1 and all decoded frames. Historical experiments used complexity 2 and their recorded preprocessing protocol. Sharing the 1,020-D descriptor does not make feature values identical, so even a new full-data run is not promised to reproduce old scores exactly.

Full-data preparation is an explicit optional operation, described in [Setup](docs/SETUP.md). It needs approximately 31 GB of videos plus processing/storage overhead. Full CPU SVM fits 2,731 one-versus-rest classifiers; default teaching runs use one worker and limit numerical threads. Historical parallel timings are not laptop estimates. Start with mini.

## Team and contribution records

The team is Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen. Xinjie organized the initial research with substantial agent assistance. Members 1–3 each own one classical method ([Member tasks](docs/roles/README.md)); the team decides who takes which role. The supplied reference code and exercise framework are not attributed to a teammate.

Record the implementation, runs, analysis, writing, and review you actually complete using [Team guide](docs/TEAM_GUIDE.md), and keep [AI-use records](docs/AI_USE_AND_CONTRIBUTIONS.md) accurate. Reference/library use is acceptable when stated; it is not a claim that you wrote the entire algorithm.

This repository includes report text and statistical figures. Original participant media, trained weights, full caches, PDF/PPTX deliverables, credentials, and private machine details are not committed. Complete PoET reproduction still needs separate assets and a protocol; it is not a one-click feature of this teaching workflow.
