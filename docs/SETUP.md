# Setup: real-data reference and student methods

[Home](../README.md) · [Data provenance](DATA_PROVENANCE.md) · [Student interface](TEAM_GUIDE.md) · [Code map](CODE_MAP.md)

Commands run from the repository root. The default workflow obtains a small official-data subset, extracts numeric features locally, and runs classical methods on CPU. It does not need PoET weights, upstream PoET vendor source, or PyTorch.

## 1. Get the private repository

Accept the invitation and authenticate with your own GitHub account:

```bash
git clone https://github.com/huiliyi-XJ/asl-word-recognition.git
cd asl-word-recognition
```

Alternatively, sign in and use **Code → Download ZIP**, extract it, and open a terminal in the top-level folder. The local workflow does not require Git history. A private ZIP still requires repository access.

## 2. Install dependencies

Use Python 3.10 or 3.11. Example for macOS/Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
```

On Windows PowerShell, create the environment with `py -3.10 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1`, then use the same pip command.

`requirements-data.txt` combines the classical dependencies with requests, MediaPipe 0.10.21, and opencv-contrib-python 4.11.0.86. MediaPipe is used for local feature preparation. If you already prepared compatible feature arrays, classifier execution needs only `requirements-classical.txt` (also selected by `requirements.txt`).

## 3. Prepare the default real mini dataset

Read the [official dataset license](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) before deciding to accept it. If you accept:

```bash
python scripts/prepare_team_data.py --accept-dataset-license
```

The default mini profile selects 10 labels in sorted gloss order. Within each label and official split, filename hashes select up to 6 training and 2 validation clips, for **at most 60 train + 20 validation = 80 clips**. A smaller group retains all its available clips; the generated manifest records actual counts. It downloads the official metadata and selected ZIP entries using checked range requests, then runs MediaPipe and the project's 1,020-D descriptor extraction locally.

The result is `data/classical-mini/`, including train/validation feature arrays, integer label arrays, local label order, and a provenance manifest. The local 0–9 labels describe this mini vocabulary; they are not arbitrary replacements for the full benchmark label IDs.

Preparation is a real video/landmark pipeline and takes time. Do not interpret successful download or feature extraction as correct recognition. Subsequent method runs reuse the arrays without repeating video preprocessing. Locally acquired clips can also be inspected under the dataset terms; they must not be committed.

If you already obtained the official dataset yourself, provide its local directory rather than downloading those clips again:

```bash
python scripts/prepare_team_data.py --accept-dataset-license --source-dir path/to/your/asl_citizen
```

The source directory must contain `splits/` and `videos/`. See [Data provenance](DATA_PROVENANCE.md) for the exact source layout, recorded identities, and licensing boundary. Real data is not redistributed through this repository or a private Release. The acceptance flag records your own explicit decision; it is not a license granted by this project.

## 4. Run a reference

With mini preparation complete:

```bash
python run.py --method knn --implementation reference --data data/classical-mini --output outputs/knn-reference
python run.py --method linear_svm --implementation reference --data data/classical-mini --output outputs/svm-reference
python run.py --method random_forest --implementation reference --data data/classical-mini --output outputs/rf-reference
```

Run only the method you need. Defaults are `--method knn`, `--implementation reference`, and `--data data/classical-mini`; `--output` is required and must name a new directory.

Parameters come from `configs/<method>.json`. Use `--config path/to/config.json` to select another saved configuration. Config files contain `method` and a `parameters` object. Keep data/settings identical for a reference-versus-student implementation comparison.

A partial configuration merges with the method defaults. Inspect the saved output `config.json` to see the resolved settings actually used.

## 5. Edit and run your student module

Edit the corresponding file in `students/`, then:

```bash
python run.py --method knn --implementation student --data data/classical-mini --output outputs/knn-student
```

For the other methods, replace `knn` with `linear_svm` or `random_forest`. Templates initially call the reference implementation. Provenance records that delegation; an unchanged template is not an independent implementation.

The runner fits StandardScaler on training rows only and passes standardized X to both reference and student methods. Do not fit preprocessing on validation. [Team guide](TEAM_GUIDE.md) describes `fit` and `predict_topk`.

## 6. Inspect the output

| File | Purpose |
| --- | --- |
| `metrics.json` | Accuracy, macro metrics, and evaluation counts |
| `per_class.json` | Class-level results |
| `predictions.npz` | Candidate IDs, scores, and validity information |
| `run.json` | Run state, implementation provenance, hashes, and reference-delegation record |
| `config.json`, `labels.json` | Resolved settings and local vocabulary |
| `model.joblib`, `scaler.joblib` | Fitted model and training-fitted scaler |

Check the run state before treating output as a completed evaluation. Preserve failed outputs and choose a new directory for the next attempt. Do not commit generated data/models indiscriminately.

Accuracy values in JSON are proportions from 0 to 1; the command-line display converts them to percentages. For example, `top1: 0.4` means 40%, not 0.4%.

The mini is a pipeline exercise, not a reliable benchmark estimate. Its 10-class scores cannot be compared with archived full-vocabulary scores. Different methods' votes, margins, and probabilities are also not calibrated to a common confidence scale.

## 7. Optional full train/validation preparation

The full profile is an explicit larger operation:

```bash
python scripts/prepare_team_data.py --profile full --accept-dataset-license
```

It covers 40,154 training and 10,304 validation clips, with approximately 31 GB of raw video plus feature/pose outputs and temporary overhead. It does not need to download the complete approximately 45.9 GB archive first, but it still needs the selected full video payload and CPU preprocessing. `--source-dir` can reuse an official copy you already obtained. This operation does not include test.

The full profile writes `data/classical-full/` by default. Use that directory with the same runner once preparation is verified. Full SVM fits 2,731 one-versus-rest classifiers; full kNN compares against 40,154 references; multiclass RF also has significant storage costs. Default SVM/RF use one worker, and the runner limits numerical thread pools to one. No short laptop runtime or measured memory ceiling is promised. Start with mini and agree on the full run budget before launching it.

The portable extractor uses MediaPipe `model_complexity=1` and all decoded frames. Historical full experiments used complexity 2 and their recorded preprocessing protocol. The descriptor dimension/formulas are shared, but the feature values can differ. Treat newly prepared full data as a separately identified run; do not promise exact historical-score reproduction.

## Optional learning and research paths

`scripts/quickstart_classical.py` remains a synthetic mechanics example that requires no dataset download. Its invented trajectories are not recognition evidence. The new real-data workflow has separate acceptance checks.

The PoET/geometry/retrieval research tools require separate optional dependencies, official source/weights, complete compatible caches, and a recorded protocol. See [Optional research setup](RESEARCH_SETUP.md) for vendor, checkpoint, MediaPipe-asset, and advanced command instructions, together with [Third-party sources](THIRD_PARTY.md), [Code map](CODE_MAP.md), and [Experiment protocol](EXPERIMENT_PROTOCOL.md). Installing dependencies alone is not full PoET reproduction.

## Verified workflow and limits

On macOS arm64, a fresh repository `.venv` using Python 3.10.18 successfully installed `requirements-data.txt`; `pip check` reported no conflicts. The automated suite completed 126 tests: 113 passed and 13 optional GPU/PoET checks were skipped.

Official HTTP Range downloads were exercised for metadata and one train plus one validation video, totaling 13,622,614 bytes, with hashes matching existing copies. Separately, all 80 selected **existing local videos** were freshly extracted into 60 training and 20 validation rows. All six reference/student method runs completed with identical saved-model reload predictions; unchanged student templates were explicitly labeled reference delegates and matched their references. Running preparation again verified cache reuse. An independent custom nearest-centroid unit test also passed; it is a contract check, not a claim of a team member's completed method.

The full 80-video network download on a fresh machine was **not** tested end to end. Neither the new full profile nor Linux/Windows execution was run. These checks do not establish final-test evaluation or optional PoET/GPU inference. See [Status](STATUS.md) and the [verification record](WORKFLOW_VERIFICATION.json) for the precise evidence.
