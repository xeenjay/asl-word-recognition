# Team repository guidance

Read README.md, docs/STATUS.md, docs/SETUP.md, and docs/EXPERIMENT_PROTOCOL.md before changing this repository.

- This is the team's editable collaboration repository. Start with `run.py`, `students/`, and `configs/`; keep the reference implementations available. Existing research results are validation results, not final-test scores.
- Keep the classical system comparison separate from matched PoET B / B+R / G / G+R ablations.
- Preserve the official signer-disjoint splits. Fit scalers and retrieval banks on training data only. Do not use test for development.
- State which implementation and data produced a result. Do not fabricate experiments, member contributions, or completed human review.
- Keep source datasets, videos, learned weights, feature arrays, caches, raw run logs, credentials, and private infrastructure details out of Git and release assets. The official ASL Citizen license restricts redistribution; `prepare_team_data.py` obtains each user's own inputs directly from the official source or their existing personal copy. Keep `data/`, `outputs/`, and `.team-staging/` ignored.
- Obtain upstream code and assets from their original sources; preserve their provenance and terms. Do not silently modify upstream code.
- Changes here do not alter the archived experiments. Record adaptations explicitly; do not claim that a modified snapshot reproduces historical hashes unchanged.
- Run checks appropriate to the change. The synthetic quickstart uses invented data. The real mini profile uses 10 classes with 60 train / 20 validation clips and is a development exercise, not the full benchmark. Its portable complexity-1 MediaPipe extraction differs from the historical complexity-2 extractor; do not promise identical historical scores.
- Student templates initially delegate to the reference implementation. Retain explicit implementation attribution and record actual changes. The runner owns train-only scaling and evaluation; student fit functions receive training data only. Never silently fall back to the reference when custom code fails.
- Record actual contributions and AI assistance as the team works. Do not automatically assign the prepared reference implementations to individual teammates.
