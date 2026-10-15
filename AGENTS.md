# Team repository guidance

Read README.md, docs/STATUS.md, docs/SETUP.md, and docs/EXPERIMENT_PROTOCOL.md before changing this repository.

- This is the team's editable collaboration repository. Start with `run.py`, `students/`, `configs/`, and `docs/roles/`; keep the reference implementations available. Research results are validation results unless labeled as the one-shot final test of October 12, 2026; never evaluate test again or tune on it.
- Keep the classical system comparison separate from matched PoET B / B+R / G / G+R ablations.
- Preserve the official signer-disjoint splits. Fit scalers and retrieval banks on training data only. Do not use test for development.
- State which implementation and data produced a result. Do not fabricate experiments, member contributions, or completed human review.
- Keep source datasets, videos, learned weights, feature arrays, caches, raw run logs, credentials, and private infrastructure details out of Git and release assets. The official ASL Citizen license restricts redistribution; `prepare_team_data.py` obtains each user's own inputs directly from the official source or their existing personal copy. Keep `data/`, `outputs/`, and `.team-staging/` ignored.
- Obtain upstream code and assets from their original sources; preserve their provenance and terms. Do not silently modify upstream code.
- Changes here do not alter the archived experiments. Record adaptations explicitly; do not claim that a modified snapshot reproduces historical hashes unchanged.
- Run checks appropriate to the change. The synthetic quickstart uses invented data. The real mini profile uses 10 classes with 60 train / 20 validation clips and is a development exercise, not the full benchmark. Its portable complexity-1 MediaPipe extraction differs from the historical complexity-2 extractor; do not promise identical historical scores.
- `students/*.py` and TODO H1–H3 in `scripts/classical_handedness.py` are learning exercises for Members 1–3 (guides in `docs/roles/`). When a member asks an AI agent for help, explain concepts, point to the failing test, and give hints first; do not write a complete solution unless the member explicitly asks, and remind them to record the help in their notes. Retain explicit implementation attribution. The runner owns train-only scaling and evaluation; student fit functions receive training data only. Never silently fall back to the reference when custom code fails.
- Record actual contributions and AI assistance as the team works. Do not automatically assign the prepared reference implementations to individual teammates.
<!-- BEGIN SOLUTION instructor -->
- Answer-key repository: answers live in `# BEGIN SOLUTION` / `<!-- BEGIN SOLUTION -->` blocks and `teaching/`. The members' repository is generated with `teaching/make_teaching_version.py` (see `teaching/README.md`). Never push this repository or its history to the teaching repository and never invite members here. Run `tests/test_teaching_version.py` after changing exercises or markers.
<!-- END SOLUTION -->
