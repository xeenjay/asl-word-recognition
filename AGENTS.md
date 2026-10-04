# Team repository guidance

Read README.md, docs/STATUS.md, docs/SETUP.md, and docs/EXPERIMENT_PROTOCOL.md before changing this repository.

- This is a learning and collaboration export of the ASL course project. Existing results are validation results, not final-test scores.
- Keep the classical system comparison separate from matched PoET B / B+R / G / G+R ablations.
- Preserve the official signer-disjoint splits. Fit scalers and retrieval banks on training data only. Do not use test for development.
- State which implementation and data produced a result. Do not fabricate experiments, member contributions, or completed human review.
- Keep source datasets, videos, learned weights, caches, raw run logs, credentials, and private infrastructure details out of Git.
- Obtain upstream code and assets from their original sources; preserve their provenance and terms. Do not silently modify upstream code.
- Changes here do not alter the archived experiments. Record adaptations explicitly; do not claim that a modified snapshot reproduces historical hashes unchanged.
- Run checks appropriate to the change. The synthetic quickstart demonstrates mechanics only and is not an ASL benchmark.
- Record actual contributions and AI assistance as the team works. Do not automatically assign the prepared reference implementations to individual teammates.
