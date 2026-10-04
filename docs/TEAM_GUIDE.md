# Team guide: learn, investigate, and record your contribution

[Home](../README.md) · [Methods](METHODS.md) · [Setup](SETUP.md) · [Code map](CODE_MAP.md) · [Status](STATUS.md)

This repository is a shared starting point. The confirmed team members are Xinjie Huang, Duncan Stadler, Leyun Kim, and Ming Chen. Xinjie's existing investigation centers on PoET and its modifications. Other individual method assignments and completed contributions have not yet been consolidated; the presence of a reference implementation does not assign its ownership.

## A practical learning path

### First: understand what a result means

Read the main README, then the [English report](../reports/report-en.md) or [Chinese review copy](../reports/report-zh.md). These are complete text analyses with statistical figures; the original video gallery and presentation files are not included. Be able to explain the input and output, the three dataset splits, and the difference between the four-system comparison and the matched PoET ablation.

Check your understanding with these questions:

- Why is a predicted word label not an English translation?
- Why must the scaler and neighbor bank use training data only?
- What does it mean when a missing-hand clip remains in the denominator?
- Why are official PoET and fine-tuned baseline B shown separately?
- Why is a 0.55-percentage-point validation gain not a final-test claim?

### Next: trace one method

Read [Methods](METHODS.md) and use the [code map](CODE_MAP.md) to follow one path through `scripts/`: input representation, fitting or bank creation, prediction ranking, serialization, and metric calculation. Read the relevant checks in `tests/` alongside the implementation.

For a classical method, distinguish the shared descriptor from the classifier. For PoET, distinguish the original model, the geometry residual, and retrieval after classification. Being able to identify a library call is a useful start; explaining its inputs, parameters, and outputs is the next step.

### Then: inspect evidence and reproduce a supported check

Use [Results](RESULTS.md) to inspect the saved outcomes before running expensive workflows. Select one correct and one incorrect example and describe what the records show, separating observation from hypotheses about why the model behaved that way.

Follow [Setup](SETUP.md) for supported checks. Ask the team to agree on how required assets will be obtained or shared before attempting an asset-dependent workflow. Raw videos, pose caches, weights, and vendor source are omitted from Git. A source file existing here does not establish that it can run without those dependencies.

Reproducing an existing output can demonstrate understanding and reproducibility. Record exactly what was reproduced; do not present it as a newly trained model or a new result.

### Finally: agree on a bounded investigation

Possible investigations include a clearly defined error slice, an explanation of model disagreements, or a small controlled comparison. Examples to discuss are neighbor-count sensitivity, the contribution of motion summaries, and a bounded forest-complexity comparison. These are options, not assigned tasks or experiments already authorized by the repository.

Write the question, fixed settings, changed factor, validation metric, and finite run budget before starting. Coordinate shared features and model versions so everyone compares compatible inputs. Keep final test reserved until the team freezes its protocol.

## What counts as useful evidence

A method-focused contribution should make it clear what the member actually did. Useful evidence may include:

- A code change or reviewed implementation, with its assumptions and provenance explained.
- An existing result independently reproduced, with the configuration and verification recorded.
- A controlled experiment whose input, changed factor, budget, and output are identifiable.
- An error analysis that distinguishes counts, visible observations, and uncertain explanations.
- A report or presentation section whose numerical claims can be traced to the evidence.

This list is a collaboration guide, not an additional course rubric. A contribution need not invent an algorithm or outperform PoET. It should go beyond attributing a supplied implementation to someone solely because their name is placed beside it.

## Contribution record

Keep one entry per meaningful completed activity in a team-agreed document or pull-request description. Use this format:

| Field | What to record |
| --- | --- |
| Date and member | Who performed the activity and when |
| Question or purpose | What the work was intended to establish |
| Actual work | Code, reproduction, experiment, analysis, review, writing, or presentation work completed |
| Evidence | Commit, file, configuration, exported result, or review link |
| Findings and limits | What the evidence supports and what remains uncertain |
| AI or other assistance | What a tool or collaborator generated, and what the member checked or changed |
| Report connection | Which figure, table, paragraph, or presentation section it supports |

A plan is not a completed contribution. Reading a report is not the same as verifying its results. Updating a contribution statement should follow the work, rather than serve as an assignment of credit in advance.

## AI assistance and review

The existing pipeline and initial materials were prepared under Xinjie's direction with substantial coding-agent and writing assistance. Assistance included data preparation, feature and adapter implementation, user-authorized experiment execution, numerical checks, demo development, plots, and draft writing. Read the [documented AI-use record](AI_USE_AND_CONTRIBUTIONS.md). The existing reference code should not be described as entirely hand-written by a team member.

The team remains responsible for the version it submits. Read and correct generated explanations, trace numerical claims to records, and be able to explain your contribution. Keep the AI-use record aligned with the actual work; it should not claim that everyone has already reviewed or mastered the materials.

## Making changes safely and reproducibly

- Keep one clear purpose per change and describe what changed, why, and how it was checked.
- Discuss changes to shared feature definitions, label order, missing-data rules, or model-selection criteria before comparing new results with the existing tables.
- Preserve existing results and negative findings. New experiments need new, identifiable records rather than overwriting a favorable or unfavorable run.
- Do not use final-test errors to tune the model. Agree on the final protocol first.
- Record execution evidence outside Git when it contains large or restricted assets, and commit only appropriate summaries or code.
- Keep videos, pose payloads, trained weights, credentials, personal machine details, and unrelated course records out of the repository.

The repository supports learning and collaboration. It does not automatically launch training, grant access to another member's machine, invite collaborators, or submit coursework.
