# Member tasks: learn, implement, compare, investigate, write

[Home](../../README.md) · [Member 1 — kNN](member1-knn.md) · [Member 2 — Linear SVM](member2-linear-svm.md) · [Member 3 — Random Forest](member3-random-forest.md)

The project compares three classical classifiers on hand-crafted pose/motion features with PoET, a pose transformer. Xinjie owns PoET, the shared data pipeline and integration. Each of the other three members owns **one classical method from start to finish**:

| Role | Method | Code you write | Self-check |
| --- | --- | --- | --- |
| Member 1 | k-nearest neighbours | `students/knn.py` TODO 1–5 | `python -m unittest tests.test_member1_knn -v` |
| Member 2 | Linear SVM, plus left/right-hand normalization | `students/linear_svm.py` TODO 1–4, `scripts/classical_handedness.py` TODO H1–H3 | `tests.test_member2_linear_svm`, `tests.test_member2_handedness` |
| Member 3 | Decision tree and random forest | `students/random_forest.py` TODO 1–5 | `python -m unittest tests.test_member3_random_forest -v` |

The team decides who takes which role. Member 1 is the gentlest start; Member 3 involves the most code.

## The five parts of every role

| Part | What you do | Evidence |
| --- | --- | --- |
| A. Learn | Read the guide's sources and answer its concept questions in your own words | `docs/roles/memberN-notes.md` |
| B. Implement | Fill in the TODOs. Each function's docstring states exactly what to return; your guide adds hints | Self-check tests pass |
| C. Compare | Run your implementation and the library reference on the same mini data; explain every difference | Notes: comparison table |
| D. Investigate | One bounded experiment on mini, then interpret the full-data results that Xinjie runs with your code/configs | Notes: experiment table and interpretation |
| E. Write | One report subsection with a table or figure, and 1–2 slides | Draft text in your notes, then in the report |

The full ASL Citizen data (40,154 train / 10,304 validation clips, 2,731 words) are not redistributed and take a long time to prepare. You therefore develop and test on the 10-word mini dataset (see [Setup](../SETUP.md)); once your tests pass, Xinjie runs your code or configs on the full train/validation data and sends you the output files (`metrics.json`, `per_class.json`).

## Suggested schedule

| By | Milestone |
| --- | --- |
| Sun Oct 26 | Environment and mini data ready; reference runs done; Part A reading started |
| Sun Nov 2 | Part B: all your self-check tests pass |
| Sun Nov 9 | Part C comparison and concept answers in your notes (used for the midpoint check-in) |
| Sun Nov 23 | Part D: mini experiment done; full-data results interpreted |
| Sun Nov 30 | Part E: report subsection and slides drafted |
| Mon Dec 7 | Final report due |

Xinjie may adjust dates; the order matters more than the exact days.

## How to work

1. Clone the repository, create the environment and prepare the mini data ([Setup](../SETUP.md)).
2. Make a branch for your role, e.g. `git switch -c member2/svm`.
3. Work one TODO at a time. Run your self-check after each one; the tests are grouped by TODO number, so you can see which TODO is still failing.
4. Commit small steps with clear messages. Open a pull request to `main` when a part is done; Xinjie reviews it.
5. Change only your own files. Discuss first before editing the shared runner, evaluation, configs used by others, or another member's files.

## When you are stuck

Use the hints in this order, and write down in your notes which level you needed:

1. The docstring of the function and the "Hints" section of your guide.
2. Ask Xinjie or the team for a hint (describe what you tried and the failing test output).
3. Ask for the relevant answer snippet. Answers are released after each milestone anyway so that everyone can compare their code with a reference solution.

The library references live in `scripts/classical_models.py`. The kNN reference there is written in NumPy, so try your own version **before** reading it. The SVM and forest references call scikit-learn.

## AI tools and honesty

Using AI assistants is allowed when recorded (course policy): you must be able to explain every line you submit, and you must say how AI was used. Ask for explanations and hints rather than complete solutions; the point of this work is to learn the method. Record AI use and help received in your notes and in [AI use and contributions](../AI_USE_AND_CONTRIBUTIONS.md). The final report states each member's actual contribution; the reference code and the prepared framework are Xinjie's and are not counted as your work.

<!-- BEGIN SOLUTION instructor -->
## Instructor notes (answer-key repository only)

- Answers sit in `# BEGIN SOLUTION memberN` blocks (code) and `<!-- BEGIN SOLUTION memberN -->` blocks (guides). Full-data numbers are tagged `results-2026-10`. See [teaching/README.md](../../teaching/README.md) for generating, releasing and pushing the teaching repository.
- Suggested release points: `member1`, `member2`, `member3` after the Nov 2 milestone (code) — or per member once their tests pass; `results-2026-10` after each member has written their Part D predictions.
<!-- END SOLUTION -->
