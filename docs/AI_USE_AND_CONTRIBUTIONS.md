# AI use and contribution record / AI 使用与贡献记录

Snapshot: 2026-10-04. This record describes documented assistance and known ownership, not a certification that every member has reviewed or mastered all material. The course syllabus requires students to explain submitted work and be able to produce a record of AI use. It does not specify a separate mandatory attachment format.

## Documented human direction

Xinjie Huang selected the ASL direction, requested implementation and bounded experiments, reviewed reported outcomes, requested negative-case analysis and a unified demo, consulted the instructor, and asked for this initial set of written materials. The four confirmed project members are Xinjie Huang, Duncan Stadler, Leyun Kim and Ming Chen. The team selected isolated ASL word recognition after its discussion. Individual method assignments and each other member's completed technical work have not yet been consolidated in the available record.

No existing kNN/SVM/RF reference implementation should be attributed to a teammate solely because it was prepared for possible use. Before team submission, replace the snapshot contribution paragraph with each person's actual code, experiments, analysis, writing and presentation contributions.

## Assistance log

| Work area | Agent assistance documented | Evidence retained | Human review responsibility |
| --- | --- | --- | --- |
| Literature and scope | Searched primary sources; explained isolated recognition versus continuous translation; suggested bounded methods | Research/scope documents and source links | Confirm the chosen task and explain relevant sources |
| Data and preprocessing | Automated public-data preparation, uploads, MediaPipe extraction, caching and completeness checks | Dated run manifests, failure accounting, environment records | Understand split isolation, representations and missing-data policy |
| Implementations and experiments | Wrote adapters and feature code, configured existing libraries, ran user-authorized training/evaluation, checked reloads and predictions | Source, configs, model hashes, logs, `EXPERIMENTS.md` | Explain algorithm/library provenance and the fixed evaluation protocol |
| Result analysis | Recomputed metrics, paired-seed differences, error overlap and descriptive slices; retained negative results | Saved prediction analyses and independent verifications | Check numerical claims and distinguish observation from causal explanation |
| Demo and figures | Built the clip interface, generated plots from existing results, selected reproducible examples and extracted real frames | `comparison-demo/`, `figures/figure-verification.json` | Interpret displayed scores and case-selection bias |
| Written materials | Drafted/revised English proposal, report, progress snapshot and slides; translated Chinese review copies; formatted PDFs | Editable Markdown/PPTX, source citations, PDF/slide QA | Read, correct, and be able to defend the actual submitted version |

AI outputs were checked using source records, saved model predictions, numerical recomputation, software tests where appropriate, and visual review. These checks reduce errors but are not a guarantee that every explanation is correct. This record does not claim that human review already occurred after file generation.

## This writing pass

The writing pass reused completed train/validation experiments. It created no new training run, test evaluation or performance improvement. Charts describe the same 10,304 validation videos. Three seed error bars are sample standard deviations, not confidence intervals. Real clip frames originate from existing ASL Citizen files with recorded identities. No generated or fabricated sign images are used as experimental evidence.

The final team contribution statement and any final-test results must be updated from actual events. The prepared documents have not been submitted or sent externally by the assistant.

## 中文核对

这份记录说明AI实际参与了调研、实现、实验执行、验证、分析、demo及文字排版。它没有声称你逐行手写代码，也没有声称所有组员已阅读/掌握或完成分工。你负责对最终提交版本进行审核并能解释；队友贡献必须按实际工作填写。本轮整理材料复用了已有validation结果，没有新训练、test或新性能提升。

## Team programming workflow update — October 4, 2026

Xinjie requested an editable repository where teammates can implement their
chosen methods, reuse the supplied reference when needed, and run experiments.
An agent prepared the student interfaces, configurations, shared runner, official
data acquisition and local feature preparation, tests, and revised instructions.
The reference-delegating templates are starting material, not completed work by
the other members. Individual assignments remain for the team to confirm.

This update includes new engineering checks using 60 real training clips and 20
validation clips from ten classes, with a separately identified portable
extractor. Each of three reference methods and its unchanged student template
was fitted/evaluated and checked after model reload. These small runs do not
replace the historical full experiments or establish a new benchmark result.
No test evaluation or full-data retraining was performed. Dataset payloads were
not uploaded; each user prepares their own inputs under the official terms.
See [workflow verification](WORKFLOW_VERIFICATION.json) for the actual checks.

组员工作入口、下载/预处理、运行器和模板由 agent 按用户要求协助准备。本轮确实做了
80 条真实视频的小规模流程验收及三方法参考/模板运行，与上面的早期写作工作区分。
不能把这些 agent 检查写成某位组员已经完成的实现、实验或人工审核。

## Teaching exercises update — October 15, 2026

Xinjie asked for a complete version for their own use and a teaching version that tells
each of three members which files to own, what to learn, and which code to fill
in. An agent (Claude) wrote, at Xinjie's request: the three from-scratch
exercise files and their answer key (`students/knn.py`, `students/linear_svm.py`,
`students/random_forest.py`), the TODO H1–H3 exercise in the ported
`scripts/classical_handedness.py`, the self-check tests `tests/test_member*.py`,
the role guides and notes templates in `docs/roles/`, and the generator that
produces the members' repository with the answers removed. The answer key
reproduces the reference exactly for kNN, matches `LinearSVC` weights to about
1e-4 relative difference at C=0.01, and gives a forest with 55–65% Top-1 on the
mini data across seeds (reference 60%). This framework and the answers are not
contributions of the other members; members' contributions start with their own
code, notes, experiments, and writing.
