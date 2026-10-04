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
