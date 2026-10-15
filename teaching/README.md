# 教学版的生成与发布（只在完整版仓库里）

本目录和 `tests/test_teaching_version.py` 不会进入教学版。

## 两个仓库

| | 完整版（本仓库） | 教学版 |
| --- | --- | --- |
| GitHub | `huiliyi-XJ/asl-word-recognition`（私有，只有你） | 建议 `huiliyi-XJ/asl-word-recognition-team`（私有，只邀请组员） |
| 本地 | `project/asl-team-repo/` | `project/asl-team-teaching/` |
| 内容 | 全部答案、讲师说明、完整结果 | 由脚本生成：TODO、提示、自查测试、任务说明 |
| git 历史 | 本仓库自己的历史 | 独立的新历史，从未包含答案 |

**不要把组员邀请进完整版仓库，也不要把完整版的 git 历史推到教学版仓库。** 教学版只能由下面的脚本生成后单独提交。

## 答案怎么标

- Python：`# BEGIN SOLUTION <tag>` … `# END SOLUTION`。教学版里，这一段会变成 `raise NotImplementedError(...)`。如果中间有 `# BEGIN STUDENT`，它后面那几行注释会去掉 `# ` 后作为教学版内容（比如 `IMPLEMENTATION_NOTE` 的占位）。
- Markdown：`<!-- BEGIN SOLUTION <tag> -->` … `<!-- END SOLUTION -->`，教学版里整段删除。
- tag：`member1`、`member2`、`member3`（各自的代码答案和概念题答案）；`results-2026-10`（完整数据上的结果，会剧透 D 部分）；`instructor`（只给你看的说明，永远不发布）。
- **公布答案时**：Markdown 里的概念题答案和结果直接显示在原文档里；代码答案写到教学版的 `answers/<原路径>`（比如 `answers/students/knn.py`），**不覆盖**组员自己写的 `students/knn.py`，方便他们对照差异。

## 常用命令（在本仓库根目录运行）

```bash
# 1. 全部测试（含教学版生成检查）
.venv/bin/python -m unittest discover -s tests

# 2. 生成或更新教学版（默认什么答案都不发布）
.venv/bin/python teaching/make_teaching_version.py --output ../asl-team-teaching

# 3. 某个阶段结束后公布答案，例如成员 1 的代码答案和全量结果
.venv/bin/python teaching/make_teaching_version.py --output ../asl-team-teaching --release member1 --release results-2026-10
```

每次生成会写 `TEACHING_BUILD.json`，记录来源 commit、发布了哪些 tag、每个文件的 SHA256。重新生成时只替换上次生成、之后没被改过的文件。教学版目录里的 `.git`、`data/`、`.venv/` 等本地内容不会动。组员已经合并进教学版的代码（和上次生成时的 SHA 不同）也会保留，并在输出里列出。确实要强制覆盖时才加 `--overwrite-changed`，一般不要用。

已经公布过的 tag 会记在教学版的 `TEACHING_BUILD.json` 里，以后重新生成时自动保持公布，不会被重新隐藏。如果确实要全部重新隐藏，加 `--reset-releases`。

生成后在教学版目录提交并推送：

```bash
cd ../asl-team-teaching
git add -A
git commit -m "Update teaching version"
git push
```

## 第一次发布前的检查

1. 在 GitHub 网页新建**空的私有仓库** `asl-word-recognition-team`。不要勾选 README、.gitignore 或 license，否则第一次推送会冲突。
2. 推送教学版的初始提交。
3. 用组员的 JHU 邮箱或 GitHub 用户名，把他们邀请进**教学版**仓库。
4. 告诉组员先读仓库首页 README，再读 `docs/roles/README.md`。

## 组员代码在完整数据上运行

组员只在 10 类 mini 数据上开发。代码通过自查测试后，由你拉取他们的分支，在完整 train/validation 数据上运行，然后把 `metrics.json`、`per_class.json` 发给他们分析。对外不写具体机器信息。

- kNN、SVM 的自写版本可以直接跑完整数据：SVM 是 L-BFGS，约 280 万个参数。
- 自写 RF 是纯 Python 递归，在 4 万条 × 2,731 类上太慢。成员 3 的完整数据结果用 reference（sklearn）配置跑，这一点在任务说明里已经写明。

## 贡献记录

组员的贡献以他们自己的提交、PR、`docs/roles/memberN-notes.md` 和报告小节为准。框架、参考实现和答案是你（和 AI 协助）完成的，不能算作组员的贡献；最终报告的贡献说明要如实写。
