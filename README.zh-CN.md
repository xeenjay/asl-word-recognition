# ASL 词级识别 — 团队工作区

[English](README.md) · [环境与运行](docs/SETUP.md) · [方法](docs/METHODS.md) · [代码导航](docs/CODE_MAP.md) · [结果](docs/RESULTS.md) · [协作](docs/TEAM_GUIDE.md)

这个仓库让组员按 **跑真实数据参考实现 → 理解方法 → 修改学生代码 → 运行并比较结果** 的顺序工作。任务是“一段已经剪好的单词手语视频 → 一个词标签”，不做连续句子翻译。

三个传统方法 **kNN、Linear SVM、Random Forest** 共享 1,020 维姿态/运动特征。三位组员各自在 `students/` 里从零实现其中一个，有 TODO、提示和自查测试引导。

**从这里开始：[成员任务](docs/roles/README.md)**（英文），写明每个角色要学什么、写什么、怎么自查、做什么实验、交什么，以及建议时间表。

## 开始运行

先接受私有仓库邀请，用自己的 GitHub 账号认证。也可以登录网页后 **Code → Download ZIP**，解压并进入项目目录运行。

```bash
git clone https://github.com/huiliyi-XJ/asl-word-recognition-team.git
cd asl-word-recognition-team
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
```

使用 Python 3.10 或 3.11，环境细节见 [Setup](docs/SETUP.md)。

先自行阅读 [ASL Citizen 数据许可](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/)。接受条款后，准备真实小样本：

```bash
python scripts/prepare_team_data.py --accept-dataset-license
```

脚本从官方来源直接获取元数据和 **最多 80 条选定视频**，再在本地提取数字特征：按 gloss 排序的前 10 类，每类最多 train 6 条、validation 2 条，不足时保留该组所有可用样本；各类各 split 内按文件名 hash 确定样本，保留官方划分。

Git 和 GitHub Release 不重新分发数据/特征包。首次需要联网下载和 CPU 提取，后续方法运行复用 `data/classical-mini/`。**不需要每位同学先下载约 31 GB 全量 train/validation 视频**，也不需要 GPU、PoET 权重或 vendor 源码。

先跑参考实现：

```bash
python run.py --method knn --implementation reference --data data/classical-mini --output outputs/knn-reference
```

按任务说明填写自己角色文件里的 TODO（例如 `students/knn.py`），先自查：

```bash
python -m unittest tests.test_member1_knn -v
```

再用同一个运行器运行：

```bash
python run.py --method knn --implementation student --data data/classical-mini --output outputs/knn-student
```

SVM/RF 分别把 `knn` 换成 `linear_svm` / `random_forest`，编辑对应文件。默认使用 `configs/<method>.json`，也可用 `--config` 指定设置。直接比较实现时保持数据和参数一致。

每次必须用**新的输出目录**。比较 `metrics.json`、`per_class.json` 和预测文件；`run.json` 记录代码来源及是否委托 reference。原样运行模板只是使用接口，不是已经自己写完算法。

## 修改代码前要知道

运行器只在 train 上拟合 StandardScaler，再把已标准化的 X 交给学生方法。不要在 validation 上重新拟合，也不要删掉难例。接口说明在 [Team guide](docs/TEAM_GUIDE.md)，代码位置在 [Code map](docs/CODE_MAP.md)。

这个 10 类 mini 是**小规模流程验收和编程练习**，不能可靠估计完整任务效果，也不能直接和历史 2,731 类分数比较。

历史完整实验为 40,154 train / 10,304 validation：kNN **18.6821%**、SVM **38.7616%**、受限 RF **16.3141%**、官方 PoET **80.9394%**。PoET 的输入和预训练不同，因此这是完整系统比较。

PoET 内部三 seed 对照另列：微调 B **81.1853%** → G+R **81.7352%**，约 **+0.55 个百分点**。B 与官方原始 checkpoint 不同；这些是 validation 结果。32,941 条最终 test 已于 2026-10-12 按事前协议只评价一次，见 [Results](docs/RESULTS.md)。

便携提取使用 MediaPipe model complexity 1 和全部解码帧；历史实验使用 complexity 2 及其记录的预处理协议。虽然描述符都是 1,020 维，特征数值不保证相同，因此新全量运行也不能承诺精确复现历史分数。

全量数据准备需要显式选择，见 [Setup](docs/SETUP.md)：约 31 GB 原视频及额外处理/存储。教学 runner 默认单 worker 并限制数值库线程，全量 SVM 有 2,731 个二元拟合任务，CPU 耗时和内存不能按历史并行 run 猜测。先用 mini。

## 阅读与团队协作

- 特征和算法：[Methods](docs/METHODS.md)。
- 完整分析：[英文报告](reports/report-en.md)、[中文报告](reports/report-zh.md)、[Results](docs/RESULTS.md)。
- 已完成、待完成及验收：[Status](docs/STATUS.md)。
- 记录本人实际工作：[Team guide](docs/TEAM_GUIDE.md)、[AI 使用记录](docs/AI_USE_AND_CONTRIBUTIONS.md)。

团队为 Xinjie Huang、Duncan Stadler、Leyun Kim、Ming Chen；组员 1–3 各负责一个传统方法，谁担任哪个角色由团队决定。Xinjie 在 agent 协助下组织初始研究，参考代码不能自动算作某位队友已完成的贡献。使用成熟库/参考实现应如实注明，再记录自己的实现、实验、分析和写作。

仓库包含文字报告与统计图，不提交人物视频/帧、权重、完整缓存、PDF/PPTX、密钥或私人机器信息。PoET 完整复现仍需独立资产与实验协议，不是当前教学入口的一键功能。
