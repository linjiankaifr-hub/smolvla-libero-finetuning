# SmolVLA LIBERO 单任务微调与评测

基于 LeRobot SmolVLA 与 LIBERO 的仿真实验：从双相机 RGB、八维机器人本体状态和英语任务指令生成七维末端动作，把饼干盒旁的黑碗放到盘子上。

本仓库提供数据适配、微调、评测和审计代码，以及本地实验记录。代码由 Codex 协助完成；使用官方策略及预训练模型，没有从零训练视觉语言基础模型，也未部署真机。

## 新增30个初始条件配对评测

2026-10-21：固定已有权重、不重训，使用官方初始状态10—39与相同种子3000—3029，串行比较任务适配前后的模型。

| 模型 | 同条件成功次数 |
|---|---:|
| 基座模型（使用本任务输入输出适配及训练集统计） | 0/30 |
| 1000步微调模型 | 18/30 |

此前正式0—9的9/10、开发40—42的1/3与这轮分别报告，不混成一个成功率。新增条件不是严格训练未见布局的证明；只有一个训练种子。

[完整实验报告](results/paired-evaluation-20261021/实验报告.md) · [30组配对CSV](results/paired-evaluation-20261021/paired-episodes.csv) · [完整审计JSON](results/paired-evaluation-20261021/paired-summary.json) · [实验讲述](docs/paired-evaluation-interview.md)

![配对实验](results/paired-evaluation-20261021/paired-comparison.png)

[改善案例对照图](results/paired-evaluation-20261021/improved-contact-sheet.png) · [共同失败案例对照图](results/paired-evaluation-20261021/both_failure-contact-sheet.png)

全部60回合视频和实际动作／状态轨迹随仓库发布。无需模型或GPU即可核对公开记录与媒体：在仓库根目录用Python 3.11或以上运行 `python analysis/verify_public_pair.py --root .`。这只核验保存记录的一致性，不代表重跑模型或验证原始训练；完整报告的生成还要求本地原模型与输入文件。

## 原10回合记录与训练配置

[模型自主成功视频](results/model-demo.mp4) · [失败视频](results/model-failure.mp4)

![训练和评测](results/training-and-evaluation.png)

| 配置或测试 | 实际记录 |
|---|---|
| 数据 | 46条示范，38条训练、8条验证 |
| 更新预算 | 1000次，batch=8，单训练种子 |
| 模型 | 约450M参数；冻结VLM，更新约99.9M参数 |
| 正式评测 | 初始状态0—9，9/10成功 |
| 额外开发条件 | 初始状态40—42，1/3成功 |
| 相同前三条件对照 | 微调前0/3，微调后3/3 |

这是单任务小样本结果，不是完整 LIBERO 基准；未证明测试布局与全部示范完全独立，也未验证任意指令泛化。详见 [实验报告](results/实验报告.md) 和 [逐回合记录](results/evaluation-step1000/episodes.json)。

## 复现与代码入口

权重、优化器、原始数据、解码帧及上游源码未上传，需下载预训练模型和公开数据，再执行微调；没有提供已训练模型下载链接。

1. 在 Windows + WSL Ubuntu 24.04 或 Linux 中准备 Python 3.12、uv 和图形依赖。
2. 按 `sources.json` 的提交号克隆 [LeRobot](https://github.com/huggingface/lerobot) 到 `official/lerobot`，克隆 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 到 `official/LIBERO` 并切换对应提交。
3. 阅读 [本机运行说明](LOCAL_EXPERIMENT.md)，调整 `scripts/setup.sh`、`common.sh` 的本机 uv、WSLg/D3D12、缓存和暂存路径。安装脚本依赖既有环境，不是全新系统一键安装器。
4. 依次运行 `run.ps1 -Task setup`、`download`、`prepare`、`train`；生成检查点后执行 `audit`、`evaluate`。精确配置和结果边界见本机说明。

主要实现：`scripts/data_pipeline.py`、`train.py`、`evaluate.py`、`environment.py`、`policy_utils.py`；契约测试在 `tests/`。来源、许可证与贡献边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。

