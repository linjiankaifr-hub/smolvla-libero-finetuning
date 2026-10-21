# SmolVLA + LIBERO：单任务机器人大模型复现

任务：根据两路 RGB、8 维机械臂本体状态和英语任务指令，把饼干盒旁的黑碗放到盘子上。

**本机已完成 1000 次实际微调，正式 10 回合评估 9/10 成功。** 微调前相同前三个条件为 0/3，微调后为 3/3；另外三个开发条件为 1/3，说明对初始条件仍然敏感。完整记录见 [实际实验报告](results/实验报告.md)，成功与失败录像分别为 [模型演示](results/model-demo.mp4)、[失败回合](results/model-failure.mp4)。这份成绩只对应记录的单任务小样本评估。

这是 Every-Embodied 教程的本机单任务适配，使用官方 `lerobot/smolvla_base` 预训练权重与未修改的 LeRobot SmolVLA 策略。训练、仿真和评测结果以 `results` 内的实际记录为准。不是完整 LIBERO 基准或论文成绩复现。

## 运行入口

在本目录的 PowerShell 中：

```powershell
.\run.ps1 -Task status
.\run.ps1 -Task test
.\run.ps1 -Task audit
.\run.ps1 -Task evaluate -Episodes 10
.\run.ps1 -Task demo
```

`evaluate` 每次新建带时间戳的结果目录，不覆盖已保存成绩。`demo` 打开 `results/model-demo.mp4`，仅在真实模型评测并汇总后提供该文件。

训练（较长时间）：

```powershell
# 默认目录已有模型时会拒绝覆盖，选择新目录重训。
.\run.ps1 -Task train -Steps 1000 -BatchSize 8 -Checkpoint checkpoints/my-run
# 从匹配的模型和优化器状态继续。
.\run.ps1 -Task train -Steps 1500 -BatchSize 8 -Checkpoint checkpoints/my-run -Resume
```

## 数据与模型

- 任务英文：`pick up the black bowl next to the cookie box and place it on the plate`。
- 数据是公开的 `aopolin-lv/libero_spatial_no_noops_lerobot_v21`。原教程的 `nikriz/...` 下载地址在本次访问中返回 401，因此改用同为 v2.1 的公开 Spatial 子集；没有使用账号凭据。
- 46 条该任务示范，38 条训练、8 条验证；按完整回合划分。训练 4773 帧、验证 1002 帧。归一化只用训练回合统计。
- 观测：两路 256×256 RGB、末端位置 3 维、轴角 3 维、夹爪关节位置 2 维、任务文本。没有物体真值坐标输入策略。
- 动作：末端平移 3 维、末端旋转 3 维、夹爪 1 维；不是七个关节角。
- 官方 SmolVLA 约 4.50 亿参数，其中约 9988 万参数参与训练。冻结 VLM，微调动作专家与状态投影；保留官方 Flow Matching 损失。
- 预测 50 步动作块，评测每执行 10 步重新观察。两路相机在仿真输入处旋转 180°，与下载示范核对一致。
- 数据夹爪 0=闭合、1=张开，转环境动作使用 `1-2*g`。保留源数据轴角大于 π 的表示，避免姿态折回产生分布错位。

## 本机环境

Windows + WSL Ubuntu 24.04，RTX 4060 Laptop 8GB；独立 Python 环境 `/home/buntu/.venvs/codex-smolvla-libero`，Python 3.12，Torch 2.7.1、Transformers 4.51.3、Robosuite 1.4.0、MuJoCo 3.3.7。完整版本见 `requirements-lock.txt`。

LeRobot 固定为教程指定的 `d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8`。模型、数据和 LIBERO 源码版本见 `sources.json`。

渲染使用本机 WSLg/GLFW/D3D12 配置；它不是通用 Linux 配置。为避免 WSL/NTFS 大量小块读取开销，数据、内容寻址权重缓存和 checkpoint 暂存位于 `/home/buntu/.cache/codex-smolvla-libero`；最终模型、数据、日志与录像保存在本项目中。原有七个实战环境没有被修改。

重新安装当前机器环境：先确保 `official/lerobot`、`official/LIBERO` 在所记录提交，然后执行 `run.ps1 -Task setup`、`download`、`prepare`。安装脚本复用本机已存在的 uv 和图形库，不是全新系统的一键安装器。

## 结果边界

验证损失衡量离线动作拟合，不能替代闭环成功率。闭环以原始 LIBERO 任务成功判定为准，每回合最多 220 步，另有 10 步物理稳定等待。

评测重置独立于训练采样，但该 RLDS 数据没有公开示范到官方初始状态的对应关系，因此不声称测试物体布局绝对未在示范中出现。单任务训练也不能证明模型理解任意中文或英语指令、具备多任务泛化或能够直接部署真机。

官方环境空动作测试保存在 `results/environment-smoke`，它不是模型成绩。3 步训练探测保存在 `checkpoints/probe`，它不是正式模型。

## 文件

| 路径 | 内容 |
|---|---|
| `data/split.json`、`statistics.json` | 完整回合划分、训练集归一化 |
| `data/raw`、`episodes.h5` | 下载示范和无损解码后的训练数据 |
| `models/base`、`processor` | 固定版本预训练权重和分词器 |
| `checkpoints/single-task` | 正式微调权重、优化器状态、每步日志 |
| `results/data-audit.json` | 数据维度、动作范围、划分审计 |
| `results/input-preview.jpg` | 训练与仿真相机输入对照 |
| `results/checkpoint-audit.json` | 实际更新、权重与归一化审计 |
| `results/eval-*` | 逐回合成绩、动作轨迹、模型录像 |
| `scripts`、`tests` | 本地适配、运行脚本和契约测试 |

## 上游来源

- [Every-Embodied 教程](https://github.com/datawhalechina/every-embodied/tree/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/01SmolVLA-LIBERO)
- [LeRobot](https://github.com/huggingface/lerobot) 与 [SmolVLA 权重](https://huggingface.co/lerobot/smolvla_base)：保留上游许可证和模型卡。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)：保留上游代码、资产及许可证。
- [公开 Spatial 数据集](https://huggingface.co/datasets/aopolin-lv/libero_spatial_no_noops_lerobot_v21)。
