# 来源与贡献边界

- LeRobot：https://github.com/huggingface/lerobot ，Apache-2.0，固定提交见 sources.json。
- LIBERO：https://github.com/Lifelong-Robot-Learning/LIBERO ，上游仓库许可证及资产条款适用，固定提交见 sources.json。
- SmolVLA：https://huggingface.co/lerobot/smolvla_base ，使用其模型卡和许可证；未重新分发权重。
- 数据：https://huggingface.co/datasets/aopolin-lv/libero_spatial_no_noops_lerobot_v21 ，下载使用须遵守源数据条款；未重新分发原始示范。
- 教程：https://github.com/datawhalechina/every-embodied ，参考其 SmolVLA-LIBERO 教程。

本地贡献范围：单任务数据筛选和回合划分、归一化、图像与动作约定适配、训练入口、检查点审计、闭环评测、结果报告及契约测试。代码由 Codex 协助完成。未声称原创 SmolVLA、Flow Matching、LIBERO 或通用机器人模型。

上游源码、模型、数据不包含在本仓库；恢复时保留各自 LICENSE 和 NOTICE。
