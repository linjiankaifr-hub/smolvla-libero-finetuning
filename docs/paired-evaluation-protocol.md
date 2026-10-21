# SmolVLA 新增初始条件配对评测协议

日期：2026-10-21。目的：评估同一任务中微调相对于基座模型的实际收益，补足此前正式10回合与开发3回合的小样本记录。沿用现有评测实现，不重训、不调参、不覆盖历史成绩。

## 预先固定的设置

- 任务：LIBERO Spatial 中把饼干盒旁的黑碗放到盘子上；使用原任务成功谓词。
- 初始状态：官方索引10—39，30个；排除此前正式0—9和开发40—42。
- 随机种子：3000—3029，按顺序与初始状态一一配对。
- 模型：`models/base`与`checkpoints/single-task`的已保存权重；两组使用相同训练集归一化、图像/动作约定。
- 每回合：10步物理稳定等待，最多220控制步，每次预测50步、执行10步后重新观察。
- 运行顺序：先基座，后微调；串行使用GPU。时间仅记录运行开销，不作公平模型速度比较。
- 输出：`results/paired-evaluation-20261021/base`和`finetuned`，每组30回合，共60次执行，只有30个不同初始条件。
- 开始前保存源码、上游版本、模型、统计量及配置的SHA256；结束后复核。结果不足或指纹不符时不发布完整评测结论。

## 统计与失败证据

按相同初始状态与种子配对，报告各组成功数、Wilson 95%区间、成功率差，以及双方成功、仅微调成功、仅基座成功、双方失败的次数。分别报告全部回合步数与成功回合步数，避免提前终止造成误读。

每回合保留视频、实际动作和本体状态轨迹。不增加物体真值作为策略输入；失败原因先通过录像与轨迹定位，无法验证时保留为现象或假设。代表案例按预定顺序选择各类第一个回合，不挑选最好看的案例。

这不是新的训练种子实验，也不是30个严格训练未见布局的证明；RLDS示范与官方初始状态之间的完整对应关系仍缺失。基座与微调模型均适配同一LIBERO输入输出和训练统计量，结果表示本地任务适配前后的表现，不能替代其他基准。

## 命令

在WSL中进入本项目目录，执行：

```bash
bash scripts/stage.sh test
bash scripts/stage.sh evaluate --base --episodes 30 --init-offset 10 --seed 3000 --max-steps 220 --n-action-steps 10 --output results/paired-evaluation-20261021/base
bash scripts/stage.sh evaluate --checkpoint checkpoints/single-task --episodes 30 --init-offset 10 --seed 3000 --max-steps 220 --n-action-steps 10 --output results/paired-evaluation-20261021/finetuned
```

已有逐回合文件时评测入口会拒绝覆盖；中断产物应保留并单独记录，不直接修改历史成功标记。
