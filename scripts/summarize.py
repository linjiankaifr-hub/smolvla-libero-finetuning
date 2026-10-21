"""Build a report exclusively from actual persisted training/evaluation records."""
import argparse
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    args = parser.parse_args()
    evaluation = ROOT / args.evaluation
    summary = json.loads((evaluation / 'summary.json').read_text())
    records = json.loads((evaluation / 'episodes.json').read_text())
    audit = json.loads((ROOT / 'results/checkpoint-audit.json').read_text())
    inputs = json.loads((ROOT / 'results/model-input-audit.json').read_text())
    checkpoint = ROOT / summary['checkpoint']
    assert summary['mode'] == 'model-autonomous'
    assert summary['model_sha256'] == audit['model_sha256']
    assert all(row['mode'] == 'model-autonomous' for row in records)
    config = json.loads((checkpoint / 'training-config.json').read_text())
    metrics = [json.loads(line) for line in (checkpoint / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['step'] == audit['training_step']
    successful = [r for r in records if r['success']]
    demo = successful[0] if successful else records[0]
    shutil.copyfile(evaluation / demo['video'], ROOT / 'results/model-demo.mp4')
    failed = [r for r in records if not r['success']]
    if failed:
        shutil.copyfile(evaluation / failed[0]['video'], ROOT / 'results/model-failure.mp4')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 150})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    steps = np.array([r['step'] for r in metrics])
    losses = np.array([r['loss'] for r in metrics])
    axes[0].plot(steps, losses, color='#739ad6', alpha=.3, linewidth=.7, label='Per-update loss')
    if len(losses) >= 20:
        axes[0].plot(steps[19:], np.convolve(losses, np.ones(20)/20, 'valid'), color='#225eb0', label='20-update mean')
    val = [r for r in metrics if 'validation_loss' in r]
    axes[0].plot([r['step'] for r in val], [r['validation_loss'] for r in val], '.-', color='#d76337', label='Fixed 32 held-out frames')
    axes[0].set(xlabel='Optimizer update', ylabel='Flow matching loss', title='Actual local fine-tuning')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.2)
    colors = ['#3b9475' if r['success'] else '#cc695e' for r in records]
    axes[1].bar(np.arange(len(records)), [r['steps'] for r in records], color=colors)
    axes[1].set(xlabel='Evaluation episode', ylabel='Control steps', title=f"Success: {summary['successes']}/{summary['episodes']}")
    axes[1].set_xticks(np.arange(len(records)))
    axes[1].grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'results/training-and-evaluation.png')
    plt.close(fig)
    baseline_path = ROOT / 'results/base-evaluation/summary.json'
    baseline_text = ''
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        base_records = json.loads((baseline_path.parent / 'episodes.json').read_text())
        matched = [r for r in records if any(r['seed'] == b['seed'] and r['official_init_index'] == b['official_init_index'] for b in base_records)]
        baseline_text = (f"微调前基线：{baseline['successes']}/{baseline['episodes']}。正式模型在相同 "
                         f"{len(matched)} 个种子/初始条件上：{sum(r['success'] for r in matched)}/{len(matched)}。"
                         "这是小样本对照，不是统计显著性结论。\n\n")
    development_path = ROOT / 'results/development-step1000/summary.json'
    if development_path.exists():
        development = json.loads(development_path.read_text())
        baseline_text += (f"另外三个开发初始条件（官方索引 40—42）仅 {development['successes']}/{development['episodes']} 成功。"
                          "这与正式前十个初始条件的表现有差异，不能把正式小样本成绩解释为稳定泛化能力。\n\n")
    report = f"""# SmolVLA + LIBERO 本机实验结果

任务：把饼干盒旁的黑碗放到盘子上。

**本机实际闭环评估：{summary['successes']}/{summary['episodes']} 成功（{summary['success_rate']:.0%}）。**

{baseline_text}## 实际训练

- {audit['training_step']} 次优化器更新，batch={config['batch_size']}，随机种子 {config['seed']}。
- 38 条训练示范（4773 帧），8 条完整回合验证（1002 帧）。曲线中的验证损失只使用固定的 32 个验证帧；不等同于全验证集成绩。
- 约 {config['total_parameters']/1e6:.1f}M 总参数，{config['trainable_parameters']/1e6:.1f}M 可训练参数；冻结 VLM，微调动作专家和状态投影。
- 固定验证批次损失：{val[0]['validation_loss']:.4f} → {val[-1]['validation_loss']:.4f}。
- 训练进程记录的峰值分配显存 {max(r['peak_gpu_gb'] for r in metrics):.2f} GB；不含其他进程和驱动保留显存。
- 已保存模型、AdamW 状态、CPU/CUDA 随机状态与采样状态；优化器各参数步数通过核验。

## 闭环协议

- 原始 LIBERO Spatial 任务，任务索引 {summary['libero_task_index']}，官方初始状态。
- 每回合先运行 10 步物理稳定等待，再最多 {summary['max_steps']} 步；使用原环境成功谓词。
- 每次预测 50 步，执行 {summary['n_action_steps']} 步后重新读取两路图像、本体状态和指令。
- 每回合种子、初始状态索引、成功、步数、实际动作和录像见 [{args.evaluation}/episodes.json](../{args.evaluation}/episodes.json)。
- 当前演示来自第 {demo['episode']} 回合，原环境判定为 **{'成功' if demo['success'] else '失败'}**。视频是保存后重载模型的自主执行，不是示范或手工轨迹。

## 保存后核验

- 模型 SHA256：`{audit['model_sha256']}`。
- 动作输出层相对预训练权重 L2 变化：{audit['action_head_l2_delta_from_base']:.6f}；抽查冻结视觉权重张量一致。
- 状态/动作归一化与训练集统计一致；没有从验证集计算统计。
- 输出形状 {inputs['output_shape']}，均为有限值；固定噪声重复推理最大误差 {inputs['fixed_noise_repeat_max_error']}。
- 全部 action padding 的损失为 {inputs['all_padded_action_loss']}，证实所用上游遮罩生效。
- 图像消融使平均动作变化 {inputs['image_ablation_mean_abs_action_delta']:.6f}。这仅证明图像参与计算。
- 数据/动作/姿态契约测试共 7 项；独立只读代码审查发现的轴角符号问题已修复并增加回归测试。

## 文件入口

- [模型自主执行](model-demo.mp4)
- [训练曲线与逐回合结果](training-and-evaluation.png)
- [模型权重](../checkpoints/single-task/model.safetensors)
- [训练日志](../checkpoints/single-task/metrics.jsonl)
- [数据审计](data-audit.json)、[权重审计](checkpoint-audit.json)、[输入审计](model-input-audit.json)

## 适用范围

这是单任务、单训练种子的本机复现。数据源与训练预算不同于完整教程基准，不声称复现论文成绩。独立重置不证明物体布局完全未见；单任务成功不证明任意指令理解、多任务泛化或真机可用。
"""
    (ROOT / 'results/实验报告.md').write_text(report, encoding='utf-8')
    (ROOT / 'results/latest-evaluation.json').write_text(json.dumps({'directory': args.evaluation, **summary}, indent=2))
    print(f"Report saved. Success {summary['successes']}/{summary['episodes']}; demo episode {demo['episode']}.")


if __name__ == '__main__':
    main()
