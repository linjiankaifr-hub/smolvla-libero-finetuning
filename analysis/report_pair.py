"""Audit and summarize the fixed, existing SmolVLA evaluator's paired outputs."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def wilson(k,n):
    if not 0<=k<=n or n<=0:raise ValueError('Invalid binomial counts')
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,center-half),min(1.,center+half)

def paired(a,b):
    def index(rows):
        d={}
        for row in rows:
            key=(row['official_init_index'],row['seed'])
            if key in d:raise ValueError('Duplicate initial-state/seed pair')
            if type(row['success']) is not bool:raise ValueError('Success flag must be boolean')
            d[key]=row
        return d
    a,b=index(a),index(b)
    if a.keys()!=b.keys() or not a:raise ValueError('Paired conditions differ or are empty')
    table=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        category=('both_success' if x['success'] else 'improved') if y['success'] else ('regressed' if x['success'] else 'both_failure')
        table.append({'official_init_index':key[0],'seed':key[1],'base_success':x['success'],'finetuned_success':y['success'],
                      'base_steps':x['steps'],'finetuned_steps':y['steps'],'category':category,
                      'base_video':f"base/{x['video']}",'finetuned_video':f"finetuned/{y['video']}"})
    counts={k:sum(r['category']==k for r in table) for k in ['both_success','improved','regressed','both_failure']}
    return table,counts

def self_check():
    def row(i,s):return {'official_init_index':i,'seed':100+i,'success':s,'steps':10,'video':f'{i}.mp4'}
    t,c=paired([row(0,False),row(1,True),row(2,True),row(3,False)],
               [row(3,False),row(2,True),row(1,False),row(0,True)])
    assert c=={'both_success':1,'improved':1,'regressed':1,'both_failure':1}
    try:paired([row(0,True)],[row(1,True)])
    except ValueError:pass
    else:raise AssertionError('Mismatched conditions accepted')
    try:paired([row(0,True),row(0,True)],[row(0,True)])
    except ValueError:pass
    else:raise AssertionError('Duplicate conditions accepted')
    lo,hi=wilson(0,30);assert abs(lo)<1e-12 and abs(hi-.1135133931739688)<1e-10
    lo,hi=wilson(30,30);assert abs(lo-.8864866068260312)<1e-10 and abs(hi-1)<1e-12
    print('Pairing, mismatch, duplication, and Wilson boundary checks passed',flush=True)

def main(root):
    import numpy as np
    root=Path(root);out=root/'results/paired-evaluation-20261021'
    protocol=json.loads((out/'input-fingerprints.json').read_text())
    original=protocol['sha256']
    for rel,digest in original.items():
        with (root/rel).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,rel
    data={};summaries={};artifacts={}
    for group in ['base','finetuned']:
        folder=out/group
        rows=json.loads((folder/'episodes.json').read_text());s=json.loads((folder/'summary.json').read_text())
        assert len(rows)==30 and s['episodes']==30
        assert [r['official_init_index'] for r in rows]==list(range(10,40))
        assert [r['seed'] for r in rows]==list(range(3000,3030))
        assert [r['episode'] for r in rows]==list(range(30))
        assert s['successes']==sum(r['success'] for r in rows)
        assert s['success_rate']==s['successes']/30 and s['mode']=='model-autonomous'
        assert s['max_steps']==220 and s['n_action_steps']==10
        assert s['checkpoint']==('models/base' if group=='base' else 'checkpoints/single-task')
        assert s['task']==json.loads((root/'data/split.json').read_text())['task']
        assert all(r['mode']=='model-autonomous' and 1<=r['steps']<=220 for r in rows)
        assert all(r['success'] or r['steps']==220 for r in rows)
        weight='models/base/model.safetensors' if group=='base' else 'checkpoints/single-task/model.safetensors'
        assert s['model_sha256']==original[weight]
        for r in rows:
            video=folder/r['video'];assert video.is_file() and video.stat().st_size>1000
            trajectory=folder/f"episode_{r['episode']:02d}_trajectory.npz"
            with np.load(trajectory) as t:
                assert t['actions'].shape==(r['steps'],7) and t['states'].shape==(r['steps'],8)
                assert np.isfinite(t['actions']).all() and np.isfinite(t['states']).all()
            artifacts[f'{group}/{video.name}']=hashlib.sha256(video.read_bytes()).hexdigest()
            artifacts[f'{group}/{trajectory.name}']=hashlib.sha256(trajectory.read_bytes()).hexdigest()
        successes=s['successes'];successful=[r['steps'] for r in rows if r['success']]
        summaries[group]={'successes':successes,'episodes':30,'wilson95':wilson(successes,30),
                          'mean_steps_all':sum(r['steps'] for r in rows)/30,
                          'mean_steps_successful':sum(successful)/len(successful) if successful else None}
        data[group]=rows
    table,counts=paired(data['base'],data['finetuned'])
    with (out/'paired-episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    representatives={}
    for row in table:
        representatives.setdefault(row['category'],row)
    audit={'complete':True,'input_hashes_unchanged':True,'unique_initial_conditions':30,'executions':60,
           'training_seeds':1,'summaries':summaries,'paired_counts':counts,
           'success_rate_difference_percentage_points':(summaries['finetuned']['successes']-summaries['base']['successes'])/30*100,
           'representatives':representatives,'artifact_sha256':artifacts,
           'unseen_training_layout_claim':False}
    (out/'paired-summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=['Base','Fine-tuned'];rates=[summaries[k]['successes']/30*100 for k in ['base','finetuned']]
    intervals=[summaries[k]['wilson95'] for k in ['base','finetuned']]
    err=np.maximum(0,np.array([[r-lo*100,hi*100-r] for r,(lo,hi) in zip(rates,intervals)]).T)
    axes[0].bar(labels,rates,color=['#8c9aa9','#3b9475'],yerr=err,capsize=5)
    axes[0].set_ylim(0,110);axes[0].set_ylabel('Success (%)');axes[0].set_title('30 matched initial conditions | Wilson 95% CI')
    axes[1].bar(['Both\nsuccess','Improved','Regressed','Both\nfailure'],list(counts.values()),color=['#3b9475','#4c8ec1','#d5835d','#8c9aa9'])
    axes[1].set_ylabel('Paired conditions');axes[1].set_ylim(0,32);axes[1].set_title('Report improvements and regressions together')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'paired-comparison.png',dpi=160);plt.close(fig)
    statrows=[]
    for k in ['base','finetuned']:
        s=summaries[k];lo,hi=s['wilson95'];good='—' if s['mean_steps_successful'] is None else f"{s['mean_steps_successful']:.1f}"
        statrows.append(f"| {'基座模型' if k=='base' else '微调模型'} | {s['successes']}/30 | {lo*100:.1f}%—{hi*100:.1f}% | {s['mean_steps_all']:.1f} | {good} |")
    examples='\n'.join(f"- {k}：初始状态{r['official_init_index']}，seed{r['seed']}；[基座视频]({r['base_video']})、[微调视频]({r['finetuned_video']})。" for k,r in representatives.items())
    report=f'''# SmolVLA 新增30个初始条件配对评测

日期：2026-10-21。固定已保存的基座与1000步微调模型，不重训。使用官方初始状态10—39、种子3000—3029，两个模型逐一配对。旧正式0—9与开发40—42的成绩不合并进本轮。

成功沿用原LIBERO任务判据，没有额外要求夹爪退出或长时间稳定；不能与ACT项目更严格的放置判据直接比较成功率。

## 完整成绩

| 模型 | 成功次数 | Wilson 95%区间 | 全部回合平均步数 | 成功回合平均步数 |
|---|---:|---|---:|---:|
{chr(10).join(statrows)}

微调相对基座成功率差为{audit['success_rate_difference_percentage_points']:.1f}个百分点。全部回合均值将失败计为220步，因此受到超时截断影响；成功回合均值只描述各自成功子集，不能当作同一任务集合上的模型速度比较。

| 配对结果 | 个数 |
|---|---:|
| 两模型都成功 | {counts['both_success']} |
| 基座失败、微调成功 | {counts['improved']} |
| 基座成功、微调失败 | {counts['regressed']} |
| 两模型都失败 | {counts['both_failure']} |

![配对评测](paired-comparison.png)

## 可核验的案例

各类按初始状态从小到大选择第一个回合，未根据展示效果挑选。

{examples}

完整配对清单见[逐回合CSV](paired-episodes.csv)，全部摘要、视频/轨迹指纹见[审计JSON](paired-summary.json)。每组保留所有视频及逐步动作、本体状态轨迹；基座与微调分别使用同一输入输出适配和训练集统计量。

## 结论边界

这轮补充回答“在新增官方初始条件中，现有微调是否比基座更有效”，不证明任意布局、多任务、语言理解或真机能力。RLDS示范与官方初始状态映射仍缺失，不能称为严格训练未见布局测试。60次执行对应30个条件，不是60个独立场景；每条件只有一个推理种子，也没有覆盖训练随机性。

Wilson区间按二项抽样作近似描述；本轮使用固定的官方初始条件子集，并非对所有场景的随机抽样，因此区间不能保证总体泛化覆盖，也不覆盖训练随机性。

此前9/10正式成绩和1/3开发成绩保留；本轮应与它们分别报告。不会从这30个条件中挑选成功子集重新计算，也不使用本轮结果选择新模型后继续当作独立测试。

本轮数据与动作契约测试7项通过；模型、统计量、配置、评测源码和历史成绩指纹在运行前后保持一致。原评测器未改动。协议见项目docs/paired-evaluation-protocol.md，输入指纹见input-fingerprints.json。
'''
    (out/'实验报告.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='artifact_sha256'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check()
    if a.root:main(a.root)
