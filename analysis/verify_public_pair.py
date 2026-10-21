"""Verify published paired records and media without model weights or a GPU."""
import argparse,csv,hashlib,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args()
out=Path(a.root)/'results/paired-evaluation-20261021'
summary=json.loads((out/'paired-summary.json').read_text(encoding='utf-8'))
inputs=json.loads((out/'input-fingerprints.json').read_text(encoding='utf-8'))
assert summary['complete'] and summary['unique_initial_conditions']==30 and summary['executions']==60
groups={}
for group,model in [('base','models/base/model.safetensors'),('finetuned','checkpoints/single-task/model.safetensors')]:
    rows=json.loads((out/group/'episodes.json').read_text(encoding='utf-8'))
    result=json.loads((out/group/'summary.json').read_text(encoding='utf-8'))
    assert len(rows)==30 and [r['official_init_index'] for r in rows]==list(range(10,40))
    assert [r['seed'] for r in rows]==list(range(3000,3030))
    assert result['model_sha256']==inputs['sha256'][model]
    assert result['successes']==sum(r['success'] for r in rows)==summary['summaries'][group]['successes']
    assert result['max_steps']==220 and result['n_action_steps']==10
    groups[group]=rows
counts={k:0 for k in ['both_success','improved','regressed','both_failure']}
for x,y in zip(groups['base'],groups['finetuned']):
    key=('both_success' if x['success'] else 'improved') if y['success'] else ('regressed' if x['success'] else 'both_failure')
    counts[key]+=1
assert counts==summary['paired_counts']
with (out/'paired-episodes.csv').open(encoding='utf-8-sig',newline='') as f:csvrows=list(csv.DictReader(f))
assert len(csvrows)==30
for i,row in enumerate(csvrows):
    assert int(row['official_init_index'])==10+i and int(row['seed'])==3000+i
    assert row['base_success']==str(groups['base'][i]['success'])
    assert row['finetuned_success']==str(groups['finetuned'][i]['success'])
for rel,digest in summary['artifact_sha256'].items():
    with (out/rel).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==digest,rel
print(json.dumps({'saved_records_verified':60,'paired_conditions':30,'published_artifacts_verified':len(summary['artifact_sha256']),
                  'paired_counts':counts,'scope':'Saved records and media only. Original weights, training, and model execution are not re-verified.'},ensure_ascii=False))
