"""Audit actual saved weights, optimizer updates and data provenance without a GPU."""
import json
import subprocess
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch

from data_pipeline import ROOT
from policy_utils import sha256


def main():
    checkpoint = ROOT / 'checkpoints/single-task'
    final_weights = checkpoint / 'model.safetensors'
    stats = json.loads((ROOT / 'data/statistics.json').read_text())
    with safe_open(final_weights, framework='pt') as final:
        for prefix, feature in [('normalize_inputs.buffer_observation_state', 'observation.state'),
                                ('normalize_targets.buffer_action', 'action'),
                                ('unnormalize_outputs.buffer_action', 'action')]:
            for stat in ['mean', 'std']:
                np.testing.assert_allclose(final.get_tensor(f'{prefix}.{stat}').numpy(), stats[feature][stat], atol=1e-7)
        with safe_open(ROOT / 'models/base/model.safetensors', framework='pt') as base:
            key = 'model.action_out_proj.weight'
            delta = (final.get_tensor(key).float() - base.get_tensor(key).float()).norm().item()
            assert delta > 0, 'No action-head update detected'
            frozen_key = next(k for k in final.keys() if '.vision_model.' in k and k.endswith('weight'))
            frozen_equal = torch.equal(final.get_tensor(frozen_key).bfloat16(), base.get_tensor(frozen_key).bfloat16())
            assert frozen_equal, 'Frozen vision weights changed'
    state = torch.load(checkpoint / 'training-state.pt', map_location='cpu', weights_only=False)
    model_hash = sha256(final_weights)
    assert state['model_sha256'] == model_hash
    steps = [int(value['step']) for value in state['optimizer']['state'].values() if 'step' in value]
    assert steps and min(steps) == max(steps) == state['step']
    manifest = json.loads((ROOT / 'sources.json').read_text())
    for name in ['lerobot', 'LIBERO']:
        directory = ROOT / 'official' / name
        revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        modified = subprocess.check_output(['git', '-C', str(directory), 'diff', '--name-only'], text=True).strip()
        assert not modified, f'Unexpected upstream modifications: {modified}'
        manifest[name] = {'revision': revision, 'unmodified_tracked_sources': True}
    (ROOT / 'sources.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'scripts').glob('*')) if p.is_file()}
    audit = {'training_step': state['step'], 'optimizer_parameter_steps': [min(steps), max(steps)],
             'model_sha256': model_hash, 'action_head_l2_delta_from_base': delta,
             'frozen_vision_tensor_unchanged': frozen_equal, 'normalization_matches_training_only': True,
             'source_fingerprints': fingerprints}
    (ROOT / 'results/checkpoint-audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
