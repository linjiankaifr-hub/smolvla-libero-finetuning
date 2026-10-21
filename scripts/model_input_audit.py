"""Exercise the saved policy and checkpoint normalization with fixed-noise inputs."""
import argparse
import json

import h5py
import torch

from data_pipeline import ROOT, CAMERAS
from policy_utils import make_policy, to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/single-task')
    args = parser.parse_args()
    policy = make_policy(args.checkpoint).eval()
    task = json.loads((ROOT / 'data/split.json').read_text())['task']
    with h5py.File(ROOT / 'data/episodes.h5', 'r') as h5:
        episode = h5['350']  # held-out episode
        batch = {'observation.state': torch.from_numpy(episode['state'][0]).unsqueeze(0), 'task': [task]}
        for key in CAMERAS:
            batch[key] = torch.from_numpy(episode[key][0]).permute(2, 0, 1).unsqueeze(0).float()/255
        batch['action'] = torch.from_numpy(episode['action'][:50]).unsqueeze(0)
        batch['actions_id_pad'] = torch.ones((1, 50), dtype=torch.bool)
    batch = to_device(batch)
    torch.manual_seed(123)
    noise = torch.randn(1, 50, 32, device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        # Upstream Euler sampling mutates the supplied noise tensor in place.
        output = policy.predict_action_chunk(batch, noise=noise.clone()).float()
        repeated = policy.predict_action_chunk(batch, noise=noise.clone()).float()
        ablated = dict(batch)
        for key in CAMERAS:
            ablated[key] = torch.zeros_like(batch[key])
        black_images = policy.predict_action_chunk(ablated, noise=noise.clone()).float()
        different_language = dict(batch, task=['put the black bowl in the drawer'])
        other_instruction = policy.predict_action_chunk(different_language, noise=noise.clone()).float()
        loss, _ = policy(batch, noise=noise.clone(), time=torch.tensor([.5], device='cuda'))
    assert output.shape == (1, 50, 7) and torch.isfinite(output).all()
    assert torch.equal(output, repeated), 'Same saved checkpoint and noise are not reproducible'
    assert loss.item() == 0, 'Upstream episode padding mask is not active'
    image_delta = (black_images-output).abs().mean().item()
    language_delta = (other_instruction-output).abs().mean().item()
    assert image_delta > 1e-7, 'Image ablation did not change actions'
    report = {'checkpoint': args.checkpoint, 'output_shape': list(output.shape), 'finite_actions': True,
              'fixed_noise_repeat_max_error': (output-repeated).abs().max().item(),
              'all_padded_action_loss': loss.item(), 'image_ablation_mean_abs_action_delta': image_delta,
              'language_change_mean_abs_action_delta': language_delta,
              'interpretation': 'Input sensitivity and technical contract only; not proof of semantic language understanding.'}
    (ROOT / 'results/model-input-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
