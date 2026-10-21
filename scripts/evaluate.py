import argparse
import json
import time

import imageio.v2 as imageio
import numpy as np
import torch

from data_pipeline import ROOT, to_environment_action
from environment import make_environment, observation_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/single-task')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--init-offset', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=220)
    parser.add_argument('--n-action-steps', type=int, default=10)
    parser.add_argument('--output', default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--base', action='store_true', help='Evaluate base SmolVLA using the same training-only LIBERO statistics')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        raise ValueError('episodes must be 1..50')
    out = ROOT / (args.output or f'results/eval-{time.strftime("%Y%m%d-%H%M%S")}')
    if out.exists() and (out / 'episodes.json').exists():
        raise FileExistsError('Choose a new --output to preserve previous evaluation')
    out.mkdir(parents=True, exist_ok=True)
    environment, initial_states, language, task_index = make_environment(args.seed)
    policy = None
    checkpoint_hash = None
    if not args.smoke:
        from policy_utils import make_policy, sha256
        policy = make_policy(None if args.base else args.checkpoint, n_action_steps=args.n_action_steps).eval()
        checkpoint_hash = sha256(ROOT / ('models/base' if args.base else args.checkpoint) / 'model.safetensors')
    records = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            environment.seed(seed)
            environment.reset()
            init_index = args.init_offset + episode
            if not 0 <= init_index < len(initial_states):
                raise IndexError('Requested initial state is outside the official list')
            observation = environment.set_init_state(initial_states[init_index])
            for _ in range(10):
                observation, _, _, _ = environment.step([0, 0, 0, 0, 0, 0, -1])
            if policy is not None:
                policy.reset()
            file = out / f'episode_{episode:02d}.mp4'
            writer = imageio.get_writer(file, fps=20, codec='libx264', quality=7)
            started = time.monotonic()
            success = False
            actions = []
            states = []
            try:
                for step in range(3 if args.smoke else args.max_steps):
                    view = np.ascontiguousarray(observation['agentview_image'][::-1, ::-1])
                    writer.append_data(view)
                    batch = observation_batch(observation, language)
                    states.append(batch['observation.state'][0].cpu().numpy())
                    if policy is None:
                        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                    else:
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            raw_action = policy.select_action(batch)[0].float().cpu().numpy()
                        action = to_environment_action(raw_action)
                    actions.append(action)
                    observation, reward, done, _ = environment.step(action.tolist())
                    if done:
                        success = True
                        break
                writer.append_data(np.ascontiguousarray(observation['agentview_image'][::-1, ::-1]))
            finally:
                writer.close()
            record = {'episode': episode, 'seed': seed, 'official_init_index': init_index,
                      'success': success, 'steps': step+1, 'elapsed_s': time.monotonic()-started,
                      'video': file.name, 'mode': 'environment-smoke' if args.smoke else 'model-autonomous'}
            records.append(record)
            np.savez_compressed(out / f'episode_{episode:02d}_trajectory.npz', actions=np.stack(actions), states=np.stack(states))
            (out / 'episodes.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
        summary = {'task': language, 'libero_suite': 'libero_spatial', 'libero_task_index': task_index,
                   'episodes': len(records), 'successes': sum(r['success'] for r in records),
                   'success_rate': sum(r['success'] for r in records)/len(records),
                   'checkpoint': 'models/base' if args.base else args.checkpoint, 'model_sha256': checkpoint_hash,
                   'max_steps': args.max_steps, 'n_action_steps': args.n_action_steps,
                   'mode': 'environment-smoke' if args.smoke else 'model-autonomous',
                   'protocol': f'official supplied initial states; 10 settling steps; max {args.max_steps} control steps; original task success predicate',
                   'generalization_limit': 'Different environment resets, but RLDS demo-to-official-init mapping is unavailable; unseen-object-layout independence is not established.'}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        environment.close()


if __name__ == '__main__':
    main()
