"""Single-GPU fine-tuning using unchanged upstream SmolVLAPolicy and flow-matching loss."""
import argparse
import json
import math
import random
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from data_pipeline import ROOT, EpisodeDataset
from policy_utils import make_policy, to_device, sha256


def batch_from(dataset, indices):
    return to_device(default_collate([dataset[int(index)] for index in indices]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', default='checkpoints/single-task')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-every', type=int, default=500)
    args = parser.parse_args()
    random.seed(20261020)
    np.random.seed(20261020)
    torch.manual_seed(20261020)
    torch.cuda.manual_seed_all(20261020)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model.safetensors').exists() and not args.resume:
        raise FileExistsError(f'{output} already contains a model; use --resume or a new --output')
    policy = make_policy(args.output if args.resume else None)
    print('Loading training episodes into CPU memory', flush=True)
    dataset = EpisodeDataset('train')
    validation = EpisodeDataset('validation')
    print(f'train={len(dataset)} validation={len(validation)}', flush=True)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
    generator = torch.Generator().manual_seed(20261020)
    start_step = 0
    if args.resume:
        state = torch.load(output / 'training-state.pt', map_location='cpu', weights_only=False)
        if state['model_sha256'] != sha256(output / 'model.safetensors'):
            raise ValueError('Interrupted or mismatched model/training-state checkpoint')
        optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['index_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_step = state['step']
    config = {'steps_target': args.steps, 'batch_size': args.batch_size, 'seed': 20261020,
              'train_frames': len(dataset), 'validation_frames': len(validation),
              'total_parameters': sum(p.numel() for p in policy.parameters()),
              'trainable_parameters': sum(p.numel() for p in trainable),
              'precision': 'frozen VLM bfloat16; action expert float32; CUDA bfloat16 autocast',
              'optimizer': 'AdamW lr=1e-4 betas=(.9,.95), 50-step warmup then cosine to 2.5e-6',
              'frozen_vlm': True, 'chunk_size': 50, 'n_action_steps_eval': 10,
              'resize_imgs_with_padding': list(policy.config.resize_imgs_with_padding),
              'loss': 'unchanged upstream flow matching, including 32-dim padded action loss',
              'padding_mask': 'actions_id_pad (exact upstream spelling at d602e816)',
              'single_task': json.loads((ROOT / 'data/split.json').read_text())['task']}
    (output / 'training-config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    started = time.monotonic()
    policy.train()
    log_path = output / 'metrics.jsonl'
    with log_path.open('a') as log:
        for step in range(start_step + 1, args.steps + 1):
            indices = torch.randint(len(dataset), (args.batch_size,), generator=generator)
            batch = batch_from(dataset, indices)
            warmup = min(1, step / 50)
            cosine = .5 * (1 + math.cos(math.pi * min(1, max(0, (step-50)/max(1, args.steps-50)))))
            lr = warmup * (2.5e-6 + (1e-4-2.5e-6) * cosine)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite training loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 10)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f'nonfinite gradient at step {step}')
            optimizer.step()
            record = {'step': step, 'loss': loss.item(), 'lr': lr, 'grad_norm': float(grad_norm),
                      'elapsed_s': time.monotonic()-started, 'peak_gpu_gb': torch.cuda.max_memory_allocated()/1e9}
            if step == 1 or step % 50 == 0 or step == args.steps:
                # Validation uses separate frames and a forked RNG; it cannot change training randomness.
                with torch.random.fork_rng(devices=[0]), torch.no_grad():
                    torch.manual_seed(7)
                    policy.eval()
                    val_losses = []
                    for vi in range(4):
                        val_indices = [(vi*37 + j*101) % len(validation) for j in range(args.batch_size)]
                        vb = batch_from(validation, val_indices)
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            vl, _ = policy(vb)
                        val_losses.append(vl.item())
                    record['validation_loss'] = float(np.mean(val_losses))
                    policy.train()
                print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if step % args.save_every == 0 or step == args.steps:
                # Serialize many tensors on the Linux filesystem, then copy contiguous files to Windows.
                # Direct tensor-by-tensor serialization to /mnt/d is much slower under WSL.
                staging = Path(os.environ.get('SMOLVLA_CHECKPOINT_STAGING', str(output))) / output.name
                staging.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(staging)
                torch.save({'step': step, 'optimizer': optimizer.state_dict(),
                            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                            'index_rng': generator.get_state(), 'model_sha256': sha256(staging / 'model.safetensors')},
                           staging / 'training-state.pt')
                for filename in ['config.json', 'model.safetensors', 'training-state.pt']:
                    destination = output / filename
                    partial = destination.with_suffix(destination.suffix + '.partial')
                    shutil.copyfile(staging / filename, partial)
                    partial.replace(destination)
                (output / 'completed-step.json').write_text(json.dumps({'step': step}))
                print(f'saved checkpoint step={step}', flush=True)
    print('Training finished', flush=True)


if __name__ == '__main__':
    main()
