import hashlib
import json
import os
import shutil
from pathlib import Path

import safetensors.torch
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, load_smolvla

from data_pipeline import ROOT, CAMERAS


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make_policy(checkpoint=None, n_action_steps=10):
    torch.set_num_threads(4)
    folder = ROOT / checkpoint if checkpoint else ROOT / 'models/base'
    config = PreTrainedConfig.from_pretrained(folder)
    config.load_vlm_weights = False  # full VLM weights already present in smolvla_base
    config.vlm_model_name = str(ROOT / 'models/processor')
    config.n_action_steps = n_action_steps
    config.push_to_hub = False
    config.device = 'cuda'
    config.input_features = {'observation.state': PolicyFeature(FeatureType.STATE, (8,)),
                             **{key: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for key in CAMERAS}}
    config.output_features = {'action': PolicyFeature(FeatureType.ACTION, (7,))}
    stats = json.loads((ROOT / 'data/statistics.json').read_text())
    stats = {k: {name: torch.tensor(values, dtype=torch.float32) for name, values in value.items()}
             for k, value in stats.items()}
    policy = SmolVLAPolicy(config, dataset_stats=stats)
    policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    weights = folder / 'model.safetensors'
    cache_dir = os.environ.get('SMOLVLA_MODEL_CACHE')
    if cache_dir:
        # mmap across WSL/NTFS is slow; verify once and use a content-addressed native cache.
        digest = sha256(weights)
        cache_path = Path(cache_dir) / f'{digest}.safetensors'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            partial = cache_path.with_suffix('.partial')
            shutil.copyfile(weights, partial)
            if sha256(partial) != digest:
                raise ValueError('Weight cache copy failed integrity check')
            partial.replace(cache_path)
        weights = cache_path
    if checkpoint:
        # Unlike upstream load_smolvla, retain this checkpoint's trained normalization buffers.
        safetensors.torch.load_model(policy, weights, strict=True)
    else:
        load_smolvla(policy, weights)
    return policy.to('cuda')


def to_device(batch):
    return {key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
