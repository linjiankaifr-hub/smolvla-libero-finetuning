"""Unmodified LIBERO task with explicit observation/action conversion."""
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from data_pipeline import ROOT, quaternion_to_axis_angle


def make_environment(seed):
    base = ROOT / 'official/LIBERO/libero/libero'
    config = {'benchmark_root': str(base), 'bddl_files': str(base / 'bddl_files'),
              'init_states': str(base / 'init_files'), 'datasets': str(ROOT / 'data'), 'assets': str(base / 'assets')}
    config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / 'config.yaml').write_text(yaml.safe_dump(config))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    language = json.loads((ROOT / 'data/split.json').read_text())['task']
    task_index = next(i for i in range(suite.n_tasks) if suite.get_task(i).language == language)
    task = suite.get_task(task_index)
    environment = OffScreenRenderEnv(bddl_file_name=str(base / 'bddl_files' / task.problem_folder / task.bddl_file),
                                      camera_heights=256, camera_widths=256)
    environment.seed(seed)
    # Explicit weights_only=False is needed for the upstream numpy init-state pickle.
    states = torch.load(base / 'init_files' / task.problem_folder / task.init_states_file,
                        map_location='cpu', weights_only=False)
    return environment, states, language, task_index


def observation_batch(observation, language):
    state = np.concatenate([observation['robot0_eef_pos'],
                            quaternion_to_axis_angle(observation['robot0_eef_quat']),
                            observation['robot0_gripper_qpos']]).astype(np.float32)
    # Follow Every-Embodied/OpenVLA LIBERO preprocessing: rotate both views by 180 degrees.
    batch = {'observation.state': torch.from_numpy(state).unsqueeze(0).cuda(), 'task': [language]}
    for target, source in [('observation.images.image', 'agentview_image'),
                           ('observation.images.wrist_image', 'robot0_eye_in_hand_image')]:
        rgb = np.ascontiguousarray(observation[source][::-1, ::-1])
        batch[target] = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    return batch
