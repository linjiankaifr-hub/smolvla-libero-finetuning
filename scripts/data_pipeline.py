"""Explicit single-task LIBERO data contract, retaining official LeRobot tensors."""
import json
import os
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ['observation.images.image', 'observation.images.wrist_image']


def future_chunk(actions, index, size):
    indices = np.arange(index, index + size)
    mask = indices >= len(actions)
    return actions[np.minimum(indices, len(actions) - 1)].copy(), mask


def training_statistics(episodes, ids):
    result = {}
    for field, feature in [('state', 'observation.state'), ('action', 'action')]:
        values = np.concatenate([episodes[ep][field] for ep in ids]).astype(np.float32)
        result[feature] = {'mean': values.mean(0).tolist(),
                           'std': np.maximum(values.std(0), 1e-4).tolist()}
    return result


def to_environment_action(action):
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError('Expected seven finite action values')
    # RLDS dataset: 0=closed, 1=open. Robosuite: -1=open, +1=closed.
    result[-1] = 1.0 - 2.0 * result[-1]
    return np.clip(result, -1, 1)


def quaternion_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    # Preserve quaternion sign: the RLDS axis-angle state includes angles > pi.
    denominator = np.sqrt(max(0, 1 - np.clip(quat[3], -1, 1) ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2 * np.arccos(np.clip(quat[3], -1, 1)) / denominator)).astype(np.float32)


def prepare():
    split = json.loads((ROOT / 'data/split.json').read_text())
    raw = ROOT / 'data/raw'
    output = ROOT / 'data/episodes.h5'
    if output.exists():
        print('Decoded dataset already exists; preserving it.')
        return
    episodes = {}
    partial = output.with_suffix('.partial.h5')
    with h5py.File(partial, 'w') as h5:
        h5.attrs['task'] = split['task']
        for ep in split['episodes']:
            table = pq.read_table(raw / f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')
            state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
            actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            tasks = np.asarray(table['task_index'].to_pylist())
            assert np.all(tasks == split['task_index_in_dataset']), 'Unexpected task label'
            assert state.shape == (len(actions), 8) and actions.shape[1] == 7
            assert np.isfinite(state).all() and np.isfinite(actions).all()
            group = h5.create_group(str(ep))
            group.create_dataset('state', data=state)
            group.create_dataset('action', data=actions)
            episodes[ep] = {'state': state, 'action': actions}
            for key in CAMERAS:
                path = raw / f'videos/chunk-{ep//1000:03d}/{key}/episode_{ep:06d}.mp4'
                with av.open(str(path)) as container:
                    frames = np.stack([frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)])
                assert len(frames) == len(actions), (ep, key, len(frames), len(actions))
                group.create_dataset(key, data=frames, compression='lzf', chunks=(1, 256, 256, 3))
            print(f'decoded episode {ep}: {len(actions)} frames', flush=True)
    statistics = training_statistics(episodes, split['train'])
    (ROOT / 'data/statistics.json').write_text(json.dumps(statistics, indent=2))
    audit = {'task': split['task'], 'train_episodes': len(split['train']),
             'validation_episodes': len(split['validation']),
             'train_frames': sum(len(episodes[ep]['state']) for ep in split['train']),
             'validation_frames': sum(len(episodes[ep]['state']) for ep in split['validation']),
             'state_dim': 8, 'action_dim': 7, 'cameras': CAMERAS,
             'images': 'RGB uint8 256x256, video decode order, no flips at training',
             'statistics_from': split['train'], 'held_out_disjoint': not bool(set(split['train']) & set(split['validation'])),
             'action_min': np.concatenate([v['action'] for v in episodes.values()]).min(0).tolist(),
             'action_max': np.concatenate([v['action'] for v in episodes.values()]).max(0).tolist()}
    (ROOT / 'results/data-audit.json').write_text(json.dumps(audit, indent=2))
    partial.replace(output)
    print(json.dumps(audit, indent=2))


class EpisodeDataset(Dataset):
    def __init__(self, split_name='train'):
        split = json.loads((ROOT / 'data/split.json').read_text())
        self.task = split['task']
        self.size = split['chunk_size']
        self.episodes = {}
        self.index = []
        h5_path = Path(os.environ.get('SMOLVLA_DECODED_PATH', ROOT / 'data/episodes.h5'))
        with h5py.File(h5_path, 'r') as h5:
            for ep in split[split_name]:
                self.episodes[ep] = {key: value[:] for key, value in h5[str(ep)].items()}
                self.index.extend((ep, step) for step in range(len(self.episodes[ep]['state'])))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        ep, step = self.index[index]
        episode = self.episodes[ep]
        actions, padding = future_chunk(episode['action'], step, self.size)
        # This fixed LeRobot revision names its padding mask `actions_id_pad`.
        batch = {'observation.state': torch.from_numpy(episode['state'][step].copy()),
                 'action': torch.from_numpy(actions), 'actions_id_pad': torch.from_numpy(padding), 'task': self.task}
        for key in CAMERAS:
            batch[key] = torch.from_numpy(episode[key][step].copy()).permute(2, 0, 1).float() / 255.0
        return batch


if __name__ == '__main__':
    prepare()
