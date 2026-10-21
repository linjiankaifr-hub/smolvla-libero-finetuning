"""Download pinned public assets; only one task's episodes and no extra VLM weights."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'aopolin-lv/libero_spatial_no_noops_lerobot_v21'


def main():
    api = HfApi()
    manifest_path = ROOT / 'sources.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            'dataset': {'repo': REPO, 'revision': api.dataset_info(REPO).sha},
            'base': {'repo': 'lerobot/smolvla_base', 'revision': api.model_info('lerobot/smolvla_base').sha},
            'processor': {'repo': 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
                          'revision': api.model_info('HuggingFaceTB/SmolVLM2-500M-Video-Instruct').sha},
            'tutorial': 'https://github.com/datawhalechina/every-embodied/blob/main/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/01SmolVLA-LIBERO/01SmolVLA-libero.md',
            'unavailable_tutorial_dataset': 'nikriz/aopoli-lv-libero_combined_no_noops_lerobot_v21: HTTP 401',
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    dst = ROOT / 'data/raw'
    meta = ['meta/info.json', 'meta/tasks.jsonl', 'meta/episodes.jsonl', 'meta/episodes_stats.jsonl']
    for name in meta:
        hf_hub_download(REPO, name, repo_type='dataset', revision=manifest['dataset']['revision'], local_dir=dst)
    records = [json.loads(line) for line in (dst / 'meta/episodes.jsonl').read_text().splitlines()]
    tasks = [json.loads(line) for line in (dst / 'meta/tasks.jsonl').read_text().splitlines()]
    task = tasks[0]['task']
    episodes = [r['episode_index'] for r in records if task in r['tasks']]
    split = {'task': task, 'task_index_in_dataset': 0, 'episodes': episodes,
             'train': episodes[:-8], 'validation': episodes[-8:], 'chunk_size': 50,
             'note': 'Complete episodes held out; no validation frames in normalization statistics.'}
    (ROOT / 'data/split.json').write_text(json.dumps(split, indent=2))
    names = []
    for ep in episodes:
        names.append(f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')
        for camera in ['observation.images.image', 'observation.images.wrist_image']:
            names.append(f'videos/chunk-{ep//1000:03d}/{camera}/episode_{ep:06d}.mp4')

    def fetch(name):
        return hf_hub_download(REPO, name, repo_type='dataset', revision=manifest['dataset']['revision'], local_dir=dst)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(fetch, names), 1):
            if index % 15 == 0:
                print(f'dataset files {index}/{len(names)}', flush=True)
    for key, directory, patterns in [
        ('base', 'models/base', ['config.json', 'model.safetensors', 'README.md']),
        ('processor', 'models/processor', ['*.json', '*.txt', '*.model', '*.jinja', 'README.md']),
    ]:
        spec = manifest[key]
        snapshot_download(spec['repo'], revision=spec['revision'], local_dir=ROOT / directory,
                          allow_patterns=patterns, max_workers=4)
    weights = ROOT / 'models/base/model.safetensors'
    manifest['base']['sha256'] = hashlib.file_digest(weights.open('rb'), 'sha256').hexdigest()
    manifest['selected_task'] = task
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({'task': task, 'episodes': len(episodes), 'download_complete': True}), flush=True)


if __name__ == '__main__':
    main()
