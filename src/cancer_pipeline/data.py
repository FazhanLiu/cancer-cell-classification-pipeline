"""Immutable split manifests and shared preprocessing."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cancer_pipeline.artifacts import object_hash, read_json

SPLITS = ('train', 'val', 'test')


def read_labels(root, group_column=None):
    frame = pd.read_csv(Path(root) / 'train_labels.csv', dtype={'id': str})
    required = ['id', 'label'] + ([group_column] if group_column else [])
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Label table requires {required}.')
    frame = frame[required].copy().sort_values('id').reset_index(drop=True)
    if frame.isna().any().any() or frame['id'].duplicated().any() or not frame['label'].isin([0, 1]).all():
        raise ValueError('Labels require unique non-null IDs and binary labels; groups cannot be null.')
    if not frame['id'].map(lambda x: bool(x) and Path(x).name == x and '/' not in x and '\\' not in x).all():
        raise ValueError('Invalid sample ID.')
    frame['label'] = frame['label'].astype(int)
    if group_column:
        frame[group_column] = frame[group_column].astype(str)
    missing = [x for x in frame['id'] if not (Path(root) / 'train' / f'{x}.tif').is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} images, e.g. {missing[:3]}.')
    return frame


def _group_split(frame, size, seed, group_column):
    # Group sizes may prevent exact ratios. Choose a deterministic, label-balanced
    # candidate without splitting any group; this is data preparation, not model selection.
    candidates = GroupShuffleSplit(n_splits=128, test_size=size, random_state=seed)
    best = None
    for left, right in candidates.split(frame, frame['label'], frame[group_column]):
        a, b = frame.iloc[left], frame.iloc[right]
        if a['label'].nunique() != 2 or b['label'].nunique() != 2:
            continue
        score = abs(len(b) / len(frame) - size) + abs(a.label.mean() - frame.label.mean()) + abs(b.label.mean() - frame.label.mean())
        if best is None or score < best[0]:
            best = score, a, b
    if best is None:
        raise ValueError('Insufficient groups/classes for a valid grouped split.')
    return best[1:]


def prepare_manifest(config):
    ds = config['dataset']
    path = Path(ds['manifest'])
    if path.exists():
        load_manifest(config)
        return path
    group = ds.get('group_column')
    labels = read_labels(ds['root'], group)
    val, test, seed = ds['val_size'], ds['test_size'], int(config['seed'])
    if group:
        train, holdout = _group_split(labels, val + test, seed, group)
        valid, testing = _group_split(holdout, test / (val + test), seed, group)
    else:
        train, holdout = train_test_split(labels, test_size=val + test, stratify=labels.label, random_state=seed)
        valid, testing = train_test_split(holdout, test_size=test / (val + test), stratify=holdout.label, random_state=seed)
    if any(part.label.nunique() != 2 for part in (train, valid, testing)):
        raise ValueError('Each split must contain both classes.')
    records = pd.concat([part.assign(split=name) for name, part in zip(SPLITS, (train, valid, testing))]).sort_values('id').to_dict('records')
    manifest = {'schema_version': 1, 'labels_hash': object_hash(labels.to_dict('records')),
                'parameters': {'seed': seed, 'val_size': val, 'test_size': test, 'group_column': group},
                'split_mode': 'grouped' if group else 'patch_random_no_group_guarantee',
                'records': records}
    manifest['manifest_hash'] = object_hash(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation makes accidental/concurrent replacement impossible.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    load_manifest(config)
    return path


def load_manifest(config):
    ds = config['dataset']
    path = Path(ds['manifest'])
    if not path.is_file():
        raise ValueError('Split manifest is missing. Run scripts/prepare_data.py first.')
    manifest = read_json(path)
    claimed = manifest['manifest_hash']
    if object_hash({k: v for k, v in manifest.items() if k != 'manifest_hash'}) != claimed:
        raise ValueError('Split manifest hash mismatch.')
    params = {'seed': int(config['seed']), 'val_size': ds['val_size'], 'test_size': ds['test_size'], 'group_column': ds.get('group_column')}
    if manifest['parameters'] != params:
        raise ValueError('Split parameters changed. Use a new manifest path to explicitly prepare a new split.')
    labels = read_labels(ds['root'], ds.get('group_column'))
    if object_hash(labels.to_dict('records')) != manifest['labels_hash']:
        raise ValueError('Label data changed. Prepare a new manifest explicitly.')
    frame = pd.DataFrame(manifest['records'])
    if frame.id.duplicated().any() or set(frame.id) != set(labels.id) or set(frame.split) != set(SPLITS):
        raise ValueError('Manifest must contain every ID exactly once across train/val/test.')
    columns = list(labels.columns)
    if frame.sort_values('id')[columns].reset_index(drop=True).to_dict('records') != labels.to_dict('records'):
        raise ValueError('Manifest labels/groups do not match the label table.')
    group = ds.get('group_column')
    if group and frame.groupby(group)['split'].nunique().max() != 1:
        raise ValueError('Groups overlap across splits.')
    frames = {s: frame[frame.split == s].drop(columns='split').reset_index(drop=True) for s in SPLITS}
    if any(f.label.nunique() != 2 for f in frames.values()):
        raise ValueError('Each split must contain both classes.')
    provenance = {k: manifest[k] for k in ('schema_version', 'manifest_hash', 'labels_hash', 'parameters', 'split_mode')}
    provenance['class_counts'] = {s: {str(k): int(v) for k, v in f.label.value_counts().items()} for s, f in frames.items()}
    return frames, provenance


class HistopathologyDataset(Dataset):
    def __init__(self, frame, image_dir, transform=None):
        self.frame, self.image_dir, self.transform = frame.reset_index(drop=True), Path(image_dir), transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.image_dir / f'{row.id}.tif') as source:
            image = source.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(int(row.label), dtype=torch.long), row.id


def build_transforms(preprocessing, train=False, hard=None):
    if isinstance(preprocessing, int):  # compatibility for callers, not artifact loading
        preprocessing = {'image_size': preprocessing, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    size = int(preprocessing['image_size'])
    ops = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR,
                             antialias=preprocessing.get('antialias', True))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)]
        if hard:
            # Preserve the full patch: random crops can discard label-defining tissue.
            ops += [transforms.RandAugment(num_ops=hard['num_ops'], magnitude=hard['magnitude'])]
    ops += [transforms.ToTensor(), transforms.Normalize(preprocessing['mean'], preprocessing['std'])]
    return transforms.Compose(ops)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(config, splits=('train', 'val'), train_weights=None):
    frames, provenance = load_manifest(config)
    loaders = {}
    for offset, split in enumerate(SPLITS):
        if split not in splits:
            continue
        generator = torch.Generator().manual_seed(int(config['seed']) + offset)
        sampler = None
        if split == 'train' and train_weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights), replacement=True, generator=generator)
        dataset = HistopathologyDataset(frames[split], Path(config['dataset']['root']) / 'train',
                                        build_transforms(config['preprocessing'], split == 'train', config.get('hard_mining', {}).get('augmentation')))
        loaders[split] = DataLoader(dataset, batch_size=int(config['train']['batch_size']),
                                    shuffle=split == 'train' and sampler is None, sampler=sampler,
                                    num_workers=int(config['dataset'].get('num_workers', 0)),
                                    pin_memory=str(config.get('device', '')).startswith('cuda'),
                                    generator=generator, worker_init_fn=seed_worker, persistent_workers=False)
    return loaders, frames, provenance


def class_weights_from_frame(frame):
    counts = frame.label.value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError('Class weighting requires both classes.')
    return torch.tensor((len(frame) / (2 * counts)).values, dtype=torch.float32)
