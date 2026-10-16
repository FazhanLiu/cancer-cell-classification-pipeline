"""Versioned, content-checked experiment artifacts. Only load trusted checkpoints."""
import copy
import hashlib
import json
import logging
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_run(config, kind='train'):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = Path(config.get('output_dir', 'outputs')) / config['run_name'] / f'{stamp}-{kind}-{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True, exist_ok=False)
    save_json(config, path / 'config.json')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    save_json({'python': sys.version, 'git_revision': revision, 'git_dirty': dirty,
               'packages': {d.metadata['Name']: d.version for d in distributions()},
               'runtime_versions': {name: getattr(sys.modules[name], '__version__', None)
                                    for name in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'sklearn')
                                    if name in sys.modules},
               'created_utc': stamp, 'kind': kind}, path / 'environment.json')
    handler = logging.FileHandler(path / 'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger().addHandler(handler)
    return path


def load_checkpoint(path, device='cpu'):
    # Full training state contains RNG tuples and cannot use weights_only=True.
    return torch.load(path, map_location=device, weights_only=False)


def checkpoint_model(path, device, fallback_config=None, allow_legacy=False):
    from cancer_pipeline.models import create_model
    checkpoint = load_checkpoint(path, 'cpu')
    config = copy.deepcopy(checkpoint.get('config') or fallback_config)
    if not config:
        raise ValueError('Checkpoint has no config; an explicit legacy config is required.')
    if not checkpoint.get('provenance') and not allow_legacy:
        raise ValueError('Legacy checkpoint has unverified provenance; use --allow-legacy explicitly.')
    cfg = dict(config['model'])
    name = cfg.pop('name')
    cfg['pretrained'] = False
    model = create_model(name, **cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    return model, config, checkpoint


def assert_source(checkpoint, provenance, allow_legacy=False):
    previous = checkpoint.get('provenance')
    if not previous:
        if not allow_legacy:
            raise ValueError('Unverified legacy source; explicit --allow-legacy is required.')
        LOGGER.warning('Legacy source is UNVERIFIED; holdout independence cannot be established.')
        return 'unverified'
    if previous['manifest_hash'] != provenance['manifest_hash']:
        raise ValueError('Checkpoint split provenance differs; refusing possible holdout contamination.')
    status = previous.get('source_status', 'verified')
    if status != 'verified' and not allow_legacy:
        raise ValueError('Source lineage is unverified; explicit --allow-legacy is required.')
    return status


def write_predictions(frame, path, metadata):
    from cancer_pipeline.metrics import validate_predictions
    validate_predictions(frame['label'], frame['prob_cancer'])
    if frame['id'].duplicated().any():
        raise ValueError('Prediction IDs must be unique.')
    path = Path(path)
    frame.to_csv(path, index=False)
    save_json({'schema_version': 1, **metadata, 'csv_hash': file_hash(path)}, str(path) + '.meta.json')


def read_predictions(path, expected=None):
    frame = pd.read_csv(path, dtype={'id': str})
    metadata = read_json(str(path) + '.meta.json')
    if metadata.get('csv_hash') != file_hash(path):
        raise ValueError('Predictions changed after provenance was recorded.')
    from cancer_pipeline.metrics import validate_predictions
    validate_predictions(frame['label'], frame['prob_cancer'])
    if frame['id'].isna().any() or frame['id'].duplicated().any():
        raise ValueError('Prediction IDs must be present and unique.')
    for key, value in (expected or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f'Prediction source mismatch: {key}')
    return frame, metadata


def prediction_metadata(checkpoint_path, config, provenance, split, source_status='verified'):
    return {'checkpoint_hash': file_hash(checkpoint_path), 'manifest_hash': provenance['manifest_hash'],
            'labels_hash': provenance['labels_hash'], 'preprocessing': config['preprocessing'],
            'split': split, 'source_status': source_status, 'decision': load_decision()}


def load_decision(path=None, expected=None):
    if path is None:
        return {'schema_version': 1, 'threshold': 0.5, 'objective': 'fixed_default', 'source': 'default'}
    result = read_json(path)
    from cancer_pipeline.metrics import validate_threshold
    validate_threshold(result['threshold'])
    for key, value in (expected or {}).items():
        if result.get(key) != value:
            raise ValueError(f'Decision source mismatch: {key}')
    if result.get('split') != 'val':
        raise ValueError('Selected decision parameters must originate from validation data.')
    return result
