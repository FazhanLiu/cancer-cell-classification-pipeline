import math
import os
from pathlib import Path

import yaml


def load_environment(path):
    """Load a chosen .env file without replacing existing environment values."""
    from dotenv import load_dotenv
    return load_dotenv(dotenv_path=Path(path), override=False, interpolate=False,
                       encoding='utf-8-sig')


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('Config must be a mapping.')
    config.setdefault('seed', 42)
    config.setdefault('device', 'auto')
    config.setdefault('deterministic', True)
    config.setdefault('output_dir', 'outputs')
    ds = config['dataset']
    ds['root'] = os.path.expandvars(os.path.expanduser(os.environ.get('CANCER_DATA_ROOT', ds.get('root', 'data/histopathologic-cancer-detection'))))
    ds.setdefault('manifest', str(Path(ds['root']) / 'splits.json'))
    ds.setdefault('num_workers', 0)
    ds.setdefault('val_size', 0.15)
    ds.setdefault('test_size', 0.15)
    if not (0 < ds['val_size'] < 1 and 0 < ds['test_size'] < 1 and ds['val_size'] + ds['test_size'] < 1):
        raise ValueError('Split fractions must be positive and leave a nonempty training set.')
    pp = config.setdefault('preprocessing', {})
    pp.setdefault('image_size', ds.get('image_size', 224))
    pp.setdefault('mean', [0.485, 0.456, 0.406])
    pp.setdefault('std', [0.229, 0.224, 0.225])
    pp.setdefault('interpolation', 'bilinear')
    pp.setdefault('antialias', True)
    if pp['interpolation'] != 'bilinear' or pp['image_size'] < 1:
        raise ValueError('Only positive-size bilinear preprocessing is supported.')
    if len(pp['mean']) != 3 or len(pp['std']) != 3 or not all(math.isfinite(x) for x in pp['mean'] + pp['std']) or min(pp['std']) <= 0:
        raise ValueError('RGB normalization requires finite means and positive standard deviations.')
    train = config['train']
    for key in ('epochs', 'batch_size', 'grad_accum_steps', 'early_stopping_patience'):
        train.setdefault(key, {'grad_accum_steps': 1, 'early_stopping_patience': 10}.get(key, 1))
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ('lr', 'weight_decay'):
        value = float(train.get(key, 0))
        if not math.isfinite(value) or value < 0 or (key == 'lr' and value == 0):
            raise ValueError(f'Invalid {key}.')
    if not 0 <= float(train.get('label_smoothing', 0)) < 1 or ds['num_workers'] < 0:
        raise ValueError('Invalid label smoothing or worker count.')
    if config['model'].get('num_classes', 2) != 2:
        raise ValueError('This pipeline supports binary classification only.')
    return config


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
