import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cancer-pipeline-mpl')

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from cancer_pipeline.config import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / 'data'
    (root / 'train').mkdir(parents=True)
    rng = np.random.default_rng(7)
    labels = []
    for i in range(40):
        image_id = f'{i:04d}'
        label = i % 2
        image = rng.integers(0, 90, (32, 32, 3), dtype=np.uint8) + label * 140
        Image.fromarray(image).save(root / 'train' / f'{image_id}.tif')
        labels.append({'id': image_id, 'label': label, 'slide': f'slide-{i//2:02d}'})
    pd.DataFrame(labels).to_csv(root / 'train_labels.csv', index=False)
    raw = {'run_name': 'synthetic', 'seed': 42, 'device': 'cpu', 'deterministic': True,
           'dataset': {'root': str(root), 'image_size': 32, 'val_size': 0.15, 'test_size': 0.15, 'num_workers': 0},
           'model': {'name': 'small_cnn', 'pretrained': False, 'num_classes': 2},
           'train': {'epochs': 3, 'batch_size': 4, 'grad_accum_steps': 2, 'lr': 0.001,
                     'weight_decay': 0.0001, 'label_smoothing': 0.05, 'amp': False,
                     'early_stopping_patience': 10, 'class_weight': True},
           'output_dir': str(tmp_path / 'outputs')}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path, load_config(path)
