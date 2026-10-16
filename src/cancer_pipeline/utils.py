import logging
import random

import numpy as np
import torch

from cancer_pipeline.artifacts import save_json


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def get_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable.')
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
