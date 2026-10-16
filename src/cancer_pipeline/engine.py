import logging
import random
import shutil
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cancer_pipeline.artifacts import file_hash, load_checkpoint, prediction_metadata, read_predictions, write_predictions
from cancer_pipeline.losses import loss_denominator, loss_numerator
from cancer_pipeline.metrics import classification_metrics

LOGGER = logging.getLogger(__name__)


def autocast(device, enabled):
    return torch.autocast(device_type='cuda', enabled=True) if enabled and device.type == 'cuda' else nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps, amp):
    if grad_accum_steps < 1:
        raise ValueError('Gradient accumulation steps must be positive.')
    model.train()
    total_numerator = total_denominator = 0.0
    iterator = iter(loader)
    while True:
        window = list(islice(iterator, grad_accum_steps))
        if not window:
            break
        # Normalize by the actual effective batch, including class-weighted CE
        # denominators and a short final microbatch. No computation graphs retained.
        denominator = sum(loss_denominator(criterion, labels.to(device)) for _, labels, _ in window)
        if denominator <= 0:
            raise ValueError('Loss denominator must be positive.')
        optimizer.zero_grad(set_to_none=True)
        for images, labels, _ in window:
            images, labels = images.to(device), labels.to(device)
            with autocast(device, amp):
                numerator = loss_numerator(criterion, model(images), labels)
            scaler.scale(numerator / denominator).backward()
            total_numerator += numerator.detach().item()
        scaler.step(optimizer)
        scaler.update()
        total_denominator += denominator
    if total_denominator == 0:
        raise ValueError('Empty training loader.')
    return total_numerator / total_denominator


@torch.no_grad()
def predict(model, loader, device, amp=False, threshold=0.5):
    model.eval()
    ids, labels_all, probabilities = [], [], []
    loss_sum = 0.0
    for images, labels, batch_ids in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(device, amp):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        loss_sum += loss.item()
        ids.extend(batch_ids)
        labels_all.extend(labels.cpu().tolist())
        probabilities.extend(logits.float().softmax(dim=1)[:, 1].cpu().tolist())
    metrics = classification_metrics(labels_all, probabilities, threshold)
    metrics['cross_entropy_unweighted'] = loss_sum / len(labels_all)
    return metrics, pd.DataFrame({'id': ids, 'label': labels_all, 'prob_cancer': probabilities})


def capture_state(loaders, device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'mps': torch.mps.get_rng_state() if device.type == 'mps' else None,
            'loaders': {k: v.generator.get_state() for k, v in loaders.items()}}


def restore_state(state, loaders):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG state cannot be resumed on CPU.')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    if state.get('mps') is not None:
        if not torch.backends.mps.is_available():
            raise ValueError('MPS RNG state cannot be resumed without MPS.')
        torch.mps.set_rng_state(state['mps'].cpu())
    for key, value in state['loaders'].items():
        loaders[key].generator.set_state(value.cpu())


def fit(model, loaders, criterion, optimizer, scheduler, device, config, output_dir,
        provenance, resume=None, stop_after_epoch=None):
    output_dir = Path(output_dir)
    train = config['train']
    amp = bool(train.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best_auc, best_epoch, first_epoch, history = -1.0, 0, 1, []
    if resume:
        state = load_checkpoint(resume, 'cpu')
        required = ('optimizer', 'scheduler', 'scaler', 'rng_state', 'history', 'best_auc', 'best_epoch', 'device_type')
        if any(key not in state for key in required):
            raise ValueError('Legacy checkpoint cannot resume; use weight-only fine-tuning.')
        if state.get('provenance', {}).get('manifest_hash') != provenance['manifest_hash']:
            raise ValueError('Resume split provenance differs.')
        if state['config']['dataset'].get('num_workers', 0) != config['dataset'].get('num_workers', 0):
            raise ValueError('Resume requires the same data worker count.')
        if state['device_type'] != device.type:
            raise ValueError('Resume requires the same device type.')
        if state.get('checkpoint_kind') != 'last':
            raise ValueError('Resume from last.pt, not best.pt.')
        if state.get('stopped_early'):
            raise ValueError('This run has already early-stopped; start an explicit fine-tuning run.')
        for section in ('train', 'model', 'preprocessing', 'loss', 'hard_mining', 'seed', 'deterministic'):
            if state['config'].get(section) != config.get(section):
                raise ValueError(f'Resume config changed: {section}; use fine-tuning instead.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_auc, best_epoch = state['best_auc'], state['best_epoch']
        first_epoch, history = state['epoch'] + 1, state['history']
        if first_epoch > train['epochs']:
            raise ValueError('Training schedule is already complete.')
        source = Path(resume).parent
        for name in ('best.pt', 'val_predictions.csv', 'val_predictions.csv.meta.json'):
            if not (source / name).is_file():
                raise ValueError(f'Resume requires sibling artifact: {name}')
            shutil.copy2(source / name, output_dir / name)
        expected_best = state.get('best_checkpoint_hash')
        if expected_best != file_hash(output_dir / 'best.pt'):
            raise ValueError('Resume best checkpoint differs from the recorded artifact.')
        read_predictions(output_dir / 'val_predictions.csv',
                         {'checkpoint_hash': expected_best, 'manifest_hash': provenance['manifest_hash'],
                          'preprocessing': config['preprocessing'], 'split': 'val'})
        restore_state(state['rng_state'], loaders)
        del state
    start = time.monotonic()
    for epoch in range(first_epoch, int(train['epochs']) + 1):
        train_loss = train_one_epoch(model, loaders['train'], criterion, optimizer, scaler, device,
                                     train['grad_accum_steps'], amp)
        metrics, predictions = predict(model, loaders['val'], device, amp)
        if metrics['auc'] is None:
            raise ValueError('Validation requires both classes for AUC selection.')
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        **{f'val_{k}': v for k, v in metrics.items() if k != 'confusion_matrix'}})
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        improved = metrics['auc'] > best_auc
        if improved:
            best_auc, best_epoch = metrics['auc'], epoch
        stopped_early = epoch - best_epoch >= train['early_stopping_patience']
        state = {'schema_version': 2, 'model': model.state_dict(), 'config': config, 'provenance': provenance,
                 'epoch': epoch, 'metrics': metrics, 'best_auc': best_auc, 'best_epoch': best_epoch,
                 'early_stopping_count': epoch - best_epoch, 'stopped_early': stopped_early,
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'rng_state': capture_state(loaders, device), 'history': history, 'device_type': device.type,
                 'checkpoint_kind': 'last'}
        if improved:
            state['checkpoint_kind'] = 'best'
            torch.save(state, output_dir / 'best.pt.tmp')
            (output_dir / 'best.pt.tmp').replace(output_dir / 'best.pt')
            metadata = prediction_metadata(output_dir / 'best.pt', config, provenance, 'val', provenance.get('source_status', 'verified'))
            write_predictions(predictions, output_dir / 'val_predictions.csv', metadata)
        state['checkpoint_kind'] = 'last'
        state['best_checkpoint_hash'] = file_hash(output_dir / 'best.pt')
        torch.save(state, output_dir / 'last.pt.tmp')
        (output_dir / 'last.pt.tmp').replace(output_dir / 'last.pt')
        del state
        LOGGER.info('epoch=%s train_loss=%.5f val_auc=%.5f val_f1=%.5f', epoch, train_loss, metrics['auc'], metrics['f1'])
        if stopped_early or (stop_after_epoch is not None and epoch >= stop_after_epoch):
            break
    return {'best_auc': best_auc, 'best_epoch': best_epoch, 'last_epoch': epoch, 'seconds': time.monotonic() - start}
