"""Shared CLI implementation for standard, focal and hard-mining experiments."""
import argparse
import copy
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cancer_pipeline.artifacts import assert_source, checkpoint_model, file_hash, load_checkpoint, new_run, save_json
from cancer_pipeline.config import load_config
from cancer_pipeline.data import class_weights_from_frame, load_manifest, make_loaders
from cancer_pipeline.engine import fit, predict
from cancer_pipeline.losses import FocalLoss
from cancer_pipeline.models import create_model
from cancer_pipeline.plots import create_training_plots
from cancer_pipeline.utils import configure_logging, get_device, set_seed

LOGGER = logging.getLogger(__name__)


def make_sample_weights(frame, fp_weight, fn_weight):
    if not np.isfinite([fp_weight, fn_weight]).all() or min(fp_weight, fn_weight) < 0:
        raise ValueError('FP/FN sampling weights must be finite and nonnegative.')
    weights = torch.ones(len(frame), dtype=torch.float64)
    weights[frame['is_fp'].to_numpy()] = fp_weight
    weights[frame['is_fn'].to_numpy()] = fn_weight
    if weights.sum() <= 0:
        raise ValueError('Sampling weights sum must be positive.')
    return weights


def add_hard_mining_columns(frame, predictions, threshold):
    from cancer_pipeline.metrics import validate_threshold
    validate_threshold(threshold)
    merged = frame.merge(predictions[['id', 'prob_cancer']], on='id', how='left', validate='one_to_one')
    if merged.prob_cancer.isna().any():
        raise ValueError('Missing hard-mining predictions.')
    merged['pred_label'] = (merged.prob_cancer >= threshold).astype(int)
    merged['is_fp'] = (merged.label == 0) & (merged.pred_label == 1)
    merged['is_fn'] = (merged.label == 1) & (merged.pred_label == 0)
    return merged


def parser_for(mode):
    parser = argparse.ArgumentParser(description=f'{mode} training (validation only; test evaluation is separate)')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--output-run-name')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--stop-after-epoch', type=int, help='Stop at an epoch boundary without changing the configured schedule.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', help='Resume complete state from last.pt, keeping the original schedule.')
    group.add_argument('--fine-tune', '--pretrained-checkpoint', dest='fine_tune', help='Start a new optimizer/schedule from weights.')
    parser.add_argument('--allow-legacy', action='store_true', help='Allow unverified weight-only sources; never permits full resume.')
    if mode == 'focal':
        parser.add_argument('--gamma', type=float, default=None)
        parser.add_argument('--alpha-negative', type=float, default=None)
        parser.add_argument('--alpha-positive', type=float, default=None)
        parser.add_argument('--label-smoothing', type=float, default=None)
    if mode == 'hard':
        parser.add_argument('--base-checkpoint')
        parser.add_argument('--decision', help='Validation-selected threshold for this base model; otherwise 0.5.')
        parser.add_argument('--fp-weight', type=float, default=None)
        parser.add_argument('--fn-weight', type=float, default=None)
        parser.add_argument('--randaugment-num-ops', type=int, default=None)
        parser.add_argument('--randaugment-magnitude', type=int, default=None)
        parser.add_argument('--class-weight', action='store_true', default=None, help='Explicitly combine class weighting with weighted sampling.')
    return parser


def main(mode='standard'):
    args = parser_for(mode).parse_args()
    configure_logging()
    config = load_config(args.config)
    for option, key in (('output_dir', 'output_dir'), ('output_run_name', 'run_name'), ('device', 'device')):
        if getattr(args, option):
            config[key] = getattr(args, option)
    for key in ('epochs', 'lr'):
        if getattr(args, key) is not None:
            config['train'][key] = getattr(args, key)
    if config['train']['epochs'] < 1 or not np.isfinite(config['train']['lr']) or config['train']['lr'] <= 0:
        raise ValueError('Epochs and learning rate must be positive.')
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= config['train']['epochs']:
        raise ValueError('stop-after-epoch must lie inside the configured schedule.')
    config['training_mode'] = mode
    device = get_device(config['device'])
    config['device'] = str(device)
    set_seed(int(config['seed']), config['deterministic'])
    frames, provenance = load_manifest(config)
    resume_state = load_checkpoint(args.resume, 'cpu') if args.resume else None
    if resume_state and not all(k in resume_state for k in ('optimizer', 'scheduler', 'scaler', 'rng_state', 'history', 'config')):
        raise ValueError('Legacy checkpoint cannot resume; use --fine-tune for weight-only initialization.')
    if resume_state and resume_state['config'].get('training_mode') != mode:
        raise ValueError('Resume must use the original training mode.')
    if mode == 'focal':
        old = resume_state['config']['loss'] if resume_state else {}
        config['loss'] = {'name': 'focal_loss', **{key: getattr(args, key) if getattr(args, key) is not None else old.get(key, default)
                          for key, default in [('gamma', 2.0), ('alpha_negative', 0.5), ('alpha_positive', 0.5), ('label_smoothing', 0.0)]}}
        config['train']['class_weight'] = False
        config['train']['label_smoothing'] = config['loss']['label_smoothing']
    else:
        config['loss'] = {'name': 'cross_entropy'}
    source = args.resume or args.fine_tune
    if mode == 'hard':
        if args.fine_tune:
            raise ValueError('Hard mining uses --base-checkpoint or --resume, not --fine-tune.')
        if args.resume and args.base_checkpoint:
            raise ValueError('Do not supply a new base checkpoint when resuming.')
        source = args.resume or args.base_checkpoint
        if not source:
            raise ValueError('Hard mining requires --base-checkpoint or --resume.')
        old = copy.deepcopy(resume_state['config']['hard_mining']) if resume_state else {}
        config['train']['class_weight'] = args.class_weight if args.class_weight is not None else (resume_state['config']['train']['class_weight'] if resume_state else False)
        config['hard_mining'] = {**old,
            'fp_weight': args.fp_weight if args.fp_weight is not None else old.get('fp_weight', 4.0),
            'fn_weight': args.fn_weight if args.fn_weight is not None else old.get('fn_weight', 6.0),
            'augmentation': {'num_ops': args.randaugment_num_ops if args.randaugment_num_ops is not None else old.get('augmentation', {}).get('num_ops', 2),
                             'magnitude': args.randaugment_magnitude if args.randaugment_magnitude is not None else old.get('augmentation', {}).get('magnitude', 7)}}
        aug = config['hard_mining']['augmentation']
        if aug['num_ops'] < 0 or not 0 <= aug['magnitude'] <= 30:
            raise ValueError('Invalid RandAugment settings.')
    if source:
        model, source_config, checkpoint = checkpoint_model(source, device, config, args.allow_legacy)
        source_arch = {k: v for k, v in source_config['model'].items() if k != 'pretrained'}
        requested_arch = {k: v for k, v in config['model'].items() if k != 'pretrained'}
        if source_arch != requested_arch or source_config.get('preprocessing', config['preprocessing']) != config['preprocessing']:
            raise ValueError('Source architecture/preprocessing differs from requested experiment.')
        provenance['source_status'] = assert_source(checkpoint, provenance, args.allow_legacy)
        config['source'] = {'path': str(Path(source).resolve()), 'checkpoint_hash': file_hash(source),
                            'mode': 'resume' if args.resume else 'fine_tune', 'status': provenance['source_status']}
    else:
        model_cfg = dict(config['model'])
        model = create_model(model_cfg.pop('name'), **model_cfg).to(device)
        provenance['source_status'] = 'verified'
    weights = None
    hard_frame = None
    if mode == 'hard':
        hard = config['hard_mining']
        if args.resume:
            hard_frame = pd.read_csv(Path(args.resume).parent / 'train_hard_examples.csv', dtype={'id': str})
            if file_hash(Path(args.resume).parent / 'train_hard_examples.csv') != hard['examples_hash']:
                raise ValueError('Hard example artifact changed.')
            if set(hard_frame.id) != set(frames['train'].id):
                raise ValueError('Hard sample IDs differ from the training split.')
            hard_frame = frames['train'][['id']].merge(hard_frame, on='id', validate='one_to_one')
            if args.decision:
                raise ValueError('Resume reuses the recorded hard-mining decision.')
        else:
            from cancer_pipeline.artifacts import load_decision
            decision = load_decision(args.decision, {'checkpoint_hash': file_hash(source), 'manifest_hash': provenance['manifest_hash'], 'preprocessing': config['preprocessing']})
            hard['threshold'] = decision['threshold']
            hard['decision'] = decision
            hard['base_checkpoint'] = config['source']
            eval_config = copy.deepcopy(config)
            eval_config.pop('hard_mining', None)
            eval_loaders, _, _ = make_loaders(eval_config, splits=('train',))
            # Training split predictions for mining must use inference preprocessing.
            from cancer_pipeline.data import build_transforms
            eval_loaders['train'].dataset.transform = build_transforms(config['preprocessing'])
            _, mining_predictions = predict(model, eval_loaders['train'], device)
            hard_frame = add_hard_mining_columns(frames['train'], mining_predictions, decision['threshold'])
        weights = make_sample_weights(hard_frame, hard['fp_weight'], hard['fn_weight'])
    # Release the loaded state before training: the model is reused, not duplicated.
    if source:
        del checkpoint
    del resume_state
    output = new_run(config)
    if hard_frame is not None:
        if args.resume:
            shutil.copy2(Path(args.resume).parent / 'train_hard_examples.csv', output / 'train_hard_examples.csv')
        else:
            hard_frame.to_csv(output / 'train_hard_examples.csv', index=False)
        config['hard_mining']['examples_hash'] = file_hash(output / 'train_hard_examples.csv')
        save_json({'train_samples': len(hard_frame), 'fp_samples': int(hard_frame.is_fp.sum()),
                   'fn_samples': int(hard_frame.is_fn.sum()), **config['hard_mining']}, output / 'hard_mining_summary.json')
    save_json(config, output / 'config.json')
    save_json(provenance, output / 'data_provenance.json')
    loaders, _, _ = make_loaders(config, train_weights=weights)
    if mode == 'focal':
        criterion = FocalLoss(**{k: v for k, v in config['loss'].items() if k != 'name'}).to(device)
    else:
        weight = class_weights_from_frame(frames['train']).to(device) if config['train'].get('class_weight') else None
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config['train'].get('label_smoothing', 0.0))
    train = config['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train['lr']), weight_decay=float(train.get('weight_decay', 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train['epochs'])
    LOGGER.info('device=%s mode=%s output=%s', device, mode, output)
    summary = fit(model, loaders, criterion, optimizer, scheduler, device, config, output, provenance,
                  resume=args.resume, stop_after_epoch=args.stop_after_epoch)
    save_json({'summary': summary, 'selection_split': 'val', 'test_evaluated': False}, output / 'metrics.json')
    create_training_plots(output)
    print(output)
