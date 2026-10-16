"""ID-aligned probability ensemble; independent of XAI heatmap fusion."""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.artifacts import assert_source, checkpoint_model, file_hash, load_decision, new_run, object_hash, save_json, write_predictions
from cancer_pipeline.data import make_loaders
from cancer_pipeline.engine import predict
from cancer_pipeline.metrics import classification_metrics
from cancer_pipeline.plots import plot_predictions
from cancer_pipeline.utils import configure_logging, get_device


def main():
    parser = argparse.ArgumentParser(description='Ensemble models on a fixed manifest, using each model preprocessing.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--decision')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    configure_logging()
    with open(args.config, encoding='utf-8') as stream:
        ensemble = yaml.safe_load(stream)
    items = ensemble['models']
    weights = np.asarray([float(item.get('weight', 1)) for item in items])
    if not len(items) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError('Ensemble weights must be finite, nonnegative, and sum to a positive number.')
    device = get_device(args.device)
    reference, probability, sources, pp = None, None, [], []
    manifest_hash = None
    for item, weight in zip(items, weights):
        model, config, checkpoint = checkpoint_model(item['checkpoint'], device)
        if item.get('name', config['model']['name']) != config['model']['name']:
            raise ValueError('Ensemble model name differs from checkpoint.')
        for key in ('root', 'manifest'):
            if key in ensemble.get('dataset', {}):
                config['dataset'][key] = ensemble['dataset'][key]
        loaders, frames, provenance = make_loaders(config, splits=(args.split,))
        status = assert_source(checkpoint, provenance)
        if status != 'verified':
            raise ValueError('Unverified historical sources are excluded from reproducible ensembles.')
        if manifest_hash is not None and manifest_hash != provenance['manifest_hash']:
            raise ValueError('Ensemble models must share the same split manifest.')
        manifest_hash = provenance['manifest_hash']
        _, frame = predict(model, loaders[args.split], device, config['train'].get('amp', True))
        frame = frame.set_index('id').sort_index()
        if reference is None:
            reference = frame[['label']]
            probability = np.zeros(len(frame))
        if not frame.index.equals(reference.index) or not frame.label.equals(reference.label):
            raise ValueError('Ensemble IDs or labels are not aligned.')
        probability += weight * frame.prob_cancer.to_numpy()
        sources.append({'checkpoint_hash': file_hash(item['checkpoint']), 'weight': float(weight)})
        pp.append(config['preprocessing'])
        del model, checkpoint, loaders
        gc.collect()
        if device.type == 'cuda':
            import torch
            torch.cuda.empty_cache()
    metadata = {'checkpoint_hash': object_hash(sources), 'manifest_hash': manifest_hash,
                'labels_hash': provenance['labels_hash'], 'preprocessing': {'ensemble': pp},
                'split': args.split, 'source_status': 'verified', 'members': sources}
    decision = load_decision(args.decision, {k: metadata[k] for k in ('checkpoint_hash', 'manifest_hash', 'preprocessing')})
    result = reference.assign(prob_cancer=probability / weights.sum()).reset_index()
    metrics = classification_metrics(result.label, result.prob_cancer, decision['threshold'])
    output = new_run({**ensemble, 'output_dir': ensemble.get('output_dir', 'outputs'),
                      'run_name': ensemble.get('run_name', 'ensemble')}, f'ensemble-{args.split}')
    path = output / f'{args.split}_predictions.csv'
    write_predictions(result, path, {**metadata, 'decision': decision})
    save_json({'metrics': metrics, 'decision': decision, 'provenance': metadata}, output / 'metrics.json')
    plot_predictions(path, output, args.split, threshold=decision['threshold'])
    print(output)

if __name__ == '__main__':
    main()
