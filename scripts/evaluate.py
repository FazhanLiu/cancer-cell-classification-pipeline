import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.artifacts import assert_source, checkpoint_model, load_decision, new_run, prediction_metadata, save_json, write_predictions
from cancer_pipeline.config import load_config
from cancer_pipeline.data import make_loaders
from cancer_pipeline.engine import predict
from cancer_pipeline.plots import plot_predictions
from cancer_pipeline.utils import configure_logging, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', help='Optional dataset location override; model and preprocessing come from checkpoint.')
    parser.add_argument('--data-root')
    parser.add_argument('--manifest')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--decision')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--allow-legacy', action='store_true')
    args = parser.parse_args()
    configure_logging()
    device = get_device(args.device)
    override = load_config(args.config) if args.config else None
    model, config, checkpoint = checkpoint_model(args.checkpoint, device, override, args.allow_legacy)
    if 'preprocessing' not in config:
        if override is None:
            raise ValueError('Legacy checkpoint requires --config with explicit preprocessing.')
        config['preprocessing'] = override['preprocessing']
    if override:
        config['dataset']['root'] = override['dataset']['root']
        config['dataset']['manifest'] = override['dataset']['manifest']
        config['output_dir'] = override['output_dir']
    if args.data_root:
        config['dataset']['root'] = args.data_root
    if args.manifest:
        config['dataset']['manifest'] = args.manifest
    loaders, _, provenance = make_loaders(config, splits=(args.split,))
    status = assert_source(checkpoint, provenance, args.allow_legacy)
    metadata = prediction_metadata(args.checkpoint, config, provenance, args.split, status)
    decision = load_decision(args.decision, {k: metadata[k] for k in ('checkpoint_hash', 'manifest_hash', 'preprocessing')})
    metrics, predictions = predict(model, loaders[args.split], device, config['train'].get('amp', True), decision['threshold'])
    output = new_run(config, f'evaluate-{args.split}')
    destination = output / f'{args.split}_predictions.csv'
    write_predictions(predictions, destination, {**metadata, 'decision': decision})
    save_json({'metrics': metrics, 'provenance': metadata, 'decision': decision}, output / 'eval_metrics.json')
    plot_predictions(destination, output, prefix=args.split, threshold=decision['threshold'])
    print(output)

if __name__ == '__main__':
    main()
