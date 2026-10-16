import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.artifacts import assert_source, checkpoint_model, file_hash, load_decision, new_run, object_hash, read_predictions, save_json
from cancer_pipeline.config import load_config
from cancer_pipeline.data import build_transforms, load_manifest
from cancer_pipeline.utils import configure_logging, get_device
from cancer_pipeline.xai import attention_region_summary, deletion_curves, explain, high_score_area_ratio, predict_prob, save_heatmap_triplet


def confusion_group(label, prediction):
    return ('TP' if prediction else 'FN') if label else ('FP' if prediction else 'TN')


def select_samples(frame, threshold, mode, count, per_group, seed):
    frame = frame.copy()
    frame['pred_label'] = (frame.prob_cancer >= threshold).astype(int)
    frame['group'] = [confusion_group(int(y), int(p)) for y, p in zip(frame.label, frame.pred_label)]
    if mode == 'quantitative':
        if count < 2:
            raise ValueError('Quantitative sampling requires at least two examples.')
        if count >= len(frame):
            return frame.sort_values('id').reset_index(drop=True)
        selected, _ = train_test_split(frame, train_size=count, stratify=frame.label, random_state=seed)
        return selected.sort_values('id').reset_index(drop=True)
    if per_group < 1:
        raise ValueError('samples-per-group must be positive.')
    frame['threshold_distance'] = abs(frame.prob_cancer - threshold)
    return frame.sort_values(['threshold_distance', 'id']).groupby('group', sort=True).head(per_group).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Cancer-score explanations with explicit provenance and quantitative comparisons.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--config', help='Optional dataset location override.')
    parser.add_argument('--decision')
    parser.add_argument('--output', default='outputs/xai')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mode', choices=['diagnostic', 'quantitative'], default='diagnostic')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--samples-per-group', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--lime-samples', type=int, default=200)
    parser.add_argument('--grid-size', type=int, default=7)
    parser.add_argument('--occlusion-patch', type=int, default=32)
    parser.add_argument('--occlusion-stride', type=int, default=16)
    parser.add_argument('--attention-weight', type=float, default=0.5)
    parser.add_argument('--lime-weight', type=float, default=0.3)
    parser.add_argument('--occlusion-weight', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    configure_logging()
    if args.repeats < 1:
        raise ValueError('repeats must be positive.')
    device = get_device(args.device)
    override = load_config(args.config) if args.config else None
    model, config, checkpoint = checkpoint_model(args.checkpoint, device)
    model.eval()
    if override:
        for key in ('root', 'manifest'):
            config['dataset'][key] = override['dataset'][key]
    frames, provenance = load_manifest(config)
    if assert_source(checkpoint, provenance) != 'verified':
        raise ValueError('XAI comparisons require verified source provenance.')
    expected = {'checkpoint_hash': file_hash(args.checkpoint), 'manifest_hash': provenance['manifest_hash'],
                'preprocessing': config['preprocessing'], 'source_status': 'verified'}
    frame, metadata = read_predictions(args.predictions, expected)
    split = metadata['split']
    if split not in frames or set(frame.id) != set(frames[split].id):
        raise ValueError('Prediction IDs differ from the recorded split.')
    joined = frame.set_index('id').label.sort_index()
    if not joined.equals(frames[split].set_index('id').label.sort_index()):
        raise ValueError('Prediction labels differ from manifest.')
    inherited = metadata.get('decision')
    decision = load_decision(args.decision, {k: expected[k] for k in ('checkpoint_hash', 'manifest_hash', 'preprocessing')}) if args.decision else (inherited or load_decision())
    if inherited and not args.decision:
        from cancer_pipeline.metrics import validate_threshold
        validate_threshold(inherited['threshold'])
        if inherited.get('objective') != 'fixed_default':
            if inherited.get('split') != 'val' or any(inherited.get(k) != expected[k] for k in ('checkpoint_hash', 'manifest_hash', 'preprocessing')):
                raise ValueError('Inherited decision source mismatch.')
    selected = select_samples(frame, decision['threshold'], args.mode, args.samples, args.samples_per_group, args.seed)
    weights = (args.attention_weight, args.lime_weight, args.occlusion_weight)
    from cancer_pipeline.xai import weighted_final_score
    weighted_final_score(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), *weights)
    run_config = {**config, 'output_dir': args.output, 'run_name': 'vit_xai', 'xai': vars(args)}
    output = new_run(run_config, args.mode)
    selected.to_csv(output / 'selected_samples.csv', index=False)
    save_json({'source': metadata, 'decision': decision, 'parameters': vars(args),
               'target': 'cancer_score', 'attention_semantics': 'class_independent_information_flow',
               'fusion_status': 'exploratory_relative_importance_not_segmentation',
               'evaluation_note': 'Deletion may favor perturbation methods; no localization accuracy is measured.'}, output / 'provenance.json')
    transform = build_transforms(config['preprocessing'])
    explanations, evaluations, timings_rows, stability = [], [], [], []
    methods = ('attention', 'lime', 'occlusion', 'equal_fusion', 'fusion')
    for row in selected.itertuples():
        with Image.open(Path(config['dataset']['root']) / 'train' / f'{row.id}.tif') as source:
            image = transform.transforms[0](source.convert('RGB'))
        prob = float(predict_prob(model, transform(image).unsqueeze(0), device)[0])
        if abs(prob - row.prob_cancer) > 0.002:
            raise ValueError(f'Prediction cannot be reproduced for {row.id}; check model/preprocessing.')
        sample_maps = {name: [] for name in methods}
        repeats = args.repeats if args.mode == 'quantitative' else 1
        for repeat in range(repeats):
            seed = args.seed + repeat + int(object_hash(row.id)[:8], 16) % 1000000
            maps, timings = explain(model, image, transform, device, seed, args.grid_size,
                                    args.lime_samples, args.occlusion_patch, args.occlusion_stride, weights)
            prefix = f'{row.id}-r{repeat}'
            np.savez_compressed(output / f'{prefix}.npz', **maps)
            timings_rows.append({'id': row.id, 'repeat': repeat, **timings})
            for name in methods:
                sample_maps[name].append(maps[name])
            if repeat == 0:
                for name in (*methods, 'disagreement'):
                    save_heatmap_triplet(image, maps[name], output / f'{prefix}-{name}.png',
                                         f'{name}: relative importance', f'cancer score={prob:.4f}')
                record = {'id': row.id, 'model': config['model']['name'], 'prob_cancer': prob,
                          'threshold': decision['threshold'], 'pred_label': int(prob >= decision['threshold']),
                          'true_label': int(row.label), 'group': confusion_group(row.label, int(prob >= decision['threshold'])),
                          'target': 'cancer_score', 'source': expected, 'decision': decision,
                          'top_attention_regions': attention_region_summary(maps['attention'])[:3],
                          'top_lime_regions': attention_region_summary(maps['lime'])[:3],
                          'top_occlusion_regions': attention_region_summary(maps['occlusion'])[:3],
                          'top_final_score_regions': attention_region_summary(maps['fusion'])[:3],
                          'fusion_weights': list(weights), 'fusion_status': 'exploratory',
                          'mean_method_disagreement': float(maps['disagreement'].mean()),
                          'relative_high_score_area_ratio_0_7': high_score_area_ratio(maps['fusion']),
                          'arrays': f'{prefix}.npz'}
                explanations.append(record)
                save_json(explanations, output / 'xai_explanations.json')
            if args.mode == 'quantitative':
                curves = deletion_curves(model, image, transform, device, maps, seed, args.grid_size)
                for method, curve in curves.items():
                    evaluations.append({'id': row.id, 'repeat': repeat, 'method': method, **curve})
        if args.mode == 'quantitative':
            for method, arrays in sample_maps.items():
                stability.append({'id': row.id, 'method': method, 'repeats': repeats,
                                  'mean_pixel_std': float(np.std(np.stack(arrays), axis=0).mean()) if repeats > 1 else None})
    pd.DataFrame(timings_rows).to_csv(output / 'timings.csv', index=False)
    pd.DataFrame(explanations).to_csv(output / 'xai_explanations.csv', index=False)
    if evaluations:
        save_json(evaluations, output / 'deletion_curves.json')
        pd.DataFrame(stability).to_csv(output / 'stability.csv', index=False)
        pd.DataFrame(stability).groupby('method')['mean_pixel_std'].agg(['mean', 'std', 'count']).to_csv(output / 'stability_summary.csv')
        results = pd.DataFrame(evaluations)
        results.to_csv(output / 'evaluation_per_sample.csv', index=False)
        # Average repeated seeds within each image before across-image statistics.
        per_image = results.groupby(['id', 'method'])[['deletion_auc', 'score_drop_at_half']].mean().reset_index()
        per_image.to_csv(output / 'evaluation_per_image.csv', index=False)
        per_image.groupby('method')[['deletion_auc', 'score_drop_at_half']].agg(['mean', 'std', 'count']).to_csv(output / 'evaluation_summary.csv')
        timing_summary = pd.DataFrame(timings_rows).drop(columns=['id', 'repeat']).agg(['mean', 'std'])
        timing_summary.to_csv(output / 'timing_summary.csv')
    print(output)

if __name__ == '__main__':
    main()
