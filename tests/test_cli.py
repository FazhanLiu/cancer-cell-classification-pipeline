import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from cancer_pipeline.artifacts import load_checkpoint, save_json

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'}
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, args)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return Path(result.stdout.strip().splitlines()[-1])


def test_synthetic_cli_end_to_end(experiment, tmp_path):
    config_path, config = experiment
    run('prepare_data.py', '--config', config_path)
    output = run('train.py', '--config', config_path, '--stop-after-epoch', 1)
    assert (output / 'best.pt').is_file() and not (output / 'test_predictions.csv').exists()
    checkpoint = load_checkpoint(output / 'last.pt')
    assert checkpoint['epoch'] == 1 and 'optimizer' in checkpoint and 'rng_state' in checkpoint
    resumed = run('train.py', '--config', config_path, '--resume', output / 'last.pt')
    assert load_checkpoint(resumed / 'last.pt')['epoch'] == config['train']['epochs']
    decision = run('threshold_tuning.py', '--predictions', resumed / 'val_predictions.csv', '--output', tmp_path / 'selection')
    evaluation = run('evaluate.py', '--checkpoint', resumed / 'best.pt', '--decision', decision, '--device', 'cpu')
    meta = json.loads((evaluation / 'test_predictions.csv.meta.json').read_text())
    chosen = json.loads(decision.read_text())
    assert meta['decision']['threshold'] == chosen['threshold']
    assert meta['split'] == 'test'
    metrics = json.loads((evaluation / 'eval_metrics.json').read_text())
    assert metrics['metrics']['threshold'] == chosen['threshold']
    samples = [{'id': 'report-only', 'model': 'small_cnn', 'pred_label': 1, 'prob_cancer': 0.8,
                'threshold': chosen['threshold'] if chosen['threshold'] <= 0.8 else 0.5,
                'top_final_score_regions': [], 'top_attention_regions': []}]
    input_path = tmp_path / 'evidence.json'
    save_json(samples, input_path)
    report = run('llm_explain_xai.py', '--input', input_path, '--output', tmp_path / 'reports')
    # Report CLI also prints a failure count on the same final line.
    report_path = Path(str(report).split(' (failed samples:')[0])
    assert (report_path / 'llm_explanations.md').exists()
    # Both fine-tuning strategies reuse the verified base split.
    hard = run('train_vit_hard_mining.py', '--config', config_path, '--base-checkpoint', resumed / 'best.pt', '--stop-after-epoch', 1)
    assert (hard / 'train_hard_examples.csv').is_file()
    hard_state = load_checkpoint(hard / 'last.pt')
    assert hard_state['config']['train']['class_weight'] is False
    assert 'hard_weight' not in hard_state['config']['hard_mining']
    hard_resumed = run('train_vit_hard_mining.py', '--config', config_path, '--resume', hard / 'last.pt')
    assert load_checkpoint(hard_resumed / 'last.pt')['epoch'] == config['train']['epochs']
    focal = run('train_vit_focal_loss.py', '--config', config_path, '--fine-tune', resumed / 'best.pt', '--epochs', 1)
    assert load_checkpoint(focal / 'last.pt')['config']['loss']['name'] == 'focal_loss'
    # Ensemble members are deliberately identical: outputs should align by IDs.
    ensemble_path = tmp_path / 'ensemble.yaml'
    ensemble_path.write_text(yaml.safe_dump({'run_name': 'ensemble', 'output_dir': str(tmp_path / 'ensemble'),
                                            'models': [{'checkpoint': str(resumed / 'best.pt'), 'weight': 1},
                                                       {'checkpoint': str(resumed / 'best.pt'), 'weight': 2}]}))
    combined = run('ensemble.py', '--config', ensemble_path)
    original = pd.read_csv(resumed / 'val_predictions.csv', dtype={'id': str}).set_index('id').sort_index()
    averaged = pd.read_csv(combined / 'val_predictions.csv', dtype={'id': str}).set_index('id').sort_index()
    torch.testing.assert_close(torch.tensor(original.prob_cancer.to_numpy()), torch.tensor(averaged.prob_cancer.to_numpy()))


def test_vit_cli_quantitative_xai_report(experiment, tmp_path):
    config_path, config = experiment
    # A tiny, unpretrained ViT keeps the entire artifact chain CPU/offline-testable.
    raw = yaml.safe_load(config_path.read_text())
    raw['model'] = {'name': 'vit_tiny_patch16_224', 'pretrained': False, 'num_classes': 2,
                    'img_size': 32, 'embed_dim': 24, 'depth': 2, 'num_heads': 3}
    raw['train']['epochs'] = 1
    config_path.write_text(yaml.safe_dump(raw))
    run('prepare_data.py', '--config', config_path)
    trained = run('train.py', '--config', config_path)
    explained = run('xai_vit.py', '--checkpoint', trained / 'best.pt', '--predictions', trained / 'val_predictions.csv',
                    '--output', tmp_path / 'xai', '--mode', 'quantitative', '--samples', 4, '--repeats', 2,
                    '--lime-samples', 8, '--grid-size', 2, '--occlusion-patch', 16, '--occlusion-stride', 16,
                    '--device', 'cpu')
    rows = json.loads((explained / 'xai_explanations.json').read_text())
    assert len(rows) == 4
    assert len(list(explained.glob('*.npz'))) == 8
    curves = json.loads((explained / 'deletion_curves.json').read_text())
    assert len(curves) == 4 * 2 * 6
    assert (explained / 'evaluation_summary.csv').is_file()
    report = run('llm_explain_xai.py', '--input', explained / 'xai_explanations.json', '--output', tmp_path / 'reports')
    report_path = Path(str(report).split(' (failed samples:')[0])
    records = json.loads((report_path / 'llm_explanations.json').read_text())
    assert len(records) == 4 and all(x['status'] == 'mock' for x in records)
    assert all('true_label' not in x['evidence'] and 'group' not in x['evidence'] for x in records)
