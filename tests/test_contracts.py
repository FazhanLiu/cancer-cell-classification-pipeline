import copy

import numpy as np
import pandas as pd
import pytest
import torch

from cancer_pipeline.artifacts import (assert_source, checkpoint_model, load_decision, object_hash,
                                      read_json, read_predictions, save_json, write_predictions)
from cancer_pipeline.data import load_manifest, prepare_manifest
from cancer_pipeline.metrics import classification_metrics
from cancer_pipeline.training import make_sample_weights
from scripts.threshold_tuning import tune


def test_manifest_fixed_and_data_changes_rejected(experiment):
    _, config = experiment
    path = prepare_manifest(config)
    original = path.read_bytes()
    frames, provenance = load_manifest(config)
    assert sum(map(len, frames.values())) == 40
    assert [len(frames[s]) for s in ('train', 'val', 'test')] == [28, 6, 6]
    assert len(set(frames['train'].id) & set(frames['val'].id)) == 0
    assert prepare_manifest(config).read_bytes() == original
    different = copy.deepcopy(config)
    different['seed'] = 123
    with pytest.raises(ValueError, match='parameters changed'):
        load_manifest(different)
    import pathlib
    labels = pathlib.Path(config['dataset']['root']) / 'train_labels.csv'
    frame = pd.read_csv(labels, dtype={'id': str})
    frame.loc[0, 'label'] = 1
    frame.to_csv(labels, index=False)
    with pytest.raises(ValueError, match='data changed'):
        load_manifest(config)
    assert provenance['split_mode'] == 'patch_random_no_group_guarantee'


def test_group_split_and_tampered_overlap(experiment):
    _, config = experiment
    config['dataset']['group_column'] = 'slide'
    path = prepare_manifest(config)
    frames, _ = load_manifest(config)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(frames[a].slide) & set(frames[b].slide)
    manifest = read_json(path)
    manifest['records'][0]['id'] = manifest['records'][1]['id']
    manifest['manifest_hash'] = object_hash({k: v for k, v in manifest.items() if k != 'manifest_hash'})
    save_json(manifest, path)
    with pytest.raises(ValueError, match='every ID exactly once'):
        load_manifest(config)


def test_missing_images_and_duplicate_ids(experiment):
    _, config = experiment
    from pathlib import Path
    labels = Path(config['dataset']['root']) / 'train_labels.csv'
    data = pd.read_csv(labels, dtype={'id': str})
    data.loc[1, 'id'] = data.loc[0, 'id']
    data.to_csv(labels, index=False)
    with pytest.raises(ValueError, match='unique'):
        prepare_manifest(config)
    data.loc[1, 'id'] = 'missing'
    data.to_csv(labels, index=False)
    with pytest.raises(ValueError, match='Missing'):
        prepare_manifest(config)


def test_prediction_threshold_contract_and_rejection(tmp_path):
    frame = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'label': [0, 1, 0, 1], 'prob_cancer': [0.1, 0.4, 0.3, 0.9]})
    metadata = {'split': 'val', 'checkpoint_hash': 'model', 'manifest_hash': 'manifest',
                'preprocessing': {'image_size': 32}, 'source_status': 'verified'}
    path = tmp_path / 'predictions.csv'
    write_predictions(frame, path, metadata)
    loaded, source = read_predictions(path)
    results, decision = tune(loaded, source, 0.3, 0.4, 0.1)
    assert decision['threshold'] == pytest.approx(0.4)
    save_json(decision, tmp_path / 'decision.json')
    assert load_decision(tmp_path / 'decision.json', {'checkpoint_hash': 'model'})['threshold'] == decision['threshold']
    assert classification_metrics(frame.label, frame.prob_cancer, decision['threshold'])['f1'] == 1
    with pytest.raises(ValueError, match='Decision source mismatch'):
        load_decision(tmp_path / 'decision.json', {'checkpoint_hash': 'other'})
    with pytest.raises(ValueError, match='validation'):
        tune(frame, {**source, 'split': 'test'})
    with pytest.raises(ValueError, match='range/step'):
        tune(frame, source, step=0)
    with pytest.raises(ValueError, match='No feasible'):
        tune(frame, source, start=0.95, stop=0.95, objective='recall_precision', target_recall=1)
    path.write_text(path.read_text().replace('0.1', '0.2'))
    with pytest.raises(ValueError, match='changed'):
        read_predictions(path)


def test_legacy_and_source_mismatch(experiment, tmp_path):
    _, config = experiment
    prepare_manifest(config)
    _, provenance = load_manifest(config)
    with pytest.raises(ValueError, match='contamination'):
        assert_source({'provenance': {'manifest_hash': 'other'}}, provenance)
    with pytest.raises(ValueError, match='Unverified'):
        assert_source({}, provenance)
    assert assert_source({}, provenance, allow_legacy=True) == 'unverified'
    from cancer_pipeline.models import create_model
    model = create_model('small_cnn', pretrained=False)
    path = tmp_path / 'legacy.pt'
    torch.save({'model': model.state_dict(), 'config': config}, path)
    with pytest.raises(ValueError, match='Legacy'):
        checkpoint_model(path, 'cpu')
    loaded, _, state = checkpoint_model(path, 'cpu', allow_legacy=True)
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_metrics_edge_cases_and_sampler():
    result = classification_metrics([0, 0], [0.1, 0.9], 0.5)
    assert result['auc'] is None and result['confusion_matrix'] == [[1, 1], [0, 0]]
    for probabilities in ([np.nan], [1.1], [-0.1]):
        with pytest.raises(ValueError):
            classification_metrics([0], probabilities)
    frame = pd.DataFrame({'is_fp': [True, False], 'is_fn': [False, True]})
    assert make_sample_weights(frame, 4, 6).tolist() == [4, 6]
    for weights in [(-1, 1), (float('nan'), 1), (0, 0)]:
        with pytest.raises(ValueError):
            make_sample_weights(frame, *weights)


def test_model_loading_forwards_architecture_and_never_downloads(experiment, tmp_path, monkeypatch):
    import timm
    from cancer_pipeline.models import create_model
    _, config = experiment
    config['model'] = {'name': 'vit_tiny_patch16_224', 'pretrained': True, 'num_classes': 2, 'img_size': 32}
    calls = []
    def factory(name, **kwargs):
        calls.append((name, kwargs))
        return torch.nn.Linear(3, 2)
    monkeypatch.setattr(timm, 'create_model', factory)
    model = create_model('vit_tiny_patch16_224', pretrained=False, img_size=32)
    assert calls[-1][1]['img_size'] == 32
    path = tmp_path / 'verified.pt'
    torch.save({'model': model.state_dict(), 'config': config,
                'provenance': {'manifest_hash': 'test-only'}}, path)
    checkpoint_model(path, 'cpu')
    assert calls[-1][1]['pretrained'] is False and calls[-1][1]['img_size'] == 32


def test_plotting_uses_recorded_threshold(tmp_path, monkeypatch):
    import cancer_pipeline.plots as plots
    from sklearn.metrics import confusion_matrix
    frame = pd.DataFrame({'id': ['a', 'b'], 'label': [0, 1], 'prob_cancer': [0.1, 0.4]})
    path = tmp_path / 'predictions.csv'
    write_predictions(frame, path, {'decision': {'threshold': 0.3}})
    captured = []
    def capture(y_true, y_pred, **kwargs):
        captured.append(list(y_pred))
        return confusion_matrix(y_true, y_pred, **kwargs)
    monkeypatch.setattr(plots, 'confusion_matrix', capture)
    plots.plot_predictions(path, tmp_path)
    assert captured == [[0, 1]]
