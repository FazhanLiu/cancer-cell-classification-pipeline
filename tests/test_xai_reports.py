import json

import numpy as np
import pytest
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from cancer_pipeline.data import build_transforms
from cancer_pipeline.reports import build_prompt, generate_reports
from cancer_pipeline.xai import (attention_region_summary, deletion_curves, explain,
                                 high_score_area_ratio, normalize_map, weighted_final_score)
from scripts.xai_vit import select_samples


def test_heatmap_edges_and_weights():
    zero = np.zeros((4, 4))
    assert not normalize_map(zero).any()
    assert attention_region_summary(zero) == []
    assert high_score_area_ratio(zero) == 0
    negative = -np.ones((4, 4))
    assert not normalize_map(negative).any()
    for weights in [(-1, 2, 1), (0, 0, 0), (np.nan, 1, 1), (np.inf, 1, 1)]:
        with pytest.raises(ValueError):
            weighted_final_score(zero, zero, zero, *weights)
    with pytest.raises(ValueError):
        normalize_map(np.array([[np.nan]]))


def test_untrained_vit_xai_and_deletion_smoke():
    torch.manual_seed(7)
    model = VisionTransformer(img_size=32, patch_size=8, embed_dim=24, depth=2, num_heads=3, num_classes=2).eval()
    image = Image.fromarray(np.random.default_rng(3).integers(0, 255, (32, 32, 3), dtype=np.uint8))
    transform = build_transforms({'image_size': 32, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    maps, timings = explain(model, image, transform, torch.device('cpu'), grid_size=2, lime_samples=8,
                            occlusion_patch=16, occlusion_stride=16)
    assert maps['fusion'].shape == (32, 32)
    assert all(np.isfinite(array).all() for array in maps.values())
    assert all(value >= 0 for value in timings.values())
    assert 'lime_raw' in maps and 'occlusion_raw' in maps
    assert all(len(block.attn.attn_drop._forward_hooks) == 0 for block in model.blocks)
    curves = deletion_curves(model, image, transform, torch.device('cpu'), maps, grid_size=2)
    assert set(curves) == {'attention', 'lime', 'occlusion', 'equal_fusion', 'fusion', 'random'}
    initial = [curve['probabilities'][0] for curve in curves.values()]
    final = [curve['probabilities'][-1] for curve in curves.values()]
    np.testing.assert_allclose(initial, initial[0], atol=1e-7)
    np.testing.assert_allclose(final, final[0], atol=1e-7)


def sample(identifier='a'):
    return {'id': identifier, 'model': 'vit', 'prob_cancer': 0.7, 'pred_label': 1, 'threshold': 0.4,
            'true_label': 0, 'group': 'FP', 'target': 'cancer_score',
            'top_attention_regions': [{'region': 'upper-left', 'score': 0.8}],
            'top_lime_regions': [], 'top_occlusion_regions': [],
            'top_final_score_regions': [{'region': 'lower-right', 'score': 0.9}],
            'fusion_weights': [0.5, 0.3, 0.2], 'mean_method_disagreement': 0.2}


def test_report_has_fusion_and_no_truth_labels(tmp_path):
    value = sample()
    prompt = build_prompt(value)
    assert 'lower-right' in prompt and 'top_lime_regions' in prompt
    assert 'true_label' not in prompt and 'FP' not in prompt and 'group' not in prompt
    altered = {**value, 'true_label': 1, 'group': 'TP'}
    assert build_prompt(altered) == prompt
    results = generate_reports([value], tmp_path / 'mock', include_review=True)
    assert results[0]['status'] == 'mock'
    assert 'lower-right' in results[0]['explanation']
    assert 'true_label' not in results[0]['evidence']
    assert results[0]['offline_review'] == {'true_label': 0, 'group': 'FP'}


def test_report_failure_incremental_retry_and_resume(tmp_path):
    calls = []
    def failing(provider, prompt, model, timeout):
        calls.append(prompt)
        if '"id": "a"' in prompt:
            raise RuntimeError('temporary error')
        return 'Result based on supplied evidence.'
    directory = tmp_path / 'failed'
    result = generate_reports([sample('a'), sample('b')], directory, 'groq', 'explicit-model', retries=1, sleep=0, caller=failing)
    assert [x['status'] for x in result] == ['failed', 'api']
    assert len(calls) == 3
    saved = json.loads((directory / 'llm_explanations.json').read_text())
    assert len(saved) == 2
    retried = []
    def success(*args):
        retried.append(args)
        return 'Recovered.'
    recovered = generate_reports([sample('a'), sample('b')], tmp_path / 'recovered', 'groq', 'explicit-model', retries=0, sleep=0, previous=saved, caller=success)
    assert len(retried) == 1 and all(x['status'] == 'api' for x in recovered)


def test_quantitative_selection_independent_of_scores():
    import pandas as pd
    frame = pd.DataFrame({'id': [str(i) for i in range(40)], 'label': [i % 2 for i in range(40)], 'prob_cancer': np.linspace(0, 1, 40)})
    a = select_samples(frame, 0.5, 'quantitative', 10, 2, 42)
    b = select_samples(frame.assign(prob_cancer=1-frame.prob_cancer), 0.2, 'quantitative', 10, 2, 42)
    assert a.id.tolist() == b.id.tolist()
    assert a.label.value_counts().to_dict() == {0: 5, 1: 5}
