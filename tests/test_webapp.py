"""Local website integration with a real tiny ViT and intercepted LLM requests."""
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cancer_pipeline.artifacts import file_hash, save_json
from cancer_pipeline.inference import PatchInference
from cancer_pipeline.models import create_model
from cancer_pipeline.webapp import CaseStore, create_app


@pytest.fixture
def runtime(tmp_path):
    config = {'model': {'name': 'vit_tiny_patch16_224', 'pretrained': False, 'num_classes': 2,
                        'img_size': 32, 'embed_dim': 24, 'depth': 2, 'num_heads': 3},
              'preprocessing': {'image_size': 32, 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                                'interpolation': 'bilinear', 'antialias': True}}
    cfg = dict(config['model'])
    model = create_model(cfg.pop('name'), **cfg)
    path = tmp_path / 'functional-test-only-untrained.pt'
    torch.save({'model': model.state_dict(), 'config': config,
                'provenance': {'manifest_hash': 'synthetic-functional-test', 'source_status': 'verified'}}, path)
    return PatchInference(path, device='cpu', xai_options={'grid_size': 2, 'lime_samples': 8,
                                                        'occlusion_patch': 16, 'occlusion_stride': 16})


def png(value=100):
    stream = io.BytesIO()
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(stream, format='PNG')
    stream.seek(0)
    return stream


def configured_app(tmp_path, monkeypatch, runtime=None, caller=None):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'unit-test-only-key')
    app = create_app({'TESTING': True, 'MODEL_PATH': None, 'OUTPUT_ROOT': str(tmp_path / 'cases')}, runtime, caller)
    return app, app.test_client(), {'X-Local-Token': app.config['LOCAL_TOKEN']}


def upload(client, headers, count=2):
    return client.post('/api/cases', data={'reference': 'PRIVATE-CASE-REFERENCE',
                                         'images': [(png(80+i), 'patient-private-name.png') for i in range(count)]},
                       headers=headers, content_type='multipart/form-data')


def wait_case(app, case_id):
    jobs = app.extensions['case_jobs']
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with jobs.lock:
            pending = case_id in jobs.active
        if not pending:
            return app.extensions['case_store'].get(case_id)
        time.sleep(0.02)
    pytest.fail('Case job timed out')


def test_no_model_fails_closed_and_drafts_survive_restart(tmp_path, monkeypatch):
    app, client, headers = configured_app(tmp_path, monkeypatch)
    assert client.get('/').status_code == 200
    assert 'PathScope' in client.get('/').text
    status = client.get('/api/status').get_json()
    assert status['model_ready'] is False and status['model'] is None
    created = upload(client, headers)
    assert created.status_code == 201
    case = created.get_json()
    assert case['status'] == 'uploaded' and all(x['prediction'] is None for x in case['images'])
    result = client.post(f"/api/cases/{case['id']}/analyze", headers=headers)
    assert result.status_code == 409
    restored = CaseStore(tmp_path / 'cases').get(case['id'])
    assert restored == case
    assert client.get(f"/api/cases/{case['id']}/images/image_001/preview.png").status_code == 200
    assert 'unit-test-only-key' not in client.get('/api/status').text
    app.extensions['case_jobs'].close()


def test_real_unlabelled_batch_xai_and_case_report(tmp_path, monkeypatch, runtime):
    prompts = []
    def caller(provider, prompt, model, timeout):
        assert provider == 'deepseek'
        prompts.append(prompt)
        return '病例图像概览\n已完成两张图像的模型分析，需由医生复核。'
    app, client, headers = configured_app(tmp_path, monkeypatch, runtime, caller)
    created = upload(client, headers).get_json()
    case_id = created['id']
    assert client.post(f'/api/cases/{case_id}/analyze', headers=headers).status_code == 202
    case = wait_case(app, case_id)
    assert case['status'] == 'complete' and case['report']['status'] == 'complete'
    assert len(prompts) == 1
    assert 'PRIVATE-CASE-REFERENCE' not in prompts[0] and 'patient-private-name' not in prompts[0]
    assert 'true_label' not in prompts[0] and 'top_final_score_regions' in prompts[0]
    for image in case['images']:
        assert image['status'] == 'complete'
        assert 0 <= image['prediction']['prob_cancer'] <= 1
        for asset in ('attention.png', 'lime.png', 'occlusion.png', 'fusion.png', 'disagreement.png', 'maps.npz'):
            assert client.get(f"/api/cases/{case_id}/images/{image['id']}/{asset}").status_code == 200
        arrays = np.load(tmp_path / 'cases' / case_id / image['id'] / 'maps.npz')
        assert 'lime_raw' in arrays and 'occlusion_raw' in arrays
    assert client.get(f'/api/cases/{case_id}/download/report').status_code == 200
    assert client.post(f'/api/cases/{case_id}/analyze', headers=headers).status_code == 409
    assert len(prompts) == 1
    for path in (tmp_path / 'cases').rglob('*.json'):
        assert 'unit-test-only-key' not in path.read_text()
    assert client.delete(f'/api/cases/{case_id}', headers=headers).status_code == 200
    assert not (tmp_path / 'cases' / case_id).exists()
    app.extensions['case_jobs'].close()


def test_failed_report_retry_reuses_image_results(tmp_path, monkeypatch, runtime):
    calls = []
    analyses = []
    original = runtime.analyze
    def analyze(*args):
        analyses.append(args[2])
        return original(*args)
    runtime.analyze = analyze
    def caller(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError('temporary unit-test-only-key error')
        return '中文病例报告：请复核模型结果。'
    app, client, headers = configured_app(tmp_path, monkeypatch, runtime, caller)
    case_id = upload(client, headers).get_json()['id']
    client.post(f'/api/cases/{case_id}/analyze', headers=headers)
    first = wait_case(app, case_id)
    assert first['status'] == 'partial' and first['report']['status'] == 'failed'
    assert 'unit-test-only-key' not in first['report']['error']
    assert client.get(f'/api/cases/{case_id}/download/report').status_code == 404
    client.post(f'/api/cases/{case_id}/analyze', headers=headers)
    second = wait_case(app, case_id)
    assert second['status'] == 'complete' and len(calls) == 2
    assert len(analyses) == 2  # two images, never recomputed on report retry
    app.extensions['case_jobs'].close()


def test_upload_limits_validation_paths_and_local_boundary(tmp_path, monkeypatch):
    app, client, headers = configured_app(tmp_path, monkeypatch)
    assert upload(client, {}).status_code == 403
    assert client.get('/api/status', environ_overrides={'REMOTE_ADDR': '192.0.2.1'}).status_code == 403
    assert client.get('/api/status', headers={'Host': 'attacker.example'}).status_code == 400
    assert upload(client, headers, count=21).status_code == 400
    invalid = client.post('/api/cases', data={'images': [(io.BytesIO(b'invalid'), 'corrupt.png')]}, headers=headers)
    assert invalid.status_code == 400
    # A corrupt later image must roll back the whole draft, not leave orphan uploads.
    mixed = client.post('/api/cases', data={'images': [(png(), '../../outside.png'), (io.BytesIO(b'x'), 'bad.png')]}, headers=headers)
    assert mixed.status_code == 400
    assert client.get('/api/cases').get_json()['cases'] == []
    created = client.post('/api/cases', data={'images': [(png(), '../../outside.png')]}, headers=headers).get_json()
    assert created['images'][0]['filename'] == 'outside.png'
    assert not (tmp_path / 'outside.png').exists()
    case_id = created['id']
    assert client.get(f'/api/cases/{case_id}/images/image_001/input.png').status_code == 404
    assert client.get(f'/api/cases/{case_id}/images/image_001/.env').status_code == 404
    assert client.get('/.env').status_code == 404
    app.config['MAX_CONTENT_LENGTH'] = 10
    assert upload(client, headers, count=1).status_code == 413
    app.extensions['case_jobs'].close()


def test_interrupted_case_is_recoverable(tmp_path, monkeypatch):
    app, client, headers = configured_app(tmp_path, monkeypatch)
    case_id = upload(client, headers).get_json()['id']
    store = app.extensions['case_store']
    def running(case):
        case['status'] = 'running'
        case['images'][0]['status'] = 'explaining'
        case['report']['status'] = 'generating'
    store.update(case_id, running)
    recovered = CaseStore(tmp_path / 'cases').get(case_id)
    assert recovered['status'] == 'interrupted'
    assert recovered['images'][0]['status'] == 'interrupted'
    assert recovered['report']['status'] == 'failed'
    app.extensions['case_jobs'].close()


def test_individual_image_failure_does_not_cancel_remaining_images(tmp_path, monkeypatch, runtime):
    original = runtime.analyze
    def analyze(path, output, image_id, progress):
        if image_id == 'image_001':
            raise RuntimeError('unreadable sample')
        return original(path, output, image_id, progress)
    runtime.analyze = analyze
    prompts = []
    def caller(provider, prompt, model, timeout):
        prompts.append(prompt)
        return '部分图片分析失败，需复核。'
    app, client, headers = configured_app(tmp_path, monkeypatch, runtime, caller)
    case_id = upload(client, headers).get_json()['id']
    client.post(f'/api/cases/{case_id}/analyze', headers=headers)
    case = wait_case(app, case_id)
    assert case['status'] == 'partial'
    assert [x['status'] for x in case['images']] == ['failed', 'complete']
    assert '"failed_image_ids": ["image_001"]' in prompts[0]
    app.extensions['case_jobs'].close()


def test_decision_identity_and_double_submit_guard(tmp_path, monkeypatch, runtime):
    gate = threading.Event()
    original = runtime.analyze
    def analyze(*args):
        gate.wait(5)
        return original(*args)
    runtime.analyze = analyze
    app, client, headers = configured_app(tmp_path, monkeypatch, runtime, lambda *args: '已分析。')
    case_id = upload(client, headers, count=1).get_json()['id']
    try:
        assert client.post(f'/api/cases/{case_id}/analyze', headers=headers).status_code == 202
        assert client.post(f'/api/cases/{case_id}/analyze', headers=headers).status_code == 409
        assert client.delete(f'/api/cases/{case_id}', headers=headers).status_code == 409
    finally:
        gate.set()
    wait_case(app, case_id)
    store = app.extensions['case_store']
    store.update(case_id, lambda c: c.update(status='partial'))
    runtime.decision = {**runtime.decision, 'threshold': 0.75}
    assert client.post(f'/api/cases/{case_id}/analyze', headers=headers).status_code == 409
    app.extensions['case_jobs'].close()
