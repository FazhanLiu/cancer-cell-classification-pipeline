"""Hosted boundaries preserve the local workbench and isolate environment data."""
import io

import pytest
from PIL import Image

from cancer_pipeline.webapp import create_app
from cancer_pipeline.wsgi import create_hosted_app


def test_environment_proxy_tokens_and_cases_are_isolated(tmp_path, monkeypatch):
    monkeypatch.delenv('CANCER_MODEL_PATH', raising=False)
    apps = []
    try:
        for environment in ('dev', 'prod'):
            app = create_app({'TESTING': True, 'MODEL_PATH': None,
                              'OUTPUT_ROOT': str(tmp_path / environment),
                              'TRUSTED_HOSTS': ['localhost'],
                              'PROXY_TOKEN': environment * 16,
                              'DEPLOYMENT_ENV': environment, 'RELEASE': 'a' * 40})
            apps.append(app)
        dev, prod = (app.test_client() for app in apps)
        assert prod.get('/api/cases').status_code == 403
        assert prod.get('/api/cases', headers={'X-Cancer-Proxy-Token': 'dev' * 16}).status_code == 403
        prod_headers = {'X-Cancer-Proxy-Token': 'prod' * 16,
                        'X-Local-Token': apps[1].config['LOCAL_TOKEN']}
        stream = io.BytesIO()
        Image.new('RGB', (32, 32)).save(stream, format='PNG')
        stream.seek(0)
        response = prod.post('/api/cases', data={'images': (stream, 'image.png')}, headers=prod_headers)
        assert response.status_code == 201
        case_id = response.get_json()['id']
        assert dev.get('/api/cases', headers={'X-Cancer-Proxy-Token': 'dev' * 16}).get_json()['cases'] == []
        assert dev.get(f'/api/cases/{case_id}', headers={'X-Cancer-Proxy-Token': 'dev' * 16}).status_code == 404
        assert prod.delete(f'/api/cases/{case_id}', headers={'X-Cancer-Proxy-Token': 'prod' * 16}).status_code == 403
        assert prod.get('/healthz').get_json() == {
            'status': 'ok', 'environment': 'prod', 'release': 'a' * 40, 'model_ready': False}
    finally:
        for app in apps:
            app.extensions['case_jobs'].close()


@pytest.mark.parametrize('environment,output,token', [
    ('other', '/var/lib/cancer-other/cases', 'x' * 64),
    ('dev', '/var/lib/cancer-prod/cases', 'x' * 64),
    ('prod', '/var/lib/cancer-prod/cases', 'short'),
])
def test_hosted_configuration_rejects_cross_environment_paths(monkeypatch, environment, output, token):
    monkeypatch.setenv('CANCER_ENV', environment)
    monkeypatch.setenv('CANCER_OUTPUT_ROOT', output)
    monkeypatch.setenv('CANCER_PROXY_TOKEN', token)
    with pytest.raises(ValueError):
        create_hosted_app()


def test_configured_model_failure_makes_release_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr('cancer_pipeline.webapp.PatchInference', lambda *args: (_ for _ in ()).throw(ValueError('bad model')))
    app = create_app({'OUTPUT_ROOT': str(tmp_path), 'MODEL_PATH': '/missing/trusted.pt'})
    try:
        assert app.test_client().get('/healthz').status_code == 503
    finally:
        app.extensions['case_jobs'].close()
