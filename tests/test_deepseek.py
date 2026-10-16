"""Exercise the real adapter/HTTP serialization without spending API credits."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cancer_pipeline.reports import call_llm, generate_reports


def test_deepseek_http_request_and_report_evidence(monkeypatch, tmp_path):
    from cancer_pipeline import reports
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only-deepseek-key')
    received = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': ' Report from API. '}}]}).encode()

    def urlopen(request, timeout):
        received.append((request, timeout))
        return Response()

    monkeypatch.setattr(reports.urllib.request, 'urlopen', urlopen)
    sample = {'id': 'example', 'model': 'vit', 'prob_cancer': 0.8, 'threshold': 0.5, 'pred_label': 1,
              'true_label': 0, 'group': 'FP', 'top_final_score_regions': [{'region': 'upper-left', 'score': 0.9}]}
    result = generate_reports([sample], tmp_path, 'deepseek', 'deepseek-flash', retries=0, sleep=0)
    request, timeout = received[0]
    assert request.full_url == 'https://api.deepseek.com/chat/completions'
    assert request.get_method() == 'POST'
    assert request.get_header('Authorization') == 'Bearer test-only-deepseek-key'
    payload = json.loads(request.data)
    assert payload['model'] == 'deepseek-flash'
    assert payload['thinking'] == {'type': 'disabled'} and payload['stream'] is False
    assert payload['max_tokens'] == 1024 and timeout == 60
    assert 'upper-left' in payload['messages'][1]['content']
    assert 'true_label' not in payload['messages'][1]['content']
    assert result[0]['status'] == 'api' and result[0]['explanation'] == 'Report from API.'
    assert 'test-only-deepseek-key' not in (tmp_path / 'llm_explanations.json').read_text()
    resumed = generate_reports([sample], tmp_path / 'resumed', 'deepseek', 'deepseek-flash', previous=result, sleep=0)
    assert len(received) == 1 and resumed[0]['status'] == 'api'


def test_deepseek_requires_key_without_network(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='DEEPSEEK_API_KEY is not set'):
        call_llm('deepseek', 'example', 'deepseek-flash', 60)


@pytest.mark.parametrize('response', [
    {'choices': []},
    {'choices': [{'finish_reason': 'length', 'message': {'content': 'Partial report'}}]},
    {'choices': [{'finish_reason': 'stop', 'message': {'content': None}}]},
])
def test_deepseek_rejects_incomplete_response(monkeypatch, response):
    from cancer_pipeline import reports
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only-key')
    monkeypatch.setattr(reports, 'request_json', lambda *args: response)
    with pytest.raises(RuntimeError, match='DeepSeek'):
        call_llm('deepseek', 'example', 'deepseek-flash', 60)


def test_deepseek_redacts_failures(monkeypatch, tmp_path):
    monkeypatch.setenv('DEEPSEEK_API_KEY', ' test-key-with-whitespace ')
    def failing(*args):
        raise RuntimeError('Provider error mentions test-key-with-whitespace')
    sample = {'id': 'a', 'pred_label': 1, 'prob_cancer': 0.8, 'threshold': 0.5}
    result = generate_reports([sample], tmp_path, 'deepseek', 'deepseek-flash', retries=0, sleep=0, caller=failing)
    assert result[0]['status'] == 'failed'
    assert '[REDACTED]' in result[0]['error']
    for name in ('llm_explanations.json', 'llm_explanations.md'):
        assert 'test-key-with-whitespace' not in (tmp_path / name).read_text()


def test_deepseek_cli_missing_key_fails_before_reading_input(monkeypatch, tmp_path):
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'llm_explain_xai.py'
    result = subprocess.run([sys.executable, str(script), '--provider', 'deepseek', '--model', 'deepseek-flash',
                             '--input', str(tmp_path / 'absent.json'), '--output', str(tmp_path / 'reports'),
                             '--env-file', str(tmp_path / 'absent.env')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Set DEEPSEEK_API_KEY' in result.stderr
    assert not (tmp_path / 'reports').exists()
