import json
import os
from pathlib import Path

import pytest

from cancer_pipeline.config import load_environment


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # dotenv must not leak test keys into later tests or read the user's real keys.
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    os.environ.pop('DEEPSEEK_API_KEY', None)


def test_env_file_quotes_bom_and_export_priority(tmp_path):
    path = tmp_path / '.env'
    path.write_text('\ufeff# local secrets\nDEEPSEEK_API_KEY="file-only-test-key" # comment\n', encoding='utf-8')
    assert load_environment(path)
    assert os.environ['DEEPSEEK_API_KEY'] == 'file-only-test-key'
    os.environ['DEEPSEEK_API_KEY'] = 'terminal-only-test-key'
    load_environment(path)
    assert os.environ['DEEPSEEK_API_KEY'] == 'terminal-only-test-key'


def test_missing_file_and_no_interpolation(tmp_path):
    assert not load_environment(tmp_path / 'absent.env')
    assert 'DEEPSEEK_API_KEY' not in os.environ
    path = tmp_path / '.env'
    path.write_text('DEEPSEEK_API_KEY="${UNRELATED_SECRET}"\n')
    os.environ['UNRELATED_SECRET'] = 'do-not-expand'
    load_environment(path)
    assert os.environ['DEEPSEEK_API_KEY'] == '${UNRELATED_SECRET}'


def test_cli_reads_project_env_from_other_directory_without_saving_key(tmp_path, monkeypatch):
    from cancer_pipeline import reports
    from scripts import llm_explain_xai as script
    project = tmp_path / 'project'
    project.mkdir()
    (project / '.env').write_text('DEEPSEEK_API_KEY=private-test-file-key\n')
    # Simulate a checkout, keeping the user's actual root .env entirely untouched.
    monkeypatch.setattr(script, '__file__', str(project / 'scripts' / 'llm_explain_xai.py'))
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    evidence = tmp_path / 'evidence.json'
    evidence.write_text(json.dumps([{'id': 'a', 'pred_label': 1, 'prob_cancer': 0.8, 'threshold': 0.5}]))
    output = tmp_path / 'reports'
    monkeypatch.setattr('sys.argv', ['llm_explain_xai.py', '--input', str(evidence), '--output', str(output),
                                     '--provider', 'deepseek', '--model', 'deepseek-flash', '--retries', '0', '--sleep', '0'])
    requests = []
    def request(url, headers, payload, timeout):
        assert headers['Authorization'] == 'Bearer private-test-file-key'
        requests.append(url)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'Report based on evidence.'}}]}
    monkeypatch.setattr(reports, 'request_json', request)
    script.main()
    assert requests == ['https://api.deepseek.com/chat/completions']
    records_path = next(output.rglob('llm_explanations.json'))
    assert json.loads(records_path.read_text())[0]['status'] == 'api'
    for path in output.rglob('*'):
        if path.is_file():
            assert 'private-test-file-key' not in path.read_text()
