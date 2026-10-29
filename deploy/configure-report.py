#!/usr/bin/env python3
"""Configure any supported report API using JSON on stdin; never print keys."""
import grp
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if os.geteuid() != 0 or len(sys.argv) != 2 or sys.argv[1] not in {'dev', 'prod'}:
        raise SystemExit('Usage: sudo python3 cancer-configure-report.py dev|prod < private-config.json')
    environment = sys.argv[1]
    raw = sys.stdin.read(65537)
    if len(raw) > 65536:
        raise SystemExit('Configuration too large')
    config = json.loads(raw)
    if not isinstance(config, dict) or set(config) - {'provider', 'model', 'base_url', 'api_key'}:
        raise SystemExit('Expected provider, model, base_url and api_key')
    provider = config.get('provider', 'compatible')
    if provider not in {'compatible', 'deepseek'}:
        raise SystemExit('Provider must be compatible or deepseek')
    for name in ('model', 'api_key'):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise SystemExit(f'Missing {name}')
    if provider == 'compatible' and not str(config.get('base_url', '')).startswith('https://'):
        raise SystemExit('An HTTPS API base_url is required')
    values = {'LLM_PROVIDER': provider, 'LLM_MODEL': config['model'],
              'LLM_BASE_URL': config.get('base_url', ''),
              'LLM_API_KEY' if provider == 'compatible' else 'DEEPSEEK_API_KEY': config['api_key']}
    if any(not isinstance(value, str) or any(ord(c) < 32 for c in value) for value in values.values()):
        raise SystemExit('Configuration values must be single-line text')
    path = Path(f'/etc/cancer/{environment}.env')
    previous = path.read_text()
    lines = [line for line in previous.splitlines() if line.split('=', 1)[0] not in values]
    for key, value in values.items():
        quoted = value.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{key}="{quoted}"')

    def save(content):
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{environment}-', dir=path.parent)
        try:
            os.fchmod(descriptor, 0o640)
            os.fchown(descriptor, 0, grp.getgrnam(f'cancer-{environment}').gr_gid)
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    save('\n'.join(lines) + '\n')
    if Path(f'/srv/cancer-{environment}/current').exists():
        result = subprocess.run(['systemctl', 'restart', f'cancer@{environment}.service'])
        if result.returncode:
            save(previous)
            subprocess.run(['systemctl', 'restart', f'cancer@{environment}.service'])
            raise SystemExit('Service restart failed; previous configuration restored')
    print(f'{environment}: report configuration updated; no credentials displayed')


if __name__ == '__main__':
    main()
