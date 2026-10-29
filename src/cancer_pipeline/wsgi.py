"""Gunicorn entrypoint for the isolated hosted workbench."""
import os
from pathlib import Path

from cancer_pipeline.webapp import create_app


def create_hosted_app():
    environment = os.environ['CANCER_ENV']
    domains = {'dev': 'dev.cancercell.otisliu.com', 'prod': 'cancercell.otisliu.com'}
    if environment not in domains:
        raise ValueError('CANCER_ENV must be dev or prod')
    output = Path(os.environ['CANCER_OUTPUT_ROOT'])
    if output != Path(f'/var/lib/cancer-{environment}/cases'):
        raise ValueError('Unexpected environment data path')
    token = os.environ['CANCER_PROXY_TOKEN']
    if len(token) < 32:
        raise ValueError('A unique proxy token of at least 32 characters is required')
    return create_app({'OUTPUT_ROOT': str(output), 'PROXY_TOKEN': token,
                       'TRUSTED_HOSTS': [domains[environment], 'localhost', '127.0.0.1'],
                       'DEPLOYMENT_ENV': environment,
                       'RELEASE': os.environ['CANCER_RELEASE']})
