import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.config import load_config
from cancer_pipeline.data import prepare_manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an immutable split manifest; existing manifests are validated, never replaced.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(prepare_manifest(load_config(args.config)))
