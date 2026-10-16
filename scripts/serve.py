"""Run the local-only pathology workbench; model/API settings come from .env."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.utils import configure_logging
from cancer_pipeline.webapp import create_app


def main():
    parser = argparse.ArgumentParser(description='本机病理图像辅助分析网站（仅监听 127.0.0.1）。')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    configure_logging()
    app = create_app()
    print(f'本机工作台：http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False, threaded=True)

if __name__ == '__main__':
    main()
