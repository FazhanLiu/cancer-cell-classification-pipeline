import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.training import add_hard_mining_columns, main, make_sample_weights

if __name__ == '__main__':
    main('hard')
