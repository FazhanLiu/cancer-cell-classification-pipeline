import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cancer_pipeline.artifacts import new_run, read_predictions, save_json
from cancer_pipeline.metrics import classification_metrics


def tune(predictions, metadata, start=0.05, stop=0.95, step=0.01, objective='f1', target_recall=None):
    if metadata.get('split') != 'val' or metadata.get('source_status') != 'verified':
        raise ValueError('Threshold selection requires verified validation predictions.')
    if not all(k in metadata for k in ('checkpoint_hash', 'manifest_hash', 'preprocessing', 'csv_hash')):
        raise ValueError('Incomplete validation provenance.')
    if not np.isfinite([start, stop, step]).all() or not 0 <= start <= stop <= 1 or step <= 0 or (stop - start) / step > 10000:
        raise ValueError('Invalid search range/step (maximum 10001 candidates).')
    if objective not in ('f1', 'recall_precision'):
        raise ValueError('Unknown selection objective.')
    if objective == 'recall_precision' and (target_recall is None or not np.isfinite(target_recall) or not 0 <= target_recall <= 1):
        raise ValueError('Recall objective needs --target-recall in [0,1].')
    if predictions.label.nunique() != 2:
        raise ValueError('Threshold selection requires both classes.')
    rows = []
    thresholds = np.arange(start, stop + step * 1e-8, step)
    for threshold in thresholds:
        m = classification_metrics(predictions.label, predictions.prob_cancer, min(float(threshold), 1.0))
        cm = m.pop('confusion_matrix')
        rows.append({**m, 'tn': cm[0][0], 'fp': cm[0][1], 'fn': cm[1][0], 'tp': cm[1][1]})
    results = pd.DataFrame(rows)
    feasible = results if objective == 'f1' else results[results.recall >= target_recall]
    if feasible.empty:
        raise ValueError('No feasible threshold; no decision artifact was created.')
    score = 'f1' if objective == 'f1' else 'precision'
    winner = feasible.sort_values([score, 'threshold'], ascending=[False, False]).iloc[0]
    decision = {'schema_version': 1, 'threshold': float(winner.threshold), 'objective': objective,
                'target_recall': target_recall, 'split': 'val',
                **{k: metadata[k] for k in ('checkpoint_hash', 'manifest_hash', 'preprocessing', 'csv_hash')},
                'search': {'start': start, 'stop': stop, 'step': step}}
    return results, decision


def main():
    parser = argparse.ArgumentParser(description='Select a threshold using verified validation predictions only.')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', default='outputs/threshold_tuning', help='Base directory; a unique run directory is created.')
    parser.add_argument('--start', type=float, default=0.05)
    parser.add_argument('--stop', type=float, default=0.95)
    parser.add_argument('--step', type=float, default=0.01)
    parser.add_argument('--objective', choices=['f1', 'recall_precision'], default='f1')
    parser.add_argument('--target-recall', type=float)
    args = parser.parse_args()
    frame, metadata = read_predictions(args.predictions)
    results, decision = tune(frame, metadata, args.start, args.stop, args.step, args.objective, args.target_recall)
    output = new_run({'output_dir': args.output, 'run_name': 'selection', **vars(args)}, 'threshold')
    results.to_csv(output / 'threshold_tuning.csv', index=False)
    save_json(decision, output / 'decision.json')
    print(output / 'decision.json')

if __name__ == '__main__':
    main()
