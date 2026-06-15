import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with label and prob_cancer columns.")
    parser.add_argument("--output", default=None, help="Where to save threshold tuning results.")
    parser.add_argument("--target-fn", type=int, default=249, help="Maximum false negatives to beat or match.")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_path = Path(args.predictions)
    df = pd.read_csv(predictions_path)
    y_true = df["label"].astype(int)
    y_prob = df["prob_cancer"].astype(float)
    auc = roc_auc_score(y_true, y_prob)

    rows = []
    threshold = args.start
    while threshold <= args.stop + 1e-9:
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "threshold": round(threshold, 4),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
                "auc": auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        threshold += args.step

    results = pd.DataFrame(rows)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = predictions_path.with_name("threshold_tuning.csv")
    results.to_csv(output_path, index=False)

    feasible = results[results["fn"] < args.target_fn].copy()
    print(f"Saved threshold tuning results to: {output_path}")
    print()
    print("Default threshold 0.50:")
    print(results.iloc[(results["threshold"] - 0.5).abs().argsort()[:1]].to_string(index=False))
    print()
    if feasible.empty:
        print(f"No threshold found with FN < {args.target_fn}. Best by minimum FN:")
        print(results.sort_values(["fn", "f1"], ascending=[True, False]).head(10).to_string(index=False))
    else:
        print(f"Best thresholds with FN < {args.target_fn}, sorted by F1:")
        print(feasible.sort_values(["f1", "f2", "recall"], ascending=[False, False, False]).head(10).to_string(index=False))
        print()
        print("Best thresholds with FN < target, sorted by F2:")
        print(feasible.sort_values(["f2", "f1", "recall"], ascending=[False, False, False]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
