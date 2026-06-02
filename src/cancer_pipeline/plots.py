from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def plot_history(history_csv: str | Path, output_dir: str | Path) -> None:
    history_csv = Path(history_csv)
    output_dir = Path(output_dir)
    if not history_csv.exists():
        return

    history = pd.read_csv(history_csv)
    if history.empty:
        return

    plt.figure(figsize=(8, 5))
    if "train_loss" in history:
        plt.plot(history["epoch"], history["train_loss"], label="train loss")
    if "val_loss" in history:
        plt.plot(history["epoch"], history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()

    metric_cols = [c for c in ["val_auc", "val_f1", "val_recall", "val_precision", "val_accuracy"] if c in history]
    if metric_cols:
        plt.figure(figsize=(8, 5))
        for col in metric_cols:
            plt.plot(history["epoch"], history[col], label=col.replace("val_", ""))
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_curve.png", dpi=180)
        plt.close()


def plot_predictions(predictions_csv: str | Path, output_dir: str | Path, prefix: str = "test") -> None:
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    if not predictions_csv.exists():
        return

    predictions = pd.read_csv(predictions_csv)
    if predictions.empty:
        return

    y_true = predictions["label"].to_numpy()
    y_prob = predictions["prob_cancer"].to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-cancer", "cancer"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{prefix.title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(f"{prefix.title()} ROC Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=180)
    plt.close()


def create_training_plots(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    plot_history(output_dir / "history.csv", output_dir)
    plot_predictions(output_dir / "val_predictions.csv", output_dir, prefix="val")
    plot_predictions(output_dir / "test_predictions.csv", output_dir, prefix="test")
