import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def validate_threshold(threshold):
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError('Threshold must be finite and in [0, 1].')


def validate_predictions(y_true, y_prob):
    labels, probs = np.asarray(y_true), np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probs.ndim != 1 or len(labels) == 0 or len(labels) != len(probs):
        raise ValueError('Predictions must be nonempty aligned vectors.')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
        raise ValueError('Labels must be binary and probabilities finite in [0, 1].')


def classification_metrics(y_true, y_prob, threshold=0.5):
    validate_threshold(threshold)
    validate_predictions(y_true, y_prob)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {'threshold': float(threshold), 'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
