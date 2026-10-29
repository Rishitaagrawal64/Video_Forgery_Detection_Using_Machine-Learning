"""Binary classification metrics (numpy only) used to evaluate the *serving* pipeline end to end.

Positive class = forged (label 1), matching ml/split.py and the API.
"""
from __future__ import annotations

import numpy as np


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based AUC with average ranks for ties. Returns None if only one class is present."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float | int | None]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be non-empty and the same length")
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    auc = roc_auc(y, p)
    return {
        "n": int(y.size),
        "threshold": float(threshold),
        "accuracy": round(div(tp + tn, y.size), 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(div(2 * precision * recall, precision + recall), 4),
        "majority_class_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
        "auc": None if auc is None else round(auc, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
