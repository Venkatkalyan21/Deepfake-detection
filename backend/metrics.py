"""
Metrics — AUC, Accuracy, EER computation for IEEE evaluation
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from typing import List, Tuple


def compute_auc(y_true: List[int], y_scores: List[float]) -> float:
    """AUC-ROC score. y_true: 0=real, 1=fake."""
    return float(roc_auc_score(y_true, y_scores))


def compute_accuracy(y_true: List[int], y_scores: List[float], threshold: float = 0.5) -> float:
    preds = [1 if s >= threshold else 0 for s in y_scores]
    return float(accuracy_score(y_true, preds))


def compute_eer(y_true: List[int], y_scores: List[float]) -> float:
    """
    Equal Error Rate — threshold where FPR == FNR.
    Lower is better.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the threshold where |FPR - FNR| is minimised
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer


def compute_all(y_true: List[int], y_scores: List[float]) -> dict:
    """Returns dict with all metrics — suitable for IEEE results table."""
    auc = compute_auc(y_true, y_scores)
    acc = compute_accuracy(y_true, y_scores)
    eer = compute_eer(y_true, y_scores)
    return {
        "auc":      round(auc * 100, 2),
        "accuracy": round(acc * 100, 2),
        "eer":      round(eer * 100, 2),
    }


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(42)
    labels = [0] * 50 + [1] * 50
    scores = list(rng.uniform(0, 0.4, 50)) + list(rng.uniform(0.6, 1.0, 50))
    print(compute_all(labels, scores))
