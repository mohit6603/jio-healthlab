"""Classification metrics for the delay model.

Metrics are computed here and nowhere else, so training, the CLI and any
future evaluation harness report the same numbers the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true, y_prob, *, threshold: float = 0.5
) -> dict[str, Any]:
    """Compute the metric set recorded with every trained model.

    ``y_prob`` is the positive-class probability, not a hard label: the risk
    bands the API exposes are thresholds over this, so calibration matters as
    much as accuracy.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "average_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        # Lower is better; a poorly calibrated model gives misleading risk bands.
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "threshold": threshold,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "support": {
            "total": len(y_true),
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum()),
        },
    }


def format_metrics(metrics: dict[str, Any]) -> str:
    """Render metrics as an aligned block for CLI output."""
    matrix = metrics["confusion_matrix"]
    return "\n".join(
        [
            f"  accuracy          {metrics['accuracy']:.4f}",
            f"  precision         {metrics['precision']:.4f}",
            f"  recall            {metrics['recall']:.4f}",
            f"  f1                {metrics['f1']:.4f}",
            f"  roc_auc           {metrics['roc_auc']:.4f}",
            f"  average_precision {metrics['average_precision']:.4f}",
            f"  brier_score       {metrics['brier_score']:.4f}",
            "  confusion matrix  "
            f"tn={matrix['true_negative']} fp={matrix['false_positive']} "
            f"fn={matrix['false_negative']} tp={matrix['true_positive']}",
        ]
    )
