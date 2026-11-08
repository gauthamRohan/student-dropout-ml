"""Metric helpers shared by the training script and the Streamlit app,
so the numbers in the README and in the app are computed the same way."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba, class_order):
    """Return the six assignment metrics as a dict.

    Precision / Recall / F1 are weighted by class support (3-class problem).
    AUC is one-vs-rest, weighted, using predicted probabilities.
    `class_order` must match the column order of `y_proba` (model.classes_).
    """
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        scores["AUC"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted", labels=list(class_order)
        )
    except ValueError:
        # e.g. an uploaded file that contains only one class
        scores["AUC"] = np.nan
    return {name: float(scores[name]) for name in METRIC_NAMES}
