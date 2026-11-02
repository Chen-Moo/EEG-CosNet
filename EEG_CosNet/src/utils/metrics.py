# -*- coding: utf-8 -*-
"""
Classification evaluation metric utilities.

Computes ACC, error rate (Er), sensitivity (Sn), specificity (Sp), precision,
and F1 score from a single confusion matrix, shared across experiment scripts.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute a set of binary-classification metrics.

    Parameters
    ----------
    y_true : array-like
        True integer labels.
    y_pred : array-like
        Predicted integer labels.
    y_score : array-like, optional
        Predicted probability of the positive class (class 1), used for AUC;
        skipped if omitted.

    Returns
    -------
    dict
        Metric dictionary containing keys ``acc``, ``er``, ``sn``, ``sp``,
        ``precision``, ``f1``, ``auc``, ``cm``, etc.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Handle the edge case where the confusion matrix is not 2x2 due to a
    # missing label
    if cm.shape == (1, 1):
        if y_true[0] == 1:
            cm = np.array([[0, 0], [0, cm[0, 0]]])
        else:
            cm = np.array([[cm[0, 0], 0], [0, 0]])

    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    er = 100 - (acc * 100)
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sn) / (precision + sn) if (precision + sn) else 0.0

    metrics = {
        'acc': acc,
        'er': er,
        'sn': sn,
        'sp': sp,
        'precision': precision,
        'f1': f1,
        'cm': cm,
    }

    if y_score is not None:
        metrics['auc'] = roc_auc_score(y_true, y_score)

    return metrics
