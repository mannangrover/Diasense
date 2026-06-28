"""
Evaluation Metrics
==================
ROC-AUC (multi-class OvR), confusion matrix, classification report,
and hierarchical evaluation at 4-class, 3-class, and 2-class levels.

Usage:
    from src.evaluation.metrics import evaluate_hierarchical
    results = evaluate_hierarchical(y_true, y_pred_proba)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    CLASS_NAMES_4, CLASS_NAMES_3, CLASS_NAMES_2,
    LABEL_MAP_4_TO_3, LABEL_MAP_4_TO_2,
)


def _remap_labels(y, label_map):
    """Remap 4-class labels to 3-class or 2-class."""
    return np.array([label_map[int(yi)] for yi in y])


def _collapse_proba(proba_4, target_classes):
    """
    Collapse 4-class probabilities to fewer classes by summing.

    For 3-class: classes 2 and 3 merge → sum their probabilities.
    For 2-class: classes 1, 2, 3 merge → sum their probabilities.
    """
    if target_classes == 3:
        # 0→0, 1→1, 2+3→2
        new_proba = np.zeros((len(proba_4), 3))
        new_proba[:, 0] = proba_4[:, 0]
        new_proba[:, 1] = proba_4[:, 1]
        new_proba[:, 2] = proba_4[:, 2] + proba_4[:, 3]
        return new_proba
    elif target_classes == 2:
        # 0→0, 1+2+3→1
        new_proba = np.zeros((len(proba_4), 2))
        new_proba[:, 0] = proba_4[:, 0]
        new_proba[:, 1] = proba_4[:, 1] + proba_4[:, 2] + proba_4[:, 3]
        return new_proba
    return proba_4


def evaluate_hierarchical(y_true, y_pred_proba, verbose=True):
    """
    Evaluate at 4-class, 3-class, and 2-class levels.

    Args:
        y_true: array of true labels (4-class: 0-3)
        y_pred_proba: array of shape (n, 4) — predicted probabilities
        verbose: if True, print results

    Returns:
        dict with keys 'auc_4', 'auc_3', 'auc_2' and classification reports
    """

    results = {}

    for n_classes, label_map, class_names, label_suffix in [
        (4, None, CLASS_NAMES_4, "4"),
        (3, LABEL_MAP_4_TO_3, CLASS_NAMES_3, "3"),
        (2, LABEL_MAP_4_TO_2, CLASS_NAMES_2, "2"),
    ]:
        if label_map is not None:
            y_mapped = _remap_labels(y_true, label_map)
            proba = _collapse_proba(y_pred_proba, n_classes)
        else:
            y_mapped = y_true
            proba = y_pred_proba

        y_pred = np.argmax(proba, axis=1)

        try:
            if n_classes == 2:
                auc = roc_auc_score(y_mapped, proba[:, 1])
            else:
                auc = roc_auc_score(y_mapped, proba, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        results[f"auc_{label_suffix}"] = auc
        results[f"report_{label_suffix}"] = classification_report(
            y_mapped, y_pred, target_names=class_names, output_dict=True
        )

        if verbose:
            print(f"\n{'='*50}")
            print(f"{n_classes}-CLASS EVALUATION (ROC-AUC: {auc:.4f})")
            print(f"{'='*50}")
            print(classification_report(y_mapped, y_pred, target_names=class_names))
            print("Confusion Matrix:")
            cm = confusion_matrix(y_mapped, y_pred)
            print(cm)

    return results
