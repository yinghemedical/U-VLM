"""
Classification Metrics Module

Calculates metrics for multi-label classification tasks:
- Per-class and macro-averaged precision/recall/f1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


def align_predictions(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    cls_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align prediction and ground truth data

    Args:
        gt_df: Ground truth DataFrame
        pred_df: Prediction DataFrame
        id_col: ID column name
        cls_cols: Classification column names

    Returns:
        (aligned GT DataFrame, aligned prediction DataFrame)
    """
    # Clean ID column suffixes
    gt_ids = gt_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)
    pred_ids = pred_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)

    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    gt_df[id_col] = gt_ids
    pred_df[id_col] = pred_ids

    # Set index
    gt_df = gt_df.set_index(id_col)
    pred_df = pred_df.set_index(id_col)

    # Find common indices
    common_idx = gt_df.index.intersection(pred_df.index)
    assert len(common_idx) > 0, "No common samples found between GT and predictions"

    # Align
    gt_aligned = gt_df.loc[common_idx, cls_cols]
    pred_aligned = pred_df.loc[common_idx, cls_cols]

    return gt_aligned, pred_aligned


def calc_cls_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    cls_cols: List[str],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate classification metrics

    Args:
        gt_df: Ground truth DataFrame
        pred_df: Prediction DataFrame (probabilities or binary)
        id_col: ID column name
        cls_cols: Classification column names
        threshold: Binarization threshold

    Returns:
        Metrics dictionary with per_pathology and macro sections
    """
    gt_aligned, pred_aligned = align_predictions(gt_df, pred_df, id_col, cls_cols)

    # Check if binarization is needed
    max_val = pred_aligned.values.max()
    if max_val <= 1.0 and pred_aligned.values.min() >= 0.0:
        # Probability values, need binarization
        pred_binary = (pred_aligned >= threshold).astype(int)
    else:
        # Already binary
        pred_binary = pred_aligned.astype(int)

    per_pathology = []
    for col in cls_cols:
        y_true = gt_aligned[col].values.astype(int)
        y_pred = pred_binary[col].values.astype(int)

        # Calculate confusion matrix elements
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Sensitivity = Recall, Specificity = TN / (TN + FP)
        sensitivity = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_pathology.append({
            'name': col,
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(f1),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })

    # Calculate macro averages
    macro_precision = np.mean([p['precision'] for p in per_pathology])
    macro_recall = np.mean([p['recall'] for p in per_pathology])
    macro_sensitivity = np.mean([p['sensitivity'] for p in per_pathology])
    macro_specificity = np.mean([p['specificity'] for p in per_pathology])
    macro_f1 = np.mean([p['f1'] for p in per_pathology])

    return {
        'per_pathology': per_pathology,
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'sensitivity': float(macro_sensitivity),
            'specificity': float(macro_specificity),
            'f1': float(macro_f1)
        },
        'num_samples': len(gt_aligned),
        'num_classes': len(cls_cols),
        'threshold': threshold
    }
