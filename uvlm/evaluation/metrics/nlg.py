"""
NLG Metrics Module — BLEU scores for report generation evaluation.

Computes BLEU-1 through BLEU-4 and BLEU-mean following the standard
pycocoevalcap approach used in CT-RATE evaluation.
"""

import re
from typing import Dict, List, Any, Tuple
import pandas as pd


def clean_report_text(text: str) -> str:
    """Clean report text by removing special tokens and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace("<|eot_id|>", "")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\.\s*\.', '.', text)
    return text.strip()


def prepare_reports_for_eval(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_report_col: str,
    pred_report_col: str
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Prepare ground-truth and prediction reports in pycocoevalcap format."""
    gt_ids = gt_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)
    pred_ids = pred_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)

    gt_map = dict(zip(gt_ids, gt_df[gt_report_col]))
    pred_map = dict(zip(pred_ids, pred_df[pred_report_col]))

    common_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    assert len(common_ids) > 0, "No common samples found between GT and predictions"

    gts, res = {}, {}
    for idx, key in enumerate(common_ids):
        gts[idx] = [clean_report_text(gt_map[key])]
        res[idx] = [clean_report_text(pred_map[key])]
    return gts, res


def calc_nlg_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_report_col: str = "report",
    pred_report_col: str = "generated_report"
) -> Dict[str, Any]:
    """Calculate BLEU metrics for report generation.

    Returns:
        dict with BLEU_1, BLEU_2, BLEU_3, BLEU_4, BLEU_mean, num_samples
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
    except ImportError:
        raise ImportError(
            "pycocoevalcap is required for BLEU metrics. "
            "Install with: pip install pycocoevalcap"
        )

    gts, res = prepare_reports_for_eval(gt_df, pred_df, id_col, gt_report_col, pred_report_col)

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)

    return {
        'BLEU_1': float(bleu_scores[0]),
        'BLEU_2': float(bleu_scores[1]),
        'BLEU_3': float(bleu_scores[2]),
        'BLEU_4': float(bleu_scores[3]),
        'BLEU_mean': float(sum(bleu_scores) / len(bleu_scores)),
        'num_samples': len(gts),
    }
