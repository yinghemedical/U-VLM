"""
NLG (Natural Language Generation) Metrics Module

Calculates report generation metrics:
- BLEU (1-4 and mean)
- ROUGE-L
- CIDEr
- METEOR (requires Java environment)
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd


def clean_report_text(text: str) -> str:
    """
    Clean report text

    Args:
        text: Raw report text

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove special tokens
    text = text.replace("<|eot_id|>", "")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")

    # Clean extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.\s*\.', '.', text)

    return text.strip()


def prepare_reports_for_eval(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_report_col: str,
    pred_report_col: str
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Prepare reports for NLG evaluation

    Args:
        gt_df: Ground truth DataFrame
        pred_df: Prediction DataFrame
        id_col: ID column name
        gt_report_col: Ground truth report column name
        pred_report_col: Prediction report column name

    Returns:
        (gts, res) - pycocoevalcap format dictionaries
    """
    # Clean ID suffixes
    gt_ids = gt_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)
    pred_ids = pred_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)

    gt_map = dict(zip(gt_ids, gt_df[gt_report_col]))
    pred_map = dict(zip(pred_ids, pred_df[pred_report_col]))

    # Find common IDs
    common_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    assert len(common_ids) > 0, "No common samples found between GT and predictions"

    gts = {}
    res = {}
    for idx, key in enumerate(common_ids):
        gt_text = clean_report_text(gt_map[key])
        pred_text = clean_report_text(pred_map[key])

        gts[idx] = [gt_text]
        res[idx] = [pred_text]

    return gts, res


def calc_nlg_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_report_col: str,
    pred_report_col: str = "generated_report"
) -> Dict[str, Any]:
    """
    Calculate NLG metrics

    Args:
        gt_df: Ground truth DataFrame
        pred_df: Prediction DataFrame
        id_col: ID column name
        gt_report_col: Ground truth report column name
        pred_report_col: Prediction report column name

    Returns:
        NLG metrics dictionary
    """
    # Lazy import to avoid dependency issues
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        raise ImportError(
            "pycocoevalcap is required for NLG metrics. "
            "Install with: pip install git+https://github.com/salaniz/pycocoevalcap.git"
        )

    gts, res = prepare_reports_for_eval(
        gt_df, pred_df, id_col, gt_report_col, pred_report_col
    )

    results = {}

    # BLEU
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts, res)
    results['BLEU_1'] = float(bleu_scores[0])
    results['BLEU_2'] = float(bleu_scores[1])
    results['BLEU_3'] = float(bleu_scores[2])
    results['BLEU_4'] = float(bleu_scores[3])
    results['BLEU_mean'] = float(sum(bleu_scores) / len(bleu_scores))

    # ROUGE-L
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts, res)
    results['ROUGE_L'] = float(rouge_score)

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    results['CIDEr'] = float(cider_score)

    # METEOR (requires Java, may fail)
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        meteor_scorer = Meteor()
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
        results['METEOR'] = float(meteor_score)
    except (FileNotFoundError, Exception) as e:
        print(f"Warning: METEOR calculation failed (Java may not be available): {e}")
        results['METEOR'] = None

    results['num_samples'] = len(gts)

    return results


def calc_nlg_metrics_per_sample(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_col: str,
    gt_report_col: str,
    pred_report_col: str = "generated_report"
) -> pd.DataFrame:
    """
    Calculate per-sample NLG metrics

    Args:
        gt_df: Ground truth DataFrame
        pred_df: Prediction DataFrame
        id_col: ID column name
        gt_report_col: Ground truth report column name
        pred_report_col: Prediction report column name

    Returns:
        DataFrame with per-sample metrics
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        raise ImportError(
            "pycocoevalcap is required for NLG metrics. "
            "Install with: pip install git+https://github.com/salaniz/pycocoevalcap.git"
        )

    # Clean IDs
    gt_ids = gt_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)
    pred_ids = pred_df[id_col].astype(str).str.replace('.nii.gz', '', regex=False).str.replace('.nii', '', regex=False)

    gt_map = dict(zip(gt_ids, gt_df[gt_report_col]))
    pred_map = dict(zip(pred_ids, pred_df[pred_report_col]))

    common_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))

    results = []
    bleu_scorer = Bleu(4)
    rouge_scorer = Rouge()

    for key in common_ids:
        gt_text = clean_report_text(gt_map[key])
        pred_text = clean_report_text(pred_map[key])

        gts = {0: [gt_text]}
        res = {0: [pred_text]}

        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        rouge_score, _ = rouge_scorer.compute_score(gts, res)

        results.append({
            id_col: key,
            'gt_report': gt_text[:200] + '...' if len(gt_text) > 200 else gt_text,
            'pred_report': pred_text[:200] + '...' if len(pred_text) > 200 else pred_text,
            'BLEU_1': float(bleu_scores[0]),
            'BLEU_4': float(bleu_scores[3]),
            'ROUGE_L': float(rouge_score)
        })

    return pd.DataFrame(results)
