"""
Evaluation Script for U-VLM

Usage:
    # Segmentation evaluation
    python -m uvlm.evaluation.evaluate \
        --task seg \
        --predictions /path/to/predictions.json \
        --gt-csv /path/to/gt.csv \
        --id-col series_id \
        --output-dir /path/to/output

    # Classification evaluation (GT vs Pred comparison)
    python -m uvlm.evaluation.evaluate \
        --task cls \
        --gt-csv /path/to/gt.csv \
        --pred-csv /path/to/pred.csv \
        --id-col series_id \
        --output-dir /path/to/output

    # Report generation evaluation (GT vs Pred comparison)
    python -m uvlm.evaluation.evaluate \
        --task report \
        --gt-csv /path/to/gt.csv \
        --pred-csv /path/to/pred.csv \
        --id-col series_id \
        --output-dir /path/to/output
"""

import argparse
import blosc2
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from uvlm.evaluation.metrics.classification import calc_cls_metrics
from uvlm.evaluation.metrics.nlg import calc_nlg_metrics


def read_blosc2(file_path: str) -> np.ndarray:
    """Load blosc2 format file."""
    b2_array = blosc2.open(file_path, mmap_mode='r')
    return b2_array[:]


# Default classification columns
CT_RATE_COLS = [
    'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
    'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
    'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
    'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
    'Mosaic attenuation pattern', 'Peribronchial thickening',
    'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
]

ABDOMEN_COLS = ['Liver lesion', 'Kidney lesion', 'Pancreas lesion']


def detect_cls_columns(df: pd.DataFrame) -> list:
    """Auto-detect classification columns from DataFrame."""
    if all(col in df.columns for col in CT_RATE_COLS[:5]):
        return CT_RATE_COLS
    if all(col in df.columns for col in ABDOMEN_COLS[:2]):
        return ABDOMEN_COLS
    return None


def compute_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """Calculate Dice coefficient for each class."""
    dice_scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c)
        dice_scores[c] = float(2.0 * intersection / union) if union > 0 else float('nan')
    return dice_scores


def evaluate_seg(predictions_json: str, gt_csv: str, output_dir: str, id_col: str = 'series_id'):
    """Evaluate segmentation - compute Dice from predictions and GT."""
    with open(predictions_json) as f:
        data = json.load(f)

    pred_cases = data['per_case_results']
    print(f"Segmentation: {len(pred_cases)} predictions")

    # Load GT from CSV
    gt_df = pd.read_csv(gt_csv)
    # Find seg column
    seg_col = None
    for col in ['seg_blosc2_path', 'seg_path', 'label_path']:
        if col in gt_df.columns:
            seg_col = col
            break
    if seg_col is None:
        raise ValueError(f"No segmentation path column found in GT CSV. Available: {list(gt_df.columns)}")

    print(f"GT column: {seg_col}")

    # Build GT lookup
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        identifier = row[id_col]
        seg_path = row.get(seg_col)
        if pd.notna(seg_path):
            gt_lookup[identifier] = seg_path

    # Compute Dice for each case
    all_dice_scores = {}
    for case in pred_cases:
        identifier = case['identifier']
        pred_path = case['pred_path']

        if identifier not in gt_lookup:
            continue

        gt_path = gt_lookup[identifier]
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue

        # Load images - predictions as nii.gz (D,H,W), GT as blosc2 (D,H,W)
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        gt_arr = read_blosc2(gt_path)

        # Ensure same shape by transposing if needed
        if pred_arr.shape != gt_arr.shape:
            gt_arr = gt_arr.T  # Transpose to match

        # Compute Dice (assume max 48 classes)
        num_classes = 48
        dice_scores = compute_dice(pred_arr, gt_arr, num_classes)
        all_dice_scores[identifier] = dice_scores

    if not all_dice_scores:
        print("No valid case pairs found for evaluation")
        return

    # Compute per-class and mean Dice
    all_classes = set()
    for scores in all_dice_scores.values():
        all_classes.update(scores.keys())

    per_class_dice = {}
    for c in sorted(all_classes):
        class_dices = [scores[c] for scores in all_dice_scores.values()
                       if c in scores and not np.isnan(scores[c])]
        if class_dices:
            per_class_dice[c] = np.mean(class_dices)

    mean_dice = np.mean(list(per_class_dice.values())) if per_class_dice else 0.0

    result = {
        "task": "segmentation",
        "cases": len(all_dice_scores),
        "mean_dice": float(mean_dice),
        "per_class_dice": per_class_dice
    }

    print(f"Mean Dice: {mean_dice:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics_seg.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")
    return result


def evaluate_cls(gt_csv: str, pred_csv: str, output_dir: str, id_col: str, cls_columns: list, threshold: float):
    """Evaluate classification - GT vs Pred comparison with per-class F1."""
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    print(f"Classification: GT={len(gt_df)}, Pred={len(pred_df)}")

    if cls_columns is None:
        cls_columns = detect_cls_columns(gt_df)
    if cls_columns is None:
        raise ValueError("Cannot auto-detect cls_columns. Please specify --cls-columns")

    print(f"Classes ({len(cls_columns)})")

    # Handle duplicate series_ids in GT by keeping only first occurrence
    gt_df_dedup = gt_df.drop_duplicates(subset=[id_col], keep='first')

    # Merge GT and Pred on id_col
    merged = pred_df.merge(gt_df_dedup[[id_col] + cls_columns], on=id_col, how='left', suffixes=('_pred', '_gt'))

    # Prepare GT and Pred DataFrames
    gt_for_eval = merged[[id_col] + [f"{c}_gt" for c in cls_columns]].copy()
    gt_for_eval = gt_for_eval.rename(columns={f"{c}_gt": c for c in cls_columns})

    pred_for_eval = merged[[id_col] + [f"{c}_pred" for c in cls_columns]].copy()
    pred_for_eval = pred_for_eval.rename(columns={f"{c}_pred": c for c in cls_columns})

    print(f"After merge: GT={len(gt_for_eval)}, Pred={len(pred_for_eval)}")

    metrics = calc_cls_metrics(gt_for_eval, pred_for_eval, id_col, cls_columns, threshold)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metrics_cls.json")
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_path}")

    print(f"\nPer-class F1:")
    for p in metrics['per_pathology']:
        print(f"  {p['name']}: P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f}")

    print(f"\nMacro: P={metrics['macro']['precision']:.3f} R={metrics['macro']['recall']:.3f} F1={metrics['macro']['f1']:.3f}")
    return metrics


def evaluate_report(gt_csv: str, pred_csv: str, output_dir: str, id_col: str,
                   gt_col: str = 'report', pred_col: str = 'generated_report'):
    """Evaluate report generation - NLG metrics."""
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    print(f"Report Generation: GT={len(gt_df)}, Pred={len(pred_df)}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        metrics = calc_nlg_metrics(gt_df, pred_df, id_col, gt_col, pred_col)
        out_path = os.path.join(output_dir, "metrics_nlg.json")
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {out_path}")

        print(f"\nNLG Metrics: BLEU-1={metrics['BLEU_1']:.3f} BLEU-4={metrics['BLEU_4']:.3f} BLEU-mean={metrics['BLEU_mean']:.3f}")
        return metrics
    except ImportError as e:
        print(f"Error: {e}")
        print("Install: pip install git+https://github.com/salaniz/pycocoevalcap.git")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate U-VLM inference results')
    parser.add_argument('--task', required=True, choices=['seg', 'cls', 'report'],
                        help='Evaluation task')
    parser.add_argument('--predictions', help='Segmentation predictions.json path')
    parser.add_argument('--gt-csv', help='Ground truth CSV path')
    parser.add_argument('--pred-csv', help='Prediction CSV path')
    parser.add_argument('--id-col', default='series_id', help='ID column name')
    parser.add_argument('--cls-columns', nargs='+', help='Classification column names')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--gt-report-col', default='report', help='GT report column')
    parser.add_argument('--pred-report-col', default='generated_report', help='Pred report column')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    if args.task == 'seg':
        if not args.predictions or not args.gt_csv:
            raise ValueError("--predictions and --gt-csv required for seg task")
        evaluate_seg(args.predictions, args.gt_csv, args.output_dir, args.id_col)
    elif args.task == 'cls':
        if not args.gt_csv or not args.pred_csv:
            raise ValueError("--gt-csv and --pred-csv required for cls task")
        evaluate_cls(args.gt_csv, args.pred_csv, args.output_dir, args.id_col, args.cls_columns, args.threshold)
    elif args.task == 'report':
        if not args.gt_csv or not args.pred_csv:
            raise ValueError("--gt-csv and --pred-csv required for report task")
        evaluate_report(args.gt_csv, args.pred_csv, args.output_dir, args.id_col, args.gt_report_col, args.pred_report_col)

    print("Done!")


if __name__ == '__main__':
    main()
