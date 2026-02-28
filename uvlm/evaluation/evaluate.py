"""
Evaluation Script for U-VLM

Evaluates classification and report generation results.

Usage:
    # Evaluate classification
    python -m uvlm.evaluation.evaluate \
        --task cls \
        --gt-csv /path/to/ground_truth.csv \
        --pred-csv /path/to/predictions.csv \
        --output-dir /path/to/output

    # Evaluate report generation
    python -m uvlm.evaluation.evaluate \
        --task report \
        --gt-csv /path/to/ground_truth.csv \
        --pred-csv /path/to/predictions.csv \
        --output-dir /path/to/output
"""

import os
import argparse
import json
from pathlib import Path

import pandas as pd

from uvlm.evaluation.metrics.classification import calc_cls_metrics
from uvlm.evaluation.metrics.nlg import calc_nlg_metrics


# Default classification columns for CT-RATE (18 classes)
CT_RATE_CLS_COLUMNS = [
    'Medical material',
    'Arterial wall calcification',
    'Cardiomegaly',
    'Pericardial effusion',
    'Coronary artery wall calcification',
    'Hiatal hernia',
    'Lymphadenopathy',
    'Emphysema',
    'Atelectasis',
    'Lung nodule',
    'Lung opacity',
    'Pulmonary fibrotic sequela',
    'Pleural effusion',
    'Mosaic attenuation pattern',
    'Peribronchial thickening',
    'Consolidation',
    'Bronchiectasis',
    'Interlobular septal thickening'
]

# Default classification columns for AbdomenAtlas (3 classes)
ABDOMEN_CLS_COLUMNS = [
    'Liver lesion',
    'Kidney lesion',
    'Pancreas lesion'
]


def evaluate_classification(
    gt_csv: str,
    pred_csv: str,
    output_dir: str,
    id_col: str = 'series_id',
    cls_columns: list = None,
    threshold: float = 0.5
):
    """
    Evaluate classification results

    Args:
        gt_csv: Ground truth CSV path
        pred_csv: Prediction CSV path
        output_dir: Output directory
        id_col: ID column name
        cls_columns: Classification column names
        threshold: Binarization threshold
    """
    print(f"\n{'='*60}")
    print("Classification Evaluation")
    print(f"{'='*60}")

    # Load data
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    print(f"Ground truth samples: {len(gt_df)}")
    print(f"Prediction samples: {len(pred_df)}")

    # Auto-detect classification columns if not provided
    if cls_columns is None:
        # Try CT-RATE columns first
        if all(col in gt_df.columns for col in CT_RATE_CLS_COLUMNS[:5]):
            cls_columns = CT_RATE_CLS_COLUMNS
            print(f"Auto-detected CT-RATE classification columns")
        # Try AbdomenAtlas columns
        elif all(col in gt_df.columns for col in ABDOMEN_CLS_COLUMNS[:5]):
            cls_columns = ABDOMEN_CLS_COLUMNS
            print(f"Auto-detected AbdomenAtlas classification columns")
        else:
            raise ValueError(
                "Could not auto-detect classification columns. "
                "Please specify --cls-columns"
            )

    print(f"Classification columns: {len(cls_columns)}")

    # Calculate metrics
    os.makedirs(output_dir, exist_ok=True)

    print("\nCalculating classification metrics...")
    cls_metrics = calc_cls_metrics(
        gt_df, pred_df, id_col, cls_columns, threshold
    )

    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics_cls.json')
    with open(metrics_path, 'w') as f:
        json.dump(cls_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Classification Results")
    print(f"{'='*60}")
    print(f"Samples: {cls_metrics['num_samples']}")
    print(f"Classes: {cls_metrics['num_classes']}")
    print(f"Threshold: {cls_metrics['threshold']}")
    print(f"\nMacro Metrics:")
    print(f"  Precision: {cls_metrics['macro']['precision']:.4f}")
    print(f"  Recall:    {cls_metrics['macro']['recall']:.4f}")
    print(f"  F1:        {cls_metrics['macro']['f1']:.4f}")
    print(f"{'='*60}\n")


def evaluate_report_generation(
    gt_csv: str,
    pred_csv: str,
    output_dir: str,
    id_col: str = 'series_id',
    gt_report_col: str = 'report',
    pred_report_col: str = 'generated_report'
):
    """
    Evaluate report generation results

    Args:
        gt_csv: Ground truth CSV path
        pred_csv: Prediction CSV path
        output_dir: Output directory
        id_col: ID column name
        gt_report_col: Ground truth report column name
        pred_report_col: Prediction report column name
    """
    print(f"\n{'='*60}")
    print("Report Generation Evaluation")
    print(f"{'='*60}")

    # Load data
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)

    print(f"Ground truth samples: {len(gt_df)}")
    print(f"Prediction samples: {len(pred_df)}")

    # Calculate NLG metrics
    os.makedirs(output_dir, exist_ok=True)

    print("\nCalculating NLG metrics...")
    try:
        nlg_metrics = calc_nlg_metrics(
            gt_df, pred_df, id_col, gt_report_col, pred_report_col
        )

        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics_nlg.json')
        with open(metrics_path, 'w') as f:
            json.dump(nlg_metrics, f, indent=2)
        print(f"Saved: {metrics_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("Report Generation Results")
        print(f"{'='*60}")
        print(f"Samples: {nlg_metrics['num_samples']}")
        print(f"\nNLG Metrics:")
        print(f"  BLEU-1:    {nlg_metrics['BLEU_1']:.4f}")
        print(f"  BLEU-2:    {nlg_metrics['BLEU_2']:.4f}")
        print(f"  BLEU-3:    {nlg_metrics['BLEU_3']:.4f}")
        print(f"  BLEU-4:    {nlg_metrics['BLEU_4']:.4f}")
        print(f"  BLEU-mean: {nlg_metrics['BLEU_mean']:.4f}")
        print(f"  ROUGE-L:   {nlg_metrics['ROUGE_L']:.4f}")
        print(f"  CIDEr:     {nlg_metrics['CIDEr']:.4f}")
        if nlg_metrics.get('METEOR') is not None:
            print(f"  METEOR:    {nlg_metrics['METEOR']:.4f}")
        print(f"{'='*60}\n")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install pycocoevalcap:")
        print("  pip install git+https://github.com/salaniz/pycocoevalcap.git")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate U-VLM classification and report generation results'
    )

    parser.add_argument('--task', type=str, required=True,
                        choices=['cls', 'report', 'both'],
                        help='Evaluation task')
    parser.add_argument('--gt-csv', type=str, required=True,
                        help='Ground truth CSV file')
    parser.add_argument('--pred-csv', type=str, required=True,
                        help='Prediction CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')

    # Common parameters
    parser.add_argument('--id-col', type=str, default='series_id',
                        help='ID column name')

    # Classification parameters
    parser.add_argument('--cls-columns', type=str, nargs='+', default=None,
                        help='Classification column names')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')

    # Report generation parameters
    parser.add_argument('--gt-report-col', type=str, default='report',
                        help='Ground truth report column name')
    parser.add_argument('--pred-report-col', type=str, default='generated_report',
                        help='Prediction report column name')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.gt_csv).exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {args.gt_csv}")
    if not Path(args.pred_csv).exists():
        raise FileNotFoundError(f"Prediction CSV not found: {args.pred_csv}")

    # Run evaluation
    if args.task in ['cls', 'both']:
        evaluate_classification(
            args.gt_csv,
            args.pred_csv,
            args.output_dir,
            args.id_col,
            args.cls_columns,
            args.threshold
        )

    if args.task in ['report', 'both']:
        evaluate_report_generation(
            args.gt_csv,
            args.pred_csv,
            args.output_dir,
            args.id_col,
            args.gt_report_col,
            args.pred_report_col
        )

    print("Evaluation completed!")


if __name__ == '__main__':
    main()
