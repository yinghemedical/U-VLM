"""
U-VLM Evaluation Entry Point

Usage:
    # Evaluate classification
    uvlm_evaluate cls \
        --gt-csv /path/to/ground_truth.csv \
        --pred-csv /path/to/predictions.csv \
        --output-dir /path/to/output

    # Evaluate report generation
    uvlm_evaluate report \
        --gt-csv /path/to/ground_truth.csv \
        --pred-csv /path/to/predictions.csv \
        --output-dir /path/to/output

    # Evaluate segmentation
    uvlm_evaluate seg \
        --gt-csv /path/to/ground_truth.csv \
        --predictions /path/to/predictions.json \
        --output-dir /path/to/output
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='U-VLM Evaluation')
    parser.add_argument('task', type=str, choices=['cls', 'seg', 'report'],
                        help='Evaluation task type')
    parser.add_argument('--gt-csv', type=str, required=True,
                        help='Path to ground truth CSV')
    parser.add_argument('--pred-csv', type=str,
                        help='Path to predictions CSV (for cls/report tasks)')
    parser.add_argument('--predictions', type=str,
                        help='Path to predictions.json (for seg task)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--id-col', type=str, default='series_id',
                        help='ID column name')
    parser.add_argument('--cls-columns', nargs='+',
                        help='Classification column names')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold for classification')
    parser.add_argument('--gt-report-col', type=str, default='report',
                        help='GT report column name')
    parser.add_argument('--pred-report-col', type=str, default='generated_report',
                        help='Predicted report column name')

    args = parser.parse_args()

    from uvlm.evaluation.evaluate import evaluate_seg, evaluate_cls, evaluate_report

    if args.task == 'seg':
        if not args.predictions:
            parser.error("--predictions is required for seg task")
        evaluate_seg(args.predictions, args.gt_csv, args.output_dir, args.id_col)
    elif args.task == 'cls':
        if not args.pred_csv:
            parser.error("--pred-csv is required for cls task")
        evaluate_cls(args.gt_csv, args.pred_csv, args.output_dir, args.id_col,
                     args.cls_columns, args.threshold)
    elif args.task == 'report':
        if not args.pred_csv:
            parser.error("--pred-csv is required for report task")
        evaluate_report(args.gt_csv, args.pred_csv, args.output_dir, args.id_col,
                        args.gt_report_col, args.pred_report_col)

    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
