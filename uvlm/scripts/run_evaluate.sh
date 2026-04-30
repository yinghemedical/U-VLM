#!/bin/bash
# U-VLM: Evaluation Examples
# Compute metrics for each task from inference outputs.

set -e

OUT_DIR="/path/to/model/inference_results"
GT_CSV="/path/to/ground_truth.csv"

# ---- Segmentation evaluation (Dice) ----
uvlm_evaluate seg \
    --gt-csv       "$GT_CSV" \
    --predictions  "${OUT_DIR}/predictions.json" \
    --output-dir   "$OUT_DIR"
# Output: metrics_seg.json

# ---- Classification evaluation (F1 / Recall / Precision) ----
uvlm_evaluate cls \
    --gt-csv       "$GT_CSV" \
    --pred-csv     "${OUT_DIR}/results.csv" \
    --output-dir   "$OUT_DIR"
# Output: metrics_cls.json

# ---- Report generation evaluation (BLEU / ROUGE-L / CIDEr) ----
uvlm_evaluate report \
    --gt-csv       "$GT_CSV" \
    --pred-csv     "${OUT_DIR}/results.csv" \
    --output-dir   "$OUT_DIR"
# Output: metrics_nlg.json

echo "Evaluation complete. Metrics in ${OUT_DIR}/"
