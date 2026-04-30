#!/bin/bash
# U-VLM: Inference Examples
# Run inference for any task with a trained model.

set -e

MODEL_DIR="/path/to/nnUNet_results/Dataset201_MyData/nnUNetTrainer_UVLM__UVLM_ResEncUNetLPlans_cls__3d_fullres"
CSV_PATH="${MODEL_DIR}/fold_0/balanced/balanced_dataset.csv"

# ---- Segmentation inference ----
uvlm_inference seg \
    --csv-path    "${CSV_PATH}" \
    --model-dir   "${MODEL_DIR}" \
    --output-dir  "${MODEL_DIR}/inference_seg"

# ---- Classification inference ----
uvlm_inference cls \
    --csv-path    "${CSV_PATH}" \
    --model-dir   "${MODEL_DIR}" \
    --output-dir  "${MODEL_DIR}/inference_cls" \
    --gpu-config  "0:1"

# ---- Report generation inference ----
uvlm_inference report \
    --csv-path    "${CSV_PATH}" \
    --model-dir   "${MODEL_DIR}" \
    --output-dir  "${MODEL_DIR}/inference_report" \
    --gpu-config  "0:1"

echo "Inference complete. Results in ${MODEL_DIR}/"
