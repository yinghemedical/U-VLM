#!/bin/bash
# U-VLM: Preprocessing Example
# Converts raw DICOM/NIfTI to Blosc2 format and generates training CSV.

set -e

# ---- Config ----
PREPROCESSED_DIR="/path/to/nnUNet_preprocessed"
DATASET_NAME="Dataset200_MyData"

# Chest segmentation preprocessing (ReXGroundingCT as example)
python -m uvlm.preprocessing.preprocess_rexgrounding_seg \
    --config-path uvlm/preprocessing/configs/rexgrounding_ct_config.json \
    all \
    --raw-input-dir   "/path/to/nnUNet_raw/${DATASET_NAME}" \
    --output-dir      "${PREPROCESSED_DIR}/${DATASET_NAME}"

# Chest classification + report preprocessing (CT-RATE as example)
python -m uvlm.preprocessing.preprocess_ct_rate_cls_report \
    --config-path uvlm/preprocessing/configs/ct_rate_config.json \
    all \
    --train-input-dir  "/path/to/train_images" \
    --val-input-dir    "/path/to/val_images" \
    --output-dir       "${PREPROCESSED_DIR}/${DATASET_NAME}" \
    --reports-input-dir "/path/to/reports" \
    --cls-input-dir    "/path/to/classification_labels"

echo "Preprocessing complete: ${PREPROCESSED_DIR}/${DATASET_NAME}"
