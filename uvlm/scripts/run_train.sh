#!/bin/bash
# U-VLM: Training Example
# Progressive three-stage training with pretrained encoder checkpoint.

set -e

DATASET="Dataset201_MyData"
FOLD=0
PLAN_DIR="/path/to/nnUNet_preprocessed/${DATASET}"

# Generate plans from templates
python uvlm/scripts/generate_plans.py \
    --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_seg_basic.json \
    --output-dir "$PLAN_DIR" \
    --var PREPROCESSED_DIR="/path/to/nnUNet_preprocessed" \
    --var DATASET_NAME="$DATASET" \
    --var CSV_FILE="dataset_train_seg_basic.csv"

# Stage 1: Segmentation pretraining
uvlm_train $DATASET 3d_fullres $FOLD \
    -tr nnUNetTrainer_ResEncoderUNet \
    -p UVLM_ResEncUNetLPlans_chest_seg_basic
SEG_CKPT="nnUNet_results/${DATASET}/nnUNetTrainer_ResEncoderUNet__*/fold_${FOLD}/checkpoint_best.pth"

# Generate classification plan
python uvlm/scripts/generate_plans.py \
    --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_cls.json \
    --output-dir "$PLAN_DIR" \
    --var PREPROCESSED_DIR="/path/to/nnUNet_preprocessed" \
    --var DATASET_NAME="$DATASET" \
    --var CSV_FILE="train_merged.csv"

# Stage 2: Classification pretraining (load seg encoder)
uvlm_train $DATASET 3d_fullres $FOLD \
    -tr nnUNetTrainer_UVLM \
    -p UVLM_ResEncUNetLPlans_chest_cls \
    --pretrained_encoder_checkpoint_path "$SEG_CKPT"
CLS_CKPT="nnUNet_results/${DATASET}/nnUNetTrainer_UVLM__*/fold_${FOLD}/checkpoint_best.pth"

# Generate report plan
python uvlm/scripts/generate_plans.py \
    --template uvlm/configs/plans/UVLM_ResEncUNetLPlans_chest_report.json \
    --output-dir "$PLAN_DIR" \
    --var PREPROCESSED_DIR="/path/to/nnUNet_preprocessed" \
    --var DATASET_NAME="$DATASET" \
    --var CSV_FILE="train_merged.csv"

# Stage 3: Report generation (load cls encoder)
uvlm_train $DATASET 3d_fullres $FOLD \
    -tr nnUNetTrainer_UVLM \
    -p UVLM_ResEncUNetLPlans_chest_report \
    --pretrained_encoder_checkpoint_path "$CLS_CKPT"

echo "Training complete. Checkpoints in nnUNet_results/${DATASET}/"
