"""
ResEncoderUNet Segmentation Inference Script

Supports two inference modes:
1. sliding_window: Sliding window inference using training patch size
2. full_image: Full image inference by directly feeding the entire image to the model

Data format:
- Input: Load preprocessed blosc2 format images from CSV file
- Output: nii.gz format

Usage:
    python inference_seg.py --csv-path /path/to/test.csv --model-folder /path/to/model
"""

import os
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional

import blosc2
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# ==================== Data Loading ====================

def read_blosc2_file(file_path: str) -> np.ndarray:
    """Read array in Blosc2 compressed format"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Blosc2 file does not exist: {file_path}")
    b2_array = blosc2.open(file_path, mmap_mode='r')
    return np.asarray(b2_array)


def detect_data_column(df: pd.DataFrame) -> str:
    """Automatically detect data path column from CSV columns"""
    for col in ['blosc2_path', 'image_blosc2_path', 'image_path']:
        if col in df.columns:
            return col
    raise ValueError(f"CSV missing data path column. Need 'blosc2_path' or 'image_blosc2_path'. Current columns: {list(df.columns)}")


def detect_seg_column(df: pd.DataFrame) -> Optional[str]:
    """Automatically detect segmentation label path column from CSV columns"""
    for col in ['seg_blosc2_path', 'label_blosc2_path', 'seg_path']:
        if col in df.columns:
            return col
    return None


def detect_id_column(df: pd.DataFrame) -> str:
    """Detect identifier column name"""
    for col in ['identifier', 'series_id', 'case_id']:
        if col in df.columns:
            return col
    raise ValueError(f"CSV missing identifier column. Need 'identifier', 'series_id' or 'case_id'. Current columns: {list(df.columns)}")


def load_image(data_path: str) -> np.ndarray:
    """
    Load preprocessed image data

    Preprocessed data format: (C, X, Y, Z) or (X, Y, Z)
    Output format: (Z, Y, X)
    """
    img = read_blosc2_file(data_path)

    # Handle channel dimension
    if img.ndim == 4:
        img = img[0]

    # Transpose from (X, Y, Z) to (Z, Y, X)
    return np.transpose(img, (2, 1, 0)).astype(np.float32)


def load_seg(seg_path: str) -> Optional[np.ndarray]:
    """Load segmentation label"""
    if not seg_path or not os.path.exists(seg_path):
        return None

    seg = read_blosc2_file(seg_path)
    if seg.ndim == 4:
        seg = seg[0]

    # Transpose from (X, Y, Z) to (Z, Y, X)
    return np.transpose(seg, (2, 1, 0)).astype(np.int16)


def normalize_zscore(image: np.ndarray) -> torch.Tensor:
    """z-score normalization and add channel dimension"""
    mean = image.mean()
    std = max(image.std(), 1e-8)
    normalized = (image - mean) / std
    return torch.from_numpy(normalized[np.newaxis, ...]).float()


# ==================== Output Saving ====================

def save_nifti(data: np.ndarray, output_path: str, spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0), is_seg: bool = False):
    """Save as NIfTI format"""
    data = data.astype(np.uint8 if is_seg else np.float32)
    sitk_image = sitk.GetImageFromArray(data)
    sitk_image.SetSpacing((spacing[2], spacing[1], spacing[0]))  # x, y, z
    sitk_image.SetOrigin((0.0, 0.0, 0.0))
    sitk_image.SetDirection(tuple(np.eye(3).flatten().tolist()))
    sitk.WriteImage(sitk_image, output_path)


# ==================== Metric Calculation ====================

def compute_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Calculate Dice coefficient for each class"""
    dice_scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (gt == c).astype(np.float32)
        intersection = np.sum(pred_c * gt_c)
        union = np.sum(pred_c) + np.sum(gt_c)
        dice_scores[c] = 2.0 * intersection / union if union > 0 else np.nan
    return dice_scores


def compute_summary_metrics(all_dice_scores: Dict) -> Dict:
    """Calculate summary metrics"""
    if not all_dice_scores:
        return {}

    all_classes = set()
    for case_scores in all_dice_scores.values():
        all_classes.update(case_scores.keys())

    per_class_dice = {}
    for c in all_classes:
        class_dices = [scores[c] for scores in all_dice_scores.values()
                       if c in scores and not np.isnan(scores[c])]
        if class_dices:
            per_class_dice[c] = np.mean(class_dices)

    summary = {'per_class_dice': per_class_dice}
    if per_class_dice:
        summary['mean_dice'] = np.mean(list(per_class_dice.values()))
    return summary


# ==================== Main Inference Pipeline ====================

def run_inference(args) -> Dict:
    """Run segmentation inference"""
    print(f"\n{'='*60}")
    print(f"Running segmentation inference...")
    print(f"{'='*60}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Create output directories
    pred_output_dir = os.path.join(args.output, 'predictions')
    image_output_dir = os.path.join(args.output, 'images')
    gt_output_dir = os.path.join(args.output, 'gt_seg')

    maybe_mkdir_p(pred_output_dir)
    if args.save_images:
        maybe_mkdir_p(image_output_dir)
        maybe_mkdir_p(gt_output_dir)

    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,
        use_gaussian=args.use_gaussian,
        use_mirroring=args.use_mirroring,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=args.verbose
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        use_folds=args.folds,
        checkpoint_name=args.checkpoint_name
    )

    # Load CSV data
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    data_col = detect_data_column(df)
    seg_col = detect_seg_column(df)
    id_col = detect_id_column(df)
    print(f"Data column: {data_col}")
    print(f"Identifier column: {id_col}")
    if seg_col:
        print(f"Segmentation label column: {seg_col}")

    # Debug mode limit
    if args.max_cases > 0:
        df = df.head(args.max_cases)
        print(f"Debug mode: processing only {args.max_cases} case(s)")

    print(f"Loaded {len(df)} cases from CSV")

    spacing = (1.5, 1.0, 1.0)
    all_results = []
    all_dice_scores = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        identifier = row[id_col]
        data_path = row[data_col]
        seg_path = row.get(seg_col) if seg_col and pd.notna(row.get(seg_col)) else None

        # Check if output already exists
        pred_output_path = os.path.join(pred_output_dir, f"{identifier}.nii.gz")
        if args.skip_existing and os.path.exists(pred_output_path):
            continue

        # Load data
        img = load_image(data_path)
        gt_seg = load_seg(seg_path) if seg_path else None

        # z-score normalization
        preprocessed_data = normalize_zscore(img)

        # Prediction
        start_time = time.time()
        with torch.no_grad():
            # Add batch dimension: [1, C, Z, Y, X]
            input_tensor = preprocessed_data.unsqueeze(0).to(predictor.device)
            pred_logits = predictor.predict_logits_from_preprocessed_data(preprocessed_data)
            pred_seg = pred_logits.argmax(0).cpu().numpy().astype(np.uint8)
        inference_time = time.time() - start_time

        # Save prediction result
        save_nifti(pred_seg, pred_output_path, spacing=spacing, is_seg=True)

        result = {
            'identifier': identifier,
            'pred_path': pred_output_path,
            'inference_time': inference_time,
            'shape': img.shape
        }

        # Save image and GT
        if args.save_images:
            image_output_path = os.path.join(image_output_dir, f"{identifier}_0000.nii.gz")
            if not os.path.exists(image_output_path):
                save_nifti(img, image_output_path, spacing=spacing, is_seg=False)
            result['image_path'] = image_output_path

        # If GT labels exist, compute Dice metrics
        if gt_seg is not None:
            gt_seg_eval = gt_seg.copy()
            gt_seg_eval[gt_seg_eval < 0] = 0
            gt_seg_eval = gt_seg_eval.astype(np.uint8)

            num_classes = predictor.label_manager.num_segmentation_heads
            dice_scores = compute_dice(pred_seg, gt_seg_eval, num_classes)
            result['dice_scores'] = dice_scores
            all_dice_scores[identifier] = dice_scores

            if args.save_images:
                gt_output_path = os.path.join(gt_output_dir, f"{identifier}.nii.gz")
                if not os.path.exists(gt_output_path):
                    save_nifti(gt_seg_eval, gt_output_path, spacing=spacing, is_seg=True)

        all_results.append(result)

    # Compute summary metrics
    summary = compute_summary_metrics(all_dice_scores)
    summary['total_cases'] = len(all_results)
    summary['total_time'] = sum(r['inference_time'] for r in all_results)
    summary['avg_time_per_case'] = summary['total_time'] / len(all_results) if all_results else 0

    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        summary['peak_memory_mb'] = peak_memory_bytes / (1024 * 1024)

    # Save results
    results_file = os.path.join(args.output, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({'summary': summary, 'per_case_results': all_results}, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Total Time: {summary['total_time']:.2f}s")
    print(f"Avg Time per Case: {summary['avg_time_per_case']:.2f}s")

    if 'mean_dice' in summary:
        print(f"\nMean Dice: {summary['mean_dice']:.4f}")
        print("Per-class Dice:")
        for c, dice in summary['per_class_dice'].items():
            print(f"  Class {c}: {dice:.4f}")

    print(f"\nResults saved to: {args.output}")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='U-VLM Segmentation Inference (Blosc2 format)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--csv-path', type=str, required=True,
                        help='Path to CSV file containing test data paths (blosc2 format)')
    parser.add_argument('--model-folder', type=str, required=True,
                        help='Path to trained model folder')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for predictions')

    # Model configuration
    parser.add_argument('--folds', type=int, nargs='+', default=[0],
                        help='Folds to use for inference')
    parser.add_argument('--checkpoint-name', type=str, default='checkpoint_best.pth',
                        help='Checkpoint filename')

    # Inference parameters
    parser.add_argument('--tile-step-size', type=float, default=0.5,
                        help='Step size for sliding window')
    parser.add_argument('--use-mirroring', action='store_true', default=True,
                        help='Use test-time augmentation (mirroring)')
    parser.add_argument('--use-gaussian', action='store_true', default=True,
                        help='Use Gaussian weighting for sliding window')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true', default=False)

    # Output options
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='Save input images and GT segmentations as NIfTI')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip cases with existing predictions')

    # Debug
    parser.add_argument('--max-cases', type=int, default=-1,
                        help='Max cases to process (-1 for all)')

    args = parser.parse_args()

    # Check paths
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    if not os.path.exists(args.model_folder):
        raise FileNotFoundError(f"Model folder not found: {args.model_folder}")

    maybe_mkdir_p(args.output)

    # Run inference
    run_inference(args)


if __name__ == '__main__':
    main()
