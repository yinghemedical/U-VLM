# -*- coding: utf-8 -*-
"""
CT-RATE Chest Classification/Report Data Preprocessing

Pipeline:
1. resize: nii.gz -> blosc2 format (resize to target size)
2. reports: Process reports CSV file, add series_id, case_id, report columns
3. classification: Process classification labels CSV file
4. merge: Merge reports and classification, generate training/validation CSV

This is the second part of chest data (first part is ReXGroundingCT segmentation data)

Usage:
    python -m uvlm.preprocessing.preprocess_ct_rate_cls_report \
        --config-path /path/to/ct_rate_config.json \
        all --train-input-dir /path/to/train --val-input-dir /path/to/val --output-dir /path/to/output
"""
import os
import json
import random
import argparse
import blosc2
import nibabel as nib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import zoom

from uvlm.preprocessing.common_utils import (
    reorient_nifti, write_blosc2_file, select_debug_nii_samples,
    format_report, split_by_filename_prefix
)


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============== Resize Related Functions ==============

def load_nii_file(file_path: str, target_orientation: str = "LPS") -> Tuple[np.ndarray, Tuple[float, ...]]:
    """
    Load nii.gz file, reorient to target orientation, return (data, spacing)
    Output data format: (x, y, z) - xyz order
    """
    nii = nib.load(file_path)
    nii = reorient_nifti(nii, target_orientation)
    data = nii.get_fdata()  # (x, y, z)
    spacing = tuple(abs(s) for s in nii.header.get_zooms()[:3])
    return data, spacing


def process_single_image(args) -> dict:
    """Process a single image, delete source file on failure"""
    nii_path, output_dir, target_size, target_orientation, save_debug_nii = args

    base_name = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]

    result = _process_single_image_impl(nii_path, output_dir, target_size, target_orientation, save_debug_nii, base_name)
    if result.get("status") == "error":
        # Processing failed, delete source file, will skip on next run
        if os.path.exists(nii_path):
            os.remove(nii_path)
        result["status"] = "failed_deleted"
        result["needs_restart"] = True
    return result


def _process_single_image_impl(nii_path: str, output_dir: str, target_size: List[int],
                                target_orientation: str, save_debug_nii: bool, base_name: str) -> dict:
    """Actual implementation of processing a single image"""
    # blosc2 output
    blosc2_dir = os.path.join(output_dir, "imagesTr")
    os.makedirs(blosc2_dir, exist_ok=True)
    output_path = os.path.join(blosc2_dir, f"{base_name}.b2nd")

    if not os.path.exists(nii_path):
        return {"case_id": base_name, "status": "error", "reason": "file not found"}

    data, original_spacing = load_nii_file(nii_path, target_orientation)

    original_shape = data.shape  # (x, y, z)

    # target_size is [D, H, W] = [z, y, x], convert to (x, y, z)
    target_shape = (target_size[2], target_size[1], target_size[0])

    # Calculate zoom factors and new spacing
    zoom_factors = [t / s for t, s in zip(target_shape, original_shape)]
    new_spacing = tuple(original_spacing[i] * original_shape[i] / target_shape[i] for i in range(3))

    # Resize
    data_resized = zoom(data, zoom_factors, order=1, mode='constant').astype(np.int16)
    write_blosc2_file(output_path, data_resized)

    # Save debug nii.gz
    debug_nii_path = None
    if save_debug_nii:
        debug_dir = os.path.join(output_dir, "debug_nii")
        os.makedirs(debug_dir, exist_ok=True)
        debug_nii_path = os.path.join(debug_dir, f"{base_name}.nii.gz")
        # LPS orientation affine: x(L) negative, y(P) negative, z(S) positive
        new_affine = np.diag([-new_spacing[0], -new_spacing[1], new_spacing[2], 1.0])
        nib.save(nib.Nifti1Image(data_resized, new_affine), debug_nii_path)

    return {
        "case_id": base_name, "status": "success",
        "original_shape": original_shape, "final_shape": data_resized.shape,
        "orientation": target_orientation,
        "blosc2_path": output_path, "debug_nii_path": debug_nii_path
    }


def resize_images(input_dir: str, output_dir: str, target_size: List[int],
                  target_orientation: str = "LPS",
                  num_workers: int = 15, save_debug_nii: bool = False,
                  debug_nii_sample_size: int = 10,
                  case_ids: List[str] = None) -> List[dict]:
    """Resize images to target size, unify orientation to target_orientation, output xyz order"""
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Target orientation: {target_orientation}")
    print(f"Save debug nii: {save_debug_nii} (sample size: {debug_nii_sample_size})")

    os.makedirs(output_dir, exist_ok=True)

    # Find all nii.gz files
    nii_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, f))

    print(f"Found {len(nii_files)} nii.gz files")

    # If case_ids specified, only process these
    if case_ids is not None:
        nii_files = [f for f in nii_files if any(cid in f for cid in case_ids)]
        print(f"Filtered to {len(nii_files)} files")

    if not nii_files:
        return []

    # Filter already processed files
    blosc2_dir = os.path.join(output_dir, "imagesTr")
    pending_files = []
    for nii_path in nii_files:
        base_name = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
        output_path = os.path.join(blosc2_dir, f"{base_name}.b2nd")
        if not os.path.exists(output_path):
            pending_files.append(nii_path)

    skip_count = len(nii_files) - len(pending_files)
    print(f"Found {len(nii_files)} files, {skip_count} already processed, {len(pending_files)} to process")

    if not pending_files:
        return []

    # Randomly select files to save debug nii (select from first 50 for earlier results)
    debug_nii_files = set()
    if save_debug_nii and pending_files:
        debug_nii_files = select_debug_nii_samples(pending_files, debug_nii_sample_size)

    tasks = [(f, output_dir, target_size, target_orientation, f in debug_nii_files) for f in pending_files]

    results = []
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Resizing images") as pbar:
            for result in pool.imap_unordered(process_single_image, tasks):
                results.append(result)
                pbar.update(1)

    success = sum(1 for r in results if r["status"] == "success")
    error = sum(1 for r in results if r["status"] == "error")
    failed_deleted = [r for r in results if r.get("status") == "failed_deleted"]
    print(f"\nComplete: Success={success}, Error={error}, Total={len(tasks)}")

    # If there are failed and deleted cases, prompt user to re-run
    if failed_deleted:
        print(f"\n[WARNING] {len(failed_deleted)} cases failed and source files deleted:")
        for r in failed_deleted[:10]:
            print(f"  - {r['case_id']}: {r.get('error', 'unknown error')}")
        if len(failed_deleted) > 10:
            print(f"  ... and {len(failed_deleted) - 10} more")
        print("\n[ACTION] These files were corrupted. Re-download or skip them.")

    return results


# ============== Reports Processing ==============

def extract_series_id(volume_name: str) -> str:
    """Extract series_id from volume_name"""
    return volume_name.replace('.nii.gz', '').replace('.nii', '')


def extract_case_id(series_id: str) -> str:
    """Extract case_id from series_id"""
    parts = series_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return series_id


def process_reports(input_csv: str, output_csv: str, blosc2_dir: str,
                    config: dict, num_workers: int = 15) -> pd.DataFrame:
    """Process reports CSV, generate report target columns"""
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Image dir: {blosc2_dir}")

    reports_config = config.get("reports", {})
    columns = reports_config.get("columns", {})
    volume_col = columns.get("volume_name", "VolumeName")
    findings_col = columns.get("findings", "Findings_EN")
    impressions_col = columns.get("impressions", "Impressions_EN")

    # processed_shape from config (resize output)
    target_size = config.get("processing_params", {}).get("target_size") or config.get("image", {}).get("patch_size")
    processed_shape = str(target_size) if target_size else ""

    # path column name based on output_format: blosc2 -> blosc2_path, zst -> zst_path
    output_format = config.get("processing_params", {}).get("output_format", "blosc2")
    path_column = f"{output_format}_path"

    # filename pattern from config
    paths_config = config.get("paths", {})
    filename_pattern = paths_config.get("image_filename_pattern", "{case_id}.b2nd")

    print(f"Output format: {output_format}, path column: {path_column}")
    print(f"Processed shape (from config): {processed_shape}")

    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"Processing {total} reports...")

    records = []
    found_count = 0
    for _, row in tqdm(df.iterrows(), total=total, desc="Processing reports"):
        volume_name = str(row.get(volume_col, ""))
        series_id = extract_series_id(volume_name)
        case_id = extract_case_id(series_id)

        findings = row.get(findings_col, "")
        impressions = row.get(impressions_col, "")

        report = format_report(findings, impressions)

        # path using config pattern
        filename = filename_pattern.replace("{case_id}", series_id)
        file_path = os.path.join(blosc2_dir, filename)

        if os.path.exists(file_path):
            found_count += 1

        records.append({
            "series_id": series_id,
            "case_id": case_id,
            "volume_name": volume_name,
            "report": report,
            "processed_shape": processed_shape,
            path_column: file_path
        })

    result_df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    print(f"\nComplete: {total} reports, found {found_count}/{total} image files")

    return result_df


# ============== Classification Processing ==============

def process_classification(input_csv: str, output_csv: str, config: dict) -> pd.DataFrame:
    """Process classification labels CSV"""
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")

    cls_config = config.get("classification", {})
    cls_columns = cls_config.get("columns", [])

    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"Processing {total} classification labels...")

    # Extract series_id and case_id
    if "VolumeName" in df.columns:
        df["series_id"] = df["VolumeName"].apply(lambda x: extract_series_id(str(x)))
        df["case_id"] = df["series_id"].apply(extract_case_id)

    # Ensure classification columns exist
    for col in cls_columns:
        if col not in df.columns:
            df[col] = 0

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Statistics
    print("\nClassification statistics:")
    for col in cls_columns:
        if col in df.columns:
            positive = df[col].sum()
            print(f"  {col}: {positive}/{total} ({100*positive/total:.1f}%)")

    return df


# ============== Merge Processing ==============

def merge_reports_and_labels(reports_csv: str, labels_csv: str, output_csv: str,
                              config: dict) -> pd.DataFrame:
    """Merge reports and classification labels"""
    print(f"Reports CSV: {reports_csv}")
    print(f"Labels CSV: {labels_csv}")
    print(f"Output CSV: {output_csv}")

    cls_config = config.get("classification", {})
    cls_columns = cls_config.get("columns", [])

    reports_df = pd.read_csv(reports_csv)
    labels_df = pd.read_csv(labels_csv)

    print(f"Reports: {len(reports_df)}, Labels: {len(labels_df)}")

    # Merge by series_id
    merged_df = pd.merge(
        reports_df, labels_df,
        on="series_id",
        how="inner",
        suffixes=("", "_label")
    )

    print(f"Merged: {len(merged_df)}")

    # Ensure classification columns exist
    for col in cls_columns:
        if col not in merged_df.columns:
            merged_df[col] = 0

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged_df.to_csv(output_csv, index=False)

    return merged_df


def split_train_val(merged_csv: str, output_dir: str, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into training and validation sets"""
    print(f"Merged CSV: {merged_csv}")
    print(f"Output dir: {output_dir}")

    merged_df = pd.read_csv(merged_csv)

    split_config = config.get("split_method", {})
    filename_col = split_config.get("filename_column", "series_id")
    train_prefix = split_config.get("train_prefix", "train_")
    val_prefix = split_config.get("val_prefix", "valid_")

    train_df, val_df = split_by_filename_prefix(
        merged_df,
        filename_column=filename_col,
        train_prefix=train_prefix,
        val_prefix=val_prefix
    )

    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_merged.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation_merged.csv"), index=False)

    return train_df, val_df


# ============== Main Functions ==============

def verify_config(config: dict, train_input_dir: str, val_input_dir: str, output_dir: str):
    """Verify configuration"""
    print("=" * 50)
    print("CT-RATE Configuration")
    print("=" * 50)

    print(f"\nPaths:")
    print(f"  Train input: {train_input_dir}")
    print(f"  Val input: {val_input_dir}")
    print(f"  Output: {output_dir}")

    print(f"\nProcessing params:")
    params = config.get("processing_params", {})
    print(f"  Target size: {params.get('target_size', [192, 256, 256])}")
    print(f"  Target orientation: {params.get('target_orientation', 'LPS')}")
    print(f"  Output format: blosc2")

    print(f"\nReports config:")
    reports = config.get("reports", {})
    print(f"  Input dir: {reports.get('input_dir')}")
    print(f"  Columns: {reports.get('columns')}")

    print(f"\nClassification config:")
    cls = config.get("classification", {})
    print(f"  Input dir: {cls.get('input_dir')}")
    print(f"  Num classes: {cls.get('num_classes')}")
    print(f"  Columns: {cls.get('columns')}")


def main():
    parser = argparse.ArgumentParser(description="CT-RATE Preprocessing")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to configuration JSON file")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # resize command
    resize_parser = subparsers.add_parser("resize", help="Resize nii.gz to blosc2")
    resize_parser.add_argument("--input-dir", type=str, required=True,
                               help="Input directory containing nii.gz files")
    resize_parser.add_argument("--output-dir", type=str, required=True,
                               help="Output directory")
    resize_parser.add_argument("--target-size", type=int, nargs=3, default=None,
                               help="Target size [D, H, W] (default: from config)")
    resize_parser.add_argument("--num-workers", type=int, default=15)
    resize_parser.add_argument("--no-debug-nii", action="store_true", help="Disable debug nii output")

    # reports command
    reports_parser = subparsers.add_parser("reports", help="Process reports CSV")
    reports_parser.add_argument("--input-csv", type=str, required=True,
                                help="Input reports CSV file")
    reports_parser.add_argument("--output-csv", type=str, required=True,
                                help="Output processed CSV file")
    reports_parser.add_argument("--blosc2-dir", type=str, required=True,
                                help="Directory containing blosc2 image files")
    reports_parser.add_argument("--num-workers", type=int, default=15)

    # classification command
    cls_parser = subparsers.add_parser("classification", help="Process classification CSV")
    cls_parser.add_argument("--input-csv", type=str, required=True,
                            help="Input classification CSV file")
    cls_parser.add_argument("--output-csv", type=str, required=True,
                            help="Output processed CSV file")

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge reports and classification")
    merge_parser.add_argument("--reports-csv", type=str, required=True,
                              help="Reports processed CSV")
    merge_parser.add_argument("--labels-csv", type=str, required=True,
                              help="Classification processed CSV")
    merge_parser.add_argument("--output-csv", type=str, required=True,
                              help="Output merged CSV")

    # all command
    all_parser = subparsers.add_parser("all", help="Run full pipeline (resize + reports + classification + merge)")
    all_parser.add_argument("--train-input-dir", type=str, required=True,
                            help="Train input directory containing nii.gz files")
    all_parser.add_argument("--val-input-dir", type=str, required=True,
                            help="Validation input directory containing nii.gz files")
    all_parser.add_argument("--output-dir", type=str, required=True,
                            help="Output directory")
    all_parser.add_argument("--reports-input-dir", type=str, required=True,
                            help="Directory containing reports CSV files")
    all_parser.add_argument("--cls-input-dir", type=str, required=True,
                            help="Directory containing classification CSV files")
    all_parser.add_argument("--target-size", type=int, nargs=3, default=None,
                            help="Target size [D, H, W] (default: from config)")
    all_parser.add_argument("--num-workers", type=int, default=15)
    all_parser.add_argument("--no-debug-nii", action="store_true", help="Disable debug nii output")

    # debug command
    debug_parser = subparsers.add_parser("debug", help="Process N random cases for debugging")
    debug_parser.add_argument("--input-dir", type=str, required=True,
                              help="Input directory containing nii.gz files")
    debug_parser.add_argument("--output-dir", type=str, required=True,
                              help="Output directory")
    debug_parser.add_argument("--target-size", type=int, nargs=3, default=None,
                              help="Target size [D, H, W] (default: from config)")
    debug_parser.add_argument("--num-cases", type=int, default=10)
    debug_parser.add_argument("--num-workers", type=int, default=10)
    debug_parser.add_argument("--seed", type=int, default=42)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify configuration")
    verify_parser.add_argument("--train-input-dir", type=str, default="",
                               help="Train input directory")
    verify_parser.add_argument("--val-input-dir", type=str, default="",
                               help="Validation input directory")
    verify_parser.add_argument("--output-dir", type=str, default="",
                               help="Output directory")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    params = config.get("processing_params", {})
    default_target_size = params.get("target_size", [192, 256, 256])
    target_orientation = params.get("target_orientation", "LPS")
    splits_config = config.get("splits", {})

    if args.command == "resize":
        target_size = args.target_size if args.target_size else default_target_size
        resize_images(args.input_dir, args.output_dir, target_size,
                     target_orientation=target_orientation,
                     num_workers=args.num_workers,
                     save_debug_nii=not args.no_debug_nii)

    elif args.command == "reports":
        process_reports(args.input_csv, args.output_csv, args.blosc2_dir, config, args.num_workers)

    elif args.command == "classification":
        process_classification(args.input_csv, args.output_csv, config)

    elif args.command == "merge":
        merge_reports_and_labels(args.reports_csv, args.labels_csv, args.output_csv, config)

    elif args.command == "all":
        target_size = args.target_size if args.target_size else default_target_size
        reports_output_dir = os.path.join(args.output_dir, "radiology_text_reports_processed")
        cls_output_dir = os.path.join(args.output_dir, "classification_processed")

        # Step 1: Resize images
        for split_name, input_dir in [("train", args.train_input_dir), ("val", args.val_input_dir)]:
            print(f"\n{'='*50}")
            print(f"Step 1: Resizing {split_name} images")
            print("=" * 50)
            resize_images(input_dir, args.output_dir, target_size,
                         target_orientation=target_orientation,
                         num_workers=args.num_workers,
                         save_debug_nii=not args.no_debug_nii)

        # Step 2: Process reports
        for split in ["train", "val"]:
            print(f"\n{'='*50}")
            print(f"Step 2: Processing {split} reports")
            print("=" * 50)
            split_info = splits_config.get(split, {})
            input_csv = os.path.join(args.reports_input_dir, split_info.get('reports_csv', f'{split}_reports.csv'))
            output_csv = os.path.join(reports_output_dir, f"{split}_reports_processed.csv")
            blosc2_dir = os.path.join(args.output_dir, "imagesTr")
            process_reports(input_csv, output_csv, blosc2_dir, config, args.num_workers)

        # Step 3: Process classification
        for split in ["train", "val"]:
            print(f"\n{'='*50}")
            print(f"Step 3: Processing {split} classification")
            print("=" * 50)
            split_info = splits_config.get(split, {})
            input_csv = os.path.join(args.cls_input_dir, split_info.get('labels_csv', f'{split}_predicted_labels.csv'))
            output_csv = os.path.join(cls_output_dir, f"{split}_cls_processed.csv")
            process_classification(input_csv, output_csv, config)

        # Step 4: Merge
        for split in ["train", "val"]:
            print(f"\n{'='*50}")
            print(f"Step 4: Merging {split} reports and classification")
            print("=" * 50)
            reports_csv = os.path.join(reports_output_dir, f"{split}_reports_processed.csv")
            cls_csv = os.path.join(cls_output_dir, f"{split}_cls_processed.csv")
            output_csv = os.path.join(args.output_dir, f"{split}_merged.csv")
            merge_reports_and_labels(reports_csv, cls_csv, output_csv, config)

        print("\nDone!")

    elif args.command == "debug":
        target_size = args.target_size if args.target_size else default_target_size

        print("=" * 50)
        print(f"Debug mode: Processing {args.num_cases} random cases")
        print("=" * 50)

        # Find all files
        nii_files = []
        for root, dirs, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith('.nii.gz'):
                    nii_files.append(os.path.join(root, f))

        print(f"Found {len(nii_files)} nii.gz files")

        # Random selection
        random.seed(args.seed)
        selected_files = random.sample(nii_files, min(args.num_cases, len(nii_files)))
        selected_ids = [os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0] for f in selected_files]
        print(f"Selected {len(selected_ids)} cases: {selected_ids[:5]}...")

        # Process
        resize_images(args.input_dir, args.output_dir, target_size,
                     target_orientation=target_orientation,
                     num_workers=args.num_workers,
                     save_debug_nii=True,
                     debug_nii_sample_size=args.num_cases,  # In debug mode, all cases save nii
                     case_ids=selected_ids)

        print("\nDone!")

    elif args.command == "verify":
        verify_config(config, args.train_input_dir, args.val_input_dir, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
