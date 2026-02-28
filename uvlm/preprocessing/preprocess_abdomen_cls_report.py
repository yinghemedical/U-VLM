# -*- coding: utf-8 -*-
"""
AbdomenAtlas Classification and Report Preprocessing

Features:
1. Extract classification labels from segmentation masks
2. Compute tumor sizes using WHO standard measurement (for RadGPT Table 2 evaluation)
3. Process metadata CSV for reports
4. Merge reports and classification labels
5. Generate training/validation CSV files

Usage:
    python -m uvlm.preprocessing.preprocess_abdomen_cls_report \
        --config-path /path/to/abdomen_atlas_config.json \
        all --output-dir /path/to/output
"""
import os
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Optional

from uvlm.preprocessing.common_utils import (
    load_nifti, split_by_ids, split_by_iid_ood
)


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def extract_cls_with_size(
    seg_dir: str,
    ct_path: str,
    cls_columns: List[str],
    lesion_mapping: Dict[str, str]
) -> Dict[str, any]:
    """Extract classification labels and tumor sizes from GT segmentation masks.

    For GT (human-annotated masks), we do NOT use volume threshold.
    If annotated, it's positive. Only erosion is applied to remove edge noise.

    Following RadGPT: GT labels come from human reports, not from thresholded masks.
    Since AbdomenAtlas has no human reports, we use annotated masks directly as GT.

    Args:
        seg_dir: Directory containing segmentation masks
        ct_path: Path to CT scan (for spacing information)
        cls_columns: List of classification column names
        lesion_mapping: Mapping from lesion file names to classification names

    Returns:
        Dictionary with classification labels and size information
    """
    import nibabel as nib
    from scipy import ndimage
    from scipy.ndimage import zoom

    results = {cls_name: 0 for cls_name in cls_columns}
    for cls_name in cls_columns:
        results[f'{cls_name}_largest_diameter_mm'] = 0
        results[f'{cls_name}_tumor_category'] = 'negative'

    # Get spacing from CT
    ct_nii = nib.load(ct_path)
    spacing = ct_nii.header.get_zooms()[:3]

    for lesion_file, cls_name in lesion_mapping.items():
        if cls_name not in cls_columns:
            continue

        lesion_path = os.path.join(seg_dir, f"{lesion_file}.nii.gz")
        if not os.path.exists(lesion_path):
            continue

        nii = nib.load(lesion_path)
        data = np.asanyarray(nii.dataobj)

        # For GT: no volume threshold, just check if any voxel is annotated
        # Apply erosion only to remove edge noise (following RadGPT)
        # Resample to isotropic 1mm spacing for consistent measurement
        target_spacing = (1.0, 1.0, 1.0)
        zoom_factors = [s / t for s, t in zip(spacing, target_spacing)]
        resampled = zoom(data, zoom_factors, order=0)
        eroded = ndimage.binary_erosion(resampled, structure=np.ones((3, 3, 3)), iterations=1)

        # GT: if any voxel remains after erosion, it's positive (no threshold)
        if eroded.sum() > 0:
            results[cls_name] = 1

            # Compute tumor size using simple diameter estimation
            # Find connected components and measure largest
            labeled, num_features = ndimage.label(eroded)
            if num_features > 0:
                # Find largest component
                component_sizes = ndimage.sum(eroded, labeled, range(1, num_features + 1))
                largest_label = np.argmax(component_sizes) + 1
                largest_mask = labeled == largest_label

                # Compute diameter as max extent in any axis (in mm, since spacing is 1mm)
                coords = np.where(largest_mask)
                if len(coords[0]) > 0:
                    extents = [coords[i].max() - coords[i].min() + 1 for i in range(3)]
                    diameter = max(extents)
                    results[f'{cls_name}_largest_diameter_mm'] = diameter
                    results[f'{cls_name}_tumor_category'] = 'small' if diameter <= 20 else 'large'

    return results


def extract_cls_from_metadata(row: pd.Series, cls_columns: List[str], volume_columns: Dict[str, str]) -> Dict[str, int]:
    """Extract classification labels from metadata"""
    results = {cls_name: 0 for cls_name in cls_columns}

    for cls_name, vol_col in volume_columns.items():
        if cls_name not in cls_columns:
            continue
        if vol_col not in row.index:
            continue

        volume = row[vol_col]
        if pd.notna(volume) and float(volume) > 0:
            results[cls_name] = 1

    return results


def process_single_cls(args) -> dict:
    """Process classification labels for a single case (GT extraction)"""
    case_id, seg_dir, ct_path, cls_columns, lesion_mapping, volume_columns, meta_row_dict = args

    result = {"case_id": case_id}

    # Prefer extraction from segmentation mask (no threshold for GT)
    if os.path.exists(seg_dir):
        cls_labels = extract_cls_with_size(seg_dir, ct_path, cls_columns, lesion_mapping)
        result.update(cls_labels)
        result["source"] = "segmentation"
    elif meta_row_dict is not None:
        meta_row = pd.Series(meta_row_dict)
        cls_labels = extract_cls_from_metadata(meta_row, cls_columns, volume_columns)
        result.update(cls_labels)
        result["source"] = "metadata"
    else:
        for cls_name in cls_columns:
            result[cls_name] = 0
        result["source"] = "default"

    return result


def extract_all_cls_labels(
    config: dict,
    mask_only_dir: str,
    image_only_dir: str,
    meta_csv: str,
    output_dir: str,
    num_workers: int = 20
) -> pd.DataFrame:
    """Batch extract classification labels with WHO tumor size measurement"""
    cls_config = config.get("classification", {})

    cls_columns = cls_config.get("columns", [])
    volume_columns = cls_config.get("meta_volume_mapping", {})
    lesion_mapping = cls_config.get("lesion_file_to_cls", {})
    case_prefix = "BDMAP_"

    # Load metadata
    meta_df = None
    if meta_csv and os.path.exists(meta_csv):
        meta_df = pd.read_csv(meta_csv)
        id_col = "BDMAP ID" if "BDMAP ID" in meta_df.columns else meta_df.columns[0]
        meta_df = meta_df.set_index(id_col)

    # Find all cases from mask_only_dir
    if not mask_only_dir or not os.path.exists(mask_only_dir):
        print(f"Error: mask_only_dir not found: {mask_only_dir}")
        return pd.DataFrame()

    if not image_only_dir or not os.path.exists(image_only_dir):
        print(f"Error: image_only_dir not found: {image_only_dir}")
        print("  Required for WHO tumor size measurement")
        return pd.DataFrame()

    case_dirs = sorted([
        d for d in os.listdir(mask_only_dir)
        if d.startswith(case_prefix) and os.path.isdir(os.path.join(mask_only_dir, d))
    ])

    print(f"Found {len(case_dirs)} cases")
    print("Extracting GT labels (no threshold, following RadGPT)...")
    print("Computing tumor sizes (WHO standard measurement)...")

    # Prepare tasks
    tasks = []
    for case_id in case_dirs:
        seg_dir = os.path.join(mask_only_dir, case_id, "segmentations")
        ct_path = os.path.join(image_only_dir, case_id, "ct.nii.gz")
        meta_row_dict = meta_df.loc[case_id].to_dict() if meta_df is not None and case_id in meta_df.index else None
        tasks.append((case_id, seg_dir, ct_path, cls_columns, lesion_mapping, volume_columns, meta_row_dict))

    # Parallel processing with Pool.imap_unordered for better progress feedback
    results = []
    print(f"Starting parallel processing with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_cls, tasks), total=len(tasks), desc="Extracting cls labels"):
            results.append(result)

    df = pd.DataFrame(results)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cls_labels.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")

    # Statistics
    print("\nClassification statistics:")
    for cls_name in cls_columns:
        if cls_name in df.columns:
            size_col = f'{cls_name}_largest_diameter_mm'
            if size_col in df.columns:
                small = ((df[cls_name] == 1) & (df[size_col] > 0) & (df[size_col] <= 20)).sum()
                large = ((df[cls_name] == 1) & (df[size_col] > 20)).sum()
                negative = (df[cls_name] == 0).sum()
                print(f"  {cls_name}: {small} small (<=2cm), {large} large (>2cm), {negative} negative")
            else:
                positive = df[cls_name].sum()
                print(f"  {cls_name}: {positive}/{len(df)} ({100*positive/len(df):.1f}%)")

    return df


def process_reports(
    config: dict,
    meta_csv: str,
    images_dir: str,
    output_dir: str,
    report_types: List[str] = None
) -> pd.DataFrame:
    """Process reports, generate multiple report type target columns"""
    report_config = config.get("reports", {})
    report_targets = report_config.get("report_targets", {})
    default_report_type = report_config.get("default_target", "structured")

    # filename pattern from config
    filename_pattern = config.get("paths", {}).get("image_filename_pattern", "{case_id}_0000.b2nd")

    # processed_shape comes from target_size in config (resize output)
    target_size = config.get("processing_params", {}).get("target_size") or config.get("image", {}).get("patch_size")
    processed_shape = str(target_size) if target_size else ""

    # path column name based on output_format: blosc2 -> blosc2_path, zst -> zst_path
    output_format = config.get("processing_params", {}).get("output_format", "blosc2")
    path_column = f"{output_format}_path"

    report_types = report_types or list(report_targets.keys())

    print(f"Loading metadata from {meta_csv}")
    df = pd.read_csv(meta_csv)
    print(f"Total records: {len(df)}")
    print(f"Report types: {report_types}")
    print(f"Default report type: {default_report_type}")
    print(f"Image directory: {images_dir}")
    print(f"Filename pattern: {filename_pattern}")
    print(f"Output format: {output_format}, path column: {path_column}")
    print(f"Processed shape (from config): {processed_shape}")

    records = []
    found_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reports"):
        case_id = row.get("BDMAP ID", row.get("bdmap_id", f"case_{idx}"))

        record = {
            "case_id": case_id,
            "series_id": case_id,
        }

        # Generate report columns + default 'report' column
        default_report_text = ""
        for report_type in report_types:
            target_config = report_targets.get(report_type)
            if not target_config:
                continue
            source_col = target_config["source_column"]
            target_col = target_config["target_column"]
            report_text = row.get(source_col, "")
            report_text = "" if pd.isna(report_text) else str(report_text)
            record[target_col] = report_text
            # Set default report column
            if report_type == default_report_type:
                default_report_text = report_text
        record["report"] = default_report_text

        # Add metadata
        for meta_col in ["spacing", "sex", "age"]:
            if meta_col in row.index:
                record[meta_col] = row[meta_col]

        # path and processed_shape
        filename = filename_pattern.replace("{case_id}", case_id)
        file_path = os.path.join(images_dir, filename)

        record[path_column] = file_path
        record["processed_shape"] = processed_shape

        if os.path.exists(file_path):
            found_count += 1
        elif found_count == 0 and len(records) < 3:
            print(f"  File not found: {file_path}")

        records.append(record)

    result_df = pd.DataFrame(records)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reports_processed.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} records to {output_path}")
    print(f"Found {found_count}/{len(result_df)} image files")

    # Statistics
    for report_type in report_types:
        target_config = report_targets.get(report_type)
        if target_config:
            target_col = target_config["target_column"]
            if target_col in result_df.columns:
                non_empty = result_df[target_col].str.len() > 0
                print(f"  {report_type}: {non_empty.sum()}/{len(result_df)} with content")

    return result_df


def merge_cls_report(
    reports_csv: str,
    cls_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """Merge reports and classification labels

    Note: processed_shape and image_path are already in reports_csv from process_reports()
    """
    reports_df = pd.read_csv(reports_csv)
    cls_df = pd.read_csv(cls_csv)

    print(f"Reports: {len(reports_df)}, Classification: {len(cls_df)}")

    # Merge
    merged_df = pd.merge(
        reports_df, cls_df,
        on="case_id",
        how="inner",
        suffixes=("", "_cls")
    )

    print(f"Merged: {len(merged_df)}")

    # Save
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved merged data to {output_csv}")

    return merged_df


def split_train_val(
    config: dict,
    split_dir: str,
    merged_csv: str,
    output_dir: str
):
    """Split dataset into train/validation"""
    merged_df = pd.read_csv(merged_csv)
    print(f"Loaded {len(merged_df)} records")

    split_config = config.get("split_method", {})
    use_iid_ood = split_config.get("method", "") == "iid_ood"

    os.makedirs(output_dir, exist_ok=True)

    if use_iid_ood and split_dir and os.path.exists(os.path.join(split_dir, "IID_train.csv")):
        print("Using IID/OOD split...")
        splits = split_by_iid_ood(merged_df, split_dir, "case_id")

        for split_name, split_df in splits.items():
            if not split_df.empty:
                output_path = os.path.join(output_dir, f"{split_name}_merged.csv")
                split_df.to_csv(output_path, index=False)
                print(f"Saved {split_name}: {len(split_df)} records")
    else:
        # Fallback to simple train/test split
        train_ids_path = os.path.join(split_dir, "IID_train.csv")
        val_ids_path = os.path.join(split_dir, "IID_test.csv")

        if os.path.exists(train_ids_path) and os.path.exists(val_ids_path):
            train_df, val_df = split_by_ids(merged_df, train_ids_path, val_ids_path, "case_id")
            train_df.to_csv(os.path.join(output_dir, "train_merged.csv"), index=False)
            val_df.to_csv(os.path.join(output_dir, "validation_merged.csv"), index=False)
            print(f"Train: {len(train_df)}, Validation: {len(val_df)}")


def generate_all(
    config: dict,
    mask_only_dir: str,
    image_only_dir: str,
    meta_csv: str,
    images_dir: str,
    split_dir: str,
    output_dir: str,
    num_workers: int = 20
):
    """Generate all CSVs in one go (includes WHO tumor size measurement)"""
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: Processing reports...")
    process_reports(config, meta_csv, images_dir, output_dir)

    print("\nStep 2: Extracting classification labels with tumor sizes...")
    extract_all_cls_labels(config, mask_only_dir, image_only_dir, meta_csv, output_dir, num_workers)

    print("\nStep 3: Merging...")
    merge_cls_report(
        os.path.join(output_dir, "reports_processed.csv"),
        os.path.join(output_dir, "cls_labels.csv"),
        os.path.join(output_dir, "merged.csv")
    )

    print("\nStep 4: Splitting train/val...")
    split_train_val(
        config,
        split_dir,
        os.path.join(output_dir, "merged.csv"),
        output_dir
    )

    print("\nDone!")


def verify_config(config: dict, mask_only_dir: str, image_only_dir: str, meta_csv: str, images_dir: str):
    """Verify configuration"""
    print("=" * 50)
    print("AbdomenAtlas Classification & Report Configuration")
    print("=" * 50)

    print(f"\nDataset ID: {config.get('dataset_id', 'N/A')}")
    print(f"Body region: {config.get('body_region', 'N/A')}")

    print(f"\nPaths:")
    print(f"  mask_only_dir: {mask_only_dir}")
    print(f"  image_only_dir: {image_only_dir}")
    print(f"  meta_csv: {meta_csv}")
    print(f"  images_dir: {images_dir}")

    print(f"\nClassification columns:")
    cls_config = config.get("classification", {})
    for col in cls_config.get("columns", []):
        print(f"  - {col}")

    print(f"\nVolume columns mapping:")
    for cls_name, vol_col in cls_config.get("meta_volume_mapping", {}).items():
        print(f"  {cls_name}: {vol_col}")

    print(f"\nLesion file mapping:")
    for file_name, cls_name in cls_config.get("lesion_file_to_cls", {}).items():
        print(f"  {file_name}: {cls_name}")

    print(f"\nReport targets:")
    report_config = config.get("reports", {})
    for target_name, target_config in report_config.get("report_targets", {}).items():
        source_col = target_config.get("source_column", "N/A")
        target_col = target_config.get("target_column", "N/A")
        print(f"  {target_name}: {source_col} -> {target_col}")

    print(f"\nSplit method: {config.get('split_method', {}).get('method', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="AbdomenAtlas Classification and Report Preprocessing")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to configuration JSON file")

    subparsers = parser.add_subparsers(dest="command")

    # reports command
    report_parser = subparsers.add_parser("reports", help="Process reports from metadata CSV")
    report_parser.add_argument("--meta-csv", type=str, required=True,
                               help="Path to metadata CSV file")
    report_parser.add_argument("--images-dir", type=str, required=True,
                               help="Directory containing preprocessed images")
    report_parser.add_argument("--output-dir", type=str, required=True,
                               help="Output directory")
    report_parser.add_argument("--report-types", type=str, nargs="+", default=None,
                               help="Report types to process (default: all from config)")

    # classification command
    cls_parser = subparsers.add_parser("classification", help="Extract classification labels from segmentation masks")
    cls_parser.add_argument("--mask-only-dir", type=str, required=True,
                            help="Directory containing mask_only/{case_id}/segmentations/")
    cls_parser.add_argument("--image-only-dir", type=str, required=True,
                            help="Directory containing image_only/{case_id}/ct.nii.gz")
    cls_parser.add_argument("--meta-csv", type=str, default="",
                            help="Optional metadata CSV for fallback labels")
    cls_parser.add_argument("--output-dir", type=str, required=True,
                            help="Output directory")
    cls_parser.add_argument("--num-workers", type=int, default=20)

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Merge reports and classification CSVs")
    merge_parser.add_argument("--reports-csv", type=str, required=True,
                              help="Path to reports_processed.csv")
    merge_parser.add_argument("--cls-csv", type=str, required=True,
                              help="Path to cls_labels.csv")
    merge_parser.add_argument("--output-csv", type=str, required=True,
                              help="Output merged CSV path")

    # split command
    split_parser = subparsers.add_parser("split", help="Split merged CSV into train/val")
    split_parser.add_argument("--merged-csv", type=str, required=True,
                              help="Path to merged CSV")
    split_parser.add_argument("--split-dir", type=str, required=True,
                              help="Directory containing IID/OOD split files")
    split_parser.add_argument("--output-dir", type=str, required=True,
                              help="Output directory")

    # all command - full pipeline
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--mask-only-dir", type=str, required=True,
                            help="Directory containing mask_only/{case_id}/segmentations/")
    all_parser.add_argument("--image-only-dir", type=str, required=True,
                            help="Directory containing image_only/{case_id}/ct.nii.gz")
    all_parser.add_argument("--meta-csv", type=str, required=True,
                            help="Path to metadata CSV file")
    all_parser.add_argument("--images-dir", type=str, required=True,
                            help="Directory containing preprocessed images")
    all_parser.add_argument("--split-dir", type=str, required=True,
                            help="Directory containing IID/OOD split files")
    all_parser.add_argument("--output-dir", type=str, required=True,
                            help="Output directory")
    all_parser.add_argument("--num-workers", type=int, default=20)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify configuration")
    verify_parser.add_argument("--mask-only-dir", type=str, default="",
                               help="Directory containing mask_only/")
    verify_parser.add_argument("--image-only-dir", type=str, default="",
                               help="Directory containing image_only/")
    verify_parser.add_argument("--meta-csv", type=str, default="",
                               help="Path to metadata CSV")
    verify_parser.add_argument("--images-dir", type=str, default="",
                               help="Directory containing preprocessed images")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)

    if args.command == "reports":
        process_reports(config, args.meta_csv, args.images_dir, args.output_dir, args.report_types)

    elif args.command == "classification":
        extract_all_cls_labels(config, args.mask_only_dir, args.image_only_dir,
                               args.meta_csv, args.output_dir, args.num_workers)

    elif args.command == "merge":
        merge_cls_report(args.reports_csv, args.cls_csv, args.output_csv)

    elif args.command == "split":
        split_train_val(config, args.split_dir, args.merged_csv, args.output_dir)

    elif args.command == "all":
        generate_all(
            config,
            args.mask_only_dir,
            args.image_only_dir,
            args.meta_csv,
            args.images_dir,
            args.split_dir,
            args.output_dir,
            args.num_workers
        )

    elif args.command == "verify":
        verify_config(config, args.mask_only_dir, args.image_only_dir, args.meta_csv, args.images_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
