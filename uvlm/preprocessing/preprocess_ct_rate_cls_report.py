"""
CT-RATE Dataset Preprocessing for Classification and Report Generation

This module preprocesses CT-RATE chest CT scans for multi-task learning:
- Converts NIfTI images to Blosc2 compressed format
- Extracts classification labels from metadata
- Processes radiology reports
- Generates CSV files for training

Usage:
    python -m uvlm.preprocessing.preprocess_ct_rate_cls_report \
        --source-dir /path/to/CT-RATE/train \
        --output-dir /path/to/output \
        --split train
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import SimpleITK as sitk
import blosc2
from tqdm import tqdm

from .common_utils import (
    reorient_nifti_file,
    compute_cumulative_strides,
    process_z_axis
)


# CT-RATE classification columns (18 pathologies)
CT_RATE_CLS_COLUMNS = [
    'lung_nodule', 'lung_mass', 'pneumonia', 'atelectasis',
    'pleural_effusion', 'pneumothorax', 'consolidation', 'pulmonary_edema',
    'emphysema', 'fibrosis', 'thickening', 'calcification',
    'cardiomegaly', 'fracture', 'lesion', 'nodule',
    'opacity', 'abnormality'
]


def load_ct_rate_metadata(metadata_path: str) -> Dict:
    """
    Load CT-RATE metadata JSON file

    Args:
        metadata_path: Path to metadata.json

    Returns:
        Metadata dictionary
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_classification_labels(metadata: Dict) -> Dict[str, int]:
    """
    Extract binary classification labels from metadata

    Args:
        metadata: Case metadata dictionary

    Returns:
        Dictionary mapping pathology names to binary labels (0/1)
    """
    labels = {}

    # Extract from 'findings' or 'labels' field
    findings = metadata.get('findings', {})
    if isinstance(findings, dict):
        for pathology in CT_RATE_CLS_COLUMNS:
            labels[pathology] = int(findings.get(pathology, 0))
    else:
        # Default to 0 if not found
        for pathology in CT_RATE_CLS_COLUMNS:
            labels[pathology] = 0

    return labels


def extract_report_text(metadata: Dict) -> str:
    """
    Extract radiology report text from metadata

    Args:
        metadata: Case metadata dictionary

    Returns:
        Report text string
    """
    # Try different possible field names
    for field in ['report', 'impression', 'findings_text', 'text']:
        if field in metadata and metadata[field]:
            return str(metadata[field]).strip()

    return ""


def preprocess_single_case(
    case_id: str,
    source_dir: str,
    output_dir: str,
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
    target_orientation: str = "LPS"
) -> Optional[Dict]:
    """
    Preprocess a single CT-RATE case

    Args:
        case_id: Case identifier
        source_dir: Source directory containing NIfTI files
        output_dir: Output directory for Blosc2 files
        target_spacing: Target voxel spacing (z, y, x)
        target_orientation: Target orientation code

    Returns:
        Case information dictionary or None if failed
    """
    try:
        # Paths
        nifti_path = os.path.join(source_dir, f"{case_id}.nii.gz")
        metadata_path = os.path.join(source_dir, f"{case_id}_metadata.json")

        if not os.path.exists(nifti_path):
            return None

        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            metadata = load_ct_rate_metadata(metadata_path)

        # Load and reorient image
        img_sitk = sitk.ReadImage(nifti_path)

        # Reorient to standard orientation
        reoriented_path = reorient_nifti_file(
            nifti_path,
            target_orientation=target_orientation,
            inplace=False
        )
        img_sitk = sitk.ReadImage(reoriented_path)

        # Get image array
        img_array = sitk.GetArrayFromImage(img_sitk)  # (Z, Y, X)

        # Resample to target spacing if needed
        original_spacing = img_sitk.GetSpacing()  # (x, y, z)
        if not np.allclose(original_spacing[::-1], target_spacing, atol=0.1):
            # Compute new size
            original_size = img_array.shape
            scale_factors = np.array(original_spacing[::-1]) / np.array(target_spacing)
            new_size = (np.array(original_size) * scale_factors).astype(int)

            # Resample using SimpleITK
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing([target_spacing[2], target_spacing[1], target_spacing[0]])
            resampler.SetSize([int(new_size[2]), int(new_size[1]), int(new_size[0])])
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputDirection(img_sitk.GetDirection())
            resampler.SetOutputOrigin(img_sitk.GetOrigin())

            img_sitk = resampler.Execute(img_sitk)
            img_array = sitk.GetArrayFromImage(img_sitk)

        # Transpose to (X, Y, Z) for Blosc2 storage
        img_array = np.transpose(img_array, (2, 1, 0)).astype(np.float32)

        # Save as Blosc2
        blosc2_dir = os.path.join(output_dir, 'blosc2')
        os.makedirs(blosc2_dir, exist_ok=True)
        blosc2_path = os.path.join(blosc2_dir, f"{case_id}.b2nd")

        blosc2.save_array(
            img_array,
            blosc2_path,
            mode='w',
            cparams={'codec': blosc2.Codec.ZSTD, 'clevel': 5}
        )

        # Extract labels and report
        cls_labels = extract_classification_labels(metadata)
        report_text = extract_report_text(metadata)

        # Build case info
        case_info = {
            'series_id': case_id,
            'case_id': case_id.rsplit('_', 1)[0] if '_' in case_id else case_id,
            'blosc2_path': blosc2_path,
            'report': report_text,
            'original_shape': list(img_array.shape),
        }

        # Add classification labels
        case_info.update(cls_labels)

        # Clean up temporary reoriented file
        if reoriented_path != nifti_path and os.path.exists(reoriented_path):
            os.remove(reoriented_path)

        return case_info

    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return None


def preprocess_ct_rate_dataset(
    source_dir: str,
    output_dir: str,
    split: str = 'train',
    num_workers: int = 8,
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0)
) -> str:
    """
    Preprocess CT-RATE dataset

    Args:
        source_dir: Source directory containing NIfTI files
        output_dir: Output directory
        split: Dataset split ('train', 'val', 'test')
        num_workers: Number of parallel workers
        target_spacing: Target voxel spacing

    Returns:
        Path to output CSV file
    """
    print(f"Preprocessing CT-RATE {split} split...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find all NIfTI files
    nifti_files = list(Path(source_dir).glob("*.nii.gz"))
    case_ids = [f.stem.replace('.nii', '') for f in nifti_files]

    print(f"Found {len(case_ids)} cases")

    # Process cases in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                preprocess_single_case,
                case_id, source_dir, output_dir, target_spacing
            ): case_id
            for case_id in case_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            case_info = future.result()
            if case_info:
                results.append(case_info)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(output_dir, f'{split}_cls_report.csv')
    df.to_csv(csv_path, index=False)

    print(f"\nPreprocessing completed!")
    print(f"Processed: {len(results)}/{len(case_ids)} cases")
    print(f"Output CSV: {csv_path}")

    # Print statistics
    print(f"\nClassification label statistics:")
    for col in CT_RATE_CLS_COLUMNS:
        if col in df.columns:
            positive_count = df[col].sum()
            print(f"  {col}: {positive_count}/{len(df)} ({positive_count/len(df)*100:.1f}%)")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CT-RATE dataset for classification and report generation'
    )
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Source directory containing NIfTI files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--target-spacing', type=float, nargs=3,
                        default=[1.5, 1.0, 1.0],
                        help='Target voxel spacing (z y x)')

    args = parser.parse_args()

    preprocess_ct_rate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split=args.split,
        num_workers=args.num_workers,
        target_spacing=tuple(args.target_spacing)
    )


if __name__ == '__main__':
    main()
