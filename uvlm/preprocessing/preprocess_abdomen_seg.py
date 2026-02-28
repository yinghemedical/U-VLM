# -*- coding: utf-8 -*-
"""
AbdomenAtlas Abdominal Segmentation Preprocessing

Pipeline:
1. Merge segmentation masks from segmentations directory to nnUNet_raw
2. Resize + blosc2 compression to nnUNet_preprocessed

Reuses common functions from common_utils

Usage:
    python -m uvlm.preprocessing.preprocess_abdomen_seg \
        --config-path /path/to/abdomen_atlas_config.json \
        all --input-dir /path/to/source --raw-output-dir /path/to/raw --preprocessed-output-dir /path/to/preprocessed
"""
import os
import json
import random
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from uvlm.preprocessing.common_utils import (
    resize_dataset_to_preprocessed,
    run_debug_mode,
    reorient_nifti,
    apply_lut,
    create_lut_from_config,
    generate_seg_plans,
    generate_iid_ood_scope_csvs
)


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_file_to_id(config: dict) -> Dict[str, int]:
    """Get filename to label ID mapping"""
    return config["original_classes"]["file_to_id"]


# ============== Mask Merging Functions ==============

def combine_segmentation_masks(
    seg_dir: str,
    file_to_id: Dict[str, int],
    reorient: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[nib.Nifti1Header]]:
    """
    Combine individual organ segmentation files into a single multi-label volume
    """
    if not os.path.exists(seg_dir):
        return None, None, None

    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')]
    if not seg_files:
        return None, None, None

    combined = None
    affine = None
    header = None

    for seg_file in seg_files:
        organ_name = seg_file.replace('.nii.gz', '')
        label_id = file_to_id.get(organ_name)
        if label_id is None:
            continue

        seg_path = os.path.join(seg_dir, seg_file)
        nii = nib.load(seg_path)

        if reorient:
            nii = reorient_nifti(nii, "LPS")

        data = nii.get_fdata().astype(np.uint8)

        if combined is None:
            combined = np.zeros(data.shape, dtype=np.uint8)
            affine = nii.affine
            header = nii.header

        combined[data > 0] = label_id

    return combined, affine, header


def process_single_case_raw(args) -> dict:
    """Process a single case to nnUNet_raw (nii.gz format)"""
    case_dir, raw_output_dir, file_to_id, image_only_dir = args

    case_id = os.path.basename(case_dir)
    seg_dir = os.path.join(case_dir, "segmentations")

    if not os.path.exists(seg_dir):
        return {"case_id": case_id, "status": "skip", "reason": "no segmentations dir"}

    # Merge masks (reorient to LPS)
    combined, affine, header = combine_segmentation_masks(seg_dir, file_to_id, reorient=True)

    if combined is None:
        return {"case_id": case_id, "status": "skip", "reason": "no valid masks"}

    # Save original labels (no class merging, will merge during resize)
    label_output_dir = os.path.join(raw_output_dir, "labelsTr")
    os.makedirs(label_output_dir, exist_ok=True)
    label_output_path = os.path.join(label_output_dir, f"{case_id}.nii.gz")

    label_nii = nib.Nifti1Image(combined.astype(np.uint8), affine, header)
    nib.save(label_nii, label_output_path)

    # Process original image (sync reorient to LPS)
    image_output_path = None
    ct_path = os.path.join(image_only_dir, case_id, "ct.nii.gz")

    if os.path.exists(ct_path):
        image_output_dir = os.path.join(raw_output_dir, "imagesTr")
        os.makedirs(image_output_dir, exist_ok=True)
        image_output_path = os.path.join(image_output_dir, f"{case_id}_0000.nii.gz")

        nii = nib.load(ct_path)
        nii = reorient_nifti(nii, "LPS")
        nib.save(nii, image_output_path)

    # Statistics
    unique_labels = np.unique(combined)
    unique_labels = unique_labels[unique_labels > 0].tolist()

    return {
        "case_id": case_id,
        "status": "success",
        "label_path": label_output_path,
        "image_path": image_output_path,
        "shape": combined.shape,
        "unique_labels": unique_labels,
        "num_labels": len(unique_labels)
    }


def merge_to_raw(
    input_dir: str,
    raw_output_dir: str,
    config: dict,
    num_workers: int = 20,
    case_ids: List[str] = None
) -> List[dict]:
    """
    Step 1: Merge segmentation labels to nnUNet_raw (nii.gz format)
    """
    file_to_id = get_file_to_id(config)

    print(f"Output dir: {raw_output_dir}")

    # Get case directories
    mask_only_dir = os.path.join(input_dir, "mask_only")
    image_only_dir = os.path.join(input_dir, "image_only")

    if case_ids is None:
        case_dirs = sorted([
            os.path.join(mask_only_dir, d)
            for d in os.listdir(mask_only_dir)
            if d.startswith("BDMAP_") and os.path.isdir(os.path.join(mask_only_dir, d))
        ])
    else:
        case_dirs = [os.path.join(mask_only_dir, cid) for cid in case_ids]

    # Filter already processed cases
    pending_dirs = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        label_path = os.path.join(raw_output_dir, "labelsTr", f"{case_id}.nii.gz")
        image_path = os.path.join(raw_output_dir, "imagesTr", f"{case_id}_0000.nii.gz")
        if not (os.path.exists(label_path) and os.path.exists(image_path)):
            pending_dirs.append(case_dir)

    skip_count = len(case_dirs) - len(pending_dirs)
    print(f"Found {len(case_dirs)} cases, {skip_count} already processed, {len(pending_dirs)} to process")

    if not pending_dirs:
        return []

    # Prepare tasks
    tasks = [
        (case_dir, raw_output_dir, file_to_id, image_only_dir)
        for case_dir in pending_dirs
    ]

    # Parallel processing
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_case_raw, task): task for task in tasks}

        with tqdm(total=len(tasks), desc="Merging to raw (image + label)") as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    # Statistics
    success_count = sum(1 for r in results if r["status"] == "success")

    print(f"\nMerge complete: Success={success_count}, Total={len(tasks)}")

    return results


def generate_dataset_json(raw_output_dir: str, config: dict) -> str:
    """
    Generate nnUNet format dataset.json file
    """
    file_to_id = get_file_to_id(config)

    # Build labels mapping (name -> id)
    labels = {"background": 0}
    for name, label_id in file_to_id.items():
        labels[name] = label_id

    # Count training samples
    labels_dir = os.path.join(raw_output_dir, "labelsTr")
    if os.path.exists(labels_dir):
        num_training = len([f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')])
    else:
        num_training = 0

    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }

    output_path = os.path.join(raw_output_dir, "dataset.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print(f"Generated dataset.json: {output_path}")
    print(f"  - Labels: {len(labels)} classes (including background)")
    print(f"  - numTraining: {num_training}")

    return output_path


def verify_config(config: dict, input_dir: str, raw_output_dir: str, preprocessed_output_dir: str):
    """Verify configuration"""
    print("=" * 50)
    print("AbdomenAtlas Segmentation Configuration")
    print("=" * 50)

    print(f"\nPaths:")
    print(f"  Input: {input_dir}")
    print(f"  Raw output: {raw_output_dir}")
    print(f"  Preprocessed output: {preprocessed_output_dir}")

    print(f"\nProcessing params:")
    params = config.get("processing_params", {})
    print(f"  Target size: {params.get('target_size', [192, 256, 256])}")
    print(f"  Output format: blosc2")

    print(f"\nOriginal classes: {config['original_classes']['num_classes']}")

    print(f"\nMerge scopes:")
    for scope_name, scope_config in config.get("merge_scopes", {}).items():
        num_classes = scope_config.get("num_merged_classes", "N/A")
        desc = scope_config.get("description", "")
        has_lut = scope_config.get("merge_lut") is not None
        print(f"  {scope_name}: {num_classes} classes (merge: {has_lut})")
        if desc:
            print(f"    {desc}")


def main():
    parser = argparse.ArgumentParser(description="AbdomenAtlas Segmentation Preprocessing")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to configuration JSON file")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # merge command - merge to nnUNet_raw
    merge_parser = subparsers.add_parser("merge", help="Merge masks to nnUNet_raw (nii.gz)")
    merge_parser.add_argument("--input-dir", type=str, required=True,
                              help="Input directory containing mask_only/ and image_only/")
    merge_parser.add_argument("--output-dir", type=str, required=True,
                              help="nnUNet_raw output directory")
    merge_parser.add_argument("--num-workers", type=int, default=20)

    # resize command - resize + blosc2 to nnUNet_preprocessed
    resize_parser = subparsers.add_parser("resize", help="Resize + blosc2 to nnUNet_preprocessed")
    resize_parser.add_argument("--raw-input-dir", type=str, required=True,
                               help="nnUNet_raw input directory")
    resize_parser.add_argument("--output-dir", type=str, required=True,
                               help="nnUNet_preprocessed output directory")
    resize_parser.add_argument("--scopes", type=str, nargs="+", default=None,
                               help="Scopes to process (default: all)")
    resize_parser.add_argument("--target-size", type=int, nargs=3, default=None,
                               help="Target size [D, H, W] (default: from config)")
    resize_parser.add_argument("--num-workers", type=int, default=20)
    resize_parser.add_argument("--no-debug-nii", action="store_true", help="Disable debug nii output")

    # debug command - randomly process N cases
    debug_parser = subparsers.add_parser("debug", help="Merge all + resize N random cases")
    debug_parser.add_argument("--input-dir", type=str, required=True,
                              help="Input directory containing mask_only/ and image_only/")
    debug_parser.add_argument("--raw-output-dir", type=str, required=True,
                              help="nnUNet_raw output directory")
    debug_parser.add_argument("--preprocessed-output-dir", type=str, required=True,
                              help="Debug preprocessed output directory")
    debug_parser.add_argument("--num-cases", type=int, default=10)
    debug_parser.add_argument("--scopes", type=str, nargs="+", default=None)
    debug_parser.add_argument("--target-size", type=int, nargs=3, default=None)
    debug_parser.add_argument("--num-workers", type=int, default=20)
    debug_parser.add_argument("--seed", type=int, default=42)

    # all command - full pipeline
    all_parser = subparsers.add_parser("all", help="Run full pipeline (merge + resize)")
    all_parser.add_argument("--input-dir", type=str, required=True,
                            help="Input directory containing mask_only/ and image_only/")
    all_parser.add_argument("--raw-output-dir", type=str, required=True,
                            help="nnUNet_raw output directory")
    all_parser.add_argument("--preprocessed-output-dir", type=str, required=True,
                            help="nnUNet_preprocessed output directory")
    all_parser.add_argument("--split-dir", type=str, default=None,
                            help="Directory containing IID/OOD split files (default: input_dir/TrainTestIDS)")
    all_parser.add_argument("--scopes", type=str, nargs="+", default=None)
    all_parser.add_argument("--target-size", type=int, nargs=3, default=None)
    all_parser.add_argument("--num-workers", type=int, default=20)
    all_parser.add_argument("--no-debug-nii", action="store_true", help="Disable debug nii output")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify configuration")
    verify_parser.add_argument("--input-dir", type=str, default="",
                               help="Input directory")
    verify_parser.add_argument("--raw-output-dir", type=str, default="",
                               help="nnUNet_raw output directory")
    verify_parser.add_argument("--preprocessed-output-dir", type=str, default="",
                               help="nnUNet_preprocessed output directory")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    processing_params = config.get("processing_params", {})
    default_target_size = tuple(processing_params.get("target_size", [192, 256, 256]))
    all_scopes = list(config.get("merge_scopes", {}).keys())

    if args.command == "merge":
        merge_to_raw(args.input_dir, args.output_dir, config, args.num_workers)
        generate_dataset_json(args.output_dir, config)

    elif args.command == "resize":
        print("=" * 50)
        print("Resize to nnUNet_preprocessed")
        print("=" * 50)
        target_orientation = processing_params.get("target_orientation", "LPS")
        target_size = tuple(args.target_size) if args.target_size else default_target_size
        scopes = args.scopes if args.scopes else all_scopes
        resize_dataset_to_preprocessed(
            raw_input_dir=args.raw_input_dir,
            preprocessed_output_dir=args.output_dir,
            config=config,
            scopes=scopes,
            target_size=target_size,
            target_orientation=target_orientation,
            num_workers=args.num_workers,
            save_debug_nii=not args.no_debug_nii
        )
        print("\nDone!")

    elif args.command == "debug":
        target_orientation = processing_params.get("target_orientation", "LPS")
        target_size = tuple(args.target_size) if args.target_size else default_target_size
        scopes = args.scopes if args.scopes else all_scopes

        # Step 1: Merge ALL cases to nnUNet_raw
        print("=" * 50)
        print("Step 1: Merge ALL cases to nnUNet_raw")
        print("=" * 50)
        raw_results = merge_to_raw(
            input_dir=args.input_dir,
            raw_output_dir=args.raw_output_dir,
            config=config,
            num_workers=args.num_workers,
            case_ids=None
        )

        # Get successfully processed cases
        success_cases = [r["case_id"] for r in raw_results if r["status"] == "success"]
        print(f"\nSuccessfully merged {len(success_cases)} cases")

        # Step 2: Randomly select N cases for resize
        print("\n" + "=" * 50)
        print(f"Step 2: Resize {args.num_cases} random cases")
        print("=" * 50)

        random.seed(args.seed)
        selected_cases = random.sample(success_cases, min(args.num_cases, len(success_cases)))
        # Convert to format with _0000 suffix
        selected_case_ids = [f"{cid}_0000" for cid in selected_cases]
        print(f"Selected cases: {selected_cases}")

        resize_dataset_to_preprocessed(
            raw_input_dir=args.raw_output_dir,
            preprocessed_output_dir=args.preprocessed_output_dir,
            config=config,
            scopes=scopes,
            target_size=target_size,
            target_orientation=target_orientation,
            num_workers=min(args.num_workers, args.num_cases),
            save_debug_nii=True,
            case_ids=selected_case_ids
        )

        print("\nDone!")

    elif args.command == "all":
        target_orientation = processing_params.get("target_orientation", "LPS")
        target_size = tuple(args.target_size) if args.target_size else default_target_size
        scopes = args.scopes if args.scopes else all_scopes

        # Step 1: Merge to nnUNet_raw
        print("=" * 50)
        print("Step 1: Merge to nnUNet_raw")
        print("=" * 50)
        merge_to_raw(args.input_dir, args.raw_output_dir, config, args.num_workers)
        generate_dataset_json(args.raw_output_dir, config)

        # Step 2: Resize to nnUNet_preprocessed
        print("\n" + "=" * 50)
        print("Step 2: Resize to nnUNet_preprocessed")
        print("=" * 50)
        resize_dataset_to_preprocessed(
            raw_input_dir=args.raw_output_dir,
            preprocessed_output_dir=args.preprocessed_output_dir,
            config=config,
            scopes=scopes,
            target_size=target_size,
            target_orientation=target_orientation,
            num_workers=args.num_workers,
            save_debug_nii=not args.no_debug_nii
        )

        # Step 3: Generate IID/OOD split CSVs
        print("\n" + "=" * 50)
        print("Step 3: Generate IID/OOD Split CSVs")
        print("=" * 50)
        split_dir = args.split_dir
        if not split_dir:
            split_dir = os.path.join(args.input_dir, "TrainTestIDS")
        generate_iid_ood_scope_csvs(
            preprocessed_output_dir=args.preprocessed_output_dir,
            scopes=scopes,
            split_dir=split_dir,
            include_paths=True
        )

        # Auto-generate plan
        generate_seg_plans("abdomen_atlas", args.preprocessed_output_dir, scopes=scopes)

        print("\nDone!")

    elif args.command == "verify":
        verify_config(config, args.input_dir, args.raw_output_dir, args.preprocessed_output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
