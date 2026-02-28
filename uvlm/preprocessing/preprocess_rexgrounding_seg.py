# -*- coding: utf-8 -*-
"""
ReXGroundingCT Chest Segmentation Preprocessing

Pipeline:
1. Read existing nii.gz files from nnUNet_raw directory
2. Resize + blosc2 compression to nnUNet_preprocessed

Reuses common functions from common_utils

Usage:
    python -m uvlm.preprocessing.preprocess_rexgrounding_seg \
        --config-path /path/to/rexgrounding_ct_config.json \
        all --raw-input-dir /path/to/raw --output-dir /path/to/output
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

from uvlm.preprocessing.common_utils import (
    resize_dataset_to_preprocessed,
    run_debug_mode,
)


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def verify_config(config: dict, raw_input_dir: str, output_dir: str):
    """Verify configuration"""
    print("=" * 50)
    print("ReXGroundingCT Segmentation Configuration")
    print("=" * 50)

    print(f"\nPaths:")
    print(f"  Raw input: {raw_input_dir}")
    print(f"  Preprocessed output: {output_dir}")

    print(f"\nProcessing params:")
    params = config.get("processing_params", {})
    print(f"  Target size: {params.get('target_size', [192, 256, 256])}")
    print(f"  Output format: blosc2")

    print(f"\nOriginal classes: {config.get('original_classes', {}).get('num_classes', 'N/A')}")

    print(f"\nMerge scopes:")
    for scope_name, scope_config in config.get("merge_scopes", {}).items():
        num_classes = scope_config.get("num_merged_classes", "N/A")
        desc = scope_config.get("description", "")
        has_lut = scope_config.get("merge_lut") is not None
        print(f"  {scope_name}: {num_classes} classes (merge: {has_lut})")
        if desc:
            print(f"    {desc}")


def main():
    parser = argparse.ArgumentParser(description="ReXGroundingCT Segmentation Preprocessing")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to configuration JSON file")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # all command - full pipeline
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--raw-input-dir", type=str, required=True,
                            help="nnUNet_raw input directory")
    all_parser.add_argument("--output-dir", type=str, required=True,
                            help="Preprocessed output directory")
    all_parser.add_argument("--scopes", type=str, nargs="+", default=None,
                            help="Scopes to process (default: all)")
    all_parser.add_argument("--target-size", type=int, nargs=3, default=None,
                            help="Target size [D, H, W] (default: from config)")
    all_parser.add_argument("--num-workers", type=int, default=20)
    all_parser.add_argument("--no-debug-nii", action="store_true", help="Disable debug nii output")
    all_parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="both",
                           help="Which split to process: train, val, or both")

    # debug command
    debug_parser = subparsers.add_parser("debug", help="Process N random cases for debugging")
    debug_parser.add_argument("--raw-input-dir", type=str, required=True,
                              help="nnUNet_raw input directory")
    debug_parser.add_argument("--output-dir", type=str, required=True,
                              help="Debug output directory")
    debug_parser.add_argument("--num-cases", type=int, default=10)
    debug_parser.add_argument("--scopes", type=str, nargs="+", default=None)
    debug_parser.add_argument("--target-size", type=int, nargs=3, default=None)
    debug_parser.add_argument("--num-workers", type=int, default=10)
    debug_parser.add_argument("--seed", type=int, default=42)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify configuration")
    verify_parser.add_argument("--raw-input-dir", type=str, default="",
                               help="nnUNet_raw input directory")
    verify_parser.add_argument("--output-dir", type=str, default="",
                               help="Preprocessed output directory")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    processing_params = config.get("processing_params", {})
    all_scopes = list(config.get("merge_scopes", {}).keys())
    default_target_size = tuple(processing_params.get("target_size", [192, 256, 256]))

    if args.command == "all":
        # Determine which splits to process
        splits_to_process = []
        if args.split in ["train", "both"]:
            splits_to_process.append("train")
        if args.split in ["val", "both"]:
            splits_to_process.append("val")

        target_orientation = processing_params.get("target_orientation", "LPS")
        target_size = tuple(args.target_size) if args.target_size else default_target_size
        scopes = args.scopes if args.scopes else all_scopes

        # Process each split
        for split in splits_to_process:
            print("=" * 50)
            print(f"ReXGroundingCT: Resize {split.upper()} set to nnUNet_preprocessed")
            print("=" * 50)
            resize_dataset_to_preprocessed(
                raw_input_dir=args.raw_input_dir,
                preprocessed_output_dir=args.output_dir,
                config=config,
                scopes=scopes,
                target_size=target_size,
                target_orientation=target_orientation,
                num_workers=args.num_workers,
                save_debug_nii=not args.no_debug_nii,
                split=split
            )

        print("\nDone!")

    elif args.command == "debug":
        target_orientation = processing_params.get("target_orientation", "LPS")
        target_size = tuple(args.target_size) if args.target_size else default_target_size
        scopes = args.scopes if args.scopes else all_scopes

        run_debug_mode(
            raw_input_dir=args.raw_input_dir,
            debug_output_dir=args.output_dir,
            config=config,
            num_cases=args.num_cases,
            scopes=scopes,
            target_size=target_size,
            target_orientation=target_orientation,
            num_workers=args.num_workers,
            seed=args.seed
        )

    elif args.command == "verify":
        verify_config(config, args.raw_input_dir, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
