"""
Common utilities for U-VLM inference scripts.

Provides shared functions for:
- Loading validation case IDs from dataset_split.json
- Case ID extraction and filtering
- GPU config parsing
- Random seed setting
- Output path building
"""

import json
import os
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from batchgenerators.utilities.file_and_folder_operations import join


def set_seed(seed: int):
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_gpu_config(raw: str) -> Dict[int, int]:
    """Parse GPU config string '0:2,1:2' -> {0: 2, 1: 2}."""
    config = {}
    if not raw:
        return config
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        gpu, procs = token.split(":")
        config[int(gpu)] = int(procs)
    return config


def build_output_paths(args) -> Tuple[str, str, str]:
    """Build model and output paths from args."""
    model_dir = join(
        args.base_results_dir,
        args.dataset_name,
        f"{args.trainer_name}__{args.plans_name}__{args.configuration_name}"
    )
    output_dir = join(model_dir, args.output_suffix)
    output_csv = join(output_dir, "results.csv")
    return model_dir, output_dir, output_csv


def load_val_case_ids(model_folder: str, fold: int) -> Optional[List[str]]:
    """Load validation case IDs from dataset_split.json.

    Args:
        model_folder: Path to model directory containing fold_X folders
        fold: Fold number to load val_case_ids for

    Returns:
        List of validation case IDs, or None if file not found
    """
    split_json_path = os.path.join(model_folder, f"fold_{fold}", "dataset_split.json")
    if not os.path.exists(split_json_path):
        print(f"Warning: dataset_split.json not found at {split_json_path}, using all cases")
        return None

    with open(split_json_path, 'r') as f:
        split_info = json.load(f)

    val_case_ids = split_info.get('val_case_ids', [])
    print(f"Loaded val_case_ids from dataset_split.json: {len(val_case_ids)} cases")
    return val_case_ids


def extract_case_id(identifier: str) -> str:
    """Extract case ID from identifier.

    Handles two formats:
    1. train_2_a_1 -> train_2_a (strip trailing slice number)
    2. train_10001_a_1_0000 -> train_10001_a_1_0000 (keep 4-digit suffix as part of ID)

    The key insight is that 4-digit suffixes like _0000 are part of the case ID
    and should NOT be stripped, while single/double digit suffixes are slice indices.
    """
    parts = identifier.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        if len(parts[1]) == 4:
            return identifier
        return parts[0]
    return identifier


def get_case_id_for_filtering(identifier: str, val_case_ids: List[str]) -> str:
    """Extract case ID by progressively stripping trailing numeric parts until a match is found.

    This handles cases like:
    - train_2_a_1_40 -> train_2_a (matches val_case_id train_2_a)

    Args:
        identifier: The full identifier
        val_case_ids: List of validation case IDs to match against

    Returns:
        The matched case ID, or the original identifier if no match found
    """
    parts = identifier.split('_')
    val_set = set(val_case_ids)

    for i in range(len(parts), 0, -1):
        candidate = '_'.join(parts[:i])
        if candidate in val_set:
            return candidate

    return identifier


def filter_df_by_val_cases(df, id_col: str, val_case_ids: List[str]):
    """Filter dataframe to only include rows belonging to validation cases.

    Args:
        df: Input dataframe
        id_col: Name of the identifier column
        val_case_ids: List of validation case IDs

    Returns:
        Filtered dataframe containing only validation cases
    """
    if val_case_ids is None:
        return df

    mask = df[id_col].apply(lambda x: get_case_id_for_filtering(x, val_case_ids) in val_case_ids)
    filtered_df = df[mask].copy()

    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        print(f"Filtered out {removed_count} non-validation rows, kept {len(filtered_df)} rows")

    return filtered_df


def get_unique_cases(df, id_col: str, case_id_col: str) -> List[str]:
    """Get unique case IDs from dataframe.

    For report generation, we need to process each case only once,
    even if the CSV has multiple rows per case (different slices/phases).

    Args:
        df: Input dataframe (already filtered to validation cases)
        id_col: Name of the series_id column (e.g., 'series_id')
        case_id_col: Name of the case_id column (e.g., 'case_id')

    Returns:
        List of unique case IDs
    """
    if case_id_col and case_id_col in df.columns:
        unique_cases = df[case_id_col].unique().tolist()
        print(f"Unique cases (by {case_id_col}): {len(unique_cases)}")
        return unique_cases
    else:
        # Fallback: extract case ID from series_id
        unique_ids = df[id_col].apply(lambda x: get_case_id_for_filtering(x, [])).unique().tolist()
        print(f"Unique cases (extracted from {id_col}): {len(unique_ids)}")
        return unique_ids
