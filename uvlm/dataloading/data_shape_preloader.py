"""
CSV Shape Preprocessor Module (Blosc2 only)

This module provides functionality to preprocess CSV files by adding shape information
for all cases, allowing subsequent training to skip invalid cases without recomputation.
"""

import os
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import ast
import blosc2
from tqdm import tqdm


def read_blosc2_file(file_path: str) -> np.ndarray:
    """Read a Blosc2-compressed NumPy array"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f".b2nd file does not exist: {file_path}")

    b2_array = blosc2.open(file_path, mmap_mode='r')
    return np.asarray(b2_array)


def process_single_case(args: Tuple[str, str]) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    """
    Process a single case to get its shape information.

    Note: Only reads and records shape, no filtering is performed.

    Returns: (identifier, shape_tuple or None)
        - shape_tuple: (Z, H, W) actual image dimensions
        - None: file read failed or format error
    """
    identifier, file_path = args

    # Check if file exists
    if not os.path.exists(file_path):
        return identifier, None

    try:
        # Read the blosc2 file
        img = read_blosc2_file(file_path)

        # Handle channel dimension
        if img.ndim == 4:
            if img.shape[0] == 1:
                img = img[0]
            else:
                return identifier, None
        elif img.ndim != 3:
            return identifier, None

        # Get the shape: (X, Y, Z) -> transpose to (Z, Y, X) for consistency
        # Note: The recorded shape is transposed to match load_case
        shape = (img.shape[2], img.shape[1], img.shape[0])  # (Z, H, W)

        # Check for invalid shapes (any dimension is 0)
        if any(s == 0 for s in shape):
            return identifier, None

        return identifier, shape
    except Exception as e:
        return identifier, None


def preprocess_csv_with_shapes(
    csv_paths: List[str],
    num_workers: Optional[int] = None,
    force_reprocess: bool = False,
    verbose: bool = True,
    data_format: str = 'blosc2'
) -> Dict[str, int]:
    """
    Preprocess CSV files by adding shape information for all cases.

    Note: This step only records shape, no filtering is performed. All cases are processed.

    Args:
        csv_paths: List of CSV file paths to process
        num_workers: Number of worker processes (default: CPU count)
        force_reprocess: Force reprocessing even if shape column exists
        verbose: Whether to print progress information
        data_format: Data format (only 'blosc2' is supported)

    Returns:
        Dictionary with statistics: {'processed': count, 'with_shape': count, 'no_shape': count}
    """
    if data_format != 'blosc2':
        raise ValueError(f"Only 'blosc2' format is supported, got: {data_format}")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)

    stats = {'processed': 0, 'with_shape': 0, 'no_shape': 0}

    # Process CSV files with progress bar
    for csv_path in tqdm(csv_paths, desc="Processing CSV files", unit="file", disable=not verbose):
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue

        if verbose:
            print(f"Processing CSV: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Check if shape column already exists
        if 'processed_shape' in df.columns and not force_reprocess:
            if verbose:
                print(f"  Shape column already exists, skipping (use force_reprocess=True to override)")
            continue

        # Collect cases to process
        cases_to_process = []
        for _, row in df.iterrows():
            identifier = row.get('series_id', row.get('identifier', str(row.name)))
            file_path = row.get('blosc2_path')
            if file_path:
                cases_to_process.append((identifier, file_path))

        if not cases_to_process:
            continue

        if verbose:
            print(f"  Processing {len(cases_to_process)} cases using {num_workers} workers...")

        # Process cases using multiprocessing with progress tracking
        start_time = time.time()
        shape_results = {}

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_case = {
                executor.submit(process_single_case, case): case[0]
                for case in cases_to_process
            }

            # Use tqdm for processing progress
            with tqdm(total=len(future_to_case), desc="Reading shapes", unit="case",
                      disable=not verbose, leave=False) as pbar:
                for future in as_completed(future_to_case):
                    identifier = future_to_case[future]
                    result_identifier, shape = future.result()
                    shape_results[result_identifier] = shape
                    pbar.update(1)

        # Update DataFrame with shape information (save as list format)
        shapes_list = []
        for _, row in df.iterrows():
            identifier = row.get('series_id', row.get('identifier', str(row.name)))
            shape = shape_results.get(identifier)
            if shape is not None:
                # Save as list format: [32, 512, 512]
                shapes_list.append(str(list(shape)))
                stats['with_shape'] += 1
            else:
                # Cases that failed to read, shape is None
                shapes_list.append('None')
                stats['no_shape'] += 1

        df['processed_shape'] = shapes_list
        stats['processed'] += len(df)

        # Save updated CSV
        df.to_csv(csv_path, index=False)
        if verbose:
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.2f} seconds")
            print(f"  Results: {stats['with_shape']} cases with shape, {stats['no_shape']} cases failed to read")

    return stats


def load_shapes_from_csv(csv_paths: List[str]) -> Tuple[Dict[str, Tuple[int, int, int]], Set[str]]:
    """
    Load shape information from CSV files that have been preprocessed with shapes.

    Args:
        csv_paths: List of CSV file paths

    Returns:
        Tuple of (valid_shapes_dict, invalid_identifiers_set)
    """
    valid_shapes = {}
    invalid_cases = set()

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # Check if shape column exists
        if 'processed_shape' not in df.columns:
            continue

        for _, row in df.iterrows():
            identifier = row.get('series_id', row.get('identifier', str(row.name)))
            shape_str = row.get('processed_shape', '')

            if shape_str and shape_str.strip() and shape_str != 'None':
                try:
                    # Parse shape tuple from string like "[64, 128, 128]"
                    shape = ast.literal_eval(shape_str)
                    if isinstance(shape, (tuple, list)) and len(shape) == 3:
                        valid_shapes[identifier] = tuple(shape)
                    else:
                        invalid_cases.add(identifier)
                except:
                    invalid_cases.add(identifier)
            else:
                invalid_cases.add(identifier)

    return valid_shapes, invalid_cases


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess CSV files with shape information')
    parser.add_argument('--csv_paths', nargs='+', required=True, help='CSV file paths to process')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if shape column exists')

    args = parser.parse_args()

    print("Starting CSV shape preprocessing...")
    print(f"CSV files: {args.csv_paths}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Force reprocess: {args.force}")

    stats = preprocess_csv_with_shapes(
        csv_paths=args.csv_paths,
        num_workers=args.workers,
        force_reprocess=args.force,
        verbose=True
    )

    print("\nPreprocessing completed!")
    print(f"Total processed: {stats['processed']}")
    print(f"Cases with shape: {stats['with_shape']}")
    print(f"Cases failed: {stats['no_shape']}")
