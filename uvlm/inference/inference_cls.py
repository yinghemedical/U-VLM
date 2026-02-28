"""
Multi-GPU Only Classification Inference.

Fully compatible with nnUNetTrainer_UVLM's only_cls mode multi-process inference code.

Compatible with:
- nnUNetTrainer_UVLM_only_cls

Features:
- Multi-seed testing for reproducibility analysis
- Only uses real vision mode (not zero/random)
- Multiple GPUs with parallel workers
- Automatically retrieves patch_size, target_z_size from checkpoint
- Outputs CSV table containing cls_columns and probability values for each case

Data Format:
- Only supports Blosc2 format (.b2nd)

Data Flow (identical to trainer's validation_step):
    1. main() -> parse_args() -> load CSV -> build paths
    2. main() -> spawn worker processes (predict_on_gpu)
    3. predict_on_gpu() -> init_predictor() -> load checkpoint & extract config
    4. predict_on_gpu() -> create dataset (uses predictor.get_data_config() to retrieve configuration)
    5. predict_on_gpu() -> load_case() -> preprocess (resize x,y to patch_size) -> predict_single_npy() -> results_queue
    6. csv_saver_thread() -> periodically save results to CSV

Output CSV Format:
    - series_id: Series-level identifier (read from column specified by --series-id-column)
    - [cls_columns]: Probability values for each classification column (column names retrieved from plan)
    - success: Whether successful

Preprocessing:
    - If input image x,y dimensions != patch_size[1], patch_size[2], resize to match
    - Z dimension is kept unchanged
    - Target shape: (patch_size[0], patch_size[1], z_original)

Key Parameters (automatically retrieved from checkpoint):
    - patch_size: Retrieved from plans
    - target_z_size: patch_size[0]
    - strides: list of stride values for each stage
    - cls_columns: List of classification column names

Usage:
    # Standard inference
    python inference_cls.py \
        --csv-path /path/to/test.csv \
        --gpu-config "4:2,6:2"

    # Debug mode
    python inference_cls.py \
        --csv-path /path/to/test.csv \
        --gpu-config "4:1" \
        --debug-max-cases 10
"""

import argparse
import os
import random
import threading
import time
from multiprocessing import Process, Queue
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from batchgenerators.utilities.file_and_folder_operations import join
from tqdm import tqdm

from uvlm.inference.predict_cls import nnUNetPredictor
from uvlm.dataloading.dataset_csv_blosc2 import nnUNetDatasetCSVBlosc2


# Default configuration
DEFAULTS = {
    "dataset_name": "Dataset140_CT-RATE",
    "trainer_name": "nnUNetTrainer_UVLM_only_cls",
    "plans_name": "nnUNetResEncUNetL_only_cls_256_384_384_Plans",
    "configuration_name": "3d_fullres",
    "fold": 0,
    "checkpoint_name": "checkpoint_best.pth",
    "base_results_dir": "/path/to/results/nnUNet_results",
    "csv_path": "/path/to/data/validation_merged.csv",
    "output_suffix": "inference_results_only_cls_best",
    "gpu_config": "2:2,3:2",
    "debug_max_cases": None,
    "random_seed": 42,
    "num_repeats": 1,  # Number of seeds to test per case
}


def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.

    Matches trainer's random seed settings exactly:
    - random.seed(seed)
    - np.random.seed(seed)
    - torch.manual_seed(seed)
    - torch.cuda.manual_seed_all(seed)
    - cudnn.deterministic = True
    - cudnn.benchmark = False
    - os.environ['PYTHONHASHSEED'] = str(seed)

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
    """Build model and output paths."""
    model_dir = join(
        args.base_results_dir,
        args.dataset_name,
        f"{args.trainer_name}__{args.plans_name}__{args.configuration_name}"
    )
    output_dir = join(model_dir, args.output_suffix)
    output_csv = join(output_dir, "results.csv")
    return model_dir, output_dir, output_csv


def init_predictor(gpu_id: int, args):
    """
    Initialize predictor on specified GPU.

    Automatically loads all configuration parameters from checkpoint (identical to trainer):
    - patch_size: Retrieved from plans
    - target_z_size: patch_size[0]
    - strides: Stride values for each encoder stage
    - data_format: 'blosc2' (Blosc2 supported)
    - cls_columns: List of classification column names
    - cls_head_num_classes_list: List of class counts for classification heads

    Args:
        gpu_id: GPU device ID
        args: Parsed command line arguments

    Returns:
        Tuple of (predictor, data_config)
    """
    torch.cuda.set_device(gpu_id)

    # Enable verbose for first GPU only
    is_first_gpu = (gpu_id == min(args.gpu_config.keys()))

    predictor = nnUNetPredictor(
        device=torch.device("cuda", gpu_id),
        verbose=is_first_gpu  # Only first GPU prints details
    )

    predictor.initialize_from_trained_model_folder(
        args.model_dir,
        use_folds=[args.fold],
        checkpoint_name=args.checkpoint_name
    )

    # Retrieve data configuration (from predictor, identical to trainer)
    data_config = predictor.get_data_config()

    if is_first_gpu:
        print(f"Data configuration from checkpoint:")
        print(f"  patch_size: {data_config['patch_size']}")
        print(f"  target_z_size: {data_config['target_z_size']}")
        print(f"  strides: {data_config['strides']}")
        print(f"  data_format: {data_config['data_format']}")
        print(f"  cls_columns: {data_config['cls_columns']}")
        print(f"  cls_head_num_classes_list: {data_config['cls_head_num_classes_list']}")
        print(f"  num_cls_task: {data_config['num_cls_task']}")

    return predictor, data_config

def predict_on_gpu(gpu_id: int, files_queue: Queue, results_queue: Queue, args):
    """
    Worker process for GPU-based prediction.

    Data flow matches trainer's validation flow:
    1. init_predictor() -> Load model and configuration parameters
    2. Select dataset class based on data_format (consistent with trainer's get_tr_and_val_datasets)
    3. dataset.load_case() -> Load and preprocess image
    4. predictor.predict_single_npy() -> Return classification probabilities
    5. results_queue.put() -> Send results to save thread

    Dataset: Blosc2 format is supported (nnUNetDatasetCSVBlosc2)

    Args:
        gpu_id: GPU device ID
        files_queue: Queue containing case keys to process
        results_queue: Queue for sending results to saver thread
        args: Parsed command line arguments
    """
    # ==================== Initialize predictor ====================
    predictor, data_config = init_predictor(gpu_id, args)

    # Get cls_columns for output
    cls_columns = data_config['cls_columns']

    # ==================== Create dataset (only Blosc2 format supported) ====================
    dataset = nnUNetDatasetCSVBlosc2(
        csv_paths=[args.csv_path],
        verbose=False,
        strides=data_config['strides'],
        target_z_size=data_config['target_z_size']
    )

    # Progress bar
    progress_bar = (
        tqdm(
            total=args.total_cases,
            desc=f"GPU {gpu_id}",
            unit="case",
            position=gpu_id
        )
        if args.total_cases
        else None
    )

    while True:
        case_key = files_queue.get()
        if case_key is None:  # Stop signal
            break

        # Load case with validation (identical to trainer's load_case)
        input_img, _, _ = dataset.load_case(case_key)

        # Validate input image
        if input_img is None:
            print(f"Warning: Case {case_key}: input_img is None (skipped due to size limit)")
            # Record skipped case
            result_dict = (
                args.csv_data[case_key].copy()
                if args.csv_data and case_key in args.csv_data
                else {"series_id": case_key}
            )
            result_dict["seed_idx"] = 0
            result_dict["random_seed"] = args.random_seed
            result_dict["success"] = False
            result_dict["skip_reason"] = "size_limit_exceeded"
            # Add null values for all cls_columns
            for col_name in cls_columns:
                result_dict[col_name] = None
            results_queue.put(result_dict)
            if progress_bar:
                progress_bar.update(1)
            continue

        if not isinstance(input_img, np.ndarray):
            raise ValueError(f"Case {case_key}: input_img is not a numpy array, got {type(input_img)}")

        if input_img.size == 0:
            raise ValueError(f"Case {case_key}: input_img is empty (size=0)")

        if not np.isfinite(input_img).all():
            raise ValueError(f"Case {case_key}: input_img contains NaN or Inf values")

        print(f"Case {case_key}: input_img.shape = {input_img.shape}, dtype = {input_img.dtype}, range = [{input_img.min():.3f}, {input_img.max():.3f}]")

        # Ensure 5D input [B, C, D, H, W]
        if input_img.ndim == 4:
            input_img = input_img[np.newaxis, ...]

        # Final validation after potential reshaping
        if input_img.ndim != 5:
            raise ValueError(f"Case {case_key}: expected 5D input after reshaping, got {input_img.ndim}D with shape {input_img.shape}")

        expected_channels = 1  # Assuming single channel CT images
        if input_img.shape[1] != expected_channels:
            print(f"Warning: Case {case_key}: expected {expected_channels} channels, got {input_img.shape[1]}")

        # Test with multiple seeds
        all_results = []

        for seed_idx in range(args.num_repeats):
            current_seed = args.random_seed + seed_idx

            # Set seed for reproducibility
            set_seed(current_seed)

            # Predict classification (identical to trainer's validation_step)
            if progress_bar and (seed_idx == 0 or args.num_repeats <= 3):  # Log first few seeds
                print(f"GPU {gpu_id}: Processing case {case_key} with seed {current_seed}")

            cls_probs_list, cls_probs_dict = predictor.predict_single_npy(input_img)

            if progress_bar and (seed_idx == 0 or args.num_repeats <= 3):  # Log first few seeds
                print(f"GPU {gpu_id}: Case {case_key} seed {current_seed} -> {len(cls_probs_dict)} classification results")

            # Store result
            result_dict = (
                args.csv_data[case_key].copy()
                if args.csv_data and case_key in args.csv_data
                else {"series_id": case_key}
            )

            result_dict["seed_idx"] = seed_idx
            result_dict["random_seed"] = current_seed
            result_dict["success"] = True

            # Add classification probabilities to result dictionary (directly using column names)
            for col_name, prob_value in cls_probs_dict.items():
                result_dict[col_name] = prob_value

            all_results.append(result_dict)

        # Send all results for this case to the queue
        for result_dict in all_results:
            results_queue.put(result_dict)

        if progress_bar:
            progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    print(f"GPU {gpu_id} worker finished")


def csv_saver_thread(results_queue: Queue, all_results: list, output_path: str, stop_event: threading.Event):
    """Background thread to save results periodically."""
    last_save_time = time.time()
    save_interval = 10  # Save every 10 seconds

    while not stop_event.is_set():
        # Collect results
        while not results_queue.empty():
            result = results_queue.get_nowait()
            all_results.append(result)

        # Save periodically
        current_time = time.time()
        should_save = (
            all_results and
            (current_time - last_save_time >= save_interval or not os.path.exists(output_path))
        )

        if should_save:
            print(f"Saving {len(all_results)} results...")
            df = pd.DataFrame(all_results)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            last_save_time = current_time

        time.sleep(1)

    # Final save
    if all_results:
        print(f"Final save: {len(all_results)} results")
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed only classification inference matching training validation (UVLM)"
    )

    # Input
    parser.add_argument("--csv-path", default=DEFAULTS["csv_path"],
                        help="Path to CSV file with test cases")
    parser.add_argument("--series-id-column", default="series_id",
                        help="Column name for series ID in CSV (default: series_id)")
    # Model configuration
    parser.add_argument("--dataset-name", default=DEFAULTS["dataset_name"],
                        help="Dataset name")
    parser.add_argument("--trainer-name", default=DEFAULTS["trainer_name"],
                        help="Trainer class name")
    parser.add_argument("--plans-name", default=DEFAULTS["plans_name"],
                        help="Plans name")
    parser.add_argument("--configuration-name", default=DEFAULTS["configuration_name"],
                        help="Configuration name")
    parser.add_argument("--fold", type=int, default=DEFAULTS["fold"],
                        help="Fold number")
    parser.add_argument("--checkpoint-name", default=DEFAULTS["checkpoint_name"],
                        help="Checkpoint filename")
    parser.add_argument("--base-results-dir", default=DEFAULTS["base_results_dir"],
                        help="Base directory for nnUNet results")

    # Output
    parser.add_argument("--output-suffix", default=DEFAULTS["output_suffix"],
                        help="Output directory suffix")

    # Execution
    parser.add_argument("--gpu-config", default=DEFAULTS["gpu_config"],
                        help="GPU config like '0:2,1:2' (gpu_id:num_workers)")
    parser.add_argument("--debug-max-cases", type=int, default=DEFAULTS["debug_max_cases"],
                        help="Limit number of cases for debugging")
    parser.add_argument("--random-seed", type=int, default=DEFAULTS["random_seed"],
                        help="Base random seed")
    parser.add_argument("--num-repeats", type=int, default=DEFAULTS["num_repeats"],
                        help="Number of seeds to test per case")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print("nnUNet Only Classification Inference - UVLM")
    print("Exact Training Validation Match (only_cls mode)")
    print(f"{'='*70}")

    # Set global random seed
    set_seed(args.random_seed)
    print(f"Base random seed: {args.random_seed}")

    # Load CSV
    print(f"Loading CSV file: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    csv_df = pd.read_csv(args.csv_path)

    # Use the specified series_id column
    series_id_col = args.series_id_column
    if series_id_col not in csv_df.columns:
        raise ValueError(f"Column '{series_id_col}' not found in CSV. Available columns: {list(csv_df.columns)}")

    case_keys = csv_df[series_id_col].tolist()
    args.csv_data = {row[series_id_col]: row.to_dict() for _, row in csv_df.iterrows()}
    args.series_id_column = series_id_col  # Save the actual column name used
    print(f"Loaded {len(case_keys)} cases from CSV")
    print(f"Sample case keys: {case_keys[:3]}...")

    # Debug limit
    if args.debug_max_cases is not None and args.debug_max_cases > 0:
        case_keys = case_keys[:args.debug_max_cases]
        print(f"DEBUG MODE: Limited to {len(case_keys)} cases")

    # Setup paths
    model_dir, output_dir, output_csv = build_output_paths(args)
    args.model_dir = model_dir
    args.output_csv = output_csv
    args.total_cases = len(case_keys)
    args.gpu_config = parse_gpu_config(args.gpu_config)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Trainer: {args.trainer_name}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  CSV: {args.csv_path}")
    print(f"  Cases: {len(case_keys)}")
    print(f"  GPU config: {args.gpu_config}")
    print(f"\nExperiment Settings:")
    print(f"  Seeds per case: {args.num_repeats}")
    print(f"  Total inferences: {len(case_keys) * args.num_repeats}")
    print(f"\nOutput: {output_csv}")
    print(f"{'='*70}")
    print(f"NOTE: First GPU will print detailed configuration")
    print(f"NOTE: All parameters (patch_size, target_z_size, strides, cls_columns) are loaded from checkpoint")
    print(f"NOTE: Output CSV will contain probability columns for each classification task")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Setup queues and threads
    files_queue = Queue()
    results_queue = Queue()
    all_results = []
    stop_event = threading.Event()

    # Start saver thread
    saver = threading.Thread(
        target=csv_saver_thread,
        args=(results_queue, all_results, output_csv, stop_event)
    )
    saver.start()

    # Enqueue cases
    for case_key in case_keys:
        files_queue.put(case_key)

    # Start worker processes
    processes = []
    total_workers = sum(args.gpu_config.values())

    for gpu_id, num_workers in args.gpu_config.items():
        for _ in range(num_workers):
            p = Process(
                target=predict_on_gpu,
                args=(gpu_id, files_queue, results_queue, args)
            )
            p.start()
            processes.append(p)

    # Send stop signals
    for _ in range(total_workers):
        files_queue.put(None)

    # Wait for completion
    for p in processes:
        p.join()

    stop_event.set()
    saver.join()

    # Print summary
    print(f"\n{'='*70}")
    print("Inference Completed!")
    print(f"{'='*70}")
    print(f"Results: {output_csv}")
    print(f"Total results: {len(all_results)}")
    print(f"Expected results: {len(case_keys) * args.num_repeats}")
    print(f"Results match expectation: {len(all_results) == len(case_keys) * args.num_repeats}")

    if all_results:
        df = pd.DataFrame(all_results)
        successful = df['success'].sum() if 'success' in df.columns else len(df)
        print(f"Successful: {successful}/{len(all_results)}")

        # Analyze classification probability statistics (cls_columns used directly as column names)
        # Exclude non-classification columns
        non_cls_columns = {'series_id', 'case_id', 'seed_idx', 'random_seed', 'success', 'skip_reason'}
        cls_prob_columns = [col for col in df.columns if col not in non_cls_columns and col not in args.csv_data.get(list(args.csv_data.keys())[0], {}).keys()]

        if cls_prob_columns:
            print(f"\nClassification Probability Statistics:")
            print(f"  Number of classification columns: {len(cls_prob_columns)}")
            success_df = df[df['success'] == True] if 'success' in df.columns else df
            if len(success_df) > 0:
                for col in cls_prob_columns[:5]:  # Only print statistics for the first 5 columns
                    col_data = success_df[col].dropna()
                    if len(col_data) > 0:
                        print(f"  {col}: mean={col_data.mean():.4f}, min={col_data.min():.4f}, max={col_data.max():.4f}")
                if len(cls_prob_columns) > 5:
                    print(f"  ... and {len(cls_prob_columns) - 5} more classification columns")

        # Analyze consistency across seeds
        if 'seed_idx' in df.columns:
            # Use the actual series_id column name to count unique cases
            id_col = args.series_id_column if args.series_id_column in df.columns else 'series_id'
            num_unique_cases = df[id_col].nunique() if id_col in df.columns else len(case_keys)
            print(f"\nCase Statistics:")
            print(f"  Unique cases: {num_unique_cases}")
            print(f"  Seeds per case: {args.num_repeats}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
