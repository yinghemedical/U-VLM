"""
Multi-GPU Report Generation Inference.

General inference code supporting two report generation models:
- UVLM lightweight LLM (UVLM)
- Qwen3 (UVLM_Qwen3)

Automatically detects model type from checkpoint, no manual specification required.

Compatible with:
- nnUNetTrainer_UVLM
- nnUNetTrainer_UVLM_Qwen3

Features:
- Multi-seed testing for reproducibility analysis
- Only uses real vision mode (not zero/random)
- Multiple GPUs with parallel workers
- Auto-detection of model type (UVLM/qwen3) from checkpoint
- Automatically retrieves patch_size, target_z_size from checkpoint

Data Format:
- Only supports Blosc2 format (.b2nd)

Data Flow (identical to trainer's validation_step):
    1. main() -> parse_args() -> load CSV -> build paths
    2. main() -> spawn worker processes (predict_on_gpu)
    3. predict_on_gpu() -> init_predictor() -> load checkpoint & extract config (auto-detect model type)
    4. predict_on_gpu() -> create dataset (uses predictor.get_data_config() to retrieve configuration)
    5. predict_on_gpu() -> load_case() -> preprocess (resize x,y to patch_size) -> predict_single_npy() -> results_queue
    6. csv_saver_thread() -> periodically save results to CSV

Preprocessing:
    - If input image x,y dimensions != patch_size[1], patch_size[2], resize to match
    - Z dimension is kept unchanged
    - Target shape: (patch_size[0], patch_size[1], z_original)

Key Parameters (automatically retrieved from checkpoint):
    - patch_size: retrieved from plans
    - target_z_size: patch_size[0]
    - model_type: 'UVLM' or 'qwen3' (auto-detected)
    - strides: list of stride values for each stage

Usage:
    # Standard inference (auto-detect model type)
    python inference_reportgen.py \
        --csv-path /path/to/test.csv \
        --gpu-config "4:2,6:2"

    # Debug mode
    python inference_reportgen.py \
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

# Use universal predictor supporting both UVLM and Qwen3 models
from uvlm.inference.predict_reportgen import nnUNetPredictor
from uvlm.dataloading.dataset_csv_blosc2 import nnUNetDatasetCSVBlosc2
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg


# Default configuration
DEFAULTS = {
    "dataset_name": "Dataset140_CT-RATE",
    "trainer_name": "nnUNetTrainer_UVLM_only_report",
    "plans_name": "nnUNetResEncUNetL_cls_seg_pretrained_256_384_384_Plans",
    "configuration_name": "3d_fullres",
    "fold": 0,
    "checkpoint_name": "checkpoint_best.pth",
    "base_results_dir": "/path/to/results/nnUNet_results",
    "csv_path": "/path/to/data/validation_reports_processed.csv",
    "output_suffix": "inference_results_best_temperature_0.1",
    "gpu_config": "5:2,6:2",
    "debug_max_cases": None,
    "random_seed": 42,
    "num_repeats": 1,  # Number of seeds to test per case
    # LLM Generation Parameters
    "max_new_tokens": 512,  # Maximum number of tokens to generate
    "temperature": 0.1,  # Temperature parameter, lower values yield more deterministic outputs
    "top_p": 0.9,  # nucleus sampling
    "top_k": 50,  # top-k sampling
    # report_prompt: None = use model default (first prompt from report_prompts.txt)
    "report_prompt": None
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


def resize_image_xy_to_patch_size(input_img: np.ndarray, patch_size: tuple, verbose: bool = False) -> np.ndarray:
    """
    Resize input image x,y dimensions to match patch_size, keep z unchanged.

    Args:
        input_img: Input image with shape [C, D, H, W] or [B, C, D, H, W]
        patch_size: Target patch size (D, H, W) - only H and W are used for resizing
        verbose: Whether to print resize information

    Returns:
        Resized image with x,y matching patch_size[1], patch_size[2], z unchanged
    """
    is_5d = input_img.ndim == 5

    # Convert to 4D for processing if needed
    if is_5d:
        batch_size = input_img.shape[0]
        input_img_4d = input_img[0]  # Take first batch
    else:
        input_img_4d = input_img

    # Current shape: [C, D, H, W]
    current_shape = input_img_4d.shape[1:]  # (D, H, W)
    target_h, target_w = patch_size[1], patch_size[2]
    current_d, current_h, current_w = current_shape

    # Check if resize is needed
    if current_h == target_h and current_w == target_w:
        if verbose:
            print(f"  No resize needed: current shape {current_shape} already matches patch_size x,y")
        return input_img

    # Target shape: keep D unchanged, resize H and W
    target_shape = (current_d, target_h, target_w)

    if verbose:
        print(f"  Resizing from {current_shape} to {target_shape} (keeping z={current_d} unchanged)")

    # Resize using resample_data_or_seg
    # This function expects [C, D, H, W] and returns [C, D', H', W']
    resized_img_4d = resample_data_or_seg(
        input_img_4d,
        target_shape,
        is_seg=False,
        axis=None,
        order=3,
        do_separate_z=False
    )

    # Convert back to 5D if needed
    if is_5d:
        resized_img = resized_img_4d[np.newaxis, ...]
    else:
        resized_img = resized_img_4d

    if verbose:
        print(f"  Resized shape: {resized_img.shape}")

    return resized_img

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

    Automatically load all configuration parameters from checkpoint (identical to trainer):
    - patch_size: obtained from plans
    - target_z_size: patch_size[0]
    - strides: Stride values for each encoder stage
    - data_format: 'blosc2' (Blosc2 supported)

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

    # Get data configuration (obtained from predictor, identical to trainer)
    data_config = predictor.get_data_config()

    if is_first_gpu:
        print(f"Data configuration from checkpoint:")
        print(f"  patch_size: {data_config['patch_size']}")
        print(f"  target_z_size: {data_config['target_z_size']}")
        print(f"  strides: {data_config['strides']}")
        print(f"  data_format: {data_config['data_format']}")

    return predictor, data_config

def predict_on_gpu(gpu_id: int, files_queue: Queue, results_queue: Queue, args):
    """
    Worker process for GPU-based prediction.

    Data flow matches trainer's validation flow:
    1. init_predictor() -> load model and configuration parameters
    2. Select dataset class based on data_format (identical to trainer's get_tr_and_val_datasets)
    3. dataset.load_case() -> load and preprocess image
    4. predictor.predict_single_npy() -> generate report
    5. results_queue.put() -> send results to saver thread

    Dataset: Blosc2 format is supported (nnUNetDatasetCSVBlosc2)

    Args:
        gpu_id: GPU device ID
        files_queue: Queue containing case keys to process
        results_queue: Queue for sending results to saver thread
        args: Parsed command line arguments
    """
    # ==================== Initialize predictor ====================
    predictor, data_config = init_predictor(gpu_id, args)

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
            result_dict["generated_report"] = ""
            result_dict["success"] = False
            result_dict["skip_reason"] = "size_limit_exceeded"
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

        # ==================== Preprocess: Resize to patch_size (x, y), keep z unchanged ====================
        # Get patch_size from predictor's data_config
        patch_size = data_config['patch_size']  # (D, H, W)

        # Resize if needed (only x and y dimensions, keep z unchanged)
        input_img = resize_image_xy_to_patch_size(
            input_img,
            patch_size,
            verbose=(progress_bar and gpu_id == min(args.gpu_config.keys()))
        )

        if progress_bar and gpu_id == min(args.gpu_config.keys()):
            print(f"Case {case_key}: After preprocessing, shape = {input_img.shape}")

        # Test with multiple seeds
        all_results = []

        for seed_idx in range(args.num_repeats):
            current_seed = args.random_seed + seed_idx

            # Set seed for reproducibility
            set_seed(current_seed)

            # Generate report (identical to trainer's validation_step)
            if progress_bar and (seed_idx == 0 or args.num_repeats <= 3):  # Log first few seeds
                print(f"GPU {gpu_id}: Processing case {case_key} with seed {current_seed}")

            generated_reports = predictor.predict_single_npy(
                input_img,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                report_prompt=args.report_prompt
            )
            generated_report = generated_reports[0] if generated_reports else ""

            if progress_bar and (seed_idx == 0 or args.num_repeats <= 3):  # Log first few seeds
                report_length = len(generated_report) if generated_report else 0
                print(f"GPU {gpu_id}: Case {case_key} seed {current_seed} -> report length: {report_length}")

            # Store result
            result_dict = (
                args.csv_data[case_key].copy()
                if args.csv_data and case_key in args.csv_data
                else {"series_id": case_key}
            )
            result_dict["seed_idx"] = seed_idx
            result_dict["random_seed"] = current_seed
            result_dict["generated_report"] = generated_report
            result_dict["success"] = bool(generated_report)

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
        description="Multi-seed inference matching training validation (UVLM)"
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

    # LLM Generation Parameters
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"],
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"],
                        help="Temperature for LLM generation (lower = more deterministic)")
    parser.add_argument("--top-p", type=float, default=DEFAULTS["top_p"],
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"],
                        help="Top-k sampling parameter")
    parser.add_argument("--report-prompt", type=str, default=DEFAULTS["report_prompt"],
                        help="Custom report prompt (overrides model default)")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print("nnUNet Report Generation Inference - UVLM")
    print("Exact Training Validation Match")
    print(f"{'='*70}")

    # Set global random seed
    set_seed(args.random_seed)
    print(f"Base random seed: {args.random_seed}")

    # Log LLM generation parameters
    print(f"LLM Generation Parameters:")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    if args.report_prompt:
        print(f"  report_prompt: custom")
    else:
        print(f"  report_prompt: model default")

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
    print(f"NOTE: All parameters (patch_size, target_z_size, strides) are loaded from checkpoint")
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

        if 'generated_report' in df.columns:
            # Analyze by seed
            success_df = df[df['success'] == True] if 'success' in df.columns else df
            if len(success_df) > 0:
                avg_len = success_df['generated_report'].str.len().mean()
                min_len = success_df['generated_report'].str.len().min()
                max_len = success_df['generated_report'].str.len().max()
                print(f"\nReport Statistics:")
                print(f"  Length: avg={avg_len:.1f}, min={min_len}, max={max_len} chars")

                # Analyze consistency across seeds
                if 'seed_idx' in df.columns:
                    # Use actual series_id column name to count unique cases
                    id_col = args.series_id_column if args.series_id_column in df.columns else 'series_id'
                    num_unique_cases = df[id_col].nunique() if id_col in df.columns else len(case_keys)
                    print(f"  Unique cases: {num_unique_cases}")
                    print(f"  Seeds per case: {args.num_repeats}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
