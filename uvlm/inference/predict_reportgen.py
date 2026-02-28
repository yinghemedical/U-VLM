"""
Report Generation Predictor for nnUNet models.

General inference code supporting two report generation models:
1. UVLM lightweight LLM (UVLM)
2. Qwen3 (UVLM_Qwen3)

Automatically detects model type from checkpoint, no manual specification required.

Compatible with:
- nnUNetTrainer_UVLM
- nnUNetTrainer_UVLM_Qwen3

Usage:
    predictor = nnUNetPredictor(device=torch.device('cuda'), verbose=True)
    predictor.initialize_from_trained_model_folder(model_dir)
    reports = predictor.predict_single_npy(input_image)
"""

import os
import torch
import numpy as np
from typing import Optional, List
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.helpers import empty_cache


class nnUNetPredictor:
    """
    General report generation predictor supporting UVLM lightweight LLM and Qwen3 models.

    Automatically detects model type, no manual specification required.

    Data Flow:
        1. initialize_from_trained_model_folder() -> Loads checkpoint and configuration, detects model type
        2. build_network_architecture() -> Builds network (consistent with trainer)
        3. predict_single_npy() -> Accepts preprocessed numpy array -> Generates report

    Supported Models:
        - UVLM: UVLM (lightweight LLM)
        - qwen3: UVLM_Qwen3 (Qwen3-4B + LoRA)

    Usage:
        predictor = nnUNetPredictor(device=torch.device('cuda'), verbose=True)
        predictor.initialize_from_trained_model_folder(model_dir)
        reports = predictor.predict_single_npy(input_image)
    """

    # Mapping of supported model types
    MODEL_TYPE_MAP = {
        'nnUNetTrainer_UVLM': 'UVLM',
        'nnUNetTrainer_UVLM_Qwen3': 'qwen3',
    }

    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False):
        """
        Initialize predictor.

        Args:
            device: Device for inference
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.device = device

        self.network = None
        self.trainer_name = None
        self.model_type = None  # 'UVLM' or 'qwen3'
        self.mode = None  # 'only_cls', 'only_report', 'both'

        # Configuration parameters (loaded from checkpoint)
        self.patch_size = None
        self.target_z_size = None
        self.strides = None
        self.data_format = None

        # Generation parameters
        self.report_prompt = None
        self.report_max_new_tokens = None
        self.generation_temperature = None
        self.generation_top_p = None
        self.generation_top_k = None

        # Enable CUDA optimizations
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def _detect_model_type(self, trainer_name: str) -> str:
        """
        Detect model type from trainer_name.

        Args:
            trainer_name: Trainer class name

        Returns:
            Model type: 'UVLM' or 'qwen3'
        """
        if trainer_name in self.MODEL_TYPE_MAP:
            return self.MODEL_TYPE_MAP[trainer_name]

        # Infer based on name
        trainer_lower = trainer_name.lower()
        if 'qwen3' in trainer_lower or 'qwen' in trainer_lower:
            return 'qwen3'
        return 'UVLM'

    def initialize_from_trained_model_folder(self,
                                             model_training_output_dir: str,
                                             use_folds: Optional[List[int]] = None,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        Initialize predictor from trained model folder.

        Automatically detects model type (UVLM or qwen3) and loads corresponding network architecture.

        Args:
            model_training_output_dir: Training output directory
            use_folds: List of folds to use (default auto-detected)
            checkpoint_name: Checkpoint filename
        """
        if use_folds is None:
            use_folds = self._auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        fold = use_folds[0]

        # ==================== Load checkpoint ====================
        fold_dir = join(model_training_output_dir, f'fold_{fold}')
        checkpoint_path = join(fold_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

        trainer_name = checkpoint['trainer_name']
        self.trainer_name = trainer_name

        # ==================== Detect model type ====================
        self.model_type = self._detect_model_type(trainer_name)
        if self.verbose:
            print(f"Detected model type: {self.model_type}")
            print(f"Trainer: {trainer_name}")

        # ==================== Load plans and dataset configuration ====================
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)
        configuration_name = checkpoint['init_args']['configuration']
        configuration_manager = plans_manager.get_configuration(configuration_name)

        # ==================== Get patch_size from plans ====================
        self.patch_size = configuration_manager.patch_size
        self.target_z_size = self.patch_size[0]

        # ==================== Get other configurations from checkpoint ====================
        network_arch_init_kwargs = checkpoint['init_args']['network_arch_init_kwargs']
        self.strides = network_arch_init_kwargs.get('strides')
        self.data_format = network_arch_init_kwargs.get('data_format', 'blosc2')
        self.mode = network_arch_init_kwargs.get('mode', 'both')

        if self.verbose:
            print(f"Loaded configuration from checkpoint:")
            print(f"  mode: {self.mode}")
            print(f"  patch_size: {self.patch_size}")
            print(f"  target_z_size: {self.target_z_size}")
            print(f"  strides: {self.strides}")
            print(f"  data_format: {self.data_format}")

        # ==================== Build network ====================
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(os.path.dirname(__file__), '..', 'training', 'nnUNetTrainer'),
            trainer_name, 'nnunetv2.training.nnUNetTrainer'
        )

        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name}')

        # Create temporary trainer instance to correctly initialize network parameters
        temp_trainer = trainer_class(plans, configuration_name, fold, dataset_json,
                                     unpack_dataset=False, device=self.device)

        # Build network using correctly initialized configuration parameters
        network = trainer_class.build_network_architecture(
            temp_trainer.configuration_manager.network_arch_class_name,
            temp_trainer.configuration_manager.network_arch_init_kwargs,
            temp_trainer.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            1,  # dummy for segmentation heads
            enable_deep_supervision=False
        )

        # ==================== Load weights ====================
        checkpoint_weights = checkpoint['network_weights']
        model_state_dict = network.state_dict()
        # Filter mismatched parameters
        filtered_weights = {}
        skipped_keys = []
        for key, value in checkpoint_weights.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_weights[key] = value
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(key)

        if skipped_keys and self.verbose:
            print(f"Skipped {len(skipped_keys)} keys from checkpoint:")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")

        network.load_state_dict(filtered_weights, strict=True)
        network.to(self.device)
        network.eval()

        self.network = network

        # ==================== Extract generation parameters ====================
        self.report_max_new_tokens = network_arch_init_kwargs.get('report_max_new_tokens', 512)
        self.generation_temperature = network_arch_init_kwargs.get('generation_temperature', 0.7)
        self.generation_top_p = network_arch_init_kwargs.get('generation_top_p', 0.9)
        self.generation_top_k = network_arch_init_kwargs.get('generation_top_k', 50)

        # ==================== Load report_prompt ====================
        if self.mode in ['only_report', 'both']:
            self.report_prompt = getattr(temp_trainer, 'report_prompt', None)
            if self.report_prompt is None:
                self.report_prompt = "<|im_start|>system\nYou are a radiologist, write standardized reports based only on images.\n<|im_start|>user\nPlease write the report:<|im_end|>\n<|im_start|>assistant\n"

        if self.verbose:
            print(f"Network loaded successfully")
            print(f"  Model type: {self.model_type}")
            print(f"  Generation params: max_new_tokens={self.report_max_new_tokens}, "
                  f"temp={self.generation_temperature}, top_p={self.generation_top_p}, top_k={self.generation_top_k}")
            if self.mode in ['only_report', 'both'] and self.report_prompt:
                print(f"  Report prompt: {self.report_prompt[:50]}...")

    def manual_initialization(self, network: torch.nn.Module, plans_manager: PlansManager,
                              configuration_manager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[tuple] = None,
                              report_prompt: Optional[str] = None,
                              mode: Optional[str] = None):
        """
        Manual initialization for trainer integration.
        """
        self.network = network
        self.trainer_name = trainer_name
        self.model_type = self._detect_model_type(trainer_name)

        # Get parameters from configuration_manager
        self.patch_size = configuration_manager.patch_size
        self.target_z_size = self.patch_size[0]

        network_arch_init_kwargs = configuration_manager.network_arch_init_kwargs
        self.report_max_new_tokens = network_arch_init_kwargs.get('report_max_new_tokens', 512)
        self.generation_temperature = network_arch_init_kwargs.get('generation_temperature', 0.7)
        self.generation_top_p = network_arch_init_kwargs.get('generation_top_p', 0.9)
        self.generation_top_k = network_arch_init_kwargs.get('generation_top_k', 50)
        self.strides = network_arch_init_kwargs.get('strides')
        self.data_format = network_arch_init_kwargs.get('data_format', 'blosc2')

        if mode is not None:
            self.mode = mode
        else:
            self.mode = network_arch_init_kwargs.get('mode', 'both')

        if report_prompt is not None:
            self.report_prompt = report_prompt
        else:
            self.report_prompt = "<|im_start|>system\nYou are a radiologist, write standardized reports based only on images.\n<|im_start|>user\nPlease write the report:<|im_end|>\n<|im_start|>assistant\n"

        if self.verbose:
            print(f"Manual initialization completed for: {trainer_name}")
            print(f"  Model type: {self.model_type}")

    def get_data_config(self) -> dict:
        """
        Get data configuration parameters for dataset creation.

        Returns:
            dict: Configuration dictionary containing strides, target_z_size, data_format, patch_size, model_type
        """
        return {
            'strides': self.strides,
            'target_z_size': self.target_z_size,
            'data_format': self.data_format,
            'patch_size': self.patch_size,
            'model_type': self.model_type,
        }

    @staticmethod
    def _auto_detect_available_folds(model_training_output_dir: str, checkpoint_name: str) -> List[int]:
        """Auto-detect available folds."""
        import glob
        fold_dirs = glob.glob(join(model_training_output_dir, 'fold_*'))
        fold_dirs = [d for d in fold_dirs if isfile(join(d, checkpoint_name))]
        available_folds = []
        for fold_dir in fold_dirs:
            fold_name = os.path.basename(fold_dir)
            if fold_name.startswith('fold_'):
                fold_idx = int(fold_name.split('_')[1])
                available_folds.append(fold_idx)
        if not available_folds:
            raise ValueError(f"No folds with {checkpoint_name} found in {model_training_output_dir}")
        return sorted(available_folds)

    def predict_single_npy(self, input_image: np.ndarray, max_new_tokens: Optional[int] = None,
                           temperature: Optional[float] = None, top_p: Optional[float] = None,
                           top_k: Optional[int] = None, report_prompt: Optional[str] = None,
                           skip_special_tokens: bool = True) -> List[str]:
        """
        Generate report for single preprocessed image.

        Supports both UVLM and Qwen3 models with identical interfaces.

        Args:
            input_image: Preprocessed image [C, D, H, W] or [B, C, D, H, W]
            max_new_tokens: Override maximum generated tokens
            temperature: Override temperature parameter
            top_p: Override top-p (nucleus) sampling parameter
            top_k: Override top-k sampling parameter
            report_prompt: Override report prompt
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List[str]: List of generated reports
        """
        # Ensure batch dimension
        if input_image.ndim == 4:
            input_image = input_image[np.newaxis, ...]

        if input_image.ndim != 5:
            raise ValueError(f"Expected 5D input [B,C,D,H,W], got {input_image.ndim}D")

        # Convert to tensor
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32)

        data = torch.from_numpy(input_image).to(self.device)

        # Data type conversion (Qwen3 uses bfloat16)
        with torch.no_grad():
            model_dtype = next(self.network.parameters()).dtype
            if model_dtype != torch.float32:
                data = data.to(model_dtype)

        if self.verbose:
            print(f"Input shape: {data.shape}, device: {data.device}, dtype: {data.dtype}")

        # Determine generation parameters
        use_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.report_max_new_tokens
        use_prompt = report_prompt if report_prompt is not None else self.report_prompt
        use_temperature = temperature if temperature is not None else self.generation_temperature
        use_top_p = top_p if top_p is not None else self.generation_top_p
        use_top_k = top_k if top_k is not None else self.generation_top_k

        if self.verbose:
            print(f"Generation params: max_tokens={use_max_new_tokens}, temperature={use_temperature}, "
                  f"top_p={use_top_p}, top_k={use_top_k}")

        # Generate report (UVLM and Qwen3 have identical interfaces)
        with torch.no_grad():
            network_output = self.network(
                data,
                generate_report=True,
                report_prompt=use_prompt,
                max_new_tokens=use_max_new_tokens,
                temperature=use_temperature,
                top_p=use_top_p,
                top_k=use_top_k,
                skip_special_tokens=skip_special_tokens
            )

        # Extract report
        if isinstance(network_output, tuple) and len(network_output) >= 1:
            report_output = network_output[0]
        else:
            report_output = network_output

        # Convert to string list
        if report_output is None:
            generated_reports = []
        elif isinstance(report_output, list):
            generated_reports = [str(r) for r in report_output]
        elif isinstance(report_output, str):
            generated_reports = [report_output]
        else:
            generated_reports = [str(report_output)]

        if self.verbose:
            print(f"Generated {len(generated_reports)} reports")
            if generated_reports:
                print(f"First report preview: {generated_reports[0][:100]}...")

        empty_cache(self.device)
        return generated_reports


# Backwards compatibility aliases
nnUNetPredictorWithReportGen = nnUNetPredictor
nnUNetReportPredictor = nnUNetPredictor
