"""
Only Classification Predictor for nnUNet models.

Fully compatible with nnUNetTrainer_UVLM's only_cls mode inference code.

Compatible with:
- nnUNetTrainer_UVLM_only_cls

Data Flow (identical to trainer's validation_step):
    1. initialize_from_trained_model_folder() -> load checkpoint and configuration
    2. build_network_architecture() -> build network (identical to trainer)
    3. predict_single_npy() -> receive preprocessed numpy array -> return classification probabilities

Key Parameters (from checkpoint):
    - patch_size: obtained from plans, used to determine target_z_size
    - target_z_size: patch_size[0]
    - cls_columns: list of classification column names (obtained from plan)
    - cls_head_num_classes_list: list of classification head class counts

Usage:
    predictor = nnUNetPredictor(device=torch.device('cuda'), verbose=True)
    predictor.initialize_from_trained_model_folder(model_dir)
    cls_probs = predictor.predict_single_npy(input_image)
"""

import os
import torch
import numpy as np
from typing import Optional, List, Tuple
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.helpers import empty_cache


class nnUNetPredictor:
    """
    Only classification predictor for nnUNet models.

    Fully compatible with nnUNetTrainer_UVLM's only_cls mode.

    Data Flow (identical to trainer's validation_step):
        1. initialize_from_trained_model_folder() -> load checkpoint and configuration
        2. build_network_architecture() -> build network (identical to trainer)
        3. predict_single_npy() -> receive preprocessed numpy array -> return classification probabilities

    Key Parameters (from checkpoint/plans):
        - patch_size: obtained from plans, used to determine target_z_size
        - target_z_size: patch_size[0]
        - cls_columns: list of classification column names (obtained from plan)
        - cls_head_num_classes_list: list of classification head class counts

    Usage:
        predictor = nnUNetPredictor(device=torch.device('cuda'), verbose=True)
        predictor.initialize_from_trained_model_folder(model_dir)
        cls_probs = predictor.predict_single_npy(input_image)
    """

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
        self.mode = 'only_cls'  # Fixed to only_cls mode

        # Configuration parameters (loaded from checkpoint)
        self.patch_size = None
        self.target_z_size = None
        self.strides = None
        self.data_format = None

        # Classification-related parameters (loaded from plan/checkpoint)
        self.cls_columns = None  # List of classification column names
        self.cls_head_num_classes_list = None  # List of classification head class counts
        self.num_cls_task = None  # Number of classification tasks

        # Enable CUDA optimizations
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def initialize_from_trained_model_folder(self,
                                             model_training_output_dir: str,
                                             use_folds: Optional[List[int]] = None,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        Initialize predictor from trained model folder.

        Consistent with trainer's initialization process:
        1. Load checkpoint to obtain trainer_name and network parameters
        2. Load plans.json and dataset.json
        3. Obtain patch_size from plans, calculate target_z_size
        4. Obtain cls_columns from network_arch_init_kwargs, dynamically compute cls_head_num_classes_list
        5. Use trainer_class.build_network_architecture() to build network
        6. Load network weights

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

        # ==================== Load plans and dataset configuration ====================
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)
        configuration_name = checkpoint['init_args']['configuration']
        configuration_manager = plans_manager.get_configuration(configuration_name)

        # ==================== Obtain patch_size from plans (identical to trainer) ====================
        # trainer: patch_size = self.configuration_manager.patch_size
        self.patch_size = configuration_manager.patch_size
        self.target_z_size = self.patch_size[0]  # z-axis target size obtained from patch_size

        # ==================== Obtain configuration from network_arch_init_kwargs ====================
        network_arch_init_kwargs = configuration_manager.network_arch_init_kwargs
        self.strides = network_arch_init_kwargs.get('strides')
        self.data_format = network_arch_init_kwargs.get('data_format', 'blosc2')

        # Obtain mode parameter to ensure network architecture matches training
        self.mode = network_arch_init_kwargs.get('mode', 'only_cls')

        # ==================== Obtain classification-related configuration from network_arch_init_kwargs (identical to trainer) ====================
        # trainer: self.cls_columns = self.configuration_manager.network_arch_init_kwargs.get("cls_columns", [])
        self.cls_columns = network_arch_init_kwargs.get('cls_columns', [])

        # Logic in trainer:
        # default_num_classes = len(self.cls_columns)
        # default_cls_head_num_classes_list = [default_num_classes]
        # self.cls_head_num_classes_list = self.configuration_manager.network_arch_init_kwargs.get(
        #     "cls_head_num_classes_list", default_cls_head_num_classes_list
        # )
        default_num_classes = len(self.cls_columns)
        default_cls_head_num_classes_list = [default_num_classes]
        self.cls_head_num_classes_list = network_arch_init_kwargs.get(
            'cls_head_num_classes_list', default_cls_head_num_classes_list
        )

        # trainer: default_pos_weights_list = [[1.0] * default_num_classes]
        # self.pos_weights_list = self.configuration_manager.network_arch_init_kwargs.get(
        #     'pos_weights_list', default_pos_weights_list
        # )
        # self.num_cls_task = len(self.pos_weights_list)
        default_pos_weights_list = [[1.0] * default_num_classes]
        pos_weights_list = network_arch_init_kwargs.get('pos_weights_list', default_pos_weights_list)
        self.num_cls_task = len(pos_weights_list)

        if self.verbose:
            print(f"Loaded configuration from checkpoint:")
            print(f"  mode: {self.mode}")
            print(f"  patch_size: {self.patch_size}")
            print(f"  target_z_size: {self.target_z_size}")
            print(f"  strides: {self.strides}")
            print(f"  data_format: {self.data_format}")
            print(f"  cls_columns: {self.cls_columns} ({len(self.cls_columns)} classes)")
            print(f"  cls_head_num_classes_list: {self.cls_head_num_classes_list}")
            print(f"  num_cls_task: {self.num_cls_task}")

        # ==================== Build network (identical to trainer) ====================
        # Use trainer_class.build_network_architecture to ensure network structure is identical
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(os.path.dirname(__file__), '..', 'training', 'nnUNetTrainer'),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')

        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name}')

        # Create temporary trainer instance to correctly initialize network parameters
        # Ensure all dynamic parameters (patch_kernel_sizes, etc.) are correctly set
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

        # ==================== Load weights and move to device ====================
        # Filter unnecessary parameters based on mode
        checkpoint_weights = checkpoint['network_weights']
        model_state_dict = network.state_dict()

        # Identify parameters to load (only load parameters present in current model)
        filtered_weights = {}
        skipped_keys = []
        for key, value in checkpoint_weights.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_weights[key] = value
                else:
                    skipped_keys.append(f"{key} (shape mismatch: checkpoint {value.shape} vs model {model_state_dict[key].shape})")
            else:
                skipped_keys.append(key)

        if skipped_keys and self.verbose:
            print(f"Skipped {len(skipped_keys)} keys from checkpoint (not in current mode '{self.mode}'):")
            for key in skipped_keys[:10]:  # Print only first 10
                print(f"  - {key}")
            if len(skipped_keys) > 10:
                print(f"  ... and {len(skipped_keys) - 10} more")

        # Load filtered weights (strict=False allows partial loading)
        network.load_state_dict(filtered_weights, strict=False)
        network.to(self.device)
        network.eval()

        # ==================== Save state ====================
        self.network = network
        self.trainer_name = trainer_name

        if self.verbose:
            print(f"Loaded trainer: {trainer_name}")
            print(f"Network moved to device: {self.device}")

    def manual_initialization(self, network: torch.nn.Module, plans_manager: PlansManager,
                              configuration_manager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[tuple] = None,
                              mode: Optional[str] = None):
        """
        Manual initialization for trainer integration.

        Args:
            mode: Training mode ('only_cls'), should be obtained from trainer.mode
        """
        self.network = network
        self.trainer_name = trainer_name

        # Get patch_size from configuration_manager (consistent with trainer)
        self.patch_size = configuration_manager.patch_size
        self.target_z_size = self.patch_size[0]

        # Set parameters (same as trainer)
        network_arch_init_kwargs = configuration_manager.network_arch_init_kwargs
        self.strides = network_arch_init_kwargs.get('strides')
        self.data_format = network_arch_init_kwargs.get('data_format', 'blosc2')

        # mode: Obtain from parameter or configuration
        if mode is not None:
            self.mode = mode
        else:
            self.mode = network_arch_init_kwargs.get('mode', 'only_cls')

        # Classification-related parameters (consistent with trainer)
        self.cls_columns = network_arch_init_kwargs.get('cls_columns', [])

        # Dynamically compute cls_head_num_classes_list (consistent with trainer)
        default_num_classes = len(self.cls_columns)
        default_cls_head_num_classes_list = [default_num_classes]
        self.cls_head_num_classes_list = network_arch_init_kwargs.get(
            'cls_head_num_classes_list', default_cls_head_num_classes_list
        )

        # Compute num_cls_task (consistent with trainer)
        default_pos_weights_list = [[1.0] * default_num_classes]
        pos_weights_list = network_arch_init_kwargs.get('pos_weights_list', default_pos_weights_list)
        self.num_cls_task = len(pos_weights_list)

        if self.verbose:
            print(f"Manual initialization completed for: {trainer_name}")
            print(f"  mode: {self.mode}")
            print(f"  patch_size: {self.patch_size}")
            print(f"  target_z_size: {self.target_z_size}")
            print(f"  cls_columns: {self.cls_columns} ({len(self.cls_columns)} classes)")
            print(f"  cls_head_num_classes_list: {self.cls_head_num_classes_list}")
            print(f"  num_cls_task: {self.num_cls_task}")

    def get_data_config(self) -> dict:
        """
        Get data configuration parameters for dataset creation.

        Returns:
            dict: Configuration dictionary containing strides, target_z_size, data_format, patch_size, cls_columns
        """
        return {
            'strides': self.strides,
            'target_z_size': self.target_z_size,
            'data_format': self.data_format,
            'patch_size': self.patch_size,
            'cls_columns': self.cls_columns,
            'cls_head_num_classes_list': self.cls_head_num_classes_list,
            'num_cls_task': self.num_cls_task
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

    def predict_single_npy(self, input_image: np.ndarray) -> Tuple[List[np.ndarray], dict]:
        """
        Predict classification probabilities for single preprocessed image.

        Consistent with classification inference flow in trainer's validation_step:
        1. Ensure input is 5D tensor [B, C, D, H, W]
        2. Convert to torch tensor and move to device
        3. Perform forward propagation using network(data)
        4. Apply sigmoid to classification logits to obtain probabilities
        5. Return classification probability list and probability dictionary

        Input format (consistent with trainer's data loading):
            - 4D: [C, D, H, W] -> automatically add batch dimension
            - 5D: [B, C, D, H, W] -> use directly

        Args:
            input_image: Preprocessed image [C, D, H, W] or [B, C, D, H, W]

        Returns:
            Tuple[List[np.ndarray], dict]:
                - cls_probs_list: Classification probability list, each element corresponds to a classification task
                - cls_probs_dict: Classification probability dictionary, key is column name from cls_columns, value is probability
        """
        # ==================== Ensure batch dimension ====================
        if input_image.ndim == 4:
            input_image = input_image[np.newaxis, ...]

        if input_image.ndim != 5:
            raise ValueError(f"Expected 5D input [B,C,D,H,W], got {input_image.ndim}D")

        # ==================== Convert to tensor and move to device ====================
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32)

        data = torch.from_numpy(input_image).to(self.device)

        # ==================== Data type conversion (consistent with trainer's validation_step) ====================
        with torch.no_grad():
            model_dtype = next(self.network.parameters()).dtype
            if model_dtype != torch.float32:
                data = data.to(model_dtype)

        if self.verbose:
            print(f"Input shape: {data.shape}, device: {data.device}")

        # ==================== Forward propagation (completely consistent with trainer's validation_step) ====================
        # trainer validation_step:
        #   with torch.no_grad():
        #       output = self.network(data)
        #       # only_cls mode: output = (None, cls_pred_list)
        #       report_output, cls_pred_list = output
        with torch.no_grad():
            network_output = self.network(data, generate_report=False)

        # ==================== Extract classification results (consistent with trainer) ====================
        # only_cls mode: network_output = (None, cls_pred_list)
        if isinstance(network_output, tuple) and len(network_output) >= 2:
            _, cls_pred_list = network_output[0], network_output[1]
        else:
            raise ValueError(f"Unexpected network output format: {type(network_output)}")
        # ==================== Compute classification probabilities (consistent with trainer) ====================
        # trainer: cls_probs = torch.sigmoid(cls_pred_logits)
        cls_probs_list = []
        for t_i in range(self.num_cls_task):
            cls_pred_logits = cls_pred_list[t_i]  # [B, num_classes]
            cls_probs = torch.sigmoid(cls_pred_logits)  # [B, num_classes]
            cls_probs_list.append(cls_probs[0].cpu().numpy())  # Take first batch, convert to numpy

        # ==================== Build probability dictionary ====================
        cls_probs_dict = {}
        col_idx = 0
        for t_i in range(self.num_cls_task):
            num_classes = self.cls_head_num_classes_list[t_i]
            for c_i in range(num_classes):
                if col_idx < len(self.cls_columns):
                    col_name = self.cls_columns[col_idx]
                    cls_probs_dict[col_name] = float(cls_probs_list[t_i][c_i])
                col_idx += 1

        if self.verbose:
            print(f"Classification results:")
            for col_name, prob in cls_probs_dict.items():
                print(f"  {col_name}: {prob:.4f}")

        empty_cache(self.device)
        return cls_probs_list, cls_probs_dict


# Backwards compatibility
nnUNetPredictorOnlyCls = nnUNetPredictor
