import os
import pydoc
import inspect
import ast
import json
import glob
import torch
import numpy as np
import pandas as pd
import random
import re
import SimpleITK as sitk
from time import time, sleep
from typing import Union, Tuple, List, Optional
from torch import autocast, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from torch import distributed as dist
from multiprocessing import Pool
from tqdm import tqdm
from nnunetv2.configuration import ANISO_THRESHOLD
# Internal modules of the UVLM project
from uvlm.networks.uvlm import UVLM
from uvlm.dataloading.class_balancer import balance_csv_files
from uvlm.dataloading.data_loader_cls_reportgen import nnUNetDataLoader3DWithGlobalClsReportgen as nnUNetDataLoader3DWithGlobalClsReportgenCSV
from uvlm.dataloading.dataset_csv_blosc2 import nnUNetDatasetCSVBlosc2
from uvlm.dataloading.data_shape_preloader import preprocess_csv_with_shapes

# nnU-Net base modules
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.logging.nnUNet_logger import nnUNetLogger
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels, LabelManager
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p, isfile
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor, RemoveLabelTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from uvlm.training.lr_scheduler.multigroup_polylr import MultiGroupPolyLRScheduler
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter


class nnUNetTrainer_UVLM(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.debug_save_input = False

        # Disable dataset unpacking since we read directly from Blosc2 files
        self.unpack_dataset = False

        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)

        # ==================== Mode Configuration ====================
        # Read mode from plan's network_arch_init_kwargs, default is 'both'
        mode = self.configuration_manager.network_arch_init_kwargs.get('mode', 'both')
        if mode not in ['only_cls', 'only_report', 'both']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: 'only_cls', 'only_report', 'both'")
        self.mode = mode
        self.print_to_log_file(f"Training mode: {self.mode}")

        # ==================== Optimizer Configuration ====================
        # Supported optimizers: 'AdamW' or 'SGD'
        self.optimizer_type = self.configuration_manager.network_arch_init_kwargs.get('optimizer_type', 'AdamW')

        if self.optimizer_type not in ['AdamW', 'SGD']:
            raise ValueError(f"Invalid optimizer_type: {self.optimizer_type}. Must be 'AdamW' or 'SGD'")

        # Set default hyperparameters based on optimizer type
        if self.optimizer_type == 'SGD':
            default_lr = 1e-3
            default_weight_decay = 3e-5
            default_momentum = 0.9
            default_nesterov = True
            default_betas = (0.9, 0.999)  # not used for SGD
        else:  # AdamW
            default_lr = 2e-5
            default_weight_decay = 0.01
            default_momentum = 0.9  # not used for AdamW
            default_nesterov = True  # not used for AdamW
            default_betas = (0.9, 0.999)

        # Read hyperparameters from plan, use defaults if not present
        self.initial_lr = self.configuration_manager.network_arch_init_kwargs.get('initial_lr', default_lr)
        self.weight_decay = self.configuration_manager.network_arch_init_kwargs.get('weight_decay', default_weight_decay)
        self.momentum = self.configuration_manager.network_arch_init_kwargs.get('momentum', default_momentum)
        self.nesterov = self.configuration_manager.network_arch_init_kwargs.get('nesterov', default_nesterov)
        # betas for AdamW: (beta1, beta2)
        betas_from_plan = self.configuration_manager.network_arch_init_kwargs.get('betas', None)
        if betas_from_plan is not None:
            self.betas = tuple(betas_from_plan)
        else:
            self.betas = default_betas

        if self.optimizer_type == 'SGD':
            self.print_to_log_file(f"Optimizer: SGD (lr={self.initial_lr}, weight_decay={self.weight_decay}, momentum={self.momentum}, nesterov={self.nesterov})")
        else:
            self.print_to_log_file(f"Optimizer: AdamW (lr={self.initial_lr}, weight_decay={self.weight_decay}, betas={self.betas})")

        # ==================== CSV Dataset Configuration ====================
        # All parameters are passed from plan's network_arch_init_kwargs, no hardcoded defaults
        # Required parameters: csv_paths, series_id_column, case_id_column, cls_columns
        # Optional parameter: report_column (for report generation)

        # CSV file paths (required)
        self.csv_paths = self.configuration_manager.network_arch_init_kwargs.get('csv_paths', None)
        if self.csv_paths is None:
            raise ValueError(
                "csv_paths must be configured in plan's network_arch_init_kwargs. "
                "Example: 'csv_paths': ['/path/to/train.csv']"
            )
        if isinstance(self.csv_paths, str):
            self.csv_paths = [self.csv_paths]
        self.print_to_log_file(f"Using CSV datasets: {self.csv_paths}")

        # Series-level identifier column name (required, used for data loading and label matching)
        self.series_id_column = self.configuration_manager.network_arch_init_kwargs.get('series_id_column', None)
        if self.series_id_column is None:
            raise ValueError(
                "series_id_column must be configured in plan's network_arch_init_kwargs. "
                "Example: 'series_id_column': 'series_id'"
            )
        self.print_to_log_file(f"Using series_id column: {self.series_id_column}")

        # Case-level identifier column name (required, used for train/val split to ensure same patient doesn't appear in both train and val)
        self.case_id_column = self.configuration_manager.network_arch_init_kwargs.get('case_id_column', None)
        if self.case_id_column is None:
            raise ValueError(
                "case_id_column must be configured in plan's network_arch_init_kwargs. "
                "Example: 'case_id_column': 'case_id'"
            )
        self.print_to_log_file(f"Using case_id column: {self.case_id_column}")

        # Classification column name list (required, for classification tasks and data balancing)
        self.cls_columns = self.configuration_manager.network_arch_init_kwargs.get('cls_columns', None)
        if self.cls_columns is None:
            raise ValueError(
                "cls_columns must be configured in plan's network_arch_init_kwargs. "
                "Example: 'cls_columns': ['cyst', 'edema', 'brain_hemorrhage', ...]"
            )
        if isinstance(self.cls_columns, str):
            self.cls_columns = [self.cls_columns]
        self.print_to_log_file(f"Using cls_columns: {self.cls_columns} ({len(self.cls_columns)} classes)")

        # Report text column name (optional, for report generation)
        self.report_column = self.configuration_manager.network_arch_init_kwargs.get('report_column', 'report')
        self.print_to_log_file(f"Using report column: {self.report_column}")

        # ==================== Classification Task Configuration ====================
        # Default: one classification task, multi-label classification with number of classes = len(cls_columns)
        # Can configure more classification tasks via plan settings cls_head_num_classes_list and pos_weights_list
        self.num_cls_task = 0
        self.checkpoint_t_index = 0

        if self.mode in ['only_cls', 'both']:
            # Default: single multi-label classification task, number of classes = len(cls_columns)
            default_num_classes = len(self.cls_columns)
            default_cls_head_num_classes_list = [default_num_classes]
            default_pos_weights_list = [[1.0] * default_num_classes]
            default_cls_drop_out_list = [0.0]
            default_cls_query_num_list = [default_num_classes]

            self.cls_head_num_classes_list = self.configuration_manager.network_arch_init_kwargs.get(
                "cls_head_num_classes_list", default_cls_head_num_classes_list
            )
            self.pos_weights_list = self.configuration_manager.network_arch_init_kwargs.get(
                'pos_weights_list', default_pos_weights_list
            )
            self.cls_drop_out_list = self.configuration_manager.network_arch_init_kwargs.get(
                'cls_drop_out_list', default_cls_drop_out_list
            )
            self.cls_query_num_list = self.configuration_manager.network_arch_init_kwargs.get(
                'cls_query_num_list', default_cls_query_num_list
            )
            self.num_cls_task = len(self.pos_weights_list)

            # Write classification head configuration back to network_arch_init_kwargs to ensure network architecture initializes correctly
            self.configuration_manager.network_arch_init_kwargs['cls_head_num_classes_list'] = self.cls_head_num_classes_list
            self.configuration_manager.network_arch_init_kwargs['cls_drop_out_list'] = self.cls_drop_out_list
            self.configuration_manager.network_arch_init_kwargs['cls_query_num_list'] = self.cls_query_num_list

            self.print_to_log_file(f"Classification tasks: {self.num_cls_task}")
            self.print_to_log_file(f"  cls_head_num_classes_list: {self.cls_head_num_classes_list}")
            self.print_to_log_file(f"  cls_drop_out_list: {self.cls_drop_out_list}")
            self.print_to_log_file(f"  cls_query_num_list: {self.cls_query_num_list}")
            self.print_to_log_file(f"  pos_weights_list: {self.pos_weights_list}")

        # ==================== Unified Image Size Configuration ====================
        # Get size configuration from patch_size, ensure consistency
        patch_size = self.configuration_manager.patch_size
        self.target_z_size = patch_size[0]  # Target z-axis size from patch_size
        self.patch_size = tuple(patch_size)  # Save patch_size for shape validation
        self.print_to_log_file(f"Using patch_size: {patch_size}, target_z_size: {self.target_z_size}")

        # Whether to force regenerate shape information (if enabled, will recalculate all case shapes on every run)
        self.force_regenerate_shapes = self.configuration_manager.network_arch_init_kwargs.get('force_regenerate_shapes', False)
        if self.force_regenerate_shapes:
            self.print_to_log_file("Force regenerate shapes: ENABLED - will recalculate all shape information")
        else:
            self.print_to_log_file("Force regenerate shapes: DISABLED - will use cached shape information if available")

        # ==================== Data Balancing Configuration ====================
        # Use columns specified by cls_columns for balancing
        self.enable_balancing = self.configuration_manager.network_arch_init_kwargs.get('enable_balancing', True)
        self.print_to_log_file(f"Data balancing: enable_balancing = {self.enable_balancing}")

        # Target sample count per class (default: 20000)
        self.balancing_target_samples_per_class = self.configuration_manager.network_arch_init_kwargs.get(
            'balancing_target_samples_per_class', 20000
        )
        # Maximum repeat multiplier, insufficient classes can be repeated up to N times their original count (default: 5)
        self.balancing_max_repeat_multiplier = self.configuration_manager.network_arch_init_kwargs.get(
            'balancing_max_repeat_multiplier', 5
        )
        self.balancing_seed = self.configuration_manager.network_arch_init_kwargs.get('balancing_seed', 42)
        # Negative sample ratio (all classification columns are 0) as proportion of total dataset (default: 0.2, i.e., 20%)
        self.balancing_negative_ratio = self.configuration_manager.network_arch_init_kwargs.get(
            'balancing_negative_ratio', 0.2
        )
        # Balanced statistics output directory (if None, use self.output_folder/balanced/)
        self.balancing_stats_output_dir = self.configuration_manager.network_arch_init_kwargs.get(
            'balancing_stats_output_dir', None
        )
        # Balanced dataset save path (if None, save to statistics output directory)
        self.balancing_output_path = self.configuration_manager.network_arch_init_kwargs.get(
            'balancing_output_path', None
        )

        if self.enable_balancing:
            self.print_to_log_file(f"Data balancing config: target={self.balancing_target_samples_per_class} per class, "
                                   f"max_repeat={self.balancing_max_repeat_multiplier}x, seed={self.balancing_seed}, "
                                   f"negative_ratio={self.balancing_negative_ratio}")
        else:
            self.print_to_log_file("Data balancing disabled")

        # ==================== Data Augmentation Configuration ====================
        # Whether to enable nnUNet-style data augmentation (including spatial transforms, mirroring, etc.)
        # Default disabled, only use intensity-related augmentation (grayscale, color, noise)
        self.enable_nnunet_augmentation = self.configuration_manager.network_arch_init_kwargs.get(
            'enable_nnunet_augmentation', False
        )
        if self.enable_nnunet_augmentation:
            self.print_to_log_file("nnUNet data augmentation: ENABLED (spatial transforms, mirroring, etc.)")
        else:
            self.print_to_log_file("nnUNet data augmentation: DISABLED (only intensity augmentation)")

        # Multiprocessing parameters (for dataset balancing)
        self.num_processes = self.configuration_manager.network_arch_init_kwargs.get('num_processes', 10)  # Default 10 processes
        self.print_to_log_file(f"Multiprocessing: {self.num_processes} processes for dataset balancing")

        print("self.configuration_manager.network_arch_init_kwargs.get('data_format'): ", self.configuration_manager.network_arch_init_kwargs.get('data_format'))
        # ==================== Data Format Configuration ====================
        # Blosc2 format is supported
        self.data_format = 'blosc2'
        self.print_to_log_file(f"Using data format: {self.data_format} (Blosc2 only)")

        # ==================== CSV Shape Preprocessing ====================
        # Automatically check and run shape preprocessing (if CSV files don't have shape columns yet)
        if self.csv_paths is not None:
            self._check_and_preprocess_csv_shapes()

        # Initialize components based on mode
        if self.mode in ['only_cls', 'both']:
            self.logger = nnUNetLogger(num_cls_task=self.num_cls_task)
            self.use_sampling_weight = 'cls_balance'
        else:
            self.logger = nnUNetLogger(num_cls_task=0)

        # ==================== Report Generation and Network Parameters (only for report modes) ====================
        if self.mode in ['only_report', 'both']:
            # Report generation parameters
            self.report_loss_weight = self.configuration_manager.network_arch_init_kwargs.get('report_loss_weight', 1.0)
            self.report_max_length = self.configuration_manager.network_arch_init_kwargs.get('report_max_length', 8192)
            self.report_max_new_tokens = self.configuration_manager.network_arch_init_kwargs.get('report_max_new_tokens', 512)

            # LLM parameters
            self.llm_embed_dim = self.configuration_manager.network_arch_init_kwargs.get('llm_embed_dim', 512)
            self.num_heads = self.configuration_manager.network_arch_init_kwargs.get('num_heads', 8)
            self.ffn_dim = self.configuration_manager.network_arch_init_kwargs.get('ffn_dim', 2048)

            # LLM generation parameters
            self.generation_temperature = self.configuration_manager.network_arch_init_kwargs.get('generation_temperature', 0.7)
            self.generation_top_p = self.configuration_manager.network_arch_init_kwargs.get('generation_top_p', 0.9)
            self.generation_top_k = self.configuration_manager.network_arch_init_kwargs.get('generation_top_k', 50)

            # DeepStack parameters
            self.deepstack_skip_stages = self.configuration_manager.network_arch_init_kwargs.get('deepstack_skip_stages', 0)  # Default 0, use all stages

            # Network optimization parameters
            self.use_weight_tying = self.configuration_manager.network_arch_init_kwargs.get('use_weight_tying', True)
            self.use_deepstack = self.configuration_manager.network_arch_init_kwargs.get('use_deepstack', False)
            self.use_vision_aware_mask = self.configuration_manager.network_arch_init_kwargs.get('use_vision_aware_mask', True)
            self.layers_per_stage = self.configuration_manager.network_arch_init_kwargs.get('layers_per_stage', 5) # 5

            # Gate and Pool parameters
            self.use_gate = self.configuration_manager.network_arch_init_kwargs.get('use_gate', False)  # Default not use gating
            self.use_adaptive_pool = self.configuration_manager.network_arch_init_kwargs.get('use_adaptive_pool', True)  # Default use AdaptivePool

            self.configuration_manager.network_arch_init_kwargs['llm_embed_dim'] = self.llm_embed_dim
            self.configuration_manager.network_arch_init_kwargs['num_heads'] = self.num_heads
            self.configuration_manager.network_arch_init_kwargs['ffn_dim'] = self.ffn_dim
            self.configuration_manager.network_arch_init_kwargs['use_weight_tying'] = self.use_weight_tying
            self.configuration_manager.network_arch_init_kwargs['use_deepstack'] = self.use_deepstack
            self.configuration_manager.network_arch_init_kwargs['use_vision_aware_mask'] = self.use_vision_aware_mask
            self.configuration_manager.network_arch_init_kwargs['generation_temperature'] = self.generation_temperature
            self.configuration_manager.network_arch_init_kwargs['generation_top_p'] = self.generation_top_p
            self.configuration_manager.network_arch_init_kwargs['generation_top_k'] = self.generation_top_k
            self.configuration_manager.network_arch_init_kwargs['layers_per_stage'] = self.layers_per_stage
            self.configuration_manager.network_arch_init_kwargs['use_gate'] = self.use_gate
            self.configuration_manager.network_arch_init_kwargs['use_adaptive_pool'] = self.use_adaptive_pool
            self.configuration_manager.network_arch_init_kwargs['deepstack_skip_stages'] = self.deepstack_skip_stages

            if self.use_deepstack:
                n_stages = self.configuration_manager.network_arch_init_kwargs.get('n_stages', 6)
                num_deepstack_stages = n_stages - self.deepstack_skip_stages
                num_layers = num_deepstack_stages * self.layers_per_stage
                self.print_to_log_file(f"DeepStack: {num_deepstack_stages} stages × {self.layers_per_stage} layers = {num_layers} total layers")
                self.print_to_log_file(f"Skipping first {self.deepstack_skip_stages} stages")
            else:
                self.print_to_log_file(f"Non-DeepStack: Single stage with {self.layers_per_stage} layers")

            # Component-specific learning rate multipliers
            self.encoder_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('encoder_lr_multiplier', 1.0)
            self.patch_embed_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('patch_embed_lr_multiplier', 1.0)
            self.vision_proj_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('vision_proj_lr_multiplier', 1.0)
            self.llm_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('llm_lr_multiplier', 1.0)

            self.print_to_log_file("Using component-specific learning rate multipliers:")
            self.print_to_log_file(f"  Encoder LR multiplier: {self.encoder_lr_multiplier}")
            self.print_to_log_file(f"  Patch Embed LR multiplier: {self.patch_embed_lr_multiplier}")
            self.print_to_log_file(f"  Vision Proj LR multiplier: {self.vision_proj_lr_multiplier}")
            self.print_to_log_file(f"  LLM LR multiplier: {self.llm_lr_multiplier}")

        # Visual token configuration
        self.visual_token_length_source_stage = self.configuration_manager.network_arch_init_kwargs.get(
            'visual_token_length_source_stage', -1
        )

        # ==================== CSV Data Configuration ====================
        # Note: CSV datasets now use dynamic sizing based on strides, no fixed target size

        # ==================== Tokenizer and Prompts (only for report modes) ====================
        if self.mode in ['only_report', 'both']:
            # Load report prompts from file for randomization
            prompts_file = os.path.join(os.path.dirname(__file__), '..', 'dataloading', 'report_prompts.txt')
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse prompts line by line
                lines = content.split('\n')
                prompts = []
                current_prompt = []
                in_prompt = False

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('<|im_start|>system'):
                        if current_prompt:
                            prompts.append('\n'.join(current_prompt))
                        current_prompt = [line]
                        in_prompt = True
                    elif in_prompt:
                        current_prompt.append(line)

                # Add the last prompt
                if current_prompt:
                    prompts.append('\n'.join(current_prompt))

                self.report_prompts = prompts
                self.print_to_log_file(f"Loaded {len(self.report_prompts)} report prompts from {prompts_file}")
            else:
                # Fallback to single prompt if file not found
                self.report_prompts = ["<|im_start|>system\nYou are a radiologist who writes standardized reports based only on images.\n<|im_start|>user\nPlease write a report:<|im_end|>\n<|im_start|>assistant\n"]
                self.print_to_log_file(f"Warning: {prompts_file} not found, using fallback prompt")

            # Keep original single prompt for backward compatibility
            self.report_prompt = self.report_prompts[0]

            self.tokenizer_path = self.configuration_manager.network_arch_init_kwargs.get(
                'tokenizer_path', "/path/to/tokenizer/"
            )
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            self.report_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, model_max_length=self.report_max_length,
                padding_side="right", trust_remote_code=True, use_fast=True
            )
            if self.report_tokenizer.pad_token is None:
                self.report_tokenizer.pad_token = self.report_tokenizer.eos_token

            self.vocab_size = len(self.report_tokenizer)

            self.print_to_log_file("Report tokenizer initialized successfully")
            self.print_to_log_file(f"Using report prompt (Qwen2.5 format): '{self.report_prompt}'")
            self.print_to_log_file(f"Tokenizer: {self.tokenizer_path}")
            self.print_to_log_file(f"Report tokenizer vocab size: {self.vocab_size}")
            self.print_to_log_file(f"Report: max_len={self.report_max_length}, max_new={self.report_max_new_tokens}")
            
            # Calculate num_layers for logging display
            if self.use_deepstack:
                n_stages = self.configuration_manager.network_arch_init_kwargs.get('n_stages', 6)
                num_layers = (n_stages - self.deepstack_skip_stages) * self.layers_per_stage
            else:
                num_layers = self.layers_per_stage
            self.print_to_log_file(f"LLM: embed_dim={self.llm_embed_dim}, layers={num_layers}, heads={self.num_heads}, ffn_dim={self.ffn_dim}")
            self.print_to_log_file(f"Gen: temp={self.generation_temperature}, top_p={self.generation_top_p}, top_k={self.generation_top_k}")
            self.print_to_log_file(f"Network: weight_tying={self.use_weight_tying}, deepstack={self.use_deepstack}, vision_aware_mask={self.use_vision_aware_mask}")
            self.print_to_log_file(f"Gate: use_gate={self.use_gate}, use_adaptive_pool={self.use_adaptive_pool}")

        # ==================== Log Output ====================
        self.print_to_log_file(f"CSV: {len(self.csv_paths)} dataset(s)")

        # ==================== Patch Kernel and pool_output_size Calculation ====================
        if self.visual_token_length_source_stage != 0:
            features_per_stage = self.configuration_manager.network_arch_init_kwargs.get('features_per_stage')
            strides = self.configuration_manager.network_arch_init_kwargs.get('strides')
            if features_per_stage is not None and strides is not None:
                patch_kernel_sizes = self._compute_patch_kernel_sizes(
                    features_per_stage, strides
                )
                if 'patch_kernel_sizes' not in self.configuration_manager.network_arch_init_kwargs:
                    self.configuration_manager.network_arch_init_kwargs['patch_kernel_sizes'] = patch_kernel_sizes

                # Calculate feature map size for each stage
                patch_size = self.configuration_manager.patch_size
                normalized_strides = [(s, s, s) if isinstance(s, int) else tuple(s) for s in strides]
                num_stages = len(features_per_stage)

                # Calculate feature map size for each stage
                stage_feature_sizes = []
                cumulative = [1, 1, 1]
                for i, s in enumerate(normalized_strides):
                    cumulative = [cumulative[0] * s[0], cumulative[1] * s[1], cumulative[2] * s[2]]
                    feature_size = (
                        patch_size[0] // cumulative[0],
                        patch_size[1] // cumulative[1],
                        patch_size[2] // cumulative[2]
                    )
                    stage_feature_sizes.append(feature_size)

                # Determine source stage index
                source_stage_idx = num_stages + self.visual_token_length_source_stage if self.visual_token_length_source_stage < 0 else self.visual_token_length_source_stage
                source_stage_idx = max(0, min(source_stage_idx, num_stages - 1))

                # pool_output_size: use feature map size of source stage
                pool_output_size = stage_feature_sizes[source_stage_idx]
                self.configuration_manager.network_arch_init_kwargs['pool_output_size'] = pool_output_size

                # When visual_token_length_source_stage != -1, multi-scale vision tokens are needed
                # Stages shallower than source stage need to pool to source stage size
                # Source stage and deeper stages keep original size (no pooling)
                if self.visual_token_length_source_stage != -1:
                    # pool_output_size_list: pool target size for each stage
                    # None means keep original size (no pooling)
                    pool_output_size_list = []
                    for i in range(num_stages):
                        if i < source_stage_idx:
                            # Stages shallower than source stage: pool to source stage size
                            pool_output_size_list.append(pool_output_size)
                        else:
                            # Source stage and deeper stages: keep original size (no pooling)
                            pool_output_size_list.append(None)
                    self.configuration_manager.network_arch_init_kwargs['pool_output_size_list'] = pool_output_size_list
                    self.print_to_log_file(f"Multi-scale vision tokens: source_stage={self.visual_token_length_source_stage} (idx={source_stage_idx})")
                    self.print_to_log_file(f"  Stage feature sizes: {stage_feature_sizes}")
                    self.print_to_log_file(f"  Pool output size list: {pool_output_size_list}")
                else:
                    self.print_to_log_file(f"Single-scale vision tokens: pool_output_size={pool_output_size}")

                self.print_to_log_file(f"Multi-scale: source_stage={self.visual_token_length_source_stage}, kernels={patch_kernel_sizes}, pool_output_size={pool_output_size}")

        if 'visual_token_length_source_stage' not in self.configuration_manager.network_arch_init_kwargs:
            self.configuration_manager.network_arch_init_kwargs['visual_token_length_source_stage'] = self.visual_token_length_source_stage

        # Set network report generation based on mode
        self.configuration_manager.network_arch_init_kwargs['enable_report_gen'] = self.mode in ['only_report', 'both']

        # ==================== Pretrained Encoder Weights Configuration ====================
        # Read pretrained encoder checkpoint path from plan for ablation experiments
        # Supports loading: only_cls pretrained weights, segmentation task pretrained weights, external pretrained weights (e.g., VoxTell)
        self.pretrained_encoder_checkpoint_path = self.configuration_manager.network_arch_init_kwargs.get(
            'pretrained_encoder_checkpoint_path', None
        )
        if self.pretrained_encoder_checkpoint_path is not None:
            self.print_to_log_file(f"Pretrained encoder checkpoint path: {self.pretrained_encoder_checkpoint_path}")
            self.print_to_log_file("Will load encoder weights with strict=True after network initialization")

        # ==================== Encoder Freezing Configuration ====================
        # Read from plan whether to freeze encoder for ablation experiments
        # freeze_encoder: True/False - whether to freeze entire encoder
        # freeze_encoder_stages: List[int] - freeze specified encoder stages (e.g., [0,1,2] freezes first 3 stages)
        self.freeze_encoder = self.configuration_manager.network_arch_init_kwargs.get('freeze_encoder', False)
        self.freeze_encoder_stages = self.configuration_manager.network_arch_init_kwargs.get('freeze_encoder_stages', None)

        if self.freeze_encoder:
            self.print_to_log_file("Encoder freezing: ENABLED (all encoder parameters will be frozen)")
        elif self.freeze_encoder_stages is not None:
            self.print_to_log_file(f"Encoder freezing: PARTIAL (stages {self.freeze_encoder_stages} will be frozen)")
        else:
            self.print_to_log_file("Encoder freezing: DISABLED (all parameters trainable)")

        # ==================== Training Configuration ====================
        self.save_every = 5
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self._val_step_idx = 0
        self.num_report_examples_per_epoch = 1

        # Gradient monitoring
        self.grad_monitoring_enabled = True  # Disabled for performance
        self.grad_history = []
        self.param_history = []
        self._train_step_idx = 0

    def get_param_info(self, params_list, name):
        """
        Calculate parameter count and size information for logging.

        Args:
            params_list: List of parameters
            name: Component name for logging

        Returns:
            tuple: (num_params, size_in_millions)
        """
        if not params_list:
            return 0, 0.0
        num_params = len(params_list)
        total_size = sum(p.numel() for p in params_list)
        return num_params, total_size / 1e6  # Convert to millions

    def format_report_text_for_training(self, report_text: str) -> str:
        prompt = random.choice(self.report_prompts)
        return f"{prompt}{report_text}<|im_end|>"

    def load_pretrained_encoder(self, checkpoint_path: str) -> None:
        """
        Load encoder weights from pretrained checkpoint for ablation experiments.

        Supported checkpoint formats:
        1. Single file checkpoint (.pth): load directly
        2. Directory checkpoint: find and load from directory
        3. External pretrained weights (e.g., VoxTell checkpoint in same directory as plans.json)

        Loading rules:
        - Only load encoder weights (encoder.stem.* and encoder.stages.*)
        - Encoder weights must match strictly (strict=True) to ensure ablation experiment consistency
        - Other weights (decoder, cls_head, llm, etc.) are ignored

        Args:
            checkpoint_path: checkpoint file path or directory path
        """
        self.print_to_log_file("")
        self.print_to_log_file("=" * 80)
        self.print_to_log_file("LOADING PRETRAINED ENCODER WEIGHTS")
        self.print_to_log_file("=" * 80)
        self.print_to_log_file(f"Checkpoint path: {checkpoint_path}")

        # Get current network module
        network_module = self.network
        if hasattr(network_module, 'module'):
            network_module = network_module.module
        if isinstance(network_module, OptimizedModule):
            network_module = network_module._orig_mod

        # Determine checkpoint file path
        checkpoint_weights = None

        if os.path.isfile(checkpoint_path):
            # Single file checkpoint
            self.print_to_log_file(f"Loading from single file checkpoint")
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # Try to get weights from different keys
            if 'network_weights' in checkpoint_data:
                checkpoint_weights = checkpoint_data['network_weights']
            elif 'state_dict' in checkpoint_data:
                checkpoint_weights = checkpoint_data['state_dict']
            elif 'model' in checkpoint_data:
                checkpoint_weights = checkpoint_data['model']
            else:
                # Assume entire checkpoint is state_dict
                checkpoint_weights = checkpoint_data

        elif os.path.isdir(checkpoint_path):
            # Directory checkpoint
            self.print_to_log_file(f"Loading from directory checkpoint")

            # Try to find checkpoint file
            possible_files = [
                os.path.join(checkpoint_path, 'mp_rank_00_model_states.pt'),
                os.path.join(checkpoint_path, 'pytorch_model.bin'),
                os.path.join(checkpoint_path, 'model.pt'),
                os.path.join(checkpoint_path, 'checkpoint.pth'),
            ]

            # Also check subdirectories (e.g., best_universal, latest_universal)
            for subdir in ['best_universal', 'latest_universal', 'checkpoint_universal']:
                subdir_path = os.path.join(checkpoint_path, subdir)
                if os.path.isdir(subdir_path):
                    possible_files.extend([
                        os.path.join(subdir_path, 'mp_rank_00_model_states.pt'),
                        os.path.join(subdir_path, 'pytorch_model.bin'),
                    ])

            checkpoint_file = None
            for pf in possible_files:
                if os.path.exists(pf):
                    checkpoint_file = pf
                    break

            if checkpoint_file is None:
                # Try to find any .pt or .pth file
                pt_files = glob.glob(os.path.join(checkpoint_path, '*.pt')) + \
                           glob.glob(os.path.join(checkpoint_path, '*.pth')) + \
                           glob.glob(os.path.join(checkpoint_path, '*', '*.pt'))
                if pt_files:
                    checkpoint_file = pt_files[0]

            if checkpoint_file is None:
                raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")

            self.print_to_log_file(f"Found checkpoint file: {checkpoint_file}")
            checkpoint_data = torch.load(checkpoint_file, map_location=self.device, weights_only=False)

            # Support multiple checkpoint formats
            if 'module' in checkpoint_data:
                checkpoint_weights = checkpoint_data['module']
            elif 'network_weights' in checkpoint_data:
                checkpoint_weights = checkpoint_data['network_weights']
            elif 'state_dict' in checkpoint_data:
                checkpoint_weights = checkpoint_data['state_dict']
            elif 'model' in checkpoint_data:
                checkpoint_weights = checkpoint_data['model']
            else:
                checkpoint_weights = checkpoint_data
        else:
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

        if checkpoint_weights is None:
            raise ValueError(f"Could not extract weights from checkpoint: {checkpoint_path}")

        # Extract encoder weights (encoder.stem.* and encoder.stages.*)
        encoder_weights = {}
        other_weights_count = 0

        for key, value in checkpoint_weights.items():
            # Remove possible prefixes (e.g., 'module.' or '_orig_mod.')
            clean_key = key
            if clean_key.startswith('module.'):
                clean_key = clean_key[7:]
            if clean_key.startswith('_orig_mod.'):
                clean_key = clean_key[10:]

            # Only keep encoder-related weights (encoder.stem.* and encoder.stages.*)
            if clean_key.startswith('encoder.'):
                encoder_weights[clean_key] = value
            else:
                other_weights_count += 1

        self.print_to_log_file(f"Found {len(encoder_weights)} encoder parameters in checkpoint")
        self.print_to_log_file(f"Ignored {other_weights_count} non-encoder parameters")

        if len(encoder_weights) == 0:
            raise ValueError("No encoder weights (encoder.*) found in checkpoint!")

        # Get current network's encoder weights
        current_encoder_weights = {}
        for key, value in network_module.state_dict().items():
            if key.startswith('encoder.'):
                current_encoder_weights[key] = value

        self.print_to_log_file(f"Current network has {len(current_encoder_weights)} encoder parameters")

        # Check if encoder weights match exactly
        missing_keys = set(current_encoder_weights.keys()) - set(encoder_weights.keys())
        unexpected_keys = set(encoder_weights.keys()) - set(current_encoder_weights.keys())

        if missing_keys:
            self.print_to_log_file(f"ERROR: Missing encoder keys in checkpoint: {list(missing_keys)[:5]}...")
            raise ValueError(f"Encoder weights mismatch! Missing {len(missing_keys)} keys. "
                           f"strict=True requires exact match for ablation study consistency.")

        if unexpected_keys:
            self.print_to_log_file(f"WARNING: Unexpected encoder keys in checkpoint (will be ignored): {list(unexpected_keys)[:5]}...")
            # Remove unexpected keys
            for key in unexpected_keys:
                del encoder_weights[key]

        # Check if shapes match
        shape_mismatches = []
        for key in encoder_weights:
            if key in current_encoder_weights:
                if encoder_weights[key].shape != current_encoder_weights[key].shape:
                    shape_mismatches.append(
                        f"{key}: checkpoint {encoder_weights[key].shape} vs model {current_encoder_weights[key].shape}"
                    )

        if shape_mismatches:
            self.print_to_log_file(f"ERROR: Shape mismatches found:")
            for mismatch in shape_mismatches[:5]:
                self.print_to_log_file(f"  {mismatch}")
            raise ValueError(f"Encoder weights shape mismatch! {len(shape_mismatches)} parameters have different shapes. "
                           f"strict=True requires exact shape match for ablation study consistency.")

        # Load encoder weights (strict mode)
        # Create a state_dict containing only encoder weights
        current_state_dict = network_module.state_dict()
        for key, value in encoder_weights.items():
            current_state_dict[key] = value

        # Load updated state_dict
        network_module.load_state_dict(current_state_dict, strict=True)

        self.print_to_log_file("")
        self.print_to_log_file("SUCCESS: Pretrained encoder weights loaded with strict=True")
        self.print_to_log_file(f"  Loaded {len(encoder_weights)} encoder parameters")
        self.print_to_log_file(f"  Other network components (decoder, cls_head, llm, etc.) initialized randomly")
        self.print_to_log_file("=" * 80)
        self.print_to_log_file("")

    def freeze_encoder_parameters(self) -> None:
        """
        Freeze encoder parameters for ablation experiments.

        Supports two freezing modes:
        1. freeze_encoder=True: freeze entire encoder (encoder.stem.* and encoder.stages.*)
        2. freeze_encoder_stages=[0,1,2]: freeze specified encoder stages

        Frozen parameters will not participate in gradient computation and optimizer updates.
        """
        if not self.freeze_encoder and self.freeze_encoder_stages is None:
            return

        self.print_to_log_file("")
        self.print_to_log_file("=" * 80)
        self.print_to_log_file("FREEZING ENCODER PARAMETERS")
        self.print_to_log_file("=" * 80)

        # Get current network module
        network_module = self.network
        if hasattr(network_module, 'module'):
            network_module = network_module.module
        if isinstance(network_module, OptimizedModule):
            network_module = network_module._orig_mod

        frozen_count = 0
        frozen_params = 0
        trainable_count = 0
        trainable_params = 0

        for name, param in network_module.named_parameters():
            should_freeze = False

            if name.startswith('encoder.'):
                if self.freeze_encoder:
                    # Freeze entire encoder
                    should_freeze = True
                elif self.freeze_encoder_stages is not None:
                    # Freeze specified stages
                    # encoder.stem.* -> stage -1 (stem)
                    # encoder.stages.X.xxx -> stage X
                    if name.startswith('encoder.stem.'):
                        # stem is treated as stage -1, freeze if -1 is in freeze_encoder_stages
                        if -1 in self.freeze_encoder_stages:
                            should_freeze = True
                    else:
                        match = re.match(r'encoder\.stages\.(\d+)', name)
                        if match:
                            stage_idx = int(match.group(1))
                            if stage_idx in self.freeze_encoder_stages:
                                should_freeze = True

            if should_freeze:
                param.requires_grad = False
                frozen_count += 1
                frozen_params += param.numel()
            else:
                trainable_count += 1
                trainable_params += param.numel()

        self.print_to_log_file(f"Frozen parameters: {frozen_count} ({frozen_params/1e6:.2f}M)")
        self.print_to_log_file(f"Trainable parameters: {trainable_count} ({trainable_params/1e6:.2f}M)")

        if self.freeze_encoder:
            self.print_to_log_file("Mode: Full encoder frozen")
        elif self.freeze_encoder_stages is not None:
            self.print_to_log_file(f"Mode: Stages {self.freeze_encoder_stages} frozen")

        # Print freezing status for each stage
        stage_status = {}
        for name, param in network_module.named_parameters():
            if name.startswith('encoder.'):

                if name.startswith('encoder.stem.'):
                    stage_idx = -1  # stem as stage -1
                else:
                    match = re.match(r'encoder\.stages\.(\d+)', name)
                    if match:
                        stage_idx = int(match.group(1))
                    else:
                        continue

                if stage_idx not in stage_status:
                    stage_status[stage_idx] = {'frozen': 0, 'trainable': 0, 'frozen_params': 0, 'trainable_params': 0}
                if not param.requires_grad:
                    stage_status[stage_idx]['frozen'] += 1
                    stage_status[stage_idx]['frozen_params'] += param.numel()
                else:
                    stage_status[stage_idx]['trainable'] += 1
                    stage_status[stage_idx]['trainable_params'] += param.numel()

        self.print_to_log_file("Encoder stages status:")
        for stage_idx in sorted(stage_status.keys()):
            status = stage_status[stage_idx]
            frozen_str = f"frozen={status['frozen']} ({status['frozen_params']/1e6:.2f}M)"
            trainable_str = f"trainable={status['trainable']} ({status['trainable_params']/1e6:.2f}M)"
            state = "FROZEN" if status['trainable'] == 0 else "TRAINABLE" if status['frozen'] == 0 else "PARTIAL"
            stage_name = "Stem" if stage_idx == -1 else f"Stage {stage_idx}"
            self.print_to_log_file(f"  {stage_name}: [{state}] {frozen_str}, {trainable_str}")

        self.print_to_log_file("=" * 80)
        self.print_to_log_file("")

    def _get_actual_network(self):
        """Get actual network model, compatible with DDP and torch.compile"""

        network = self.network

        # DDP wrapper
        if hasattr(network, 'module'):
            network = network.module

        # torch.compile wrapper
        if isinstance(network, OptimizedModule):
            network = network._orig_mod

        return network

    def plot_network_architecture(self):
        """Override to disable network architecture plotting (avoid complex input issues for report generation models)"""
        if self.local_rank == 0:
            self.print_to_log_file("Network architecture plotting disabled for report generation models")
            self.print_to_log_file("(Complex input requirements make visualization impractical)")

    def _check_and_preprocess_csv_shapes(self):
        """
        Check if CSV files have shape column, run preprocessing if missing.

        Supports multiple column names for backward compatibility:
        - 'shape' (preferred, generic)
        - 'blosc2_shape' (blosc2 format)
        """
        # Possible shape column names in priority order
        shape_columns = ['processed_shape']

        needs_preprocessing = self.force_regenerate_shapes
        reason = "force_regenerate_shapes is enabled" if needs_preprocessing else ""
        found_column = None

        if not needs_preprocessing:
            # Check all CSV files for any shape column
            for csv_path in self.csv_paths:
                if not os.path.exists(csv_path):
                    self.print_to_log_file(f"Warning: CSV file not found: {csv_path}")
                    continue

                df = pd.read_csv(csv_path)
                # Find any existing shape column
                for col in shape_columns:
                    if col in df.columns:
                        found_column = col
                        break

                if found_column is None:
                    needs_preprocessing = True
                    reason = "CSV files missing shape column"
                    break

        if needs_preprocessing:
            self.print_to_log_file(f"Running shape preprocessing ({reason})...")
            self.print_to_log_file(f"Note: This step only reads and records shape information, no filtering applied")

            num_workers = self.configuration_manager.network_arch_init_kwargs.get('preprocess_workers', min(self.num_processes, 8))

            if self.local_rank == 0:
                stats = preprocess_csv_with_shapes(
                    csv_paths=self.csv_paths,
                    num_workers=num_workers,
                    force_reprocess=self.force_regenerate_shapes,
                    verbose=True,
                    data_format=self.data_format
                )

                self.print_to_log_file("Shape preprocessing completed!")
                self.print_to_log_file(f"Total processed: {stats['processed']}")
                self.print_to_log_file(f"Cases with shape: {stats['with_shape']}")
                self.print_to_log_file(f"Cases failed to read: {stats['no_shape']}")

            if self.is_ddp:
                if dist.is_initialized():
                    dist.barrier()
        else:
            self.print_to_log_file(f"CSV files already have '{found_column}' column, skipping shape preprocessing")

    # ==================== CSV Loading Methods ====================
    @staticmethod
    def _load_single_csv_file(csv_path):
        """
        Load single CSV file (for multiprocessing)

        Args:
            csv_path: CSV file path

        Returns:
            (csv_path, DataFrame)
        """
        if not os.path.exists(csv_path):
            return csv_path, None
        df = pd.read_csv(csv_path)
        return csv_path, df

    def _compute_patch_kernel_sizes(
        self,
        features_per_stage: List[int],
        strides: List[Union[Tuple[int, int, int], int]]
    ) -> List[Tuple[int, int, int]]:
        num_stages = len(features_per_stage)
        visual_token_length_source_stage = self.visual_token_length_source_stage
        if visual_token_length_source_stage < 0:
            source_stage_idx = num_stages + visual_token_length_source_stage
        else:
            source_stage_idx = visual_token_length_source_stage
        source_stage_idx = max(0, min(source_stage_idx, num_stages - 1))

        normalized_strides = []
        for s in strides:
            if isinstance(s, int):
                normalized_strides.append((s, s, s))
            else:
                normalized_strides.append(s)

        patch_kernel_sizes = []
        for stage_idx in range(num_stages):
            if stage_idx >= source_stage_idx:
                patch_kernel_sizes.append((1, 1, 1))
            else:
                cumulative = [1, 1, 1]
                for i in range(stage_idx + 1, source_stage_idx + 1):
                    if i < len(normalized_strides):
                        cumulative[0] *= normalized_strides[i][0]
                        cumulative[1] *= normalized_strides[i][1]
                        cumulative[2] *= normalized_strides[i][2]
                patch_kernel_sizes.append(tuple(cumulative))

        self.print_to_log_file(
            f"Source stage {source_stage_idx}, patch kernel sizes based on stride accumulation: {patch_kernel_sizes}"
        )
        return patch_kernel_sizes

    def initialize(self):
        if not self.was_initialized:
            # First create label manager and other basic components (order unchanged)
            simple_label_dict = {'background': 0}
            self.label_manager = LabelManager(label_dict=simple_label_dict, regions_class_order=None)

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                self.dataset_json)

            # !!!Most critical: create network first
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                1,  # dummy
                self.enable_deep_supervision
            ).to(self.device)

            self.print_to_log_file(f"Network initialized with {sum(p.numel() for p in self.network.parameters())/1e6:.2f}M parameters")

            # ==================== Load Pretrained Encoder Weights (for ablation experiments) ====================
            # Must load before torch.compile and DDP wrapping
            if self.pretrained_encoder_checkpoint_path is not None:
                self.load_pretrained_encoder(self.pretrained_encoder_checkpoint_path)

            # ==================== Freeze Encoder Parameters (for ablation experiments) ====================
            # Must freeze after loading pretrained weights and before configuring optimizer
            if self.freeze_encoder or self.freeze_encoder_stages is not None:
                self.freeze_encoder_parameters()

            # Compile (if needed)
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            # Configure optimizer and scheduler
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            # Loss initialization (order unchanged)
            if self.mode in ['only_cls', 'both']:
                self.cls_loss_list = []
                for i in range(self.num_cls_task):
                    pos_weights_tensor = torch.tensor(self.pos_weights_list[i]).to(self.device)
                    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor, reduction='none')
                    self.cls_loss_list.append(cls_loss)

            if self.mode in ['only_report', 'both']:
                self.report_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
                self.print_to_log_file("Report generation loss function initialized")

            self.was_initialized = True
        else:
            raise RuntimeError("Trainer already initialized")

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Configure data augmentation parameters.

        By default, only mirror on axes 0 and 1 (not on z-axis),
        to maintain vision token order consistency.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        if dim == 2:
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0,)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            # Only mirror on axes 0 and 1, not on z-axis
            mirror_axes = (0, 1)
        else:
            raise RuntimeError()

        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.print_to_log_file(f'mirror_axes: {mirror_axes}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            enable_nnunet_augmentation: bool = False,
    ) -> AbstractTransform:
        """
        Data augmentation transforms.

        Args:
            enable_nnunet_augmentation: Whether to enable nnUNet-style data augmentation
                - False (default): Only use grayscale, color, and noise augmentation, no spatial transforms or mirroring
                - True: Use full nnUNet data augmentation including spatial transforms, mirroring, etc.
        """
        tr_transforms = []

        if enable_nnunet_augmentation:
            # ==================== Full nnUNet-style data augmentation ====================
            if do_dummy_2d_data_aug:
                ignore_axes = (0,)
                tr_transforms.append(Convert3DTo2DTransform())
                patch_size_spatial = patch_size[1:]
            else:
                patch_size_spatial = patch_size
                ignore_axes = None

            # Spatial transforms (rotation, scaling)
            tr_transforms.append(SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=None,
                do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
                do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
                p_rot_per_axis=1,
                do_scale=True, scale=(0.7, 1.4),
                border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
                border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
                random_crop=False,
                p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False
            ))

            if do_dummy_2d_data_aug:
                tr_transforms.append(Convert2DTo3DTransform())

            # Grayscale and color augmentation
            tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
            tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                       p_per_channel=0.5))
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
            tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
            tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                                p_per_channel=0.5,
                                                                order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                                ignore_axes=ignore_axes))
            tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
            tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

            # Mirror transform
            if mirror_axes is not None and len(mirror_axes) > 0:
                tr_transforms.append(MirrorTransform(mirror_axes))

            if use_mask_for_norm is not None and any(use_mask_for_norm):
                tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                                   mask_idx_in_seg=0, set_outside_to=0))

            tr_transforms.append(RemoveLabelTransform(-1, 0))

            if is_cascaded:
                assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
                tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    p_per_sample=0.4,
                    key="data",
                    strel_size=(1, 8),
                    p_per_label=1))
                tr_transforms.append(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        key="data",
                        p_per_sample=0.2,
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15))

            tr_transforms.append(RenameTransform('seg', 'target', True))

            if regions is not None:
                tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                           if ignore_label is not None else regions,
                                                                           'target', 'target'))

            if deep_supervision_scales is not None:
                tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                                  output_key='target'))
        else:
            # ==================== Simple data augmentation (default) ====================
            # Only use grayscale, color, and noise augmentation, no spatial transforms or mirroring
            tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
            tr_transforms.append(GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=True,
                                                       p_per_sample=0.15, p_per_channel=0.5))
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25),
                                                                   p_per_sample=0.15))
            tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
            tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True,
                                               p_per_sample=0.1))
            tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True,
                                               p_per_sample=0.1))

            tr_transforms.append(RenameTransform('seg', 'target', True))

        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_dataloaders(self):
        """
        Override parent's get_dataloaders to pass enable_nnunet_augmentation to get_training_transforms.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline - pass enable_nnunet_augmentation parameter
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
            enable_nnunet_augmentation=self.enable_nnunet_augmentation)

        self.print_to_log_file(f"get_dataloaders: enable_nnunet_augmentation={self.enable_nnunet_augmentation}")

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)
        return mt_gen_train, mt_gen_val

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Override the base class method since UVLM
        is an encoder-only architecture without a traditional decoder.
        Deep supervision doesn't apply to this architecture.
        """
        pass  # No deep supervision needed for encoder-only architecture

    def configure_optimizers(self):
        """
        Configure optimizer with different learning rates for encoder, patch embedding, vision projection, and LLM parameters.
        This allows fine-grained control over different network components.
        """
        # Separate parameters into different groups
        encoder_params = []
        patch_embed_params = []
        vision_proj_params = []
        llm_params = []
        other_params = []

        # Get the actual network module (handle compiled models, but not DDP yet as it's called before DDP wrapping)
        network_module = self.network
        if isinstance(network_module, OptimizedModule):
            network_module = network_module._orig_mod

        # Get learning rate multipliers from instance variables (set during initialization)
        encoder_lr_multiplier = getattr(self, 'encoder_lr_multiplier', 1.0)
        patch_embed_lr_multiplier = getattr(self, 'patch_embed_lr_multiplier', 1.0)
        vision_proj_lr_multiplier = getattr(self, 'vision_proj_lr_multiplier', 1.0)
        llm_lr_multiplier = getattr(self, 'llm_lr_multiplier', 1.0)

        # Identify parameters for different components
        for name, param in network_module.named_parameters():
            if 'llm_report_gen' in name:
                llm_params.append(param)
            elif 'patch_embed' in name and ('patch_embed_list' in name or 'patch_embed.' in name):
                patch_embed_params.append(param)
            elif 'pool_embed' in name and ('pool_embed_list' in name or 'pool_embed.' in name):
                patch_embed_params.append(param)  # pool_embed uses the same learning rate as patch_embed
            elif 'vision_proj' in name or 'vision_projection' in name:
                vision_proj_params.append(param)
            elif 'encoder.' in name:  # encoder.stem.* and encoder.stages.*
                encoder_params.append(param)
            else:
                other_params.append(param)

        # Calculate parameter counts and sizes for logging
        encoder_num_params, encoder_size = self.get_param_info(encoder_params, "Encoder")
        patch_embed_num_params, patch_embed_size = self.get_param_info(patch_embed_params, "Patch Embed")
        vision_proj_num_params, vision_proj_size = self.get_param_info(vision_proj_params, "Vision Proj")
        llm_num_params, llm_size = self.get_param_info(llm_params, "LLM")
        other_num_params, other_size = self.get_param_info(other_params, "Other")

        # Calculate network layer counts
        n_stages = self.configuration_manager.network_arch_init_kwargs.get('n_stages', 6)
        encoder_layers = n_stages
        patch_embed_layers = n_stages if patch_embed_params else 0
        vision_proj_layers = n_stages if vision_proj_params else 0

        # Calculate LLM layers based on deepstack configuration
        if hasattr(self, 'use_deepstack') and self.use_deepstack:
            llm_layers = (n_stages - getattr(self, 'deepstack_skip_stages', 2)) * getattr(self, 'layers_per_stage', 1)
        else:
            llm_layers = getattr(self, 'layers_per_stage', 1)

        # Print detailed parameter and layer information
        self.print_to_log_file("=" * 80)
        self.print_to_log_file("NETWORK COMPONENT PARAMETER AND LAYER ANALYSIS:")
        self.print_to_log_file("=" * 80)
        if encoder_params:
            self.print_to_log_file(f"  Encoder:      {encoder_num_params:>6} params ({encoder_size:>6.2f}M), {encoder_layers:>2} layers, LR={self.initial_lr * encoder_lr_multiplier:.1e}")
        if patch_embed_params:
            self.print_to_log_file(f"  Patch Embed:  {patch_embed_num_params:>6} params ({patch_embed_size:>6.2f}M), {patch_embed_layers:>2} layers, LR={self.initial_lr * patch_embed_lr_multiplier:.1e}")
        if vision_proj_params:
            self.print_to_log_file(f"  Vision Proj:  {vision_proj_num_params:>6} params ({vision_proj_size:>6.2f}M), {vision_proj_layers:>2} layers, LR={self.initial_lr * vision_proj_lr_multiplier:.1e}")
        if llm_params:
            self.print_to_log_file(f"  LLM:          {llm_num_params:>6} params ({llm_size:>6.2f}M), {llm_layers:>2} layers, LR={self.initial_lr * llm_lr_multiplier:.1e}")
        if other_params:
            self.print_to_log_file(f"  Other:        {other_num_params:>6} params ({other_size:>6.2f}M), LR={self.initial_lr:.1e}")

        total_params = sum([encoder_num_params, patch_embed_num_params, vision_proj_num_params, llm_num_params, other_num_params])
        total_size = sum([encoder_size, patch_embed_size, vision_proj_size, llm_size, other_size])
        self.print_to_log_file(f"  Total:        {total_params:>6} params ({total_size:>6.2f}M)")
        self.print_to_log_file("=" * 80)

        # Create parameter groups with different learning rates
        param_groups = []

        # Encoder parameters
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.initial_lr * encoder_lr_multiplier,
                'weight_decay': self.weight_decay
            })

        # Patch embedding parameters
        if patch_embed_params:
            param_groups.append({
                'params': patch_embed_params,
                'lr': self.initial_lr * patch_embed_lr_multiplier,
                'weight_decay': self.weight_decay
            })

        # Vision projection parameters
        if vision_proj_params:
            param_groups.append({
                'params': vision_proj_params,
                'lr': self.initial_lr * vision_proj_lr_multiplier,
                'weight_decay': self.weight_decay
            })

        # LLM parameters
        if llm_params:
            param_groups.append({
                'params': llm_params,
                'lr': self.initial_lr * llm_lr_multiplier,
                'weight_decay': self.weight_decay
            })

        # Other parameters (classification heads, etc.)
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.initial_lr,
                'weight_decay': self.weight_decay
            })

        # ==================== Create optimizer ====================
        # Select optimizer based on optimizer_type
        if self.optimizer_type == 'SGD':
            # SGD optimizer
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=self.momentum,
                nesterov=self.nesterov
            )
            self.print_to_log_file(f"Optimizer: SGD")
            self.print_to_log_file(f"  - Momentum: {self.momentum}")
            self.print_to_log_file(f"  - Nesterov: {self.nesterov}")

        else:  # AdamW
            # AdamW optimizer
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=self.betas,
                eps=1e-8
            )
            self.print_to_log_file(f"Optimizer: AdamW")
            self.print_to_log_file(f"  - Betas: {self.betas}")
            self.print_to_log_file(f"  - Eps: 1e-8")

        # Store initial LRs for the scheduler
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        lr_scheduler = MultiGroupPolyLRScheduler(optimizer, self.num_epochs)

        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # Ensure visual_token_length_source_stage is set
        if 'visual_token_length_source_stage' not in architecture_kwargs:
            architecture_kwargs['visual_token_length_source_stage'] = -1

        # Compute patch_kernel_sizes if missing
        if 'patch_kernel_sizes' not in architecture_kwargs:
            # Compute patch_kernel_sizes similar to trainer initialization
            features_per_stage = architecture_kwargs.get('features_per_stage')
            strides = architecture_kwargs.get('strides')
            visual_token_length_source_stage = architecture_kwargs.get('visual_token_length_source_stage', -1)
            
            # Compute patch_kernel_sizes using the same logic as the trainer
            num_stages = len(features_per_stage)
            if visual_token_length_source_stage < 0:
                source_stage_idx = num_stages + visual_token_length_source_stage
            else:
                source_stage_idx = visual_token_length_source_stage
            source_stage_idx = max(0, min(source_stage_idx, num_stages - 1))

            normalized_strides = []
            for s in strides:
                if isinstance(s, int):
                    normalized_strides.append((s, s, s))
                else:
                    normalized_strides.append(s)

            patch_kernel_sizes = []
            for stage_idx in range(num_stages):
                if stage_idx >= source_stage_idx:
                    patch_kernel_sizes.append((1, 1, 1))
                else:
                    cumulative = [1, 1, 1]
                    for i in range(stage_idx + 1, source_stage_idx + 1):
                        if i < len(normalized_strides):
                            cumulative[0] *= normalized_strides[i][0]
                            cumulative[1] *= normalized_strides[i][1]
                            cumulative[2] *= normalized_strides[i][2]
                    patch_kernel_sizes.append(tuple(cumulative))

            architecture_kwargs['patch_kernel_sizes'] = patch_kernel_sizes
            # Only print on rank 0 in distributed training
            if int(os.environ.get('LOCAL_RANK', '0')) == 0:
                print(f"Computed patch_kernel_sizes: {patch_kernel_sizes}")

        # Filter kwargs to only include parameters accepted by UVLM
        sig = inspect.signature(UVLM.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in architecture_kwargs.items() if k in valid_params}

        filtered_out = set(architecture_kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out and int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print(f"Warning: Filtered out unsupported arguments for UVLM: {filtered_out}")

        network = UVLM(
                in_channels=num_input_channels,
                **filtered_kwargs
            )

        return network

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        properties = batch['properties']
        keys = batch['keys']

        # Debug mode: save input images as nii.gz for visualization
        if self.debug_save_input == True:
            debug_dir = os.path.join(self.output_folder, 'debug_input')
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[DEBUG] Saving input data to {debug_dir}")
            print(f"[DEBUG] data shape: {data.shape}, target shape: {target.shape if not isinstance(target, list) else [t.shape for t in target]}")
            print(f"[DEBUG] keys: {keys}")

            # Save each sample in the batch
            for b in range(data.shape[0]):
                # Save input image (all channels)
                for c in range(data.shape[1]):
                    img_np = data[b, c].cpu().numpy()
                    img_sitk = sitk.GetImageFromArray(img_np)
                    sitk.WriteImage(img_sitk, os.path.join(debug_dir, f'batch{b}_channel{c}_input.nii.gz'))

                # Save target
                if isinstance(target, list):
                    for t_idx, t in enumerate(target):
                        target_np = t[b, 0].cpu().numpy().astype(np.uint8)
                        target_sitk = sitk.GetImageFromArray(target_np)
                        sitk.WriteImage(target_sitk, os.path.join(debug_dir, f'batch{b}_target{t_idx}.nii.gz'))
                else:
                    target_np = target[b, 0].cpu().numpy().astype(np.uint8)
                    target_sitk = sitk.GetImageFromArray(target_np)
                    sitk.WriteImage(target_sitk, os.path.join(debug_dir, f'batch{b}_target.nii.gz'))

            print(f"[DEBUG] Saved {data.shape[0]} samples. Exiting for debug inspection.")
            exit()

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Prepare inputs based on mode
        cls_task_list = None
        if self.mode in ['only_cls', 'both']:
            cls_all = torch.from_numpy(batch['cls_all']).float().to(self.device, non_blocking=True)
            cls_task_list = [cls_all]

        # Prepare report text tokens if available
        report_text_ids = None
        report_text_attention_mask = None
        if ('report_texts' in batch and hasattr(self, 'report_tokenizer') and self.report_tokenizer is not None and
            len(batch['report_texts']) > 0 and any(text.strip() for text in batch['report_texts'] if text)):
            report_texts = batch['report_texts']
            formatted_texts = [self.format_report_text_for_training(report_text) for report_text in report_texts]

            tokenized = self.report_tokenizer(
                formatted_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.report_max_length
            )
            report_text_ids = tokenized['input_ids'].to(self.device)
            report_text_attention_mask = tokenized['attention_mask'].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            network_output = self.network(
                data,
                generate_report=False,
                report_text_ids=report_text_ids,
                report_text_attention_mask=report_text_attention_mask
            )

            # New architecture returns (report_output, cls_pred_list)
            report_output, cls_pred_list = network_output

            total_loss = torch.tensor(0.0, device=self.device)
            total_cls_loss = torch.tensor(0.0, device=self.device)
            total_report_loss = torch.tensor(0.0, device=self.device)

            # Classification loss (only_cls or both mode)
            if self.mode in ['only_cls', 'both']:
                for t_index in range(self.num_cls_task):
                    cls_pred_logits = cls_pred_list[t_index]
                    cls_target = cls_task_list[t_index]
                    cls_loss = self.cls_loss_list[t_index](cls_pred_logits, cls_target)
                    total_loss += cls_loss.mean()
                    total_cls_loss += cls_loss.mean()

            # Report generation loss (only_report or both mode)
            if self.mode in ['only_report', 'both']:
                if report_output is not None and report_text_ids is not None:
                    llm_logits = report_output.get('llm_logits')
                    text_start_idx = report_output.get('text_start_idx')

                    if llm_logits is not None and text_start_idx is not None:
                        text_len = report_text_ids.shape[1]
                        shift_logits = llm_logits[:, text_start_idx:text_start_idx + text_len - 1, :].contiguous()
                        shift_labels = report_text_ids[:, 1:text_len].contiguous()

                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)

                        mask = shift_labels != self.report_tokenizer.pad_token_id
                        if mask.any():
                            valid_logits = shift_logits[mask]
                            valid_labels = shift_labels[mask]
                            report_loss = self.report_loss_fn(valid_logits, valid_labels)
                            total_loss += self.report_loss_weight * report_loss
                            total_report_loss = report_loss

            # Normalize loss by number of active tasks
            num_active_tasks = 0
            if self.mode in ['only_cls', 'both']:
                num_active_tasks += self.num_cls_task
            if self.mode in ['only_report', 'both']:
                num_active_tasks += 1

            if num_active_tasks > 0:
                l = total_loss / num_active_tasks
            else:
                l = total_loss

        # Gradient computation and optimization
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)

            # Gradient monitoring (standard PyTorch with grad_scaler)
            if self.grad_monitoring_enabled:
                grad_norms = {}
                param_norms = {}
                has_nan_grad = False
                actual_network = self._get_actual_network()

                for name, param in actual_network.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            self.print_to_log_file(f"⚠️  NaN/Inf in gradient: {name}")
                            param.grad.zero_()
                            has_nan_grad = True
                            continue
                        grad_norms[name] = param.grad.data.norm(2).item()
                        param_norms[name] = param.data.norm(2).item()

                self._store_gradient_stats_cls(grad_norms, param_norms)

                if not has_nan_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
                    self.grad_scaler.step(self.optimizer)
                else:
                    self.print_to_log_file("⚠️  Skipping optimizer step due to NaN gradients")
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
                self.grad_scaler.step(self.optimizer)

            self.grad_scaler.update()
        else:
            l.backward()

            # Gradient monitoring (standard PyTorch without grad_scaler)
            if self.grad_monitoring_enabled:
                grad_norms = {}
                param_norms = {}
                has_nan_grad = False
                actual_network = self._get_actual_network()

                for name, param in actual_network.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            self.print_to_log_file(f"⚠️  NaN/Inf in gradient: {name}")
                            param.grad.zero_()
                            has_nan_grad = True
                            continue
                        grad_norms[name] = param.grad.data.norm(2).item()
                        param_norms[name] = param.data.norm(2).item()

                self._store_gradient_stats_cls(grad_norms, param_norms)

                if not has_nan_grad:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
                    self.optimizer.step()
                else:
                    self.print_to_log_file("⚠️  Skipping optimizer step due to NaN gradients")
            else:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
                self.optimizer.step()

        # Increment training step counter
        self._train_step_idx += 1

        # Return results based on mode
        result_dict = {'loss': l.detach().float().cpu().numpy()}
        if self.mode in ['only_cls', 'both']:
            result_dict['total_cls_loss'] = total_cls_loss.detach().float().cpu().numpy()
        if self.mode in ['only_report', 'both']:
            result_dict['total_report_loss'] = total_report_loss.detach().float().cpu().numpy()

        return result_dict

    @staticmethod
    def _validate_shapes(df: pd.DataFrame, patch_size: Tuple[int, int, int], data_format: str = 'blosc2') -> pd.DataFrame:
        """
        Validate shape column in DataFrame, filter cases where shape doesn't match patch_size.

        Args:
            df: DataFrame with shape column
            patch_size: patch size (Z, H, W)
            data_format: data format (unused, kept for compatibility)

        Returns:
            Filtered DataFrame with shape_valid column
        """
        # Find shape column
        shape_columns = ['processed_shape']
        shape_column = None
        for col in shape_columns:
            if col in df.columns:
                shape_column = col
                break

        if shape_column is None:
            raise ValueError(f"CSV missing shape column. Expected one of: {shape_columns}")

        # patch_size is (Z, H, W)
        expected_z, expected_h, expected_w = patch_size

        def check_shape_valid(shape_str):
            """Check if shape matches patch_size"""
            if pd.isna(shape_str) or shape_str == 'None' or shape_str == '':
                return False

            # Parse shape string: "[192, 256, 256]"
            shape = ast.literal_eval(str(shape_str))
            if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                return False

            # Shape from CSV is in same order as config target_size: (Z, H, W)
            z, h, w = shape

            if z == 0 or h == 0 or w == 0:
                return False

            # H and W must match (Z is flexible - will be cropped/padded)
            if h != expected_h or w != expected_w:
                return False

            return True

        df['shape_valid'] = df[shape_column].apply(check_shape_valid)
        valid_df = df[df['shape_valid'] == True].copy()

        return valid_df

    def _wait_for_file_sync(self, file_path: str, ready_flag_path: str, timeout: int = 600):
        """Wait for file to be created using barrier or file-based sync fallback."""

        if self.is_ddp and dist.is_initialized():
            dist.barrier()
        else:
            # File-based sync when dist not initialized
            poll_interval = 2
            start_time = time()
            while not os.path.exists(ready_flag_path):
                elapsed = time() - start_time
                if elapsed > timeout:
                    raise RuntimeError(f"[Rank {self.local_rank}] Timeout ({timeout}s) waiting for {file_path}")
                sleep(poll_interval)

        if not os.path.exists(file_path):
            raise RuntimeError(f"[Rank {self.local_rank}] File not found: {file_path}")

    def get_tr_and_val_datasets(self):
        """
        Load CSV datasets and split, with optional AISU balancing.

        Data flow:
        1. Load CSV files (with shape column)
        2. Filter: check if shape matches patch_size, add shape_valid column
        3. Keep only rows with shape_valid=True
        4. If balancing enabled, balance the filtered data
        5. Load dataset and split
        """
        if self.csv_paths is None:
            raise ValueError("csv_paths must be provided for this trainer")

        csv_paths_to_use = self.csv_paths

        # ==================== Stage 1: Load CSV and filter invalid shapes ====================
        self.print_to_log_file(f"Loading {len(self.csv_paths)} CSV files with {self.num_processes} processes...")
        with Pool(processes=self.num_processes) as pool:
            csv_results = list(tqdm(
                pool.imap(self._load_single_csv_file, self.csv_paths),
                total=len(self.csv_paths),
                desc="Loading CSV files",
                unit="files"
            ))

        # Merge DataFrames
        all_dfs = []
        for csv_path, df in csv_results:
            if df is not None and not df.empty:
                all_dfs.append(df)
                self.print_to_log_file(f"Loaded {len(df)} samples from {csv_path}")

        if not all_dfs:
            raise ValueError("No valid CSV data found")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.print_to_log_file(f"Combined dataset has {len(combined_df)} samples")

        # Find shape column (supports multiple column names)
        shape_columns = ['processed_shape']
        shape_column = None
        for col in shape_columns:
            if col in combined_df.columns:
                shape_column = col
                break

        if shape_column is None:
            raise ValueError(f"CSV missing shape column. Expected one of: {shape_columns}")

        # Validate shape x/y matches patch_size
        self.print_to_log_file(f"Validating shapes against patch_size={self.patch_size} using '{shape_column}' column...")
        filtered_df = self._validate_shapes(combined_df, self.patch_size, self.data_format)
        filtered_count = len(combined_df) - len(filtered_df)

        if filtered_count > 0:
            self.print_to_log_file(f"Filtered out {filtered_count} samples (shape x/y does not match patch_size x/y)")

        self.print_to_log_file(f"Remaining {len(filtered_df)} valid samples")

        if len(filtered_df) == 0:
            raise ValueError("No valid samples after shape validation")

        # ==================== Stage 1.5: Filter empty report samples (only_report or both mode) ====================
        if self.mode in ['only_report', 'both'] and self.report_column in filtered_df.columns:
            before_count = len(filtered_df)
            empty_mask = filtered_df[self.report_column].isna() | (filtered_df[self.report_column].astype(str).str.strip() == '')
            empty_count = empty_mask.sum()
            if empty_count > 0:
                empty_ids = filtered_df.loc[empty_mask, self.series_id_column].tolist()
                filtered_df = filtered_df[~empty_mask].reset_index(drop=True)
                self.print_to_log_file(f"Filtered out {empty_count} samples with empty reports: {empty_ids[:10]}{'...' if len(empty_ids) > 10 else ''}")
                self.print_to_log_file(f"Remaining {len(filtered_df)} samples after empty report filtering")

            if len(filtered_df) == 0:
                raise ValueError("No valid samples after empty report filtering")

        # ==================== Stage 2: Data balancing (optional) ====================
        if self.enable_balancing:
            self.print_to_log_file(f"[Rank {self.local_rank}] Data balancing enabled, starting...")

            # Set output path (all ranks need to know the path)
            if self.balancing_stats_output_dir is None:
                stats_output_dir = join(self.output_folder, 'balanced')
            else:
                stats_output_dir = self.balancing_stats_output_dir

            if self.balancing_output_path is None:
                balanced_output_path = join(stats_output_dir, 'balanced_dataset.csv')
            else:
                balanced_output_path = self.balancing_output_path

            self.print_to_log_file(f"[Rank {self.local_rank}] Target path: {balanced_output_path}")

            # Only rank 0 performs balancing to avoid file conflicts
            if self.local_rank == 0:
                self.print_to_log_file(f"[Rank 0] Creating output directory and starting balancing...")

                # Create output directory
                maybe_mkdir_p(stats_output_dir)
                if self.balancing_output_path is not None and os.path.dirname(balanced_output_path):
                    maybe_mkdir_p(os.path.dirname(balanced_output_path))

                # Save filtered DataFrame for balancing
                temp_filtered_path = join(stats_output_dir, 'temp_filtered_for_balancing.csv')
                self.print_to_log_file(f"[Rank 0] Saving filtered data to {temp_filtered_path}...")
                filtered_df.to_csv(temp_filtered_path, index=False)

                # Perform data balancing (using cls_columns)
                self.print_to_log_file(f"[Rank 0] Performing data balancing (target={self.balancing_target_samples_per_class} per class)...")
                self.print_to_log_file(f"[Rank 0] Using cls_columns: {self.cls_columns}")
                balance_csv_files(
                    csv_paths=[temp_filtered_path],
                    cls_columns=self.cls_columns,  # Use cls_columns for balancing
                    target_samples_per_class=self.balancing_target_samples_per_class,
                    max_times_per_sample=self.balancing_max_repeat_multiplier,
                    random_seed=self.balancing_seed,
                    output_path=balanced_output_path,
                    negative_ratio=self.balancing_negative_ratio,
                    stats_output_dir=stats_output_dir,
                    logger=self.print_to_log_file
                )

                # Clean up temporary files
                if os.path.exists(temp_filtered_path):
                    os.remove(temp_filtered_path)

                # Verify file was created
                if not os.path.exists(balanced_output_path):
                    raise RuntimeError(f"[Rank 0] Balancing failed: output file not created at {balanced_output_path}")

                # Create ready flag file to signal other ranks
                ready_flag_path = balanced_output_path + '.ready'
                with open(ready_flag_path, 'w') as f:
                    f.write('ready')
                self.print_to_log_file(f"[Rank 0] Created ready flag: {ready_flag_path}")

                self.print_to_log_file(f"[Rank 0] Balancing completed, saved to {balanced_output_path}")
            else:
                self.print_to_log_file(f"[Rank {self.local_rank}] Waiting for rank 0 to complete balancing...")

            # Wait for rank 0 to finish balancing
            ready_flag_path = balanced_output_path + '.ready'
            self._wait_for_file_sync(balanced_output_path, ready_flag_path)

            # All ranks read the balanced dataset
            self.print_to_log_file(f"[Rank {self.local_rank}] Loading balanced dataset from {balanced_output_path}...")
            balanced_df = pd.read_csv(balanced_output_path, encoding='utf-8', low_memory=False)
            self.print_to_log_file(f"[Rank {self.local_rank}] Balanced dataset: {len(balanced_df)} samples")

            csv_paths_to_use = [balanced_output_path]
        else:
            self.print_to_log_file(f"[Rank {self.local_rank}] Data balancing disabled, using filtered dataset directly")
            temp_filtered_path = join(self.output_folder, 'filtered_dataset.csv')
            ready_flag_path = temp_filtered_path + '.ready'

            # Only rank 0 saves the file
            if self.local_rank == 0:
                self.print_to_log_file(f"[Rank 0] Saving filtered dataset to {temp_filtered_path}...")
                filtered_df.to_csv(temp_filtered_path, index=False)
                if not os.path.exists(temp_filtered_path):
                    raise RuntimeError(f"[Rank 0] Failed to save filtered dataset to {temp_filtered_path}")

                # Create ready flag file to signal other ranks
                with open(ready_flag_path, 'w') as f:
                    f.write('ready')
                self.print_to_log_file(f"[Rank 0] Filtered dataset saved, ready flag created")
            else:
                self.print_to_log_file(f"[Rank {self.local_rank}] Waiting for rank 0 to save filtered dataset...")

            # Wait for rank 0 to finish saving
            self._wait_for_file_sync(temp_filtered_path, ready_flag_path)

            csv_paths_to_use = [temp_filtered_path]

        # ==================== Stage 3: Load dataset ====================
        # Blosc2 format is supported
        full_dataset = nnUNetDatasetCSVBlosc2(
            csv_paths=csv_paths_to_use,
            strides=self.configuration_manager.network_arch_init_kwargs.get('strides'),
            target_z_size=self.target_z_size,
            series_id_column=self.series_id_column
        )
        self.print_to_log_file(f"Loaded CSV dataset with {len(full_dataset)} samples")

        # Split 80/20 - ensure same case only appears in one dataset
        all_keys = list(full_dataset.dataset.keys())

        # ==================== Split based on CSV case_id ====================
        # Read series_id to case_id mapping from CSV to ensure all sequences from the same patient are in the same set
        #
        # ID format explanation:
        #   - series_id: Series-level ID, e.g., train_18073_a_1
        #   - case_id: Case-level ID, e.g., train_18073_a
        #   - balanced_key: Balanced key, e.g., train_18073_a_1_10787 (series_id + copy number)

        # Load CSV to get series_id to case_id mapping
        df_list = []
        for csv_path in csv_paths_to_use:
            df_list.append(pd.read_csv(csv_path))
        df_full = pd.concat(df_list, ignore_index=True) if len(df_list) > 1 else df_list[0]

        # Build series_id to case_id mapping
        series_to_case = {}
        if self.series_id_column in df_full.columns and self.case_id_column in df_full.columns:
            for _, row in df_full.iterrows():
                series_id = str(row[self.series_id_column])
                case_id = str(row[self.case_id_column])
                series_to_case[series_id] = case_id
            self.print_to_log_file(f"Loaded {len(series_to_case)} series_id to case_id mappings from CSV")
        else:
            raise ValueError(
                f"CSV must contain both '{self.series_id_column}' and '{self.case_id_column}' columns. "
                f"Available columns: {list(df_full.columns)}"
            )

        def extract_series_id_from_key(key: str) -> str:
            """Extract series_id from dataset key (handle balanced key format)"""
            if '_' not in key:
                return key
            # Try to extract the part after the last underscore
            parts = key.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                potential_series_id = parts[0]
                if potential_series_id in series_to_case:
                    return potential_series_id
            return key

        # Build case_id to all keys mapping
        case_to_keys = {}
        for key in all_keys:
            series_id = extract_series_id_from_key(key)
            case_id = series_to_case.get(series_id, series_id)  # If mapping not found, use series_id as case_id
            if case_id not in case_to_keys:
                case_to_keys[case_id] = []
            case_to_keys[case_id].append(key)

        # Get all unique case IDs
        unique_cases_list = list(case_to_keys.keys())

        # Handle fold == "all": use all data for both training and validation (same as nnUNetTrainer)
        if self.fold == "all":
            train_cases = set(unique_cases_list)
            val_cases = train_cases  # Same as train
            train_keys = all_keys
            val_keys = train_keys  # Same as train
            self.print_to_log_file(f"Fold 'all': Using all {len(unique_cases_list)} cases ({len(all_keys)} samples) for both training and validation")
        else:
            random.seed(42)
            random.shuffle(unique_cases_list)

            # Split cases into train/val (80/20)
            train_case_size = int(0.8 * len(unique_cases_list))
            train_cases = set(unique_cases_list[:train_case_size])
            val_cases = set(unique_cases_list[train_case_size:])

            # Assign all keys belonging to each case to the appropriate dataset
            train_keys = []
            val_keys = []

            for case_id, keys in case_to_keys.items():
                if case_id in train_cases:
                    train_keys.extend(keys)
                else:
                    val_keys.extend(keys)

            self.print_to_log_file(f"Split: {len(unique_cases_list)} unique cases -> {len(train_cases)} train cases ({len(train_keys)} samples), {len(val_cases)} val cases ({len(val_keys)} samples)")

        # Save training case count for early stopping logic
        self.num_train_cases = len(train_cases)

        train_dict = {k: full_dataset.dataset[k] for k in train_keys}
        val_dict = {k: full_dataset.dataset[k] for k in val_keys}

        # Blosc2 format is supported
        DatasetClass = nnUNetDatasetCSVBlosc2

        dataset_tr = DatasetClass.__new__(DatasetClass)
        dataset_tr.dataset = train_dict
        dataset_tr.csv_paths = csv_paths_to_use
        dataset_tr.strides = full_dataset.strides
        dataset_tr.cum_stride_z = full_dataset.cum_stride_z
        dataset_tr.cum_stride_y = full_dataset.cum_stride_y
        dataset_tr.cum_stride_x = full_dataset.cum_stride_x
        dataset_tr.target_z_size = full_dataset.target_z_size

        dataset_val = DatasetClass.__new__(DatasetClass)
        dataset_val.dataset = val_dict
        dataset_val.csv_paths = csv_paths_to_use
        dataset_val.strides = full_dataset.strides
        dataset_val.cum_stride_z = full_dataset.cum_stride_z
        dataset_val.cum_stride_y = full_dataset.cum_stride_y
        dataset_val.cum_stride_x = full_dataset.cum_stride_x
        dataset_val.target_z_size = full_dataset.target_z_size

        # Save split info
        split_info = {
            'fold': self.fold,
            'total_samples': len(all_keys),
            'total_unique_cases': len(unique_cases_list),
            'train_cases': len(train_cases),
            'val_cases': len(val_cases),
            'train_samples': len(train_keys),
            'val_samples': len(val_keys),
            'train_keys': train_keys,
            'val_keys': val_keys,
            'train_case_ids': sorted(list(train_cases)),
            'val_case_ids': sorted(list(val_cases)),
            'balanced': self.enable_balancing,
            'csv_paths_used': csv_paths_to_use
        }

        # Only rank 0 saves split info to avoid file conflicts
        if self.local_rank == 0:
            save_json(split_info, join(self.output_folder, 'dataset_split.json'))

        return dataset_tr, dataset_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim != 3:
            raise NotImplementedError("Only 3D supported")

        # Use patch_size instead of initial_patch_size
        # Because this Trainer disables spatial transforms (rotation, scaling), no extra space needed
        patch_size = self.configuration_manager.patch_size

        dl_tr = nnUNetDataLoader3DWithGlobalClsReportgenCSV(
            dataset_tr, self.batch_size, patch_size, patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            csv_paths=self.csv_paths,
            series_id_column=self.series_id_column,
            cls_columns=self.cls_columns,
            report_column=self.report_column
        )
        dl_val = nnUNetDataLoader3DWithGlobalClsReportgenCSV(
            dataset_val, self.batch_size, patch_size, patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            csv_paths=self.csv_paths,
            series_id_column=self.series_id_column,
            cls_columns=self.cls_columns,
            report_column=self.report_column
        )
        return dl_tr, dl_val

    def on_validation_epoch_start(self):
        """Reset validation batch index at the start of each validation epoch"""
        super().on_validation_epoch_start()
        self._val_step_idx = 0
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        properties = batch['properties']
        keys = batch['keys']

        validation_dict = {}

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Prepare inputs based on mode
        cls_task_list = None
        if self.mode in ['only_cls', 'both']:
            cls_all = torch.from_numpy(batch['cls_all']).float().to(self.device, non_blocking=True)
            cls_task_list = [cls_all]

        # Prepare report text tokens if available (like v1, for validation)
        report_text_ids = None
        report_text_attention_mask = None
        report_texts = None
        if 'report_texts' in batch and hasattr(self, 'report_tokenizer') and self.report_tokenizer is not None:
            report_texts = batch['report_texts']
            # Use unified method to format report text (same format for validation)
            formatted_texts = [self.format_report_text_for_training(report_text) for report_text in report_texts]

            # Tokenize report texts
            tokenized = self.report_tokenizer(
                formatted_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.report_max_length
            )
            report_text_ids = tokenized['input_ids'].to(self.device)
            report_text_attention_mask = tokenized['attention_mask'].to(self.device)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            network_output = self.network(
                data,
                generate_report=False,
                report_text_ids=report_text_ids,
                report_text_attention_mask=report_text_attention_mask
            )

        report_output, cls_pred_list = network_output

        total_loss = torch.tensor(0.0, device=self.device)
        total_cls_loss = torch.tensor(0.0, device=self.device)
        total_report_loss = torch.tensor(0.0, device=self.device)

        # Classification loss (only_cls or both mode)
        if self.mode in ['only_cls', 'both']:
            for t_index in range(self.num_cls_task):
                cls_pred_logits = cls_pred_list[t_index]
                cls_target = cls_task_list[t_index]

                cls_loss = self.cls_loss_list[t_index](cls_pred_logits, cls_target)
                total_loss += cls_loss.mean()
                total_cls_loss += cls_loss.mean()

                cls_probs = torch.sigmoid(cls_pred_logits)

                # Convert to float32 before numpy (BFloat16 not supported)
                validation_dict[f'cls_task_{t_index}_probs'] = cls_probs.detach().float().cpu().numpy()
                validation_dict[f'cls_task_{t_index}_targets'] = cls_target.detach().float().cpu().numpy()
                validation_dict[f'cls_task_{t_index}_loss'] = cls_loss.mean().detach().float().cpu().numpy()

        # Report generation loss (only_report or both mode)
        if self.mode in ['only_report', 'both']:
            if report_output is not None and report_text_ids is not None:
                llm_logits = report_output.get('llm_logits')
                text_start_idx = report_output.get('text_start_idx')

                if llm_logits is not None and text_start_idx is not None:
                    text_len = report_text_ids.shape[1]
                    shift_logits = llm_logits[:, text_start_idx:text_start_idx + text_len - 1, :]
                    shift_labels = report_text_ids[:, 1:text_len]

                    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.reshape(-1)

                    mask = shift_labels != self.report_tokenizer.pad_token_id
                    if mask.any():
                        valid_logits = shift_logits[mask]
                        valid_labels = shift_labels[mask]
                        report_loss = self.report_loss_fn(valid_logits, valid_labels)
                        total_loss += self.report_loss_weight * report_loss
                        total_report_loss = report_loss

                    if report_texts is not None:
                        validation_dict['report_texts'] = report_texts

        # Normalize loss by number of active tasks (same as train_step)
        num_active_tasks = 0
        if self.mode in ['only_cls', 'both']:
            num_active_tasks += self.num_cls_task
        if self.mode in ['only_report', 'both']:
            num_active_tasks += 1

        if num_active_tasks > 0:
            l = total_loss / num_active_tasks
        else:
            l = total_loss

        validation_dict['loss'] = l.detach().float().cpu().numpy()

        # Return results based on mode (same as train_step)
        if self.mode in ['only_cls', 'both']:
            validation_dict['total_cls_loss'] = total_cls_loss.detach().float().cpu().numpy()
        if self.mode in ['only_report', 'both']:
            validation_dict['total_report_loss'] = total_report_loss.detach().float().cpu().numpy()

        validation_dict['keys'] = keys

        tp_hard = np.zeros(self.num_cls_task) if self.num_cls_task > 0 else np.array([])
        fp_hard = np.zeros(self.num_cls_task) if self.num_cls_task > 0 else np.array([])
        fn_hard = np.zeros(self.num_cls_task) if self.num_cls_task > 0 else np.array([])

        validation_dict['tp_hard'] = tp_hard
        validation_dict['fp_hard'] = fp_hard
        validation_dict['fn_hard'] = fn_hard

        # Generate report examples for validation
        generated_reports = None
        ground_truth_reports = None
        if self._val_step_idx < self.num_report_examples_per_epoch and self.mode in ['only_report', 'both']:
            with torch.no_grad():
                first_sample_data = data[0:1]
                generated_reports = self.network(
                    first_sample_data,
                    generate_report=True,
                    report_prompt=self.report_prompt,
                    max_new_tokens=self.report_max_new_tokens,
                    skip_special_tokens=False  # Keep special tokens during training validation
                )[0]  # First element is the generated reports

            if report_texts is not None and len(report_texts) > 0:
                ground_truth_reports = report_texts[0] if isinstance(report_texts[0], str) else str(report_texts[0])

        validation_dict['generated_reports'] = generated_reports
        validation_dict['generated_report_keys'] = [keys[0]] if len(keys) > 0 and generated_reports is not None else None
        validation_dict['ground_truth_reports'] = ground_truth_reports

        # Print report example if generated
        if generated_reports is not None:
            key = keys[0] if len(keys) > 0 else f"sample_{self._val_step_idx}"
            generated_text = generated_reports[0] if isinstance(generated_reports, list) else str(generated_reports)
            ground_truth_text = ground_truth_reports if ground_truth_reports else "N/A"

            self.print_to_log_file("=" * 80)
            self.print_to_log_file(f"Report Example {self._val_step_idx + 1} (Epoch {self.current_epoch}) - Key: {key}")
            self.print_to_log_file("=" * 80)
            self.print_to_log_file("Generated Report:")
            self.print_to_log_file("-" * 80)
            self.print_to_log_file(generated_text)
            self.print_to_log_file("-" * 80)
            self.print_to_log_file("Ground Truth Report:")
            self.print_to_log_file("-" * 80)
            self.print_to_log_file(ground_truth_text if ground_truth_text != "N/A" else "Ground truth report not available")
            self.print_to_log_file("-" * 80)
            self.print_to_log_file("=" * 80)

        self._val_step_idx += 1
        return validation_dict

    def on_train_start(self):
        """Override to add network shape tracing before training."""
        super().on_train_start()
        self.trace_network_shapes(save_to_file=True)

    def trace_network_shapes(self, save_to_file: bool = True):
        """
        Trace shape changes at major modules based on current mode.
        - only_cls: only trace encoder and cls_head
        - only_report: only trace encoder, vision_projection, and llm
        - both: trace all components
        """
        if self.local_rank != 0:
            return

        self.print_to_log_file("")
        self.print_to_log_file("=" * 70)
        self.print_to_log_file(f"NETWORK SHAPE TRACE (mode: {self.mode})")
        self.print_to_log_file("=" * 70)

        network = self.network
        if hasattr(network, 'module'):
            network = network.module
        if isinstance(network, OptimizedModule):
            network = network._orig_mod

        # Base mode: encoder always needs tracing
        base_patterns = [
            r'^encoder\.stem$',
            r'^encoder\.stages\.\d+$',
        ]

        # Add additional tracing patterns based on mode
        if self.mode in ['only_cls', 'both']:
            base_patterns.append(r'^cls_head_list\.\d+$')

        if self.mode in ['only_report', 'both']:
            base_patterns.extend([
                r'^pool_embed_list\.\d+$',
                r'^vision_projection_list\.\d+$',
                r'^llm_report_gen$',
                r'^llm_report_gen\.layers\.\d+$',
                r'^llm_report_gen\.embed_tokens$',
                r'^llm_report_gen\.lm_head$',
                r'^llm_report_gen\.norm$',
            ])

        patterns = [re.compile(p) for p in base_patterns]

        shape_trace = []
        traced_names = set()

        def make_hook(name):
            def hook(module, input, output):
                if name in traced_names:
                    return
                traced_names.add(name)
                in_shape = next((list(x.shape) for x in (input if isinstance(input, tuple) else [input]) if isinstance(x, torch.Tensor)), None)
                out_shape = list(output.shape) if isinstance(output, torch.Tensor) else \
                           next((list(x.shape) for x in output if isinstance(x, torch.Tensor)), None) if isinstance(output, tuple) else None
                shape_trace.append({'name': name, 'in': in_shape, 'out': out_shape})
            return hook

        # Register hooks temporarily
        hooks = [module.register_forward_hook(make_hook(name))
                 for name, module in network.named_modules()
                 if name and any(p.match(name) for p in patterns)]

        # Single forward pass for tracing
        batch = next(iter(self.dataloader_train))
        data = batch['data'].to(self.device)
        self.print_to_log_file(f"Input: {list(data.shape)}")

        report_text_ids, report_text_attention_mask = None, None
        if self.mode in ['only_report', 'both']:
            if hasattr(self, 'report_tokenizer') and self.report_tokenizer and batch.get('report_texts'):
                formatted = [self.format_report_text_for_training(t) for t in batch['report_texts']]
                tok = self.report_tokenizer(formatted, return_tensors='pt', padding=True, truncation=True,
                                            max_length=getattr(self, 'report_max_length', 512))
                report_text_ids = tok['input_ids'].to(self.device)
                report_text_attention_mask = tok['attention_mask'].to(self.device)

        network.eval()
        with torch.no_grad():
            if report_text_ids is not None:
                network(data, generate_report=False, report_text_ids=report_text_ids,
                       report_text_attention_mask=report_text_attention_mask)
            else:
                network(data)

        # Remove hooks immediately - no impact on subsequent training
        for h in hooks:
            h.remove()
        network.train()

        # Output results
        self.print_to_log_file("-" * 70)
        for t in shape_trace:
            self.print_to_log_file(f"{t['name']:40s} | {str(t['in']):25s} -> {str(t['out'])}")
        self.print_to_log_file("=" * 70)

    def on_train_epoch_start(self):
        """Reset training batch index at the start of each training epoch"""
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')

        # Print learning rates for parameter groups with proper names (skip "Other" group)
        group_names = {
            0: "Encoder",
            1: "Patch Embed",
            2: "Vision Proj",
            3: "LLM"
        }

        for i, group in enumerate(self.optimizer.param_groups):
            if i in group_names:  # Skip "Other" group (index 4)
                lr = group['lr']
                group_name = group_names[i]
                self.print_to_log_file(
                    f"Current learning rate ({group_name}): {np.round(lr, decimals=10)}")
                # Log the first group (Encoder) to maintain backward compatibility
                if i == 0:
                    self.logger.log('lrs', lr, self.current_epoch)

        self._train_step_idx = 0

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()

            total_cls_loss_here = 0.0
            if self.mode in ['only_cls', 'both']:
                total_cls_train_losses_tr = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(total_cls_train_losses_tr, outputs['total_cls_loss'])
                total_cls_loss_here = np.vstack(total_cls_train_losses_tr).mean()

            total_report_loss_here = 0.0
            if self.mode in ['only_report', 'both']:
                total_report_train_losses_tr = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(total_report_train_losses_tr, outputs['total_report_loss'])
                total_report_loss_here = np.vstack(total_report_train_losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

            total_cls_loss_here = 0.0
            if self.mode in ['only_cls', 'both']:
                total_cls_loss_here = np.mean(outputs['total_cls_loss'])

            total_report_loss_here = 0.0
            if self.mode in ['only_report', 'both']:
                total_report_loss_here = np.mean(outputs['total_report_loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

        if self.mode in ['only_cls', 'both']:
            if 'total_cls_train_losses' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['total_cls_train_losses'] = list()
            self.logger.log('total_cls_train_losses', total_cls_loss_here, self.current_epoch)

        if self.mode in ['only_report', 'both']:
            if 'total_report_train_losses' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['total_report_train_losses'] = list()
            self.logger.log('total_report_train_losses', total_report_loss_here, self.current_epoch)

    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # Extract generated_reports, ground_truth_reports and generated_report_keys before collation
        # since they may contain strings/None which collate_outputs doesn't handle well
        generated_reports_list = []
        ground_truth_reports_list = []
        generated_report_keys_list = []
        for output in val_outputs:
            if 'generated_reports' in output and output['generated_reports'] is not None:
                generated_reports_list.append(output['generated_reports'])
            if 'ground_truth_reports' in output and output['ground_truth_reports'] is not None:
                ground_truth_reports_list.append(output['ground_truth_reports'])
            if 'generated_report_keys' in output and output['generated_report_keys'] is not None:
                if isinstance(output['generated_report_keys'], list):
                    generated_report_keys_list.extend(output['generated_report_keys'])
                else:
                    generated_report_keys_list.append(output['generated_report_keys'])

        # Handle DDP: gather generated reports from all processes
        if self.is_ddp:
            world_size = dist.get_world_size()
            all_generated_reports = [[] for _ in range(world_size)]
            all_ground_truth_reports = [[] for _ in range(world_size)]
            all_generated_keys = [[] for _ in range(world_size)]
            dist.all_gather_object(all_generated_reports, generated_reports_list)
            dist.all_gather_object(all_ground_truth_reports, ground_truth_reports_list)
            dist.all_gather_object(all_generated_keys, generated_report_keys_list)
            generated_reports_list = sum(all_generated_reports, [])
            ground_truth_reports_list = sum(all_ground_truth_reports, [])
            generated_report_keys_list = sum(all_generated_keys, [])

        # Remove these keys temporarily for collate_outputs
        val_outputs_for_collate = []
        for output in val_outputs:
            output_copy = {k: v for k, v in output.items()
                          if k not in ['generated_reports', 'ground_truth_reports', 'generated_report_keys']}
            val_outputs_for_collate.append(output_copy)

        outputs_collated = collate_outputs(val_outputs_for_collate)
        keys = outputs_collated['keys']

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            # Only gather classification losses if in classification mode
            total_cls_losses_here = 0.0
            if self.mode in ['only_cls', 'both']:
                total_cls_losses_val = [None for _ in range(world_size)]
                dist.all_gather_object(total_cls_losses_val, outputs_collated['total_cls_loss'])
                total_cls_losses_here = np.vstack(total_cls_losses_val).mean()

            # Only gather report losses if in report mode
            total_report_losses_here = 0.0
            if self.mode in ['only_report', 'both']:
                total_report_losses_val = [None for _ in range(world_size)]
                dist.all_gather_object(total_report_losses_val, outputs_collated['total_report_loss'])
                total_report_losses_here = np.vstack(total_report_losses_val).mean()

            cls_losses_mean = []
            if self.mode in ['only_cls', 'both']:
                for t_index in range(self.num_cls_task):
                    cls_task_losses = [None for _ in range(world_size)]
                    dist.all_gather_object(cls_task_losses, [output[f'cls_task_{t_index}_loss'] for output in val_outputs if f'cls_task_{t_index}_loss' in output])
                    cls_losses_mean.append(np.mean([np.mean(losses) for losses in cls_task_losses if losses is not None]))
        else:
            loss_here = np.mean(outputs_collated['loss'])

            # Only get classification losses if in classification mode
            total_cls_losses_here = 0.0
            if self.mode in ['only_cls', 'both']:
                total_cls_losses_here = np.mean(outputs_collated['total_cls_loss'])

            # Only get report losses if in report mode
            total_report_losses_here = 0.0
            if self.mode in ['only_report', 'both']:
                total_report_losses_here = np.mean(outputs_collated['total_report_loss'])

            cls_losses_mean = []
            if self.mode in ['only_cls', 'both']:
                cls_losses_mean = [np.mean([output[f'cls_task_{t_index}_loss'] for output in val_outputs if f'cls_task_{t_index}_loss' in output])
                                for t_index in range(self.num_cls_task)]

        # Initialize checkpoint_auc
        checkpoint_auc = 0.5  # Default value for modes without classification

        if self.mode in ['only_cls', 'both']:
            for t_index in range(self.num_cls_task):
                self.logger.log(f'cls_task_{t_index}_loss', cls_losses_mean[t_index], self.current_epoch)

                cls_probs_list = [output[f'cls_task_{t_index}_probs'] for output in val_outputs if f'cls_task_{t_index}_probs' in output]
                cls_targets_list = [output[f'cls_task_{t_index}_targets'] for output in val_outputs if f'cls_task_{t_index}_targets' in output]

                if self.is_ddp:
                    world_size = dist.get_world_size()
                    all_cls_probs = [[] for _ in range(world_size)]
                    all_cls_targets = [[] for _ in range(world_size)]
                    dist.all_gather_object(all_cls_probs, cls_probs_list)
                    dist.all_gather_object(all_cls_targets, cls_targets_list)
                    cls_probs = np.concatenate(sum(all_cls_probs, []))
                    cls_targets = np.concatenate(sum(all_cls_targets, []))
                else:
                    cls_probs = np.concatenate(cls_probs_list)
                    cls_targets = np.concatenate(cls_targets_list)

                if cls_targets.ndim == 1:
                    cls_targets = cls_targets.reshape(-1, 1)
                    cls_probs = cls_probs.reshape(-1, 1)

                auc_list = []
                for i in range(cls_probs.shape[1]):
                    if len(np.unique(cls_targets[:, i])) > 1:
                        auc_list.append(roc_auc_score(cls_targets[:, i], cls_probs[:, i]))
                    else:
                        auc_list.append(0.5)

                auc = np.mean(auc_list)
                if t_index == self.checkpoint_t_index:
                    checkpoint_auc = auc

                cls_preds = (cls_probs > 0.5).astype(int)
                acc = accuracy_score(cls_targets.flatten(), cls_preds.flatten())

                self.logger.log(f'cls_task_{t_index}_acc', acc, self.current_epoch)
                self.logger.log(f'cls_task_{t_index}_auc', auc, self.current_epoch)

        # Log metrics based on mode
        self.logger.log('mean_auc', checkpoint_auc, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        if self.mode in ['only_cls', 'both']:
            if 'total_cls_val_losses' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['total_cls_val_losses'] = list()
            self.logger.log('total_cls_val_losses', total_cls_losses_here, self.current_epoch)

        if self.mode in ['only_report', 'both']:
            if 'total_report_val_losses' not in self.logger.my_fantastic_logging:
                self.logger.my_fantastic_logging['total_report_val_losses'] = list()
            self.logger.log('total_report_val_losses', total_report_losses_here, self.current_epoch)
    
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))

        if self.mode in ['only_cls', 'both']:
            self.print_to_log_file('total_cls_train_losses', np.round(self.logger.my_fantastic_logging['total_cls_train_losses'][-1], decimals=4))
            self.print_to_log_file('total_cls_val_losses', np.round(self.logger.my_fantastic_logging['total_cls_val_losses'][-1], decimals=4))

        if self.mode in ['only_report', 'both']:
            self.print_to_log_file('total_report_train_losses', np.round(self.logger.my_fantastic_logging['total_report_train_losses'][-1], decimals=4))
            self.print_to_log_file('total_report_val_losses', np.round(self.logger.my_fantastic_logging['total_report_val_losses'][-1], decimals=4))

        if self.mode in ['only_cls', 'both']:
            for task_i in range(self.num_cls_task):
                self.print_to_log_file(f'cls_task_{task_i}_loss', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_loss'][-1], decimals=4))
                self.print_to_log_file(f'cls_task_{task_i}_acc', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_acc'][-1], decimals=4))
                self.print_to_log_file(f'cls_task_{task_i}_auc', np.round(self.logger.my_fantastic_logging[f'cls_task_{task_i}_auc'][-1], decimals=4))

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_epoch = self.current_epoch

        # Only save latest and best checkpoints
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest'))

        # Early stopping logic: stop when train_loss and val_loss gap > 0.15 (prevent overfitting)
        # Only enable early stopping check when both train_loss and val_loss are below 0.95, avoid false triggers in early training
        # And epoch count must be greater than num_train_cases*8/(250*batch_size)
        train_loss = self.logger.my_fantastic_logging['train_losses'][-1]
        val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
        loss_gap = train_loss - val_loss
        early_stopping_threshold = 0.95  # Only check early stopping when both losses are below this threshold

        # Calculate minimum epoch requirement
        if hasattr(self, 'num_train_cases') and self.num_train_cases is not None:
            min_epochs = self.num_train_cases * 8 / (250 * self.batch_size)
        else:
            min_epochs = 0  # If not set, no minimum epoch limit

        if train_loss < early_stopping_threshold and val_loss < early_stopping_threshold:
            if abs(loss_gap) > 0.15:  # Check gap first (core condition)
                if self.current_epoch >= min_epochs:
                    # Trigger early stopping
                    self.print_to_log_file(f"Early stopping triggered! Epoch {self.current_epoch} >= min_epochs {min_epochs:.1f}, train_loss ({np.round(train_loss, decimals=4)}) - val_loss ({np.round(val_loss, decimals=4)}) = {np.round(loss_gap, decimals=4)}, gap > 0.15")
                    # Save checkpoint_final.pth when early stopping (consistent with normal training end)
                    self.save_checkpoint(join(self.output_folder, 'checkpoint_final'))
                    # Delete checkpoint_latest.pth (consistent with on_train_end behavior)
                    if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
                        os.remove(join(self.output_folder, "checkpoint_latest.pth"))
                    self.print_to_log_file("Training done (early stopped).")

                    # Properly clean up data loaders to avoid core dump from improperly terminated multiprocessing
                    if hasattr(self, 'dataloader_train') and self.dataloader_train is not None:
                        if hasattr(self.dataloader_train, '_finish'):
                            self.dataloader_train._finish()
                    if hasattr(self, 'dataloader_val') and self.dataloader_val is not None:
                        if hasattr(self.dataloader_val, '_finish'):
                            self.dataloader_val._finish()

                    self.current_epoch += 1
                    os._exit(0)  # Use os._exit() instead of exit() for more thorough termination, avoiding conflicts with subprocesses
                else:
                    # Gap condition met but epoch not enough
                    self.print_to_log_file(f"Early stopping condition met but epoch {self.current_epoch} < min_epochs {min_epochs:.1f}, continuing training...")

        # Best model saving logic
        if self.mode == 'only_cls':
            # only_cls mode: save best model based on ema_auc (higher AUC is better)
            current_ema_auc = self.logger.my_fantastic_logging['ema_auc'][-1]
            if self._best_ema is None or current_ema_auc > self._best_ema:
                self._best_ema = current_ema_auc
                self.print_to_log_file(f"Yayy! New best EMA AUC: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best'))
        else:
            # Other modes: save best model based on val_loss
            if self._best_ema is None or val_loss < self._best_ema:
                self._best_ema = val_loss
                self.print_to_log_file(f"Yayy! New best validation loss: {np.round(self._best_ema, decimals=4)}")
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best'))

        # Enhanced Gradient Monitoring (summary every 5 epochs)
        if self.grad_monitoring_enabled and self.grad_history and self.current_epoch % 5 == 0:
            self._log_gradient_summary_cls()

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def _compute_component_gradient_stats_cls(self, grad_norms: dict, param_norms: dict):
        """Compute gradient statistics for different network components in ResEncoderUVLM classification mode."""
        stats = {}

        # Initialize component dictionaries
        encoder_blocks = {}
        decoder_blocks = {}
        bottleneck_blocks = {}
        cls_head_stats = {}
        seg_head_stats = {}
        stem_stats = {}
        patch_embed_stats = {}
        vision_proj_stats = {}
        llm_stats = {}
        llm_layers = {}  # For per-layer LLM statistics
        other_stats = {}

        for name, grad_norm in grad_norms.items():
            param_norm = param_norms.get(name, 0)

            # Stem/Initial convolution
            if any(keyword in name.lower() for keyword in ['stem', 'initial', 'conv_blocks.0']):
                stem_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

            # Encoder blocks - more comprehensive detection (encoder.stages.*)
            # Note: encoder.stem.* already handled in stem detection above
            elif any(pattern in name for pattern in ['encoder.stages', 'downsample', 'down_conv']):
                # Extract stage/block index using multiple patterns
                stage_idx = None

                # Pattern 1: encoder.stages.X (new naming)
                stage_match = re.search(r'encoder\.stages\.(\d+)', name)
                if stage_match:
                    stage_idx = int(stage_match.group(1))
                else:
                    # Pattern 2: Look for any digit sequence that might indicate stage
                    digits = re.findall(r'\d+', name)
                    if digits:
                        # Use the first digit found (usually indicates the main stage)
                        stage_idx = int(digits[0])

                if stage_idx is not None:
                    if stage_idx not in encoder_blocks:
                        encoder_blocks[stage_idx] = {'grad_norms': [], 'param_norms': []}
                    encoder_blocks[stage_idx]['grad_norms'].append(grad_norm)
                    encoder_blocks[stage_idx]['param_norms'].append(param_norm)
                else:
                    # If no stage found, put in other
                    other_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

            # Decoder blocks - more comprehensive detection
            elif any(pattern in name for pattern in ['conv_decoder_blocks', 'decoder.', 'upsample', 'up_conv', 'transpconv']):
                # Extract stage/block index
                stage_idx = None

                # Pattern 1: conv_decoder_blocks.X or decoder.X
                stage_match = re.search(r'(?:conv_decoder_blocks|decoder)\.(\d+)', name)
                if stage_match:
                    stage_idx = int(stage_match.group(1))
                else:
                    # Pattern 2: Look for any digit sequence
                    digits = re.findall(r'\d+', name)
                    if digits:
                        stage_idx = int(digits[0])

                if stage_idx is not None:
                    if stage_idx not in decoder_blocks:
                        decoder_blocks[stage_idx] = {'grad_norms': [], 'param_norms': []}
                    decoder_blocks[stage_idx]['grad_norms'].append(grad_norm)
                    decoder_blocks[stage_idx]['param_norms'].append(param_norm)
                else:
                    other_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

            # Bottleneck blocks (between encoder and decoder)
            elif any(pattern in name for pattern in ['bottleneck', 'bridge', 'center']):
                if 0 not in bottleneck_blocks:
                    bottleneck_blocks[0] = {'grad_norms': [], 'param_norms': []}
                bottleneck_blocks[0]['grad_norms'].append(grad_norm)
                bottleneck_blocks[0]['param_norms'].append(param_norm)

            # Classification head - expanded patterns
            elif any(pattern in name.lower() for pattern in ['cls_head', 'classifier', 'classification', 'cls.', 'class_head']):
                cls_head_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

            # Segmentation head - detect seg_layers
            elif any(pattern in name.lower() for pattern in ['seg_layers', 'seg_layer', 'segmentation']):
                seg_head_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

            # Vision components specific to ResEncoderUVLM
            elif 'patch_embed' in name or 'pool_embed' in name:
                patch_embed_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}
            elif 'vision_proj' in name or 'vision_projection' in name:
                vision_proj_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}
            elif 'llm_report_gen' in name or 'llm_model' in name:
                llm_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

            else:
                other_stats[name] = {'grad_norm': grad_norm, 'param_norm': param_norm}

        # Aggregate stem statistics
        if stem_stats:
            grad_norms_stem = [v['grad_norm'] for v in stem_stats.values()]
            param_norms_stem = [v['param_norm'] for v in stem_stats.values()]
            stats['stem'] = {
                'grad_mean': np.mean(grad_norms_stem),
                'grad_std': np.std(grad_norms_stem),
                'param_mean': np.mean(param_norms_stem),
                'param_count': len(stem_stats)
            }

        # Aggregate encoder block statistics
        if encoder_blocks:
            stats['encoder_stages'] = {}
            for stage_idx, stage_data in encoder_blocks.items():
                grad_norms_stage = stage_data['grad_norms']
                param_norms_stage = stage_data['param_norms']
                stats['encoder_stages'][stage_idx] = {
                    'grad_mean': np.mean(grad_norms_stage),
                    'grad_std': np.std(grad_norms_stage),
                    'grad_max': np.max(grad_norms_stage),
                    'grad_min': np.min(grad_norms_stage),
                    'param_mean': np.mean(param_norms_stage),
                    'param_count': len(grad_norms_stage)
                }

        # Aggregate bottleneck statistics
        if bottleneck_blocks:
            stats['bottleneck'] = {}
            for stage_idx, stage_data in bottleneck_blocks.items():
                grad_norms_stage = stage_data['grad_norms']
                param_norms_stage = stage_data['param_norms']
                stats['bottleneck'][stage_idx] = {
                    'grad_mean': np.mean(grad_norms_stage),
                    'grad_std': np.std(grad_norms_stage),
                    'grad_max': np.max(grad_norms_stage),
                    'grad_min': np.min(grad_norms_stage),
                    'param_mean': np.mean(param_norms_stage),
                    'param_count': len(grad_norms_stage)
                }

        # Aggregate decoder block statistics
        if decoder_blocks:
            stats['decoder_stages'] = {}
            for stage_idx, stage_data in decoder_blocks.items():
                grad_norms_stage = stage_data['grad_norms']
                param_norms_stage = stage_data['param_norms']
                stats['decoder_stages'][stage_idx] = {
                    'grad_mean': np.mean(grad_norms_stage),
                    'grad_std': np.std(grad_norms_stage),
                    'grad_max': np.max(grad_norms_stage),
                    'grad_min': np.min(grad_norms_stage),
                    'param_mean': np.mean(param_norms_stage),
                    'param_count': len(grad_norms_stage)
                }

        # Classification head statistics
        if cls_head_stats:
            grad_norms_cls = [v['grad_norm'] for v in cls_head_stats.values()]
            param_norms_cls = [v['param_norm'] for v in cls_head_stats.values()]
            stats['cls_head'] = {
                'grad_mean': np.mean(grad_norms_cls),
                'grad_std': np.std(grad_norms_cls),
                'param_mean': np.mean(param_norms_cls),
                'param_count': len(cls_head_stats)
            }

        # Segmentation head statistics
        if seg_head_stats:
            grad_norms_seg = [v['grad_norm'] for v in seg_head_stats.values()]
            param_norms_seg = [v['param_norm'] for v in seg_head_stats.values()]
            stats['seg_head'] = {
                'grad_mean': np.mean(grad_norms_seg),
                'grad_std': np.std(grad_norms_seg),
                'param_mean': np.mean(param_norms_seg),
                'param_count': len(seg_head_stats)
            }

        # Vision components specific to ResEncoderUVLM - separate monitoring for different stages
        if patch_embed_stats:
            # Group patch embedding by stages (from patch_embed_list.X or pool_embed_list.X format)
            patch_embed_stages = {}
            for name, stats_dict in patch_embed_stats.items():
                stage_idx = None
                # Extract stage index from names like patch_embed_list.0.xxx, pool_embed_list.0.xxx, or patch_embed.0.xxx
                match = re.search(r'(?:patch_embed_list|pool_embed_list)\.(\d+)', name)
                if match:
                    stage_idx = int(match.group(1))
                else:
                    match = re.search(r'(?:patch_embed|pool_embed)\.(\d+)', name)
                    if match:
                        stage_idx = int(match.group(1))

                if stage_idx is not None:
                    # Adjust stage index for deepstack mode (add deepstack_skip_stages offset)
                    if hasattr(self, 'use_deepstack') and self.use_deepstack and hasattr(self, 'deepstack_skip_stages'):
                        stage_idx += self.deepstack_skip_stages

                    if stage_idx not in patch_embed_stages:
                        patch_embed_stages[stage_idx] = {'grad_norms': [], 'param_norms': []}
                    patch_embed_stages[stage_idx]['grad_norms'].append(stats_dict['grad_norm'])
                    patch_embed_stages[stage_idx]['param_norms'].append(stats_dict['param_norm'])

            # Aggregate per-stage statistics
            if patch_embed_stages:
                stats['patch_embed_stages'] = {}
                for stage_idx in sorted(patch_embed_stages.keys()):
                    stage_data = patch_embed_stages[stage_idx]
                    grad_norms = stage_data['grad_norms']
                    param_norms = stage_data['param_norms']
                    stats['patch_embed_stages'][stage_idx] = {
                        'grad_mean': np.mean(grad_norms),
                        'grad_std': np.std(grad_norms),
                        'grad_max': np.max(grad_norms),
                        'grad_min': np.min(grad_norms),
                        'param_mean': np.mean(param_norms),
                        'param_count': len(grad_norms)
                    }

            # Overall patch embedding statistics (for backward compatibility)
            grad_norms_patch = [v['grad_norm'] for v in patch_embed_stats.values()]
            param_norms_patch = [v['param_norm'] for v in patch_embed_stats.values()]
            stats['patch_embed'] = {
                'grad_mean': np.mean(grad_norms_patch),
                'grad_std': np.std(grad_norms_patch),
                'param_mean': np.mean(param_norms_patch),
                'param_count': len(patch_embed_stats)
            }

        if vision_proj_stats:
            # Group vision projection by stages (from vision_projection_list.X format)
            vision_proj_stages = {}
            for name, stats_dict in vision_proj_stats.items():
                stage_idx = None
                # Extract stage index from names like vision_projection_list.0.xxx
                match = re.search(r'vision_projection_list\.(\d+)', name)
                if match:
                    stage_idx = int(match.group(1))
                else:
                    match = re.search(r'vision_proj\.(\d+)', name)
                    if match:
                        stage_idx = int(match.group(1))

                if stage_idx is not None:
                    # Adjust stage index for deepstack mode (add deepstack_skip_stages offset)
                    if hasattr(self, 'use_deepstack') and self.use_deepstack and hasattr(self, 'deepstack_skip_stages'):
                        stage_idx += self.deepstack_skip_stages

                    if stage_idx not in vision_proj_stages:
                        vision_proj_stages[stage_idx] = {'grad_norms': [], 'param_norms': []}
                    vision_proj_stages[stage_idx]['grad_norms'].append(stats_dict['grad_norm'])
                    vision_proj_stages[stage_idx]['param_norms'].append(stats_dict['param_norm'])

            # Aggregate per-stage statistics
            if vision_proj_stages:
                stats['vision_proj_stages'] = {}
                for stage_idx in sorted(vision_proj_stages.keys()):
                    stage_data = vision_proj_stages[stage_idx]
                    grad_norms = stage_data['grad_norms']
                    param_norms = stage_data['param_norms']
                    stats['vision_proj_stages'][stage_idx] = {
                        'grad_mean': np.mean(grad_norms),
                        'grad_std': np.std(grad_norms),
                        'grad_max': np.max(grad_norms),
                        'grad_min': np.min(grad_norms),
                        'param_mean': np.mean(param_norms),
                        'param_count': len(grad_norms)
                    }

            # Overall vision projection statistics (for backward compatibility)
            grad_norms_vision = [v['grad_norm'] for v in vision_proj_stats.values()]
            param_norms_vision = [v['param_norm'] for v in vision_proj_stats.values()]
            stats['vision_proj'] = {
                'grad_mean': np.mean(grad_norms_vision),
                'grad_std': np.std(grad_norms_vision),
                'param_mean': np.mean(param_norms_vision),
                'param_count': len(vision_proj_stats)
            }

        # LLM components - per layer statistics
        if llm_stats:
            # Group LLM parameters by layer
            llm_layers = {}
            llm_other = {}

            for name, stats_dict in llm_stats.items():
                # Extract layer index from patterns like: layers.0.xxx, layers.1.xxx, etc.
                layer_match = re.search(r'layers\.(\d+)', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx not in llm_layers:
                        llm_layers[layer_idx] = {'grad_norms': [], 'param_norms': []}
                    llm_layers[layer_idx]['grad_norms'].append(stats_dict['grad_norm'])
                    llm_layers[layer_idx]['param_norms'].append(stats_dict['param_norm'])
                else:
                    llm_other[name] = stats_dict

            # Store per-layer statistics
            if llm_layers:
                stats['llm_layers'] = {}
                for layer_idx in sorted(llm_layers.keys()):
                    layer_data = llm_layers[layer_idx]
                    grad_norms = layer_data['grad_norms']
                    param_norms = layer_data['param_norms']
                    stats['llm_layers'][layer_idx] = {
                        'grad_mean': np.mean(grad_norms),
                        'grad_std': np.std(grad_norms),
                        'grad_max': np.max(grad_norms),
                        'grad_min': np.min(grad_norms),
                        'param_mean': np.mean(param_norms),
                        'param_count': len(grad_norms)
                    }

            # Store overall LLM statistics (including non-layer parameters)
            grad_norms_llm = [v['grad_norm'] for v in llm_stats.values()]
            param_norms_llm = [v['param_norm'] for v in llm_stats.values()]
            stats['llm'] = {
                'grad_mean': np.mean(grad_norms_llm),
                'grad_std': np.std(grad_norms_llm),
                'grad_max': np.max(grad_norms_llm),
                'grad_min': np.min(grad_norms_llm),
                'param_mean': np.mean(param_norms_llm),
                'param_count': len(llm_stats)
            }

        # Other components
        if other_stats:
            grad_norms_other = [v['grad_norm'] for v in other_stats.values()]
            param_norms_other = [v['param_norm'] for v in other_stats.values()]
            stats['other_components'] = {
                'grad_mean': np.mean(grad_norms_other),
                'grad_std': np.std(grad_norms_other),
                'param_mean': np.mean(param_norms_other),
                'param_count': len(other_stats)
            }

        return stats

    def _store_gradient_stats_cls(self, grad_norms: dict, param_norms: dict):
        """Store gradient and parameter statistics for monitoring."""
        if not grad_norms:
            return

        # Calculate comprehensive gradient statistics
        grad_values = list(grad_norms.values())
        param_values = list(param_norms.values()) if param_norms else []

        stats = {
            'epoch': self.current_epoch,
            'step_idx': self._train_step_idx,
            'grad_norms': grad_norms.copy(),
            'param_norms': param_norms.copy() if param_norms else {},
            'grad_stats': {
                'mean': np.mean(grad_values),
                'std': np.std(grad_values),
                'max': np.max(grad_values),
                'min': np.min(grad_values),
                'total_params': len(grad_values)
            }
        }

        if param_values:
            stats['param_stats'] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'max': np.max(param_values),
                'min': np.min(param_values)
            }

        # Store component-specific statistics for classification mode
        component_stats = self._compute_component_gradient_stats_cls(grad_norms, param_norms)
        stats.update(component_stats)

        self.grad_history.append(stats)

        # Keep only last 50 batches to avoid memory issues
        if len(self.grad_history) > 50:
            self.grad_history = self.grad_history[-50:]

        # Log detailed stats every 50 batches
        if self._train_step_idx % 50 == 0:
            self._log_gradient_stats_cls(stats)

    def _log_gradient_stats_cls(self, stats: dict):
        """Log detailed gradient statistics for ResEncoderUVLM classification/report generation mode."""
        # Determine title based on mode
        if self.mode == 'only_cls':
            title = "ResEncoderUVLM Classification Gradient Stats"
        elif self.mode == 'only_report':
            title = "ResEncoderUVLM Report Generation Gradient Stats"
        else:  # both
            title = "ResEncoderUVLM Gradient Stats"

        self.print_to_log_file(f"{title} - Epoch {stats['epoch']}, Step {stats['step_idx']}")

        # Overall statistics
        grad_stats = stats['grad_stats']
        self.print_to_log_file(f"  Overall: mean={grad_stats['mean']:.6f}, std={grad_stats['std']:.6f}, "
                              f"max={grad_stats['max']:.6f}, min={grad_stats['min']:.6f} "
                              f"(params: {grad_stats['total_params']})")

        # Stem/Initial convolution
        if 'stem' in stats:
            stem_stats = stats['stem']
            self.print_to_log_file(f"  Stem: grad_mean={stem_stats['grad_mean']:.6f}, "
                                  f"params={stem_stats['param_count']}")

        # Encoder stages
        if 'encoder_stages' in stats:
            self.print_to_log_file("  Encoder Stages:")
            for stage_idx in sorted(stats['encoder_stages'].keys()):
                stage_stats = stats['encoder_stages'][stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: grad_mean={stage_stats['grad_mean']:.6f}, "
                                      f"grad_std={stage_stats['grad_std']:.6f}, "
                                      f"params={stage_stats['param_count']}")

        # Bottleneck
        if 'bottleneck' in stats:
            self.print_to_log_file("  Bottleneck:")
            for stage_idx in sorted(stats['bottleneck'].keys()):
                stage_stats = stats['bottleneck'][stage_idx]
                self.print_to_log_file(f"    Bottleneck {stage_idx}: grad_mean={stage_stats['grad_mean']:.6f}, "
                                      f"grad_std={stage_stats['grad_std']:.6f}, "
                                      f"params={stage_stats['param_count']}")

        # Decoder stages
        if 'decoder_stages' in stats:
            self.print_to_log_file("  Decoder Stages:")
            for stage_idx in sorted(stats['decoder_stages'].keys()):
                stage_stats = stats['decoder_stages'][stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: grad_mean={stage_stats['grad_mean']:.6f}, "
                                      f"grad_std={stage_stats['grad_std']:.6f}, "
                                      f"params={stage_stats['param_count']}")

        # Classification head
        if 'cls_head' in stats:
            cls_stats = stats['cls_head']
            self.print_to_log_file(f"  Classification Head: grad_mean={cls_stats['grad_mean']:.6f}, "
                                  f"params={cls_stats['param_count']}")

        # Segmentation head
        if 'seg_head' in stats:
            seg_stats = stats['seg_head']
            self.print_to_log_file(f"  Segmentation Head: grad_mean={seg_stats['grad_mean']:.6f}, "
                                  f"params={seg_stats['param_count']}")

        # Vision components specific to ResEncoderUVLM
        if 'patch_embed_stages' in stats:
            self.print_to_log_file("  Patch Embed Stages:")
            for stage_idx in sorted(stats['patch_embed_stages'].keys()):
                stage_stats = stats['patch_embed_stages'][stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: grad_mean={stage_stats['grad_mean']:.6f}, "
                                      f"grad_std={stage_stats['grad_std']:.6f}, "
                                      f"params={stage_stats['param_count']}")
        elif 'patch_embed' in stats:
            patch_stats = stats['patch_embed']
            self.print_to_log_file(f"  Patch Embed: grad_mean={patch_stats['grad_mean']:.6f}, "
                                  f"params={patch_stats['param_count']}")

        if 'vision_proj_stages' in stats:
            self.print_to_log_file("  Vision Proj Stages:")
            for stage_idx in sorted(stats['vision_proj_stages'].keys()):
                stage_stats = stats['vision_proj_stages'][stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: grad_mean={stage_stats['grad_mean']:.6f}, "
                                      f"grad_std={stage_stats['grad_std']:.6f}, "
                                      f"params={stage_stats['param_count']}")
        elif 'vision_proj' in stats:
            vision_stats = stats['vision_proj']
            self.print_to_log_file(f"  Vision Proj: grad_mean={vision_stats['grad_mean']:.6f}, "
                                  f"params={vision_stats['param_count']}")

        # LLM components
        if 'llm_layers' in stats:
            self.print_to_log_file("  LLM Layers:")
            for layer_idx in sorted(stats['llm_layers'].keys()):
                layer_stats = stats['llm_layers'][layer_idx]
                self.print_to_log_file(f"    Layer {layer_idx}: grad_mean={layer_stats['grad_mean']:.6f}, "
                                      f"grad_std={layer_stats['grad_std']:.6f}, "
                                      f"params={layer_stats['param_count']}")
        elif 'llm' in stats:
            llm_stats = stats['llm']
            self.print_to_log_file(f"  LLM: grad_mean={llm_stats['grad_mean']:.6f}, "
                                  f"grad_std={llm_stats['grad_std']:.6f}, "
                                  f"params={llm_stats['param_count']}")

        # Other components
        if 'other_components' in stats:
            other_stats = stats['other_components']
            self.print_to_log_file(f"  Other Components: grad_mean={other_stats['grad_mean']:.6f}, "
                                  f"params={other_stats['param_count']}")

        # Check for potential issues
        if grad_stats['max'] > 10.0:
            self.print_to_log_file(f"  WARNING: Very large gradients detected (max: {grad_stats['max']:.6f})")
        elif grad_stats['max'] < 1e-6:
            self.print_to_log_file(f"  WARNING: Very small gradients detected (max: {grad_stats['max']:.6f})")

    def _log_gradient_summary_cls(self):
        """Log comprehensive gradient monitoring summary for ResEncoderUVLM classification mode."""
        if not self.grad_history:
            return

        # Get statistics from the last few batches of this epoch
        recent_stats = [s for s in self.grad_history if s['epoch'] == self.current_epoch]
        if not recent_stats:
            return

        self.print_to_log_file(f"ResEncoderUVLM Classification Gradient Monitoring Summary - Epoch {self.current_epoch}")

        # Overall statistics across recent batches
        all_grad_means = [s['grad_stats']['mean'] for s in recent_stats]
        all_grad_stds = [s['grad_stats']['std'] for s in recent_stats]
        all_grad_maxs = [s['grad_stats']['max'] for s in recent_stats]

        self.print_to_log_file(f"  Overall Gradients (last {len(recent_stats)} batches):")
        self.print_to_log_file(f"    Mean: {np.mean(all_grad_means):.6f} ± {np.std(all_grad_means):.6f}")
        self.print_to_log_file(f"    Std:  {np.mean(all_grad_stds):.6f} ± {np.std(all_grad_stds):.6f}")
        self.print_to_log_file(f"    Max:  {np.max(all_grad_maxs):.6f}")

        # Encoder stages analysis (most important for classification)
        encoder_stage_data = {}
        for stats in recent_stats:
            if 'encoder_stages' in stats:
                for stage_idx, stage_stats in stats['encoder_stages'].items():
                    if stage_idx not in encoder_stage_data:
                        encoder_stage_data[stage_idx] = []
                    encoder_stage_data[stage_idx].append(stage_stats['grad_mean'])

        if encoder_stage_data:
            self.print_to_log_file("  Encoder Stages:")
            for stage_idx in sorted(encoder_stage_data.keys()):
                stage_grads = encoder_stage_data[stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: {np.mean(stage_grads):.6f} ± {np.std(stage_grads):.6f}")

        # Vision components (patch_embed and vision_proj) - per stage monitoring
        vision_components = {}
        patch_embed_stages_data = {}
        vision_proj_stages_data = {}

        for stats in recent_stats:
            # Overall vision components
            for comp_name in ['patch_embed', 'vision_proj']:
                if comp_name in stats:
                    if comp_name not in vision_components:
                        vision_components[comp_name] = []
                    vision_components[comp_name].append(stats[comp_name]['grad_mean'])

            # Per-stage vision components
            if 'patch_embed_stages' in stats:
                for stage_idx, stage_stats in stats['patch_embed_stages'].items():
                    if stage_idx not in patch_embed_stages_data:
                        patch_embed_stages_data[stage_idx] = []
                    patch_embed_stages_data[stage_idx].append(stage_stats['grad_mean'])

            if 'vision_proj_stages' in stats:
                for stage_idx, stage_stats in stats['vision_proj_stages'].items():
                    if stage_idx not in vision_proj_stages_data:
                        vision_proj_stages_data[stage_idx] = []
                    vision_proj_stages_data[stage_idx].append(stage_stats['grad_mean'])

        if patch_embed_stages_data:
            self.print_to_log_file("  Patch Embed Stages:")
            for stage_idx in sorted(patch_embed_stages_data.keys()):
                stage_grads = patch_embed_stages_data[stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: {np.mean(stage_grads):.6f} ± {np.std(stage_grads):.6f}")

        if vision_proj_stages_data:
            self.print_to_log_file("  Vision Proj Stages:")
            for stage_idx in sorted(vision_proj_stages_data.keys()):
                stage_grads = vision_proj_stages_data[stage_idx]
                self.print_to_log_file(f"    Stage {stage_idx}: {np.mean(stage_grads):.6f} ± {np.std(stage_grads):.6f}")

        # Overall vision components (if no per-stage data available)
        if vision_components and not (patch_embed_stages_data or vision_proj_stages_data):
            self.print_to_log_file("  Vision Components:")
            for comp_name, grads in vision_components.items():
                comp_display_name = "Patch Embed" if comp_name == "patch_embed" else "Vision Proj"
                self.print_to_log_file(f"    {comp_display_name}: {np.mean(grads):.6f} ± {np.std(grads):.6f}")

        # Classification head
        cls_grads = []
        for stats in recent_stats:
            if 'cls_head' in stats:
                cls_grads.append(stats['cls_head']['grad_mean'])

        if cls_grads:
            self.print_to_log_file("  Classification Head:")
            self.print_to_log_file(f"    Gradients: {np.mean(cls_grads):.6f} ± {np.std(cls_grads):.6f}")

        # Segmentation head
        seg_grads = []
        for stats in recent_stats:
            if 'seg_head' in stats:
                seg_grads.append(stats['seg_head']['grad_mean'])

        if seg_grads:
            self.print_to_log_file("  Segmentation Head:")
            self.print_to_log_file(f"    Gradients: {np.mean(seg_grads):.6f} ± {np.std(seg_grads):.6f}")

        # LLM components
        llm_grads = []
        for stats in recent_stats:
            if 'llm' in stats:
                llm_grads.append(stats['llm']['grad_mean'])

        if llm_grads:
            self.print_to_log_file("  LLM Components:")
            self.print_to_log_file(f"    LLM: {np.mean(llm_grads):.6f} ± {np.std(llm_grads):.6f}")

        # Other components
        other_grads = []
        for stats in recent_stats:
            if 'other_components' in stats:
                other_grads.append(stats['other_components']['grad_mean'])

        if other_grads:
            self.print_to_log_file("  Other Components:")
            self.print_to_log_file(f"    Gradients: {np.mean(other_grads):.6f} ± {np.std(other_grads):.6f}")

        # Check for gradient health issues
        avg_max_grad = np.mean(all_grad_maxs)
        if avg_max_grad > 10.0:
            self.print_to_log_file(f"  WARNING: Large gradients detected (avg max: {avg_max_grad:.6f})")
        elif avg_max_grad < 1e-6:
            self.print_to_log_file(f"  WARNING: Very small gradients detected (avg max: {avg_max_grad:.6f})")

        # Gradient stability analysis
        if len(all_grad_means) > 1:
            grad_stability = np.std(all_grad_means) / (np.mean(all_grad_means) + 1e-8)
            self.print_to_log_file(f"  Gradient Stability: {grad_stability:.4f} (lower is better)")

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save checkpoint

        Args:
            checkpoint_path: checkpoint path (without .pth suffix)
                Example: "checkpoint_best" or "checkpoint_latest"

        File structure:
            checkpoint_best.pth       # Single file checkpoint
        """
        if self.disable_checkpointing:
            return

        # Prepare checkpoint data
        init_args_with_network_kwargs = self.my_init_kwargs.copy()
        init_args_with_network_kwargs['network_arch_init_kwargs'] = dict(
            self.configuration_manager.network_arch_init_kwargs
        )

        checkpoint_data = {
            'logging': self.logger.get_checkpoint(),
            '_best_ema': self._best_ema,
            'current_epoch': self.current_epoch + 1,
            'init_args': init_args_with_network_kwargs,
            'trainer_name': self.__class__.__name__,
            'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
        }

        # Get actual network module (remove DDP wrapper)
        if self.is_ddp:
            network_module = self.network.module
        else:
            network_module = self.network

        num_params = sum(p.numel() for p in network_module.parameters())

        # Only rank 0 saves in DDP mode, or always save in single GPU mode
        if not self.is_ddp or self.local_rank == 0:
            # Add network state to checkpoint data
            checkpoint_data['network_weights'] = network_module.state_dict()

            # Add optimizer and scheduler state
            if self.optimizer is not None:
                checkpoint_data['optimizer_state'] = self.optimizer.state_dict()
            if self.lr_scheduler is not None:
                checkpoint_data['lr_scheduler_state'] = self.lr_scheduler.state_dict()
            if self.grad_scaler is not None:
                checkpoint_data['grad_scaler_state'] = self.grad_scaler.state_dict()

            # Save as single .pth file (avoid double suffix)
            if checkpoint_path.endswith('.pth'):
                checkpoint_file = checkpoint_path
            else:
                checkpoint_file = f"{checkpoint_path}.pth"
            torch.save(checkpoint_data, checkpoint_file)

            self.print_to_log_file(f"")
            self.print_to_log_file(f"Checkpoint saved: {checkpoint_file}")
            self.print_to_log_file(f"   Epoch: {self.current_epoch}, Parameters: {num_params/1e6:.2f}M")

        # Synchronize across ranks in DDP mode
        if self.is_ddp and dist.is_initialized():
            dist.barrier()

    def load_checkpoint(self, checkpoint_path: Optional[str] = None,
                       load_optimizer: bool = True) -> None:
        """
        Load checkpoint

        Args:
            checkpoint_path: checkpoint file path. If None, automatically select the latest checkpoint
                Example: "checkpoint_best.pth" or "checkpoint_best" (automatically add .pth)
            load_optimizer: whether to load optimizer state

        Note:
            LR scheduler state will not be loaded, scheduler will automatically calculate learning rate based on current_epoch and initial_lr in plan.
            This allows using new learning rate when resuming training after modifying initial_lr in plan.
        """
        if not self.was_initialized:
            self.initialize()

        # Automatically select checkpoint path
        if checkpoint_path is None:
            candidates = [
                os.path.join(self.output_folder, 'checkpoint_latest.pth'),
                os.path.join(self.output_folder, 'checkpoint_latest'),
                os.path.join(self.output_folder, 'checkpoint_best.pth'),
                os.path.join(self.output_folder, 'checkpoint_best'),
            ]

            for candidate_path in candidates:
                if os.path.exists(candidate_path):
                    checkpoint_path = candidate_path
                    self.print_to_log_file(f"Auto-selected checkpoint: {checkpoint_path}")
                    break
            else:
                self.print_to_log_file("No checkpoint found (latest or best)")
                return

        # Determine checkpoint file path
        if checkpoint_path.endswith('.pth'):
            checkpoint_file = checkpoint_path
        else:
            checkpoint_file = f"{checkpoint_path}.pth"

        # Check if file exists
        if not os.path.exists(checkpoint_file):
            self.print_to_log_file(f"Checkpoint file not found: {checkpoint_file}")
            return

        self.print_to_log_file(f"")
        self.print_to_log_file(f"Loading checkpoint from {checkpoint_file}")

        # Load checkpoint data
        # PyTorch 2.6+ requires weights_only=False for full checkpoint loading
        checkpoint_data = torch.load(checkpoint_file, map_location=self.device, weights_only=False)

        # Get actual network module (remove DDP wrapper)
        network_module = self.network.module if self.is_ddp else self.network

        # Load network weights (filter unnecessary parameters based on mode)
        if 'network_weights' in checkpoint_data:
            checkpoint_weights = checkpoint_data['network_weights']
            model_state_dict = network_module.state_dict()

            # Find parameters to load (only load parameters that exist in current model and have matching shapes)
            filtered_weights = {}
            skipped_keys = []
            for key, value in checkpoint_weights.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_weights[key] = value
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: ckpt {value.shape} vs model {model_state_dict[key].shape})")
                else:
                    skipped_keys.append(key)

            if skipped_keys:
                self.print_to_log_file(f"   Skipped {len(skipped_keys)} keys from checkpoint (not in current mode '{self.mode}'):")
                for key in skipped_keys[:5]:
                    self.print_to_log_file(f"      - {key}")
                if len(skipped_keys) > 5:
                    self.print_to_log_file(f"      ... and {len(skipped_keys) - 5} more")

            # Load filtered weights (strict=True ensures complete match)
            network_module.load_state_dict(filtered_weights, strict=True)
            self.print_to_log_file(f"   Network weights loaded ({len(filtered_weights)}/{len(checkpoint_weights)} keys)")
        else:
            self.print_to_log_file(f"   Warning: No network weights found in checkpoint")

        # Load optimizer state
        if load_optimizer and 'optimizer_state' in checkpoint_data:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
                self.print_to_log_file(f"   Optimizer state loaded")

        # Don't load scheduler state, let scheduler automatically calculate learning rate based on current_epoch and initial_lr in plan
        # optimizer.load_state_dict will restore old lr and initial_lr, need to overwrite with new values from plan
        if self.optimizer is not None:
            # Get lr_multiplier configuration
            encoder_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('encoder_lr_multiplier', 1.0)
            patch_embed_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('patch_embed_lr_multiplier', 1.0)
            vision_proj_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('vision_proj_lr_multiplier', 1.0)
            llm_lr_multiplier = self.configuration_manager.network_arch_init_kwargs.get('llm_lr_multiplier', 1.0)

            # Parameter group order: Encoder, Patch Embed, Vision Proj, LLM, Other
            lr_multipliers = [encoder_lr_multiplier, patch_embed_lr_multiplier, vision_proj_lr_multiplier, llm_lr_multiplier, 1.0]

            for i, group in enumerate(self.optimizer.param_groups):
                multiplier = lr_multipliers[i] if i < len(lr_multipliers) else 1.0
                new_initial_lr = self.initial_lr * multiplier
                group['initial_lr'] = new_initial_lr
                group['lr'] = new_initial_lr  # Will be recalculated by scheduler.step()

            self.print_to_log_file(f"   LR reset to plan values (initial_lr={self.initial_lr}, will recalculate based on current_epoch)")

        # Load grad scaler state
        if 'grad_scaler_state' in checkpoint_data:
            if self.grad_scaler is not None:
                self.grad_scaler.load_state_dict(checkpoint_data['grad_scaler_state'])

        # Restore training state
        if 'logging' in checkpoint_data:
            self.logger.load_checkpoint(checkpoint_data['logging'])
        if '_best_ema' in checkpoint_data:
            self._best_ema = checkpoint_data['_best_ema']
        if 'current_epoch' in checkpoint_data:
            self.current_epoch = checkpoint_data['current_epoch']
        if 'init_args' in checkpoint_data:
            self.my_init_kwargs = checkpoint_data['init_args']
        if 'inference_allowed_mirroring_axes' in checkpoint_data:
            self.inference_allowed_mirroring_axes = checkpoint_data['inference_allowed_mirroring_axes']

        self.print_to_log_file(f"Checkpoint loaded successfully")
        self.print_to_log_file(f"   Resuming from epoch: {self.current_epoch}")

        if self.is_ddp and dist.is_initialized():
            current_world_size = dist.get_world_size()
            self.print_to_log_file(f"   Current GPUs: {current_world_size}")
            # Synchronize across ranks
            dist.barrier()

    def on_train_end(self):
        """Cleanup work at the end of training"""
        # Call parent's on_train_end
        super().on_train_end()

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Skip validation set validation step.

        Because this Trainer uses CSV datasets and doesn't depend on nnUNet's preprocessed data directory structure,
        validation will be performed on a separate test set through inference scripts.
        """
        self.print_to_log_file("Skipping perform_actual_validation (will evaluate on separate test set)")
        return