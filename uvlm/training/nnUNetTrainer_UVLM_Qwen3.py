"""
nnUNetTrainer_UVLM_Qwen3: Trainer using Qwen3-4B as the LLM

Inherits from nnUNetTrainer_UVLM, only replacing the LLM-related parts.
Supports two modes:
1. Full parameter fine-tuning (use_lora=False)
2. LoRA fine-tuning (use_lora=True)

Other components (Encoder, data loading, training pipeline, etc.) remain unchanged.
"""

import os
import pydoc
import inspect
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn

# Internal modules of the UVLM project
from uvlm.training.nnUNetTrainer_UVLM import nnUNetTrainer_UVLM
from uvlm.networks.uvlm_qwen3 import UVLM_Qwen3


class nnUNetTrainer_UVLM_Qwen3(nnUNetTrainer_UVLM):
    """
    Trainer using Qwen3-4B as the LLM

    Inherits from nnUNetTrainer_UVLM, only modifying:
    1. build_network_architecture: Uses the Qwen3 network
    2. configure_optimizers: Supports LoRA parameter grouping
    3. Reads Qwen3-related configurations during initialization
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        # First call parent class initialization
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Qwen3 specific configurations
        self.use_lora = self.configuration_manager.network_arch_init_kwargs.get('use_lora', True)
        self.lora_r = self.configuration_manager.network_arch_init_kwargs.get('lora_r', 64)
        self.lora_alpha = self.configuration_manager.network_arch_init_kwargs.get('lora_alpha', 128)
        self.lora_dropout = self.configuration_manager.network_arch_init_kwargs.get('lora_dropout', 0.05)
        self.load_in_8bit = self.configuration_manager.network_arch_init_kwargs.get('load_in_8bit', False)
        self.load_in_4bit = self.configuration_manager.network_arch_init_kwargs.get('load_in_4bit', False)

        # LLM model path
        self.llm_model_path = self.configuration_manager.network_arch_init_kwargs.get(
            'llm_model_path', '/path/to/model/')

        self.print_to_log_file(f"Qwen3 Trainer initialized:")
        self.print_to_log_file(f"  - LLM model path: {self.llm_model_path}")
        self.print_to_log_file(f"  - use_lora: {self.use_lora}")
        if self.use_lora:
            self.print_to_log_file(f"  - lora_r: {self.lora_r}")
            self.print_to_log_file(f"  - lora_alpha: {self.lora_alpha}")
            self.print_to_log_file(f"  - lora_dropout: {self.lora_dropout}")
        self.print_to_log_file(f"  - load_in_8bit: {self.load_in_8bit}")
        self.print_to_log_file(f"  - load_in_4bit: {self.load_in_4bit}")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build the Qwen3 network architecture
        """
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # Ensure visual_token_length_source_stage is set
        if 'visual_token_length_source_stage' not in architecture_kwargs:
            architecture_kwargs['visual_token_length_source_stage'] = -1

        # Compute patch_kernel_sizes if missing
        if 'patch_kernel_sizes' not in architecture_kwargs:
            features_per_stage = architecture_kwargs.get('features_per_stage')
            strides = architecture_kwargs.get('strides')
            visual_token_length_source_stage = architecture_kwargs.get('visual_token_length_source_stage', -1)

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
            if int(os.environ.get('LOCAL_RANK', '0')) == 0:
                print(f"Computed patch_kernel_sizes: {patch_kernel_sizes}")

        # Filter kwargs to only include parameters accepted by UVLM_Qwen3
        sig = inspect.signature(UVLM_Qwen3.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in architecture_kwargs.items() if k in valid_params}

        filtered_out = set(architecture_kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out and int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print(f"Warning: Filtered out unsupported arguments for UVLM_Qwen3: {filtered_out}")

        network = UVLM_Qwen3(
            in_channels=num_input_channels,
            **filtered_kwargs
        )

        return network

    def configure_optimizers(self):
        """
        Configure optimizers, supporting LoRA parameter grouping

        Parameter groups:
        - encoder: Encoder parameters
        - vision_proj_outer: Outer vision_proj (320->512)
        - vision_proj_inner: Inner vision_proj (512->2560, inside llm_report_gen)
        - llm_lora: LoRA parameters
        - llm_other: Other trainable parameters of the LLM
        """
        # Get the actual network
        network = self._get_actual_network()

        # Collect parameters
        encoder_params = []
        cls_head_params = []
        vision_proj_outer_params = []  # Outer vision_proj (320->512)
        vision_proj_inner_params = []  # Inner vision_proj (512->2560)
        llm_lora_params = []
        llm_other_params = []
        other_params = []

        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith('encoder.'):
                encoder_params.append(param)
            elif 'cls_head' in name:
                cls_head_params.append(param)
            elif name.startswith('vision_proj.') or name.startswith('vision_norm.') or name.startswith('pool_'):
                # Outer vision_proj (not inside llm_report_gen)
                vision_proj_outer_params.append(param)
            elif 'llm_report_gen' in name:
                if 'lora' in name.lower():
                    llm_lora_params.append(param)
                elif 'vision_proj' in name or 'vision_norm' in name:
                    # Inner vision_proj (inside llm_report_gen)
                    vision_proj_inner_params.append(param)
                else:
                    llm_other_params.append(param)
            else:
                other_params.append(param)

        # Print parameter statistics
        encoder_info = self.get_param_info(encoder_params, "Encoder")
        cls_head_info = self.get_param_info(cls_head_params, "Classification Head")
        vision_proj_outer_info = self.get_param_info(vision_proj_outer_params, "Vision Proj Outer")
        vision_proj_inner_info = self.get_param_info(vision_proj_inner_params, "Vision Proj Inner")
        llm_lora_info = self.get_param_info(llm_lora_params, "LLM LoRA")
        llm_other_info = self.get_param_info(llm_other_params, "LLM Other")
        other_info = self.get_param_info(other_params, "Other")
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("Qwen3 Parameter Groups:")
        self.print_to_log_file(f"  Encoder: {encoder_info[0]} params ({encoder_info[1]:.2f}M)")
        self.print_to_log_file(f"  Classification Head: {cls_head_info[0]} params ({cls_head_info[1]:.2f}M)")
        self.print_to_log_file(f"  Vision Proj Outer (320->512): {vision_proj_outer_info[0]} params ({vision_proj_outer_info[1]:.2f}M)")
        self.print_to_log_file(f"  Vision Proj Inner (512->2560): {vision_proj_inner_info[0]} params ({vision_proj_inner_info[1]:.2f}M)")
        self.print_to_log_file(f"  LLM LoRA: {llm_lora_info[0]} params ({llm_lora_info[1]:.2f}M)")
        self.print_to_log_file(f"  LLM Other: {llm_other_info[0]} params ({llm_other_info[1]:.2f}M)")
        self.print_to_log_file(f"  Other: {other_info[0]} params ({other_info[1]:.2f}M)")
        self.print_to_log_file("=" * 60)

        # Build parameter groups
        param_groups = []

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.initial_lr,
                'name': 'encoder'
            })

        if cls_head_params:
            param_groups.append({
                'params': cls_head_params,
                'lr': self.initial_lr,
                'name': 'cls_head'
            })

        # Outer vision_proj uses standard learning rate
        if vision_proj_outer_params:
            param_groups.append({
                'params': vision_proj_outer_params,
                'lr': self.initial_lr,
                'name': 'vision_proj_outer'
            })

        # Inner vision_proj uses standard learning rate
        if vision_proj_inner_params:
            param_groups.append({
                'params': vision_proj_inner_params,
                'lr': self.initial_lr,
                'name': 'vision_proj_inner'
            })

        # LoRA parameters use the same learning rate (no longer 2x to avoid instability)
        if llm_lora_params:
            param_groups.append({
                'params': llm_lora_params,
                'lr': self.initial_lr,
                'name': 'llm_lora'
            })
            self.print_to_log_file(f"LoRA parameters LR: {self.initial_lr:.2e}")

        # Other trainable parameters in LLM
        if llm_other_params:
            param_groups.append({
                'params': llm_other_params,
                'lr': self.initial_lr,
                'name': 'llm_other'
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.initial_lr,
                'name': 'other'
            })

        # Create optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.initial_lr,
            betas=self.betas,
            eps=1e-8
        )
        self.print_to_log_file("Optimizer: AdamW")

        # Store initial LRs for the scheduler
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        # Use parent class LR scheduler
        from uvlm.training.lr_scheduler.multigroup_polylr import MultiGroupPolyLRScheduler
        lr_scheduler = MultiGroupPolyLRScheduler(optimizer, self.num_epochs)

        return optimizer, lr_scheduler

    def initialize(self):
        """
        Initialize the trainer

        Same as parent class, but uses Qwen3 network
        """
        # Call parent class initialization
        super().initialize()

        # Qwen3 uses bfloat16, no GradScaler needed
        # bfloat16 has sufficient dynamic range, no loss scaling required
        # Also, GradScaler does not support bfloat16 and will error:
        # "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
        # Disable GradScaler for both full-parameter and LoRA training
        self.grad_scaler = None
        self.print_to_log_file("GradScaler disabled for Qwen3 training (bfloat16)")

        # Print Qwen3-specific information
        if self.was_initialized:
            network = self._get_actual_network()

            # Count trainable parameters
            total_params = sum(p.numel() for p in network.parameters())
            trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

            self.print_to_log_file("=" * 60)
            self.print_to_log_file("Qwen3 Network Summary:")
            self.print_to_log_file(f"  Total parameters: {total_params / 1e6:.2f}M")
            self.print_to_log_file(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
            self.print_to_log_file(f"  Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
            self.print_to_log_file(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")
            self.print_to_log_file("=" * 60)

    def train_step(self, batch: dict) -> dict:
        """
        Training step

        Same as parent class; Qwen3 network interface is compatible with original network
        """
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        """
        Validation step

        Same as parent class; Qwen3 network interface is compatible with original network
        """
        return super().validation_step(batch)

    def _log_gradient_stats_cls(self, stats: dict):
        """
        Log detailed gradient statistics for Qwen3 mode.

        Same as parent class, but skips gradient printing for LLM Layers because Qwen3 uses a pre-trained LLM,
        and its gradient information is extensive and less meaningful compared to training from scratch.
        """
        # Determine title based on mode
        if self.mode == 'only_cls':
            title = "Qwen3 Classification Gradient Stats"
        elif self.mode == 'only_report':
            title = "Qwen3 Report Generation Gradient Stats"
        else:  # both
            title = "Qwen3 Gradient Stats"

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

        # Classification head
        if 'cls_head' in stats:
            cls_stats = stats['cls_head']
            self.print_to_log_file(f"  Cls Head: grad_mean={cls_stats['grad_mean']:.6f}, "
                                  f"params={cls_stats['param_count']}")

        # Vision projection
        if 'vision_proj' in stats:
            vision_stats = stats['vision_proj']
            self.print_to_log_file(f"  Vision Proj: grad_mean={vision_stats['grad_mean']:.6f}, "
                                  f"params={vision_stats['param_count']}")

        # Skip detailed printing for LLM Layers (Qwen3 uses pre-trained LLM, gradient info is extensive and less meaningful)
        if 'llm_layers' in stats:
            # Only print summary information
            total_llm_params = sum(layer['param_count'] for layer in stats['llm_layers'].values())
            grad_means = [layer['grad_mean'] for layer in stats['llm_layers'].values()]
            if grad_means:
                import numpy as np
                avg_grad_mean = np.mean(grad_means)
                self.print_to_log_file(f"  LLM Layers (Qwen3): {len(stats['llm_layers'])} layers, "
                                      f"avg_grad_mean={avg_grad_mean:.6f}, total_params={total_llm_params}")

        # Check for potential issues
        if grad_stats['max'] > 10.0:
            self.print_to_log_file(f"  WARNING: Very large gradients detected (max: {grad_stats['max']:.6f})")
        elif grad_stats['max'] < 1e-6:
            self.print_to_log_file(f"  WARNING: Very small gradients detected (max: {grad_stats['max']:.6f})")
