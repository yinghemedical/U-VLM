#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
U-VLM Training Entry Point

Similar to nnUNet_train but specifically for U-VLM trainers.
Searches for trainers only in uvlm.training directory.

Usage:
    uvlm_train Dataset156_CT_YH_Chest_852 3d_fullres 0 \\
        -tr nnUNetTrainer_UVLM \\
        -p ctyhchest852_cls_192x256x256_cls_seg_full_aug_fr
"""

import argparse
import os
import torch
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer_UVLM',
                          plans_identifier: str = 'nnUNetResEncUNetLPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda')):
    """
    Load U-VLM trainer from uvlm.training directory

    Args:
        dataset_name_or_id: Dataset name (e.g., Dataset156_CT_YH_Chest_852)
        configuration: Configuration name (e.g., 3d_fullres)
        fold: Fold number
        trainer_name: Trainer class name (default: nnUNetTrainer_UVLM)
        plans_identifier: Plans file name prefix
        use_compressed: Whether to use compressed data
        device: torch device

    Returns:
        Initialized trainer instance
    """
    # Load U-VLM trainer class from uvlm.training
    import uvlm
    uvlm_trainer = recursive_find_python_class(
        join(uvlm.__path__[0], "training"),
        trainer_name,
        'uvlm.training'
    )

    if uvlm_trainer is None:
        raise RuntimeError(
            f'Could not find requested U-VLM trainer {trainer_name} in '
            f'uvlm.training ({join(uvlm.__path__[0], "training")}). '
            f'Available trainers: nnUNetTrainer_UVLM, nnUNetTrainer_UVLM_Qwen3, nnUNetTrainer_ResEncoderUNet'
        )

    assert issubclass(uvlm_trainer, nnUNetTrainer), \
        f'The requested trainer class must inherit from nnUNetTrainer'

    # Handle dataset input
    if isinstance(dataset_name_or_id, str) and not dataset_name_or_id.startswith('Dataset'):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(
                f'dataset_name_or_id must either be an integer or a valid dataset name '
                f'with the pattern DatasetXXX_YYY where XXX are the three(!) task ID digits. '
                f'Your input: {dataset_name_or_id}'
            )

    # Initialize trainer
    preprocessed_dataset_folder_base = join(
        nnUNet_preprocessed,
        maybe_convert_to_dataset_name(dataset_name_or_id)
    )
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))

    uvlm_trainer = uvlm_trainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device
    )

    return uvlm_trainer


def run_training(dataset_name_or_id: Union[int, str],
                 configuration: str,
                 fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer_UVLM',
                 plans_identifier: str = 'nnUNetResEncUNetLPlans',
                 pretrained_weights: str = None,
                 pretrained_encoder_checkpoint_path: str = None,
                 num_gpus: int = 1,
                 use_compressed: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    """
    Run U-VLM training

    This is a simplified version that only supports single GPU training.
    For multi-GPU training, please use the nnUNet multi-GPU training script.
    """
    if num_gpus > 1:
        raise NotImplementedError(
            'Multi-GPU training is not supported in uvlm_train. '
            'Please use nnUNet_train with the registered U-VLM trainers.'
        )

    # Convert fold to int if needed
    if fold == 'all':
        raise NotImplementedError(
            'Training all folds is not supported in uvlm_train. '
            'Please train folds individually.'
        )
    fold = int(fold)

    # Get trainer
    nnunet_trainer = get_trainer_from_args(
        dataset_name_or_id,
        configuration,
        fold,
        trainer_class_name,
        plans_identifier,
        use_compressed,
        device
    )

    # Initialize trainer
    nnunet_trainer.initialize()

    # Load pretrained encoder checkpoint if specified (for progressive training)
    if pretrained_encoder_checkpoint_path is not None:
        nnunet_trainer.load_pretrained_encoder(pretrained_encoder_checkpoint_path)

    # Load checkpoint if needed
    # Follow nnUNet original implementation: prioritize checkpoint_final -> checkpoint_latest -> checkpoint_best
    if continue_training or only_run_validation:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print(f"WARNING: Cannot continue training because there seems to be no checkpoint available to "
                  f"continue from. Starting a new training...")
            expected_checkpoint_file = None

        if expected_checkpoint_file is not None:
            nnunet_trainer.load_checkpoint(expected_checkpoint_file)

    # Load pretrained weights if specified
    if pretrained_weights is not None:
        nnunet_trainer.load_checkpoint(pretrained_weights)

    # Run training or validation
    if not only_run_validation:
        nnunet_trainer.run_training()

    # Validation
    nnunet_trainer.perform_actual_validation(
        join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
    )

    if val_with_best:
        nnunet_trainer.perform_actual_validation(
            join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        )


def run_training_entry():
    """Entry point for uvlm_train command"""
    parser = argparse.ArgumentParser(
        description='U-VLM Training Entry Point. Searches for trainers in uvlm.training directory.'
    )
    parser.add_argument('dataset_name_or_id', type=str,
                        help='Dataset name (e.g., Dataset156_CT_YH_Chest_852) or ID')
    parser.add_argument('configuration', type=str,
                        help='Configuration (e.g., 3d_fullres)')
    parser.add_argument('fold', type=str,
                        help='Fold number (e.g., 0)')
    parser.add_argument('-tr', '--trainer_class_name', type=str, default='nnUNetTrainer_UVLM',
                        help='Trainer class name (default: nnUNetTrainer_UVLM)')
    parser.add_argument('-p', '--plans_identifier', type=str, default='nnUNetResEncUNetLPlans',
                        help='Plans identifier (default: nnUNetResEncUNetLPlans)')
    parser.add_argument('-pretrained_weights', type=str, default=None,
                        help='Path to pretrained checkpoint (full model weights)')
    parser.add_argument('--pretrained_encoder_checkpoint_path', type=str, default=None,
                        help='Path to pretrained encoder checkpoint (for progressive training)')
    parser.add_argument('--c', action='store_true',
                        help='Continue training from checkpoint')
    parser.add_argument('--val', action='store_true',
                        help='Only run validation')
    parser.add_argument('--val_best', action='store_true',
                        help='Validate with best checkpoint')
    parser.add_argument('--disable_checkpointing', action='store_true',
                        help='Disable checkpointing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("U-VLM Training")
    print("=" * 80)
    print(f"Dataset: {args.dataset_name_or_id}")
    print(f"Configuration: {args.configuration}")
    print(f"Fold: {args.fold}")
    print(f"Trainer: {args.trainer_class_name}")
    print(f"Plans: {args.plans_identifier}")
    print(f"Device: {device}")
    if args.pretrained_encoder_checkpoint_path:
        print(f"Pretrained Encoder: {args.pretrained_encoder_checkpoint_path}")
    print("=" * 80)
    print()

    run_training(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name=args.trainer_class_name,
        plans_identifier=args.plans_identifier,
        pretrained_weights=args.pretrained_weights,
        pretrained_encoder_checkpoint_path=args.pretrained_encoder_checkpoint_path,
        continue_training=args.c,
        only_run_validation=args.val,
        val_with_best=args.val_best,
        disable_checkpointing=args.disable_checkpointing,
        device=device
    )


if __name__ == '__main__':
    run_training_entry()
