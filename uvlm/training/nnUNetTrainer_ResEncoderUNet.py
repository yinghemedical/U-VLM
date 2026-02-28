"""
nnUNetTrainer_ResEncoderUNet_only_seg.py

ResEncoderUNet trainer for segmentation only task.
Data flow and encoder processing follows nnUNetTrainer_UVLM.
Segmentation loss and validation follows nnUNetTrainer.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydoc
import inspect

from time import time
from typing import Union, Tuple, List, Optional
from torch import autocast, nn
from torch._dynamo import OptimizedModule
from torch import distributed as dist
from multiprocessing import Pool
from tqdm import tqdm

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from uvlm.dataloading.dataset_csv_blosc2 import nnUNetDatasetCSVBlosc2
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import LimitedLenWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from uvlm.networks.res_encoder_unet import ResEncoderUNet
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch.nn.parallel import DistributedDataParallel as DDP

# ==================== Whole Image Data Loader ====================
class nnUNetDataLoaderWholeImage(DataLoader):
    """
    Data loader for whole image training (no cropping).
    Returns full images without any spatial cropping.
    Used when training_mode='whole'.
    """
    def __init__(self, data, batch_size: int, label_manager, sampling_probabilities=None):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        self.indices = list(data.keys())
        self.list_of_keys = list(self._data.keys())
        self.label_manager = label_manager

    def generate_train_batch(self):
        selected_keys = self.get_indices()

        # Load first case to determine shape
        data_0, seg_0, _ = self._data.load_case(selected_keys[0])

        # Preallocate memory
        data_shape = (self.batch_size, *data_0.shape)
        seg_shape = (self.batch_size, *seg_0.shape)

        data_all = np.zeros(data_shape, dtype=np.float32)
        seg_all = np.zeros(seg_shape, dtype=np.int16)
        case_properties = []

        for j, key in enumerate(selected_keys):
            data, seg, properties = self._data.load_case(key)
            case_properties.append(properties)

            data_all[j] = data
            seg_all[j] = seg

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetTrainer_ResEncoderUNet_only_seg(nnUNetTrainer):
    """
    ResEncoderUNet trainer for segmentation only.

    - Data flow and encoder: follows nnUNetTrainer_UVLM
    - Segmentation loss and validation: follows nnUNetTrainer
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        """Initialize the trainer."""
        # If the plan specifies a dataset_json_file, reload the corresponding dataset_json
        # This allows different scopes to use their own dataset_seg_xxx.json
        if 'dataset_json_file' in plans:
            from nnunetv2.paths import nnUNet_preprocessed
            from batchgenerators.utilities.file_and_folder_operations import load_json, join
            dataset_name = plans['dataset_name']
            dataset_json_file = plans['dataset_json_file']
            dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, dataset_json_file))

        # Call parent init
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.debug_save_input = False

        # Disable dataset unpacking since we read directly from Blosc2 files
        self.unpack_dataset = False

        # Disable cuDNN benchmark to avoid huge workspace allocation for large 3D convolutions
        torch.backends.cudnn.benchmark = False

        # Get arch_kwargs from configuration (where csv_paths etc. are stored in Plans JSON)
        arch_kwargs = self.configuration_manager.network_arch_init_kwargs

        # CSV paths for data loading (from arch_kwargs)
        self.csv_paths = arch_kwargs.get('csv_paths', [])
        self.identifier_column = arch_kwargs.get('identifier_column', 'case_id')

        # Data format: Blosc2 is supported
        self.data_format = 'blosc2'

        # Patch size settings
        self.patch_size = self.configuration_manager.patch_size
        self.target_z_size = arch_kwargs.get('target_z_size', self.patch_size[0])

        # Other settings from arch_kwargs
        self.training_mode = arch_kwargs.get('training_mode', 'patch')
        self.force_regenerate_shapes = arch_kwargs.get('force_regenerate_shapes', False)
        self.skip_fusion_mode = arch_kwargs.get('skip_fusion_mode', 'add')  # 'concat' or 'add'

        # Optimizer settings
        self.optimizer_type = arch_kwargs.get('optimizer_type', 'AdamW')
        if self.optimizer_type not in ['AdamW', 'SGD']:
            raise ValueError(f"Invalid optimizer_type: {self.optimizer_type}. Must be 'AdamW' or 'SGD'")

        # Segmentation CE class weights (from arch_kwargs in plans)
        # If not set in plans, all classes get weight 1.0
        # Example in plans: "seg_ce_class_weights": [1.0, 1.0, ..., 5.0, 6.0, ...]
        # Array length should match num_classes, index corresponds to class ID
        self.seg_ce_class_weights = arch_kwargs.get('seg_ce_class_weights', None)

        # Set default hyperparameters based on optimizer type
        if self.optimizer_type == 'SGD':
            default_lr = 1e-2
            default_weight_decay = 3e-5
            default_momentum = 0.99
            default_nesterov = True
            default_betas = (0.9, 0.999)  # not used for SGD
        else:  # AdamW
            default_lr = 2e-4
            default_weight_decay = 0.01
            default_momentum = 0.99  # not used for AdamW
            default_nesterov = True  # not used for AdamW
            default_betas = (0.9, 0.999)

        # Read hyperparameters from plan; use defaults if not present
        self.initial_lr = arch_kwargs.get('initial_lr', default_lr)
        self.weight_decay = arch_kwargs.get('weight_decay', default_weight_decay)
        self.momentum = arch_kwargs.get('momentum', default_momentum)
        self.nesterov = arch_kwargs.get('nesterov', default_nesterov)
        # betas for AdamW: (beta1, beta2)
        betas_from_plan = arch_kwargs.get('betas', None)
        if betas_from_plan is not None:
            self.betas = tuple(betas_from_plan)
        else:
            self.betas = default_betas

        self.print_to_log_file(f"Using CSV datasets: {self.csv_paths}")
        self.print_to_log_file(f"Using identifier column: {self.identifier_column}")
        self.print_to_log_file(f"Using patch_size: {self.patch_size}, target_z_size: {self.target_z_size}")
        self.print_to_log_file(f"Using data format: {self.data_format} (Blosc2 only)")
        self.print_to_log_file(f"Using skip_fusion_mode: {self.skip_fusion_mode}")
        self.print_to_log_file(f"Using training_mode: {self.training_mode}")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build ResEncoderUNet network architecture.
        """

        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        # Filter kwargs to only include parameters accepted by ResEncoderUNet
        sig = inspect.signature(ResEncoderUNet.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_kwargs = {k: v for k, v in architecture_kwargs.items() if k in valid_params}

        filtered_out = set(architecture_kwargs.keys()) - set(filtered_kwargs.keys())
        if filtered_out and int(os.environ.get('LOCAL_RANK', '0')) == 0:
            print(f"Filtered out unsupported arguments for ResEncoderUNet: {filtered_out}")

        network = ResEncoderUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **filtered_kwargs
        )

        return network

    def _load_single_csv_file(self, csv_path: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Load a single CSV file."""
        df = pd.read_csv(csv_path)
        return csv_path, df

    def get_tr_and_val_datasets(self):
        """
        Load CSV datasets and split. Following reportgen's approach.
        """
        if not self.csv_paths:
            raise ValueError("csv_paths must be provided for this trainer")

        # Load CSV files
        self.print_to_log_file(f"Loading {len(self.csv_paths)} CSV files...")
        all_dfs = []
        for csv_path in self.csv_paths:
            csv_path, df = self._load_single_csv_file(csv_path)
            if df is not None and not df.empty:
                all_dfs.append(df)
                self.print_to_log_file(f"Loaded {len(df)} samples from {csv_path}")

        if not all_dfs:
            raise ValueError("No valid CSV data found")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.print_to_log_file(f"Combined dataset has {len(combined_df)} samples")

        # Blosc2 format is supported
        DatasetClass = nnUNetDatasetCSVBlosc2

        # Get strides from configuration (following reportgen)
        strides = self.configuration_manager.network_arch_init_kwargs.get('strides')

        # Get all_labels from arch_kwargs for class_locations computation
        all_labels = self.configuration_manager.network_arch_init_kwargs.get('all_labels', None)

        # Create full dataset with training_mode and patch_size for patch-based loading
        full_dataset = DatasetClass(
            csv_paths=self.csv_paths,
            strides=strides,
            target_z_size=self.target_z_size,
            training_mode=self.training_mode,
            patch_size=tuple(self.patch_size) if self.training_mode == 'patch' else None,
            all_labels=all_labels,
        )

        # Split by case_id (following reportgen)
        all_keys = list(full_dataset.dataset.keys())
        case_to_keys = {}
        for key in all_keys:
            case_id = key.split('_stride_')[0] if '_stride_' in key else key
            if case_id not in case_to_keys:
                case_to_keys[case_id] = []
            case_to_keys[case_id].append(key)

        unique_cases_list = sorted(list(case_to_keys.keys()))

        # Handle fold == "all": use all data for both training and validation (same as nnUNetTrainer)
        if self.fold == "all":
            train_cases = set(unique_cases_list)
            val_cases = train_cases  # Same as train
            train_keys = all_keys
            val_keys = train_keys  # Same as train
            self.print_to_log_file(f"Fold 'all': Using all {len(unique_cases_list)} cases ({len(all_keys)} samples) for both training and validation")
        else:
            num_val = max(1, len(unique_cases_list) // 5)  # 20% for validation

            # Use fold for reproducible split
            np.random.seed(self.fold)
            np.random.shuffle(unique_cases_list)

            val_cases = set(unique_cases_list[:num_val])
            train_cases = set(unique_cases_list[num_val:])

            train_keys = []
            val_keys = []
            for case_id, keys in case_to_keys.items():
                if case_id in train_cases:
                    train_keys.extend(keys)
                else:
                    val_keys.extend(keys)

            self.print_to_log_file(f"Split: {len(unique_cases_list)} unique cases -> {len(train_cases)} train ({len(train_keys)} samples), {len(val_cases)} val ({len(val_keys)} samples)")

        train_dict = {k: full_dataset.dataset[k] for k in train_keys}
        val_dict = {k: full_dataset.dataset[k] for k in val_keys}

        # Create train/val datasets
        dataset_tr = DatasetClass.__new__(DatasetClass)
        dataset_tr.dataset = train_dict
        dataset_tr.csv_paths = self.csv_paths
        dataset_tr.strides = full_dataset.strides
        dataset_tr.cum_stride_z = full_dataset.cum_stride_z
        dataset_tr.cum_stride_y = full_dataset.cum_stride_y
        dataset_tr.cum_stride_x = full_dataset.cum_stride_x
        dataset_tr.target_z_size = full_dataset.target_z_size
        dataset_tr.training_mode = full_dataset.training_mode
        dataset_tr.patch_size = full_dataset.patch_size
        dataset_tr.all_labels = full_dataset.all_labels
        dataset_tr.has_seg = full_dataset.has_seg

        dataset_val = DatasetClass.__new__(DatasetClass)
        dataset_val.dataset = val_dict
        dataset_val.csv_paths = self.csv_paths
        dataset_val.strides = full_dataset.strides
        dataset_val.cum_stride_z = full_dataset.cum_stride_z
        dataset_val.cum_stride_y = full_dataset.cum_stride_y
        dataset_val.cum_stride_x = full_dataset.cum_stride_x
        dataset_val.target_z_size = full_dataset.target_z_size
        dataset_val.training_mode = full_dataset.training_mode
        dataset_val.patch_size = full_dataset.patch_size
        dataset_val.all_labels = full_dataset.all_labels
        dataset_val.has_seg = full_dataset.has_seg

        # Save split info
        split_info = {
            'fold': self.fold,
            'total_samples': len(all_keys),
            'total_unique_cases': len(unique_cases_list),
            'train_cases': len(train_cases),
            'val_cases': len(val_cases),
            'train_samples': len(train_keys),
            'val_samples': len(val_keys),
            'train_case_ids': sorted(list(train_cases)),
            'val_case_ids': sorted(list(val_cases)),
        }

        if self.local_rank == 0:
            save_json(split_info, join(self.output_folder, 'dataset_split.json'))

        return dataset_tr, dataset_val

    def get_plain_dataloaders(self, dim: int):
        """
        Get data loaders based on training_mode.
        - 'patch': Use nnUNetDataLoader3D with center cropping and foreground oversampling
        - 'whole': Use simple data loader that returns full images without cropping
        """
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim != 3:
            raise NotImplementedError("Only 3D supported")

        patch_size = self.configuration_manager.patch_size

        if self.training_mode == 'patch':
            # Patch mode: use standard nnUNetDataLoader3D with cropping and oversampling
            dl_tr = nnUNetDataLoader3D(
                dataset_tr, self.batch_size,
                patch_size,
                patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None
            )

            dl_val = nnUNetDataLoader3D(
                dataset_val, self.batch_size,
                patch_size,
                patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None
            )
        else:
            # Whole mode: use simple data loader without cropping
            dl_tr = nnUNetDataLoaderWholeImage(
                dataset_tr, self.batch_size,
                self.label_manager,
            )

            dl_val = nnUNetDataLoaderWholeImage(
                dataset_val, self.batch_size,
                self.label_manager,
            )

        self.print_to_log_file(f"Using training_mode='{self.training_mode}' data loaders")
        if self.training_mode == 'patch':
            self.print_to_log_file(f"  patch_size: {patch_size}")

        return dl_tr, dl_val

    def get_dataloaders(self):
        """
        Get data loaders with data augmentation.
        For training_mode='patch': follows nnUNetTrainer.py with full data augmentation
        For training_mode='whole': minimal augmentation (no spatial transforms)
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # Get deep supervision scales
        deep_supervision_scales = self._get_deep_supervision_scales()

        if self.training_mode == 'patch':
            # Patch mode: use full data augmentation like nnUNetTrainer
            (
                rotation_for_DA,
                do_dummy_2d_data_aug,
                _,  # initial_patch_size not used
                mirror_axes,
            ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

            # Training transforms with spatial augmentation
            tr_transforms = self.get_training_transforms(
                patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                order_resampling_data=3, order_resampling_seg=1,
                use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
                is_cascaded=False,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )

            # Validation transforms (no spatial augmentation)
            val_transforms = self.get_validation_transforms(
                deep_supervision_scales,
                is_cascaded=False,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )
        else:
            # Whole mode: minimal augmentation
            mirror_axes = (0, 1, 2)
            self.inference_allowed_mirroring_axes = mirror_axes

            # Minimal transforms for whole image mode
            tr_transforms = self.get_validation_transforms(
                deep_supervision_scales,
                is_cascaded=False,
                foreground_labels=self.label_manager.foreground_labels,
                regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
                ignore_label=self.label_manager.ignore_label
            )
            val_transforms = tr_transforms

        # Get plain data loaders
        dl_tr, dl_val = self.get_plain_dataloaders(dim)

        # Wrap with augmentation

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                num_processes=allowed_num_processes, num_cached=6, seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.02
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch, data_loader=dl_val, transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2), num_cached=3, seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.02
            )

        # Warm up the generators
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)

        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        """
        Perform a single training step.
        Following base nnUNetTrainer's approach.
        """
        data = batch['data']
        target = batch['target']

        # Debug mode: save input images as nii.gz for visualization
        if self.debug_save_input == True:
            debug_dir = os.path.join(self.output_folder, 'debug_input')
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[DEBUG] Saving input data to {debug_dir}")
            print(f"[DEBUG] data shape: {data.shape}, target shape: {target.shape if not isinstance(target, list) else [t.shape for t in target]}")

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

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        Perform a single validation step.
        Consistent with nnUNetTrainer.validation_step().
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs to be one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        """Called at the end of each training epoch."""
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])

        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Called at the end of each validation epoch. Consistent with nnUNetTrainer."""
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def plot_network_architecture(self):
        """Plot network architecture (simplified for segmentation)."""
        if self.local_rank == 0:
            self.print_to_log_file("Network architecture: ResEncoderUNet for segmentation")
            self.print_to_log_file(f"  Input channels: {self.num_input_channels}")
            self.print_to_log_file(f"  Output channels: {self.label_manager.num_segmentation_heads}")
            self.print_to_log_file(f"  Deep supervision: {self.enable_deep_supervision}")

    def configure_optimizers(self):
        """Configure optimizer based on optimizer_type setting."""

        if self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov
            )
            self.print_to_log_file(f"Optimizer: SGD (lr={self.initial_lr}, weight_decay={self.weight_decay}, momentum={self.momentum}, nesterov={self.nesterov})")
        else:  # AdamW
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=1e-8
            )
            self.print_to_log_file(f"Optimizer: AdamW (lr={self.initial_lr}, weight_decay={self.weight_decay}, betas={self.betas})")

        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def _build_loss(self):
        """
        Build loss function for segmentation.
        Uses standard DC_and_CE_loss with DeepSupervisionWrapper.
        Consistent with nnUNetTrainer._build_loss().

        Supports class weights for CE loss via seg_ce_class_weights in plans.
        """
        # Prepare class weights for CE loss
        seg_ce_class_weights_gpu = None
        if self.seg_ce_class_weights is not None:
            self.print_to_log_file(f"seg_ce_class_weights: {self.seg_ce_class_weights}")
            seg_ce_class_weights_gpu = torch.tensor(self.seg_ce_class_weights, dtype=torch.float32).to(self.device)

        # Full resolution loss with deep supervision
        ce_kwargs = {}
        if seg_ce_class_weights_gpu is not None:
            ce_kwargs['weight'] = seg_ce_class_weights_gpu

        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss  # Use memory efficient dice (consistent with nnUNetTrainer)
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            # Handle DDP mode (consistent with nnUNetTrainer)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
            self.print_to_log_file(f"Using DC_and_CE_loss with DeepSupervisionWrapper")
        else:
            self.print_to_log_file(f"Using DC_and_CE_loss without deep supervision")

        return loss

    def _get_deep_supervision_scales(self):
        """Get deep supervision scales from configuration.

        Returns cumulative downsampling scales for each decoder level.
        This is used by DownsampleSegForDSTransform2 to create multi-scale targets.

        Follows nnUNetTrainer.py exactly:
        - Uses pool_op_kernel_sizes (which equals strides in arch_kwargs)
        - Computes 1 / cumulative_product of strides
        - Removes the last element (lowest resolution)
        """
        if self.enable_deep_supervision:
            # Use pool_op_kernel_sizes directly (same as nnUNetTrainer.py)
            # pool_op_kernel_sizes = strides from arch_kwargs
            pool_op_kernel_sizes = self.configuration_manager.pool_op_kernel_sizes

            # Compute cumulative scales (same as nnUNetTrainer.py line 387-388)
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

    def initialize(self):
        """Initialize network, optimizer, loss, etc. Consistent with nnUNetTrainer."""
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            # Build network
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                enable_deep_supervision=self.enable_deep_supervision
            ).to(self.device)

            # Compile network if supported
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            # Configure optimizer
            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # if ddp, wrap in DDP wrapper (consistent with nnUNetTrainer)
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)

                self.network = DDP(self.network, device_ids=[self.local_rank])

            # Build loss
            self.loss = self._build_loss()

            # Grad scaler for mixed precision
            if self.device.type == 'cuda':
                self.grad_scaler = torch.amp.GradScaler('cuda')
            else:
                self.grad_scaler = None
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Configure data augmentation parameters.

        Both patch mode and whole mode: Only mirrors along spatial axes 0 and 1 for 3D
        (following nnUNetTrainer_onlyMirror01).
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # Both patch and whole mode: only mirror along axes 0 and 1
        if dim == 2:
            mirror_axes = (0,)
        else:
            mirror_axes = (0, 1)

        self.print_to_log_file(f"Training mode: {self.training_mode}, using mirror_axes = {mirror_axes}")
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def _do_i_compile(self):
        """Check if we should compile the network."""
        # Disable compilation for now - can cause issues with some architectures
        return False

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        Load checkpoint and reset learning rate to the value specified in the plan.
        """
        # Call parent class to load checkpoint
        super().load_checkpoint(filename_or_checkpoint)

        # Reset initial_lr and lr in optimizer to the new values from the plan
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group['initial_lr'] = self.initial_lr
                group['lr'] = self.initial_lr
            self.print_to_log_file(f"LR reset to plan value (initial_lr={self.initial_lr})")

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Skip the validation step on the validation set.

        Since this Trainer uses CSV datasets and does not rely on nnUNet's preprocessed data directory structure,
        validation will be performed on a separate test set via inference scripts.
        """
        self.print_to_log_file("Skipping perform_actual_validation (will evaluate on separate test set)")
        return
