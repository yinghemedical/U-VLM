import os
import ast
import blosc2
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional


def read_blosc2_file(file_path: str, mmap_mode: Optional[str] = 'r') -> np.ndarray:
    """
    Read array in Blosc2 compressed format

    Args:
        file_path: Path to Blosc2 file (.b2nd)
        mmap_mode: Memory mapping mode ('r', 'r+', 'c', None)

    Returns:
        numpy array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f".b2nd file does not exist: {file_path}")

    b2_array = blosc2.open(file_path, mmap_mode=mmap_mode)
    return np.asarray(b2_array)


def compute_cumulative_strides(strides: List[List[int]]) -> Tuple[int, int, int]:
    """
    Calculate cumulative stride multiples to determine the minimum size multiple for image padding.

    Args:
        strides: List of stride lists, e.g., [[1,1,1], [1,2,2], [1,2,2], ...]

    Returns:
        Tuple of (z_stride, y_stride, x_stride) - cumulative stride for each axis
    """
    if not strides:
        raise ValueError("strides list cannot be empty")

    # Initialize cumulative strides
    cum_z, cum_y, cum_x = 1, 1, 1

    for stride in strides:
        cum_z *= stride[0]
        cum_y *= stride[1]
        cum_x *= stride[2]

    return cum_z, cum_y, cum_x


def process_z_axis(img: np.ndarray, target_z: int = 36) -> np.ndarray:
    """
    Process z-axis: standardize to target_z size
    - If larger than target_z: truncate from the end
    - If smaller than target_z: pad at the end (ensures consistent vision token order)

    Args:
        img: Input image (Z, H, W)
        target_z: Target z-axis size

    Returns:
        Processed image (target_z, H, W)
    """
    current_z = img.shape[0]

    if current_z > target_z:
        # Truncation: keep the first target_z slices
        img = img[:target_z]
    elif current_z < target_z:
        # Padding: pad with minimum value at the end (ensures consistent vision token order)
        pad_z = target_z - current_z
        img = np.pad(img, ((0, pad_z), (0, 0), (0, 0)), mode='constant', constant_values=img.min())

    return img


def process_z_axis_seg(seg: np.ndarray, target_z: int = 36) -> np.ndarray:
    """
    Process z-axis for segmentation labels: standardize to target_z size
    - If larger than target_z: truncate from the end
    - If smaller than target_z: pad at the end using -1 as ignore label

    Args:
        seg: Input segmentation labels (Z, H, W)
        target_z: Target z-axis size

    Returns:
        Processed segmentation labels (target_z, H, W)
    """
    current_z = seg.shape[0]

    if current_z > target_z:
        # Truncation: keep the first target_z slices
        seg = seg[:target_z]
    elif current_z < target_z:
        # Padding: pad with -1 (ignore label) at the end
        pad_z = target_z - current_z
        seg = np.pad(seg, ((0, pad_z), (0, 0), (0, 0)), mode='constant', constant_values=-1)

    return seg


class nnUNetDatasetCSVBlosc2:
    """
    Dataset class for reading data from CSV files (Blosc2 version)
    Each load_case reads from disk and processes the data

    Supports two training modes:
    - training_mode='whole': Load full image without cropping
    - training_mode='patch': Supports patch cropping based on class_locations and foreground oversampling

    Key changes:
    - z-axis fixed to target_z_size: truncate if larger, pad if smaller
    - x/y must match patch_size (filtered during CSV preprocessing)
    - Ensures consistent vision token count and order
    - Supports loading segmentation labels and computing class_locations
    """
    def __init__(
        self,
        csv_paths: Union[str, List[str]],
        verbose: bool = True,
        strides: List[List[int]] = None,
        target_z_size: int = 36,
        training_mode: str = 'whole',  # 'whole' or 'patch'
        patch_size: Tuple[int, int, int] = None,  # Required for patch mode
        all_labels: List[int] = None,  # All possible label values for class_locations
        series_id_column: str = 'series_id',  # Series-level ID column name
    ):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        self.csv_paths = csv_paths
        if strides is None:
            raise ValueError("strides parameter must be provided")
        self.strides = strides

        self.target_z_size = target_z_size
        self.training_mode = training_mode
        self.patch_size = patch_size
        self.all_labels = all_labels or []
        self.series_id_column = series_id_column

        if training_mode == 'patch' and patch_size is None:
            raise ValueError("patch_size must be provided for training_mode='patch'")

        # Compute cumulative strides for padding
        self.cum_stride_z, self.cum_stride_y, self.cum_stride_x = compute_cumulative_strides(self.strides)

        # Load and merge all CSV files
        all_dfs = []
        for csv_path in csv_paths:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
            df = pd.read_csv(csv_path)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No data loaded")

        # Merge DataFrame
        self.df = pd.concat(all_dfs, ignore_index=True)

        # Ensure required columns exist
        required_columns = [self.series_id_column, 'blosc2_path']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")

        # Check if seg_blosc2_path column exists
        self.has_seg = 'seg_blosc2_path' in self.df.columns

        # Use series_id_column as key
        self.dataset = {}
        # Identify all columns starting with 'class_'
        class_columns = [col for col in self.df.columns if col.startswith('class_')]

        for idx, row in self.df.iterrows():
            key = str(row[self.series_id_column])  # Ensure key is a string
            if key in self.dataset:
                key = f"{key}_{idx}"

            # Extract class info
            class_info = {col: row[col] for col in class_columns}

            # report is optional
            report = ''
            if 'report' in self.df.columns and pd.notna(row.get('report')):
                report = str(row['report'])

            # seg_blosc2_path is optional
            seg_blosc2_path = None
            if self.has_seg and pd.notna(row.get('seg_blosc2_path')):
                seg_blosc2_path = row['seg_blosc2_path']

            self.dataset[key] = {
                'blosc2_path': row['blosc2_path'],
                'seg_blosc2_path': seg_blosc2_path,
                'report': report,
                'class_info': class_info,
                'properties': {
                    'origin': [0.0, 0.0, 0.0],
                    'direction': np.eye(3).flatten().tolist(),
                    'class_locations': None,  # Will be computed on first load if needed
                    'original_shape': None,
                }
            }

        if verbose:
            print(f"Loaded {len(self.dataset)} samples from {len(self.csv_paths)} CSV files")
            print(f"Training mode: {training_mode}, has segmentation labels: {self.has_seg}")

    def __getitem__(self, key):
        if key not in self.dataset:
            raise KeyError(f"Key {key} not found in dataset")
        return {**self.dataset[key]}

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return len(self.dataset)

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        """
        Load and preprocess a single case.
        Processing pipeline: read blosc2 -> transpose -> z-axis processing -> z-score normalization.

        Supports two modes:
        - training_mode='whole': returns the full image.
        - training_mode='patch': returns the full image + class_locations (cropped by data_loader).

        Key rules:
        - x/y dimensions have been verified to match patch_size during CSV preprocessing.
        - z-axis is uniformly processed to target_z_size: truncate if exceeding, pad if insufficient.
        - Ensures consistent vision token count per batch.

        Returns: data (1, D, H, W), seg (1, D, H, W), properties
        """
        if key not in self.dataset:
            raise KeyError(f"Key {key} not found in dataset")

        entry = self.dataset[key]

        # Read image from disk
        img = read_blosc2_file(entry['blosc2_path'])

        # Handle channel dimension
        if img.ndim == 4:
            img = img[0]

        # Transpose to (D, H, W) = (Z, Y, X)
        img = np.transpose(img, (2, 1, 0)).astype(np.float32)

        # Save original shape (only on first load)
        if entry['properties']['original_shape'] is None:
            entry['properties']['original_shape'] = img.shape

        # z-score normalization (global statistics, performed before z-axis processing)
        mean = img.mean()
        std = max(img.std(), 1e-8)
        img_normalized = (img - mean) / std

        # z-axis processing: unify to target_z_size
        img_processed = process_z_axis(img_normalized, self.target_z_size)

        # Add channel dimension: (1, D, H, W)
        data = img_processed[np.newaxis, ...]

        # Load segmentation labels
        if entry['seg_blosc2_path'] is not None and os.path.exists(entry['seg_blosc2_path']):
            seg = read_blosc2_file(entry['seg_blosc2_path'])
            # Handle channel dimension
            if seg.ndim == 4:
                seg = seg[0]
            # Transpose to (D, H, W) = (Z, Y, X)
            seg = np.transpose(seg, (2, 1, 0)).astype(np.int16)
            # z-axis processing: consistent with image
            seg = process_z_axis_seg(seg, self.target_z_size)
            # Add channel dimension: (1, D, H, W)
            seg = seg[np.newaxis, ...]
        else:
            # No segmentation labels (all -1)
            seg = np.full_like(data, -1, dtype=np.int16)

        # class_locations are precomputed in the pkl file during preprocessing; no need to recompute.
        # If class_locations is None, set to empty dict.
        if entry['properties']['class_locations'] is None:
            entry['properties']['class_locations'] = {}

        return data, seg, entry['properties']
