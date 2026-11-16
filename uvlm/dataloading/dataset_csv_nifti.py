"""NIfTI input for U-VLM report/classification inference.

The training pipeline stores preprocessed volumes as Blosc2. This loader adds
an inference-only path for NIfTI files (``.nii`` / ``.nii.gz``) so users do not
need to convert their data first.

CSV columns:
    series_id, image_path

The image is transposed to (D, H, W), z-score normalized, and padded/truncated
on the z axis to the checkpoint patch size. The report inference code resizes
H/W to the patch size before the forward pass.
"""

import os
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd


def _read_nifti(path: str) -> np.ndarray:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    # nibabel (X, Y, Z) -> nnU-Net (D, H, W)
    data = np.transpose(data, (2, 1, 0))
    return np.ascontiguousarray(data, dtype=np.float32)


def _process_z_axis(img: np.ndarray, target_z: int) -> np.ndarray:
    if img.shape[0] > target_z:
        return img[:target_z]
    if img.shape[0] < target_z:
        pad = target_z - img.shape[0]
        return np.pad(img, ((0, pad), (0, 0), (0, 0)), mode="constant", constant_values=float(img.min()))
    return img


class nnUNetDatasetCSVNifti:
    """Single-channel NIfTI dataset with the same interface as the Blosc2 loader."""

    def __init__(
        self,
        csv_paths: Union[str, List[str]],
        strides: List[List[int]],
        target_size: Tuple[int, int, int],
        series_id_column: str = "series_id",
        image_path_column: str = "image_path",
        verbose: bool = True,
    ):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        frames = [pd.read_csv(p) for p in csv_paths]
        self.df = pd.concat(frames, ignore_index=True)
        if series_id_column not in self.df.columns:
            raise ValueError(f"CSV is missing '{series_id_column}'")
        if image_path_column not in self.df.columns:
            raise ValueError(f"CSV is missing '{image_path_column}'")
        self.series_id_column = series_id_column
        self.image_path_column = image_path_column
        self.target_z_size = int(target_size[0])
        self.strides = strides
        self.dataset = {}
        for _, row in self.df.iterrows():
            key = str(row[series_id_column])
            self.dataset[key] = {
                "image_path": str(row[image_path_column]),
                "properties": {
                    "origin": [0.0, 0.0, 0.0],
                    "direction": np.eye(3).flatten().tolist(),
                    "class_locations": {},
                    "original_shape": None,
                },
            }
        if verbose:
            print(f"Loaded {len(self.dataset)} NIfTI cases from {len(csv_paths)} CSV(s)")

    def __len__(self):
        return len(self.dataset)

    def keys(self):
        return self.dataset.keys()

    @property
    def identifiers(self):
        return list(self.dataset.keys())

    def load_case(self, key):
        entry = self.dataset[key]
        img = _read_nifti(entry["image_path"])
        if entry["properties"]["original_shape"] is None:
            entry["properties"]["original_shape"] = img.shape
        mean = float(img.mean())
        std = max(float(img.std()), 1e-8)
        img = (img - mean) / std
        img = _process_z_axis(img, self.target_z_size)
        data = img[np.newaxis, ...].astype(np.float32)
        seg = np.full_like(data, -1, dtype=np.int16)
        return data, seg, None, entry["properties"]
