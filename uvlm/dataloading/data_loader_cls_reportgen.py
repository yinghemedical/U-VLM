"""
General data loader supporting classification and report generation.

All parameters are passed from the plan's network_arch_init_kwargs for easy customization when training different datasets.

Required parameters (passed from plan):
    - csv_paths: List of CSV file paths
    - series_id_column: Column name for series-level identifiers in CSV (used for data loading and label matching)
    - cls_columns: List of classification column names (used for classification tasks and data balancing)
    - report_column: Column name for report text (used for report generation)

ID format explanation:
    - series_id: Series-level ID, e.g., train_18073_a_1
    - case_id: Case-level ID, e.g., train_18073_a (used for train/val split to ensure the same patient does not appear in both train and val)
    - balanced_key: Balanced key after augmentation, e.g., train_18073_a_1_10787 (series_id + replication number)

Usage example:
    Configure in plans.json's network_arch_init_kwargs:
    {
        "csv_paths": ["/path/to/train.csv"],
        "series_id_column": "series_id",
        "case_id_column": "case_id",
        "cls_columns": ["Liver lesion", "Kidney lesion", "Pancreas lesion"],
        "report_column": "report"
    }
"""

import os
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase


class nnUNetDataLoader3DWithGlobalClsReportgen(nnUNetDataLoaderBase):
    def __init__(
        self,
        *args,
        # Required parameters (passed from plan)
        csv_paths: Union[str, List[str]],
        series_id_column: str,
        cls_columns: List[str],
        report_column: str,
        # Optional parameters
        use_sampling_weight: Optional[str] = None,
        **kwargs
    ):
        """
        General data loader supporting classification and report generation.

        Parameters:
            csv_paths: CSV file path(s) (single file or list of files), must be passed from plan
            series_id_column: Column name for series-level identifiers in CSV, must be passed from plan
            cls_columns: List of classification column names, must be passed from plan
            report_column: Column name for report text, must be passed from plan
            use_sampling_weight: Sampling weight option
                - None: No weighting, uniform sampling
                - 'balanced': Use class-balanced sampling
        """
        # Set attributes before calling super().__init__()
        self.csv_paths = csv_paths if isinstance(csv_paths, list) else [csv_paths]
        self.series_id_column = series_id_column
        self.cls_columns = cls_columns
        self.report_column = report_column
        self.use_sampling_weight = use_sampling_weight

        super().__init__(*args, **kwargs)

        # Load and merge all CSV files
        self.df_full = self._load_csv_files()

        # Validate CSV columns
        self._validate_csv_columns()

    def _load_csv_files(self) -> pd.DataFrame:
        """Load and merge all CSV files"""
        dfs = []
        for csv_path in self.csv_paths:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"Loaded CSV: {csv_path} ({len(df)} rows)")

        if len(dfs) == 1:
            combined = dfs[0]
        else:
            combined = pd.concat(dfs, ignore_index=True)
            print(f"Combined {len(dfs)} CSV files: {len(combined)} total rows")

        return combined

    def _validate_csv_columns(self):
        """Validate that CSV contains required columns"""
        missing_columns = []

        # Check series identifier column
        if self.series_id_column not in self.df_full.columns:
            missing_columns.append(f"series_id_column: '{self.series_id_column}'")

        # Check classification columns
        for col in self.cls_columns:
            if col not in self.df_full.columns:
                missing_columns.append(f"cls_column: '{col}'")

        # Check report column
        if self.report_column not in self.df_full.columns:
            missing_columns.append(f"report_column: '{self.report_column}'")

        if missing_columns:
            available_columns = list(self.df_full.columns)
            raise ValueError(
                f"CSV is missing the following columns: {missing_columns}\n"
                f"Available columns: {available_columns}"
            )

    def get_indices(self):
        return np.random.choice(
            self.indices, self.batch_size, replace=True, p=self.sampling_probabilities
        )

    def _extract_series_id_from_key(self, key: str) -> str:
        """
        Extract series_id from dataset key.

        Balanced key format: series_id_replication_number (e.g., train_18073_a_1_10787)
        Original key format: series_id (e.g., train_18073_a_1)

        Need to determine if key is in balanced format; if so, remove the replication number at the end.
        """
        if '_' not in key:
            return key

        # Attempt to extract the part after the last underscore
        parts = key.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            # Last part is a number, possibly a replication number
            potential_series_id = parts[0]
            # Check if this potential_series_id exists in CSV
            if potential_series_id in self.df_full[self.series_id_column].values:
                return potential_series_id

        # If not balanced format, or extracted series_id not in CSV, return original key
        return key

    def _get_sample_data(self, key: str) -> Optional[pd.Series]:
        """Retrieve data row for specified key from CSV"""
        # First extract series_id from key
        series_id = self._extract_series_id_from_key(key)

        matches = self.df_full[self.df_full[self.series_id_column] == series_id]
        if len(matches) == 0:
            # If extracted series_id not found, try original key
            if series_id != key:
                matches = self.df_full[self.df_full[self.series_id_column] == key]
            if len(matches) == 0:
                print(f"WARNING: No data found for case {key} (series_id: {series_id}) in CSV")
                return None
        return matches.iloc[0]

    def generate_train_batch(self) -> Dict[str, Any]:
        """
        Generate training batch

        Returns:
            dict: Contains the following fields:
                - data: Image data [B, C, D, H, W]
                - seg: Segmentation labels [B, 1, D, H, W]
                - properties: List of properties
                - keys: List of sample identifiers
                - cls_all: Classification labels [B, num_cls]
                - report_texts: List of report texts
        """
        # 1. Select samples for current batch
        selected_keys = self.get_indices()

        # 2. Pre-allocate memory
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        # Classification labels
        cls_all = np.zeros((self.batch_size, len(self.cls_columns)), dtype=np.float32)

        # Report texts
        report_texts = []

        valid_count = 0
        for j, key in enumerate(selected_keys):
            # Foreground oversampling logic
            force_fg = self.get_do_oversample(j)

            # Load 3D data, segmentation, and properties
            data, seg, properties = self._data.load_case(key)

            if data is None:
                print(f"WARNING: Failed to load case {key}, skipping")
                continue

            case_properties.append(properties)
            # Get metadata from CSV
            sample_data = self._get_sample_data(key)

            # Extract classification labels
            if sample_data is not None:
                for k, cls_col in enumerate(self.cls_columns):
                    try:
                        cls_all[valid_count, k] = float(sample_data[cls_col])
                    except (ValueError, TypeError):
                        cls_all[valid_count, k] = 0.0

            # Extract report text
            if sample_data is not None and pd.notna(sample_data[self.report_column]):
                report_text = str(sample_data[self.report_column])
            else:
                report_text = ""
            report_texts.append(report_text)

            # Process image cropping and padding
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            this_slice = tuple(
                [slice(0, data.shape[0])] +
                [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            data = data[this_slice]

            this_slice = tuple(
                [slice(0, seg.shape[0])] +
                [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)]
            )
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[valid_count] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[valid_count] = np.pad(seg, padding, 'constant', constant_values=-1)

            valid_count += 1

        # Build return dictionary (always includes cls_all and report_texts)
        return {
            'data': data_all[:valid_count] if valid_count < self.batch_size else data_all,
            'seg': seg_all[:valid_count] if valid_count < self.batch_size else seg_all,
            'properties': case_properties,
            'keys': selected_keys[:valid_count] if valid_count < self.batch_size else selected_keys,
            'cls_all': cls_all[:valid_count] if valid_count < self.batch_size else cls_all,
            'report_texts': report_texts,
        }


if __name__ == '__main__':
    # Test example
    from nnunetv2.training.dataloading.nnUNet_dataset import nnUNetDataset

    # These parameters should be passed from the plan
    test_config = {
        'csv_paths': ['/path/to/your/train.csv'],
        'series_id_column': 'series_id',
        'cls_columns': ['lung_opacity', 'pleural_effusion'],
        'report_column': 'report',
    }

    folder = '/path/to/preprocessed/data'
    ds = nnUNetDataset(folder, 0)
    dl = nnUNetDataLoader3DWithGlobalClsReportgen(
        ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None,
        **test_config
    )
    batch = next(dl)
    print(f"Batch keys: {batch.keys()}")
