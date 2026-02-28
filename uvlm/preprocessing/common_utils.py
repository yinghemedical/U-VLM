# -*- coding: utf-8 -*-
"""
Unified medical imaging data preprocessing toolkit
Supports preprocessing for chest and abdominal datasets
Blosc2 format is supported.
"""
import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
from scipy.ndimage import zoom
import blosc2


# ============== Configuration Paths ==============

CONFIG_DIR = Path(__file__).parent / "configs"


# ============== Error Handling: Delete Corrupted Files ==============

def delete_raw_case(raw_dir: str, case_id: str) -> None:
    """
    Delete image and label files for a specified case in the raw directory
    Used for cleanup on failure; will be regenerated automatically on next run
    """
    label_id = case_id[:-5] if case_id.endswith('_0000') else case_id

    img_path = os.path.join(raw_dir, "imagesTr", f"{case_id}.nii.gz")
    label_path = os.path.join(raw_dir, "labelsTr", f"{label_id}.nii.gz")

    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(label_path):
        os.remove(label_path)


# ============== NIfTI Orientation Standardization ==============

def is_valid_affine(affine: np.ndarray) -> bool:
    """Check if the affine matrix is valid"""
    if affine is None:
        return False
    if not np.isfinite(affine).all():
        return False
    # Check if the 3x3 rotation/scaling part is singular
    det = np.linalg.det(affine[:3, :3])
    if abs(det) < 1e-10:
        return False
    return True


def reorient_nifti(orig_nii: nib.Nifti1Image, target_orientation: str = "LPS") -> nib.Nifti1Image:
    """
    Reorient NIfTI to a standard orientation (default LPS+)

    Args:
        orig_nii: Original NIfTI image
        target_orientation: Target orientation code, e.g., "LPS", "RAS"

    Returns:
        Reoriented NIfTI image
    """
    # Check if affine is valid
    if not is_valid_affine(orig_nii.affine):
        # Affine invalid, attempt to reconstruct from header or use identity matrix
        spacing = orig_nii.header.get_zooms()[:3]
        if all(s > 0 and np.isfinite(s) for s in spacing):
            new_affine = np.diag(list(spacing) + [1.0])
        else:
            new_affine = np.eye(4)
        orig_nii = nib.Nifti1Image(orig_nii.get_fdata(), new_affine, orig_nii.header)

    # Check if already in target orientation
    current_orientation = "".join(nib.aff2axcodes(orig_nii.affine))
    if current_orientation == target_orientation:
        return orig_nii

    # Compute transformation
    orig_ornt = nib.io_orientation(orig_nii.affine)
    targ_ornt = nib.orientations.axcodes2ornt(target_orientation)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)

    # Apply transformation
    reoriented = orig_nii.as_reoriented(transform)

    return reoriented


def reorient_nifti_file(file_path: str, target_orientation: str = "LPS", inplace: bool = True) -> str:
    """
    Reorient a NIfTI file

    Args:
        file_path: Path to the NIfTI file
        target_orientation: Target orientation
        inplace: Whether to modify in place

    Returns:
        Output file path
    """
    nii = nib.load(file_path)
    reoriented = reorient_nifti(nii, target_orientation)

    if inplace:
        output_path = file_path
    else:
        base, ext = os.path.splitext(file_path)
        if ext == ".gz":
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext
        output_path = f"{base}_reoriented{ext}"

    reoriented.to_filename(output_path)
    return output_path


def batch_reorient_nifti(
    input_dir: str,
    target_orientation: str = "LPS",
    num_workers: int = 20,
    inplace: bool = True
) -> List[str]:
    """
    Batch reorient NIfTI files

    Args:
        input_dir: Input directory
        target_orientation: Target orientation
        num_workers: Number of parallel processes
        inplace: Whether to modify in place
    """
    # Find all NIfTI files
    nii_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(('.nii.gz', '.nii')):
                nii_files.append(os.path.join(root, f))

    print(f"Found {len(nii_files)} NIfTI files to reorient")

    def process_single(file_path):
        return reorient_nifti_file(file_path, target_orientation, inplace)

    results = parallel_process(nii_files, process_single, num_workers, "Reorienting")
    return results


def write_nifti_file(file_path: str, array: np.ndarray, affine: np.ndarray = None, header: nib.Nifti1Header = None):
    """
    Save array as NIfTI format (.nii.gz)

    Args:
        file_path: Output file path
        array: NumPy array to save
        affine: Affine matrix, defaults to identity matrix
        header: NIfTI header information, optional
    """
    if affine is None:
        affine = np.eye(4)

    # Ensure file path ends with .nii.gz
    if not file_path.endswith('.nii.gz'):
        if file_path.endswith('.nii'):
            file_path = file_path + '.gz'
        else:
            file_path = file_path + '.nii.gz'

    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    nii = nib.Nifti1Image(array, affine, header)
    nib.save(nii, file_path)

    return file_path


# ============== Blosc2 Compression/Decompression ==============

def comp_blosc2_params(image_size: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute Blosc2 chunks and blocks parameters

    Args:
        image_size: Image dimensions

    Returns:
        (chunks, blocks) tuple
    """
    # Use image dimensions as chunks
    chunks = tuple(image_size)

    # Default block settings (adaptive based on image size)
    blocks = tuple(min(s, 64) for s in image_size)

    return chunks, blocks


def write_blosc2_file(file_path: str, array: np.ndarray, compression_level: int = 8):
    """
    Save array in Blosc2 compressed format (.b2nd)

    Args:
        file_path: Output file path (without extension or ending with .b2nd)
        array: NumPy array to save
        compression_level: Compression level (1-9)
    """
    # Ensure file path ends with .b2nd
    if not file_path.endswith('.b2nd'):
        file_path = file_path + '.b2nd'

    dir_path = os.path.dirname(file_path)
    if dir_path:  # Create directory only if dirname is not empty
        os.makedirs(dir_path, exist_ok=True)

    # Calculate chunks and blocks
    chunks, blocks = comp_blosc2_params(array.shape)

    # Set compression parameters
    cparams = blosc2.CParams(
        codec=blosc2.Codec.ZSTD,
        clevel=compression_level,
        typesize=array.dtype.itemsize
    )

    # Save in Blosc2 format
    blosc2.asarray(
        np.ascontiguousarray(array),
        urlpath=file_path,
        chunks=chunks,
        blocks=blocks,
        cparams=cparams,
        mode='w'
    )

    return file_path


def read_blosc2_file(file_path: str, mmap_mode: Optional[str] = 'r') -> np.ndarray:
    """
    Read array from Blosc2 compressed format

    Args:
        file_path: Blosc2 file path
        mmap_mode: Memory mapping mode ('r', 'r+', 'c', None)

    Returns:
        NumPy array
    """
    b2_array = blosc2.open(file_path, mmap_mode=mmap_mode)
    return np.asarray(b2_array)


def get_blosc2_shape(file_path: str) -> Tuple[int, ...]:
    """Quickly get shape information from Blosc2 file"""
    b2_array = blosc2.open(file_path, mmap_mode='r')
    return tuple(b2_array.shape)


def get_blosc2_dtype(file_path: str) -> np.dtype:
    """Get dtype information from Blosc2 file"""
    b2_array = blosc2.open(file_path, mmap_mode='r')
    return b2_array.dtype


def add_processed_shape_columns(
    df: pd.DataFrame,
    image_dir: str,
    case_id_column: str = "case_id",
    rename_original_shape: bool = True
) -> pd.DataFrame:
    """
    Add processed image shape and path columns to DataFrame.

    Blosc2 (.b2nd) format is supported.

    Column naming:
    - 'original_shape': original shape from source data (if exists and rename_original_shape=True)
    - 'processed_shape': processed/resized image shape (xyz order)
    - 'image_path': path to processed image file

    Args:
        df: DataFrame with case_id column
        image_dir: directory containing processed images (blosc2)
        case_id_column: column name for case ID
        rename_original_shape: if True, rename existing 'shape' column to 'original_shape'

    Returns:
        DataFrame with processed_shape and image_path columns added
    """
    from tqdm import tqdm

    # Rename existing 'shape' column to 'original_shape' if it exists
    if rename_original_shape and 'shape' in df.columns:
        df = df.rename(columns={'shape': 'original_shape'})

    # Remove legacy columns if they exist
    legacy_columns = ['processed_shape', 'image_path']
    for col in legacy_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    if not image_dir:
        print(f"Warning: image_dir is empty, skipping shape extraction")
        df["processed_shape"] = ""
        df["image_path"] = ""
        return df

    if not os.path.exists(image_dir):
        print(f"Warning: image_dir not found: {image_dir}")
        df["processed_shape"] = ""
        df["image_path"] = ""
        return df

    print(f"Looking for processed images in: {image_dir}")

    shapes = []
    image_paths = []
    found_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Getting processed shapes"):
        case_id = row[case_id_column]

        # Blosc2 format is supported
        b2nd_path = os.path.join(image_dir, f"{case_id}.b2nd")

        if os.path.exists(b2nd_path):
            shape = get_blosc2_shape(b2nd_path)
            shapes.append(str(list(shape)))
            image_paths.append(b2nd_path)
            found_count += 1
        else:
            shapes.append("")
            image_paths.append("")

    df["processed_shape"] = shapes
    df["image_path"] = image_paths

    print(f"Found {found_count}/{len(df)} cases with processed images")

    # Show sample of missing files if any
    if found_count == 0 and len(df) > 0:
        sample_id = df[case_id_column].iloc[0]
        print(f"  Example: looking for {os.path.join(image_dir, f'{sample_id}.b2nd')}")
        # List what files actually exist
        if os.path.exists(image_dir):
            files = os.listdir(image_dir)[:5]
            print(f"  Files in directory: {files}")

    return df


# ============== Report & Classification Processing ==============

def process_reports_common(
    df: pd.DataFrame,
    case_id_column: str,
    report_columns: Dict[str, str],
    image_dir: Optional[str] = None,
    metadata_columns: List[str] = None
) -> pd.DataFrame:
    """
    Common report processing logic for both chest and abdomen datasets.

    Args:
        df: source DataFrame
        case_id_column: column name for case ID
        report_columns: dict mapping source_column -> target_column for reports
        image_dir: directory containing processed images for shape extraction
        metadata_columns: list of metadata columns to keep (e.g., ['spacing', 'sex', 'age'])

    Returns:
        DataFrame with case_id, series_id, report columns, processed_shape, image_path
    """
    from tqdm import tqdm

    if metadata_columns is None:
        metadata_columns = []

    records = []
    found_shape_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reports"):
        case_id = row.get(case_id_column, f"case_{idx}")

        record = {
            "case_id": case_id,
            "series_id": case_id,
        }

        # Copy report columns
        for source_col, target_col in report_columns.items():
            value = row.get(source_col, "")
            if pd.isna(value):
                value = ""
            record[target_col] = str(value)

        # Copy metadata columns
        for col in metadata_columns:
            if col in row.index:
                record[col] = row[col]

        # Get processed shape and image path (Blosc2 format supported)
        if image_dir:
            b2nd_path = os.path.join(image_dir, f"{case_id}.b2nd")

            if os.path.exists(b2nd_path):
                shape = get_blosc2_shape(b2nd_path)
                record["processed_shape"] = str(list(shape))
                record["image_path"] = b2nd_path
                found_shape_count += 1
            else:
                record["processed_shape"] = ""
                record["image_path"] = ""
        else:
            record["processed_shape"] = ""
            record["image_path"] = ""

        records.append(record)

    result_df = pd.DataFrame(records)
    print(f"Processed {len(result_df)} records, {found_shape_count} with processed_shape")

    return result_df


def merge_reports_and_labels(
    reports_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    merge_on: str = "case_id"
) -> pd.DataFrame:
    """
    Merge reports and classification labels.

    Args:
        reports_df: DataFrame with reports (case_id, series_id, report columns, processed_shape, image_path)
        labels_df: DataFrame with classification labels (case_id, label columns)
        merge_on: column to merge on

    Returns:
        Merged DataFrame
    """
    merged_df = pd.merge(
        reports_df, labels_df,
        on=merge_on,
        how="inner",
        suffixes=("", "_cls")
    )

    print(f"Merged: {len(merged_df)} records")
    return merged_df


# ============== NIfTI File Processing ==============

def load_nifti(file_path: str, reorient: bool = True, target_orientation: str = "LPS") -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load NIfTI file, optionally reorient."""
    nii = nib.load(file_path)

    if reorient:
        nii = reorient_nifti(nii, target_orientation)

    data = np.asanyarray(nii.dataobj)
    return data, nii.affine, nii.header


def save_nifti(file_path: str, data: np.ndarray, affine: np.ndarray,
               header: Optional[nib.Nifti1Header] = None):
    """Save NIfTI file."""
    dir_path = os.path.dirname(file_path)
    if dir_path:  # Create directory only if dirname is not empty
        os.makedirs(dir_path, exist_ok=True)
    if header is not None:
        nii = nib.Nifti1Image(data, affine, header)
    else:
        nii = nib.Nifti1Image(data, affine)
    nib.save(nii, file_path)


def get_nifti_spacing(nii: nib.Nifti1Image) -> Tuple[float, float, float]:
    """Get NIfTI spacing (x, y, z)."""
    return tuple(nii.header.get_zooms()[:3])


def get_nifti_orientation(nii: nib.Nifti1Image) -> str:
    """Get NIfTI orientation code."""
    return "".join(nib.aff2axcodes(nii.affine))


# ============== Label Processing ==============

def create_lut(mapping: Dict[int, int], max_label: int = 256) -> np.ndarray:
    """Create label look-up table (LUT)."""
    lut = np.zeros(max_label, dtype=np.uint8)
    for src, dst in mapping.items():
        if src < max_label:
            lut[int(src)] = int(dst)
    return lut


def create_lut_from_config(merge_lut: Dict[str, int], max_label: int = 256) -> np.ndarray:
    """Create LUT from configured merge_lut."""
    lut = np.zeros(max_label, dtype=np.uint8)
    for src, dst in merge_lut.items():
        src_int = int(src)
        if src_int < max_label:
            lut[src_int] = int(dst)
    return lut


def apply_lut(mask: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Quickly convert labels using LUT."""
    clipped = np.clip(mask.astype(np.int32), 0, len(lut) - 1)
    return lut[clipped].astype(np.uint8)


def combine_segmentation_masks(
    seg_dir: str,
    file_to_id: Dict[str, int],
    lesion_priority: bool = True,
    reorient: bool = True
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Combine multiple mask files in a segmentation directory into a single multi-class mask.
    """
    combined = None
    affine = None
    header = None

    # Separate lesions and non-lesions
    lesion_names = {"liver_lesion", "kidney_lesion", "pancreatic_lesion"}
    non_lesion_items = []
    lesion_items = []

    for name, label_id in file_to_id.items():
        file_path = os.path.join(seg_dir, f"{name}.nii.gz")
        if not os.path.exists(file_path):
            continue

        if name in lesion_names:
            lesion_items.append((file_path, label_id))
        else:
            non_lesion_items.append((file_path, label_id))

    # Process non-lesions first
    for file_path, label_id in non_lesion_items:
        data, aff, hdr = load_nifti(file_path, reorient=reorient)
        if combined is None:
            combined = np.zeros(data.shape, dtype=np.uint8)
            affine = aff
            header = hdr
        mask = data > 0
        combined[mask] = label_id

    # Then process lesions (highest priority)
    if lesion_priority:
        for file_path, label_id in lesion_items:
            data, aff, hdr = load_nifti(file_path, reorient=reorient)
            if combined is None:
                combined = np.zeros(data.shape, dtype=np.uint8)
                affine = aff
                header = hdr
            mask = data > 0
            combined[mask] = label_id

    return combined, affine, header


# ============== Debug NII Selection ==============

def select_debug_nii_samples(items: list, sample_size: int = 10, early_limit: int = 50) -> set:
    """
    Randomly select sample_size samples from the first early_limit items for saving debug nii.

    Args:
        items: List of all items
        sample_size: Number of items to select
        early_limit: Number of early items to select from

    Returns:
        Set of selected items
    """
    import random
    if not items:
        return set()
    early_items = items[:early_limit]
    actual_size = min(sample_size, len(early_items))
    return set(random.sample(early_items, actual_size))

# ============== Resampling ==============

def compute_new_spacing(
    original_shape: Tuple[int, int, int],
    new_shape: Tuple[int, int, int],
    original_spacing: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Calculate new spacing based on original size, new size, and original spacing

    When an image is resized, spacing needs to be adjusted accordingly to maintain correct physical dimensions.
    New spacing = original spacing * (original size / new size)

    Args:
        original_shape: Original image size (x, y, z) or (D, H, W)
        new_shape: New image size, same format as original_shape
        original_spacing: Original spacing (x, y, z) or (D, H, W)

    Returns:
        New spacing, same format as original_spacing
    """
    new_spacing = tuple(
        orig_sp * (orig_sz / new_sz)
        for orig_sp, orig_sz, new_sz in zip(original_spacing, original_shape, new_shape)
    )
    return new_spacing


def compute_new_affine(
    original_affine: np.ndarray,
    original_shape: Tuple[int, int, int],
    new_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Calculate new affine matrix based on original affine and resize ratio

    Args:
        original_affine: Original 4x4 affine matrix
        original_shape: Original image size (x, y, z)
        new_shape: New image size (x, y, z)

    Returns:
        New 4x4 affine matrix
    """
    scale_factors = np.array([
        original_shape[0] / new_shape[0],
        original_shape[1] / new_shape[1],
        original_shape[2] / new_shape[2]
    ])

    new_affine = original_affine.copy()
    # Scale rotation/scaling part
    new_affine[:3, :3] = original_affine[:3, :3] * scale_factors
    # Translation part remains unchanged (origin position unchanged)

    return new_affine


def resample_to_spacing(
    data: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    is_label: bool = False
) -> np.ndarray:
    """Resample to target spacing"""
    order = 0 if is_label else 3
    mode = 'nearest' if is_label else 'constant'

    zoom_factors = [
        orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
    ]

    resampled = zoom(data, zoom_factors, order=order, mode=mode)
    return resampled


def resize_to_shape(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    is_label: bool = False
) -> np.ndarray:
    """
    Resize to target shape

    Args:
        data: Input array, shape (X, Y, Z) i.e., (W, H, D)
        target_shape: Target shape, format (D, H, W) i.e., zyx order (consistent with config)
        is_label: Whether it is a label (use nearest neighbor interpolation)

    Returns:
        Resized array
    """
    order = 0 if is_label else 3
    mode = 'nearest' if is_label else 'constant'

    # target_shape is in (D, H, W) format, need to convert to (W, H, D) to match numpy array (X, Y, Z)
    # config: [D, H, W] = [192, 256, 256]
    # numpy:  (X, Y, Z) = (W, H, D)
    # So need to reverse: (D, H, W) -> (W, H, D)
    target_shape_xyz = (target_shape[2], target_shape[1], target_shape[0])

    current_shape = data.shape
    zoom_factors = [
        tgt / cur for cur, tgt in zip(current_shape, target_shape_xyz)
    ]

    resized = zoom(data, zoom_factors, order=order, mode=mode)
    return resized


def save_nifti_with_spacing(
    data: np.ndarray,
    output_path: str,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, ...] = None,
    dtype: np.dtype = np.int16
) -> None:
    """
    Save data as NIfTI format with correct spacing

    Args:
        data: Image data, shape (X, Y, Z) or (Z, Y, X)
        output_path: Output file path (.nii.gz)
        spacing: Voxel spacing (x, y, z), unit mm
        origin: Image origin coordinates, default (0, 0, 0)
        direction: Direction cosine matrix (9 elements), default identity matrix
        dtype: Output data type, default int16
    """
    # Ensure correct data type
    data = data.astype(dtype)

    # Create SimpleITK image
    # Note: SimpleITK expects data in (Z, Y, X) order
    if data.ndim == 3:
        # Assume input is (X, Y, Z), need to transpose to (Z, Y, X)
        data_zyx = np.transpose(data, (2, 1, 0))
    else:
        data_zyx = data

    image = sitk.GetImageFromArray(data_zyx)

    # Set spacing (SimpleITK uses x, y, z order)
    image.SetSpacing(spacing)

    # Set origin
    image.SetOrigin(origin)

    # Set direction
    if direction is not None:
        image.SetDirection(direction)
    else:
        # Default identity direction matrix
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    # Save
    sitk.WriteImage(image, output_path)


def resize_and_save_nifti(
    input_path: str,
    output_path: str,
    target_shape: Tuple[int, int, int],
    is_label: bool = False
) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Read NIfTI file, resize to target shape, and save as new NIfTI file (with correct spacing)

    Args:
        input_path: Input NIfTI file path
        output_path: Output NIfTI file path
        target_shape: Target shape (X, Y, Z)
        is_label: Whether it is a label image (use nearest neighbor interpolation)

    Returns:
        (original_shape, new_spacing): Original shape and new spacing
    """
    # Read original image
    image = sitk.ReadImage(input_path)
    data = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    original_spacing = image.GetSpacing()  # (x, y, z)
    origin = image.GetOrigin()
    direction = image.GetDirection()

    # Convert to (X, Y, Z)
    data_xyz = np.transpose(data, (2, 1, 0))
    original_shape = data_xyz.shape

    # Calculate new spacing
    new_spacing = compute_new_spacing(original_shape, target_shape, original_spacing)

    # Resize
    resized_data = resize_to_shape(data_xyz, target_shape, is_label=is_label)

    # Save
    save_nifti_with_spacing(
        resized_data,
        output_path,
        spacing=new_spacing,
        origin=origin,
        direction=direction
    )

    return original_shape, new_spacing


# ============== Parallel Processing ==============

def parallel_process(
    tasks: List,
    process_func: Callable,
    num_workers: int = 20,
    desc: str = "Processing"
) -> List:
    """General parallel processing framework"""
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_func, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=desc) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    return results


# ============== Configuration Loading ==============

def load_json_config(config_path: str) -> dict:
    """Load JSON configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_config(config: dict, config_path: str, indent: int = 2):
    """Save JSON configuration file"""
    dir_path = os.path.dirname(config_path)
    if dir_path:  # Create directory only if dirname is not empty
        os.makedirs(dir_path, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=indent, ensure_ascii=False)


def get_merge_lut(config: dict, scope: str) -> Optional[np.ndarray]:
    """Get the merged LUT for the specified scope from the configuration"""
    merge_scopes = config.get("merge_scopes", {})
    scope_config = merge_scopes.get(scope, {})
    merge_lut_dict = scope_config.get("merge_lut")

    if merge_lut_dict is None:
        return None

    return create_lut_from_config(merge_lut_dict)


def get_merged_label_names(config: dict, scope: str) -> Dict[int, str]:
    """Get merged label names"""
    merge_scopes = config.get("merge_scopes", {})
    scope_config = merge_scopes.get(scope, {})
    names = scope_config.get("merged_label_names", {})
    return {int(k): v for k, v in names.items()}


def get_config_path(config_name: str) -> Path:
    """Get configuration file path"""
    return CONFIG_DIR / f"{config_name}.json"


# ============== Report Processing ==============

def format_report(findings: str = "", impressions: str = "") -> str:
    """
    Format report text

    Args:
        findings: Findings text
        impressions: Impressions text

    Returns:
        Formatted report text in the format "Findings: {findings} Impressions: {impressions}"
    """
    parts = []

    # Process findings
    findings_str = str(findings) if findings is not None else ""
    if findings_str and findings_str.lower() not in ('nan', 'none', ''):
        parts.append(f"Findings: {findings_str}")

    # Process impressions
    impressions_str = str(impressions) if impressions is not None else ""
    if impressions_str and impressions_str.lower() not in ('nan', 'none', ''):
        parts.append(f"Impressions: {impressions_str}")

    return " ".join(parts) if parts else ""


# ============== CSV Utilities ==============

def generate_csv_index(
    data_dir: str,
    output_csv: str,
    id_column: str = "series_id",
    pattern: str = "*.nii.gz"
) -> pd.DataFrame:
    """Generate CSV index"""
    from glob import glob

    files = sorted(glob(os.path.join(data_dir, "**", pattern), recursive=True))
    records = []

    for file_path in tqdm(files, desc="Generating CSV"):
        filename = os.path.basename(file_path)
        case_id = filename.replace(".nii.gz", "").replace(".b2nd", "")

        nii = nib.load(file_path)
        shape = nii.shape

        records.append({
            id_column: case_id,
            "file_path": file_path,
            "shape": str(shape)
        })

    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")
    return df


def generate_seg_csv(
    label_dir: str,
    output_csv: str,
    image_dir: Optional[str] = None,
    split_file: Optional[str] = None,
    id_column: str = "series_id"
) -> pd.DataFrame:
    """
    Generate CSV index for segmentation tasks (general version)

    Args:
        label_dir: Label file directory
        output_csv: Output CSV path
        image_dir: Image file directory (optional, for adding image_path column)
        split_file: Split file path (optional, for filtering)
        id_column: ID column name

    Returns:
        Generated DataFrame
    """
    # Get all label files
    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.endswith((".nii.gz", ".nii"))
    ])

    # Filter if split file exists
    if split_file and os.path.exists(split_file):
        split_df = pd.read_csv(split_file)
        id_col = split_df.columns[0]
        valid_ids = set(split_df[id_col].astype(str).tolist())
        label_files = [
            f for f in label_files
            if f.replace(".nii.gz", "").replace(".nii", "") in valid_ids
        ]

    records = []
    for label_file in tqdm(label_files, desc="Generating seg CSV"):
        case_id = label_file.replace(".nii.gz", "").replace(".nii", "")
        label_path = os.path.join(label_dir, label_file)

        nii = nib.load(label_path)
        shape = nii.shape

        record = {
            id_column: case_id,
            "case_id": case_id,
            "seg_path": label_path,
            "seg_shape": str(shape)
        }

        # Add image path (if provided)
        if image_dir:
            image_path = os.path.join(image_dir, label_file)
            if os.path.exists(image_path):
                record["image_path"] = image_path

        records.append(record)
    df = pd.DataFrame(records)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")

    return df


# ============== Dataset Splitting ==============

def split_by_ids(
    df: pd.DataFrame,
    train_ids_file: str,
    val_ids_file: str,
    id_column: str = "case_id"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset based on ID files"""
    train_ids_df = pd.read_csv(train_ids_file)
    val_ids_df = pd.read_csv(val_ids_file)

    train_id_col = train_ids_df.columns[0]
    val_id_col = val_ids_df.columns[0]

    train_ids = set(train_ids_df[train_id_col].tolist())
    val_ids = set(val_ids_df[val_id_col].tolist())

    train_df = df[df[id_column].isin(train_ids)]
    val_df = df[df[id_column].isin(val_ids)]

    return train_df, val_df


def load_split_ids(split_file: str) -> set:
    """
    Load split ID file and return ID set

    Supported formats:
    - Single-column CSV (e.g., IID_train.csv from AbdomenAtlas, column name "BDMAP ID")
    - Multi-column CSV (use first column as ID)

    Args:
        split_file: Path to split file

    Returns:
        Set of IDs
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    df = pd.read_csv(split_file)
    id_col = df.columns[0]
    return set(df[id_col].astype(str).tolist())


def split_by_iid_ood(
    df: pd.DataFrame,
    split_dir: str,
    id_column: str = "case_id"
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset based on IID/OOD split files (for AbdomenAtlas)

    Note: IID and OOD are two independent experiments:
    - IID experiment: Train on IID_train, test on IID_test
    - OOD experiment: Train on OOD_train, test on OOD_test
    - IID_train + IID_test = OOD_train + OOD_test (same total count)

    Expected files in split_dir:
    - IID_train.csv
    - IID_test.csv
    - OOD_train.csv
    - OOD_test.csv

    Args:
        df: DataFrame to split
        split_dir: Directory containing split files
        id_column: ID column name in DataFrame

    Returns:
        Dictionary {"iid_train": df, "iid_test": df, "ood_train": df, "ood_test": df}
    """
    split_files = {
        "iid_train": os.path.join(split_dir, "IID_train.csv"),
        "iid_test": os.path.join(split_dir, "IID_test.csv"),
        "ood_train": os.path.join(split_dir, "OOD_train.csv"),
        "ood_test": os.path.join(split_dir, "OOD_test.csv"),
    }

    result = {}
    for split_name, split_file in split_files.items():
        if os.path.exists(split_file):
            ids = load_split_ids(split_file)
            split_df = df[df[id_column].astype(str).isin(ids)]
            result[split_name] = split_df
            print(f"  {split_name}: {len(split_df)} samples")
        else:
            result[split_name] = pd.DataFrame()

    return result


def split_by_filename_prefix(
    df: pd.DataFrame,
    filename_column: str = "volume_name",
    train_prefix: str = "train_",
    val_prefix: str = "valid_"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset based on filename prefix (for CT-RATE)

    CT-RATE volume_name format: train_1_a_1.nii.gz, valid_1_a_1.nii.gz

    Args:
        df: DataFrame to split
        filename_column: Filename column
        train_prefix: Training set prefix
        val_prefix: Validation set prefix

    Returns:
        (train_df, val_df)
    """
    # Handle possible column name variations
    if filename_column not in df.columns:
        for col in ["VolumeName", "volume_name", "series_id"]:
            if col in df.columns:
                filename_column = col
                break

    if filename_column not in df.columns:
        raise ValueError(f"Cannot find filename column in DataFrame. Columns: {df.columns.tolist()}")

    train_mask = df[filename_column].astype(str).str.startswith(train_prefix)
    val_mask = df[filename_column].astype(str).str.startswith(val_prefix)

    train_df = df[train_mask]
    val_df = df[val_mask]

    return train_df, val_df


def generate_splits_csv(
    merged_df: pd.DataFrame,
    output_dir: str,
    split_config: dict,
    id_column: str = "case_id"
) -> Dict[str, str]:
    """
    Unified function to generate split CSV files

    Args:
        merged_df: Merged DataFrame
        output_dir: Output directory
        split_config: Split configuration, format:
            {
                "method": "iid_ood" | "prefix" | "ids",
                "split_dir": "...",  # for iid_ood
                "filename_column": "...",  # for prefix
                "train_ids_file": "...",  # for ids
                "val_ids_file": "..."  # for ids
            }
        id_column: ID column name

    Returns:
        Dictionary of generated CSV file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}

    method = split_config.get("method", "ids")

    if method == "iid_ood":
        # AbdomenAtlas style: IID/OOD splits
        split_dir = split_config.get("split_dir")
        if not split_dir:
            raise ValueError("split_dir required for iid_ood method")

        splits = split_by_iid_ood(merged_df, split_dir, id_column)

        for split_name, split_df in splits.items():
            if not split_df.empty:
                output_path = os.path.join(output_dir, f"{split_name}_merged.csv")
                split_df.to_csv(output_path, index=False)
                output_paths[split_name] = output_path
                print(f"Saved {split_name}: {len(split_df)} records to {output_path}")

    elif method == "prefix":
        # CT-RATE style: filename prefix
        filename_column = split_config.get("filename_column", "volume_name")
        train_prefix = split_config.get("train_prefix", "train_")
        val_prefix = split_config.get("val_prefix", "valid_")

        train_df, val_df = split_by_filename_prefix(
            merged_df, filename_column, train_prefix, val_prefix
        )

        train_path = os.path.join(output_dir, "train_merged.csv")
        val_path = os.path.join(output_dir, "validation_merged.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        output_paths["train"] = train_path
        output_paths["validation"] = val_path

        print(f"Saved train: {len(train_df)} records to {train_path}")
        print(f"Saved validation: {len(val_df)} records to {val_path}")

    else:  # method == "ids"
        # Standard: separate ID files
        train_ids_file = split_config.get("train_ids_file")
        val_ids_file = split_config.get("val_ids_file")

        if not train_ids_file or not val_ids_file:
            raise ValueError("train_ids_file and val_ids_file required for ids method")

        train_df, val_df = split_by_ids(merged_df, train_ids_file, val_ids_file, id_column)

        train_path = os.path.join(output_dir, "train_merged.csv")
        val_path = os.path.join(output_dir, "validation_merged.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        output_paths["train"] = train_path
        output_paths["validation"] = val_path

        print(f"Saved train: {len(train_df)} records to {train_path}")
        print(f"Saved validation: {len(val_df)} records to {val_path}")

    return output_paths


# ============== Foreground Segmentation Cropping ==============

def get_foreground_bbox(
    mask: np.ndarray,
    margin: int = 5
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calculate the foreground bounding box from a mask.

    Args:
        mask: Segmentation mask (D, H, W) or (H, W, D)
        margin: Number of pixels to expand the boundary

    Returns:
        ((z_start, z_end), (y_start, y_end), (x_start, x_end))
    """
    nonzero = np.where(mask > 0)

    if len(nonzero[0]) == 0:
        return ((0, mask.shape[0]), (0, mask.shape[1]), (0, mask.shape[2]))

    z_min, z_max = nonzero[0].min(), nonzero[0].max()
    y_min, y_max = nonzero[1].min(), nonzero[1].max()
    x_min, x_max = nonzero[2].min(), nonzero[2].max()

    z_start = max(0, z_min - margin)
    z_end = min(mask.shape[0], z_max + margin + 1)
    y_start = max(0, y_min - margin)
    y_end = min(mask.shape[1], y_max + margin + 1)
    x_start = max(0, x_min - margin)
    x_end = min(mask.shape[2], x_max + margin + 1)

    return ((z_start, z_end), (y_start, y_end), (x_start, x_end))


def crop_array_by_bbox(
    data: np.ndarray,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
) -> np.ndarray:
    """Crop an array according to a bounding box."""
    (z_start, z_end), (y_start, y_end), (x_start, x_end) = bbox
    return data[z_start:z_end, y_start:y_end, x_start:x_end]


def load_combined_mask(
    mask_dir: str,
    target_orientation: str = "LPS",
    mask_files: Optional[List[str]] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[nib.Nifti1Header]]:
    """
    Load and combine all masks from a directory.

    Args:
        mask_dir: Path to the mask directory containing .nii.gz files
        target_orientation: Unified orientation for all masks
        mask_files: Optional list of mask filenames (without extension) to load.
                    If None, load all .nii.gz files in the directory.

    Returns:
        (combined_mask, affine, header) Returns (None, None, None) if no masks are found.
    """
    combined = None
    affine = None
    header = None

    # Get the list of files to process
    if mask_files is not None:
        files_to_load = [f"{name}.nii.gz" for name in mask_files]
    else:
        if not os.path.exists(mask_dir):
            return None, None, None
        files_to_load = [f for f in os.listdir(mask_dir) if f.endswith(('.nii.gz', '.nii'))]

    for filename in files_to_load:
        file_path = os.path.join(mask_dir, filename)
        if not os.path.exists(file_path):
            continue

        # Load and reorient
        nii = nib.load(file_path)
        nii = reorient_nifti(nii, target_orientation)
        data = np.asanyarray(nii.dataobj)

        if combined is None:
            combined = np.zeros(data.shape, dtype=np.uint8)
            affine = nii.affine
            header = nii.header

        combined[data > 0] = 1

    return combined, affine, header


def crop_by_mask(
    image_path: str,
    mask_path: Optional[str] = None,
    output_path: Optional[str] = None,
    margin: int = 10,
    target_orientation: str = "LPS",
    mask_files: Optional[List[str]] = None
) -> Tuple[np.ndarray, dict]:
    """
    Generic mask-based cropping function.

    Args:
        image_path: Path to the image file (.nii.gz)
        mask_path: Mask path, which can be:
                   - None: Skip cropping and return the original image (optionally saved)
                   - Directory path: Combine all .nii.gz files in the directory as the mask
                   - Single .nii.gz file path: Use this mask directly
        output_path: Output path (optional)
        margin: Number of pixels to expand the boundary
        target_orientation: Unified orientation (for aligning image and mask)
        mask_files: Optional list of mask filenames to use when mask_path is a directory

    Returns:
        (data, info_dict)
    """
    # Load image and reorient
    nii = nib.load(image_path)
    nii = reorient_nifti(nii, target_orientation)
    data = np.asanyarray(nii.dataobj)
    affine = nii.affine
    header = nii.header

    # If mask_path is None, skip cropping
    if mask_path is None:
        info = {
            "bbox": None,
            "original_shape": data.shape,
            "cropped_shape": data.shape,
            "skipped": True
        }
        # If saving is required, save the original image directly
        if output_path:
            save_nifti(output_path, data, affine, header)
        return data, info

    # Load mask
    if os.path.isdir(mask_path):
        mask, _, _ = load_combined_mask(mask_path, target_orientation, mask_files)
    else:
        mask_nii = nib.load(mask_path)
        mask_nii = reorient_nifti(mask_nii, target_orientation)
        mask = np.asanyarray(mask_nii.dataobj)

    if mask is None:
        print(f"Warning: No mask found at {mask_path}")
        if output_path:
            save_nifti(output_path, data, affine, header)
        return data, {"bbox": None, "original_shape": data.shape, "cropped_shape": data.shape}

    # Check if shapes match
    if mask.shape != data.shape:
        print(f"Warning: Shape mismatch - image {data.shape}, mask {mask.shape}")
        if output_path:
            save_nifti(output_path, data, affine, header)
        return data, {"bbox": None, "original_shape": data.shape, "cropped_shape": data.shape, "error": "shape_mismatch"}

    # Calculate bbox and crop
    bbox = get_foreground_bbox(mask, margin)
    cropped = crop_array_by_bbox(data, bbox)
    info = {
        "bbox": bbox,
        "original_shape": data.shape,
        "cropped_shape": cropped.shape
    }

    # Save
    if output_path:
        (z_start, _), (y_start, _), (x_start, _) = bbox
        new_affine = affine.copy()
        new_affine[:3, 3] += new_affine[:3, :3] @ np.array([x_start, y_start, z_start])

        save_nifti(output_path, cropped, new_affine, header)

    return cropped, info


def batch_crop_by_mask(
    input_dir: str,
    output_dir: str,
    mask_base_dir: Optional[str] = None,
    margin: int = 10,
    target_orientation: str = "LPS",
    mask_subdir: Optional[str] = None,
    mask_files: Optional[List[str]] = None,
    num_workers: int = 20
) -> List[dict]:
    """
    Batch cropping

    Args:
        input_dir: Input image directory
        output_dir: Output directory
        mask_base_dir: Base directory for masks; if None, skip cropping (only reorient and copy)
        margin: Margin expansion
        target_orientation: Unified orientation
        mask_subdir: Mask subdirectory name (e.g., "segmentations"). If None, search directly under mask_base_dir/case_id/
        mask_files: Optional, list of specified mask filenames to use
        num_workers: Number of parallel processes

    Directory structure example (when mask_base_dir is not None):
    - If mask_subdir="segmentations":
        mask_base_dir/case_id/segmentations/*.nii.gz
    - If mask_subdir=None:
        mask_base_dir/case_id/*.nii.gz
        or mask_base_dir/case_id.nii.gz (single mask file)
    """
    os.makedirs(output_dir, exist_ok=True)

    nii_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.nii.gz', '.nii'))
    ])

    print(f"Found {len(nii_files)} NIfTI files to process")
    if mask_base_dir is None:
        print("mask_base_dir is None, skipping crop (reorient only)")

    def process_single(filename):
        case_id = filename.replace('.nii.gz', '').replace('.nii', '')
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # If mask_base_dir is None, skip cropping
        if mask_base_dir is None:
            _, info = crop_by_mask(
                input_path, None, output_path,
                margin=margin,
                target_orientation=target_orientation,
                mask_files=mask_files
            )
            return {
                "case_id": case_id,
                "status": "success",
                **info
            }

        # Determine mask path
        if mask_subdir:
            mask_path = os.path.join(mask_base_dir, case_id, mask_subdir)
        else:
            mask_path = os.path.join(mask_base_dir, case_id)

        # If folder does not exist, try to find a single mask file
        if not os.path.exists(mask_path):
            single_mask = os.path.join(mask_base_dir, f"{case_id}.nii.gz")
            if os.path.exists(single_mask):
                mask_path = single_mask
            else:
                return {"case_id": case_id, "status": "skipped", "reason": "mask_not_found"}

        _, info = crop_by_mask(
            input_path, mask_path, output_path,
            margin=margin,
            target_orientation=target_orientation,
            mask_files=mask_files
        )

        return {
            "case_id": case_id,
            "status": "success",
            **info
        }

    desc = "Reorienting" if mask_base_dir is None else "Cropping by mask"
    results = parallel_process(nii_files, process_single, num_workers, desc)

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"Successfully processed: {success}/{len(nii_files)}")

    return results


if __name__ == "__main__":
    print("Common utils loaded successfully!")
    print(f"Config dir: {CONFIG_DIR}")


# ============== Comprehensive Image Processing ==============

def process_single_image(
    input_path: str,
    output_path: str,
    processing_params: Dict,
    mask_path: Optional[str] = None
) -> Dict:
    """
    Comprehensive image processing: orientation unification -> cropping (optional) -> resize -> format conversion

    Args:
        input_path: Input image path
        output_path: Output path (without extension; will be added automatically based on output_format)
        processing_params: Processing parameters dictionary, containing:
            - target_orientation: Target orientation (default "LPS")
            - target_size: Target size [D, H, W] (optional)
            - resize_method: Resize method "trilinear"/"nearest" (default "trilinear")
            - crop_abdomen/crop_chest: Whether to crop (default False)
            - crop_margin: Cropping margin [D, H, W] (default [10, 10, 10])
            - output_format: Output format "nifti"/"blosc2" (default "blosc2")
            - compression_level: Compression level (default 5)
        mask_path: Mask path for cropping (optional)

    Returns:
        Processing information dictionary
    """
    # Parse parameters
    target_orientation = processing_params.get('target_orientation', 'LPS')
    target_size = processing_params.get('target_size')
    resize_method = processing_params.get('resize_method', 'trilinear')
    crop_abdomen = processing_params.get('crop_abdomen', False)
    crop_chest = processing_params.get('crop_chest', False)
    crop_margin = processing_params.get('crop_margin', [10, 10, 10])
    output_format = processing_params.get('output_format', 'blosc2')
    compression_level = processing_params.get('compression_level', 5)

    info = {
        'input_path': input_path,
        'output_path': output_path,
        'status': 'success'
    }

    # 1. Load and reorient
    nii = nib.load(input_path)
    nii = reorient_nifti(nii, target_orientation)
    data = np.asanyarray(nii.dataobj)
    original_dtype = data.dtype  # Preserve original data type
    affine = nii.affine
    header = nii.header

    info['original_shape'] = data.shape
    info['original_dtype'] = str(original_dtype)
    info['orientation'] = target_orientation

    # 2. Cropping (optional)
    if (crop_abdomen or crop_chest) and mask_path:
        if os.path.exists(mask_path):
            # Load mask
            if os.path.isdir(mask_path):
                mask, _, _ = load_combined_mask(mask_path, target_orientation)
            else:
                mask_nii = nib.load(mask_path)
                mask_nii = reorient_nifti(mask_nii, target_orientation)
                mask = np.asanyarray(mask_nii.dataobj)

            if mask is not None and mask.shape == data.shape:
                margin = crop_margin[0] if isinstance(crop_margin, list) else crop_margin
                bbox = get_foreground_bbox(mask, margin)
                data = crop_array_by_bbox(data, bbox)
                info['bbox'] = bbox
                info['cropped_shape'] = data.shape

    # 3. Resize (optional)
    if target_size:
        target_size = tuple(target_size)
        if data.shape != target_size:
            is_label = False
            data = resize_to_shape(data, target_size, is_label=is_label)
            info['resized_shape'] = data.shape

    # 4. Save
    final_shape = data.shape
    info['final_shape'] = final_shape

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Preserve original data type when saving (only blosc2 and nifti supported)
    if output_format == 'blosc2':
        final_path = write_blosc2_file(output_path, data.astype(original_dtype), compression_level)
    else:  # nifti
        final_path = output_path + '.nii.gz'
        # Update affine to reflect resize
        if target_size and info.get('original_shape') != target_size:
            scale_factors = [o / t for o, t in zip(info['original_shape'], target_size)]
            new_affine = affine.copy()
            new_affine[:3, :3] = affine[:3, :3] * np.diag(scale_factors)
        else:
            new_affine = affine
        save_nifti(final_path, data.astype(original_dtype), new_affine, header)

    info['final_path'] = final_path
    info['output_format'] = output_format
    info['final_dtype'] = str(original_dtype)

    return info


def process_single_mask(
    input_path: str,
    output_path: str,
    processing_params: Dict,
    reference_bbox: Optional[Tuple] = None
) -> Dict:
    """
    Process segmentation mask: orientation standardization -> cropping (optional) -> resize -> format conversion

    Args:
        input_path: Input mask path
        output_path: Output path
        processing_params: Processing parameters
        reference_bbox: Reference bounding box (consistent with image)

    Returns:
        Processing information dictionary
    """
    target_orientation = processing_params.get('target_orientation', 'LPS')
    target_size = processing_params.get('target_size')
    output_format = processing_params.get('output_format', 'blosc2')
    compression_level = processing_params.get('compression_level', 5)

    info = {
        'input_path': input_path,
        'output_path': output_path,
        'status': 'success'
    }

    # 1. Load and reorient
    nii = nib.load(input_path)
    nii = reorient_nifti(nii, target_orientation)
    data = np.asanyarray(nii.dataobj).astype(np.uint8)
    affine = nii.affine
    header = nii.header

    info['original_shape'] = data.shape

    # 2. Crop (using reference bounding box)
    if reference_bbox:
        data = crop_array_by_bbox(data, reference_bbox)
        info['cropped_shape'] = data.shape

    # 3. Resize (using nearest neighbor interpolation)
    if target_size:
        target_size = tuple(target_size)
        if data.shape != target_size:
            data = resize_to_shape(data, target_size, is_label=True)
            info['resized_shape'] = data.shape

    # 4. Save
    info['final_shape'] = data.shape

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Only blosc2 and nifti supported
    if output_format == 'blosc2':
        final_path = write_blosc2_file(output_path, data.astype(np.uint8), compression_level)
    else:  # nifti
        final_path = output_path + '.nii.gz'
        save_nifti(final_path, data, affine, header)

    info['final_path'] = final_path

    return info


def batch_process_images(
    input_dir: str,
    output_dir: str,
    processing_params: Dict,
    mask_base_dir: Optional[str] = None,
    mask_subdir: Optional[str] = None,
    num_workers: int = 20
) -> List[Dict]:
    """
    Batch process images

    Args:
        input_dir: Input image directory
        output_dir: Output directory
        processing_params: Processing parameters
        mask_base_dir: Base directory for masks (used for cropping)
        mask_subdir: Mask subdirectory name
        num_workers: Number of parallel processes

    Returns:
        List of processing results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all NIfTI files
    nii_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.nii.gz', '.nii'))
    ])

    print(f"Found {len(nii_files)} images to process")
    print(f"Processing params: {processing_params}")

    def process_single(filename):
        case_id = filename.replace('.nii.gz', '').replace('.nii', '')
        input_path = os.path.join(input_dir, filename)

        # Output path (without extension)
        output_path = os.path.join(output_dir, case_id)

        # Determine mask path
        mask_path = None
        if mask_base_dir:
            if mask_subdir:
                mask_path = os.path.join(mask_base_dir, case_id, mask_subdir)
            else:
                mask_path = os.path.join(mask_base_dir, case_id)

            if not os.path.exists(mask_path):
                single_mask = os.path.join(mask_base_dir, f"{case_id}.nii.gz")
                if os.path.exists(single_mask):
                    mask_path = single_mask
                else:
                    mask_path = None

        result = process_single_image(input_path, output_path, processing_params, mask_path)
        result['case_id'] = case_id
        return result

    results = parallel_process(nii_files, process_single, num_workers, "Processing images")

    success = sum(1 for r in results if r.get('status') == 'success')
    print(f"Successfully processed: {success}/{len(nii_files)}")

    return results


def batch_process_masks(
    input_dir: str,
    output_dir: str,
    processing_params: Dict,
    bbox_info: Optional[Dict[str, Tuple]] = None,
    num_workers: int = 20
) -> List[Dict]:
    """
    Batch process segmentation masks

    Args:
        input_dir: Input mask directory
        output_dir: Output directory
        processing_params: Processing parameters
        bbox_info: Mapping of case_id -> bounding box (consistent with images)
        num_workers: Number of parallel processes

    Returns:
        List of processing results
    """
    os.makedirs(output_dir, exist_ok=True)

    nii_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.nii.gz', '.nii'))
    ])

    print(f"Found {len(nii_files)} masks to process")

    def process_single(filename):
        case_id = filename.replace('.nii.gz', '').replace('.nii', '')
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, case_id)

        reference_bbox = bbox_info.get(case_id) if bbox_info else None

        result = process_single_mask(input_path, output_path, processing_params, reference_bbox)
        result['case_id'] = case_id
        return result

    results = parallel_process(nii_files, process_single, num_workers, "Processing masks")

    success = sum(1 for r in results if r.get('status') == 'success')
    print(f"Successfully processed: {success}/{len(nii_files)}")

    return results


# ============== Complete Preprocessing General Functions ==============

def compute_class_locations(
    seg: np.ndarray,
    all_labels: List[int] = None,
    num_samples: int = 10000,
    min_percent_coverage: float = 0.01,
    seed: int = 1234
) -> dict:
    """
    Compute foreground locations (sampled) for each class, used for foreground oversampling.

    Reference nnUNet implementation, sampling only a subset of coordinates to reduce pkl file size.

    Args:
        seg: Segmentation mask
        all_labels: List of all labels
        num_samples: Maximum number of points to sample per class (default 10000)
        min_percent_coverage: Minimum coverage percentage (default 1%)
        seed: Random seed

    Returns:
        {label: sampled_locations, ...}
    """
    if seg.ndim == 4:
        seg = seg[0]

    rndst = np.random.RandomState(seed)

    foreground_mask = seg > 0
    if not np.any(foreground_mask):
        if all_labels is None:
            all_labels = []
        return {label: np.array([], dtype=np.int32).reshape(0, 3) for label in all_labels}

    if all_labels is None:
        all_labels = sorted(set(int(v) for v in seg[foreground_mask]))

    class_locations = {}
    for label in all_labels:
        all_locs = np.argwhere(seg == label)
        if len(all_locs) == 0:
            class_locations[label] = np.array([], dtype=np.int32).reshape(0, 3)
            continue

        # Sampling: at most num_samples, but at least covering min_percent_coverage
        target_num = min(num_samples, len(all_locs))
        target_num = max(target_num, int(np.ceil(len(all_locs) * min_percent_coverage)))
        target_num = min(target_num, len(all_locs))  # Cannot exceed total count

        if target_num < len(all_locs):
            selected_idx = rndst.choice(len(all_locs), target_num, replace=False)
            selected = all_locs[selected_idx]
        else:
            selected = all_locs

        # Use int32 instead of int64 to save space
        class_locations[label] = selected.astype(np.int32)

    return class_locations


def load_and_reorient_nifti(
    file_path: str,
    target_orientation: str = "LPS"
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load NIfTI and reorient."""
    nii = nib.load(file_path)
    nii = reorient_nifti(nii, target_orientation)
    data = np.asanyarray(nii.dataobj)
    return data, nii.affine, nii.header


def save_compressed(
    file_path: str,
    data: np.ndarray,
    output_format: str = "blosc2",
    compression_level: int = 5
) -> str:
    """Save as Blosc2 compressed format."""
    if not file_path.endswith('.b2nd'):
        file_path = file_path + '.b2nd'
    write_blosc2_file(file_path, data, compression_level)
    return file_path


def save_pkl(file_path: str, properties: dict):
    """Save pkl file."""
    import pickle
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(properties, f)


def save_processing_log(results: List[dict], log_path: str):
    """Save processing log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, tuple):
                sr[k] = list(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable_results.append(sr)
    with open(log_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Log saved to: {log_path}")


# ============== Multi-Scope Batch Processing Support ==============

def get_all_merge_scopes(config: Dict) -> Dict[str, Dict]:
    """
    Get all merge_scopes from the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dict[scope_name, scope_config] containing configuration for each scope
    """
    return config.get("merge_scopes", {})


def get_all_merge_luts(config: Dict, max_label: int = 256) -> Dict[str, Optional[np.ndarray]]:
    """
    Get merge LUTs for all scopes.

    Args:
        config: Configuration dictionary
        max_label: Maximum label value for LUT

    Returns:
        Dict[scope_name, lut] - lut being None indicates no merging (keep original labels)
    """
    merge_scopes = get_all_merge_scopes(config)
    luts = {}

    for scope_name, scope_config in merge_scopes.items():
        merge_lut_dict = scope_config.get("merge_lut")
        if merge_lut_dict is None:
            luts[scope_name] = None
        else:
            luts[scope_name] = create_lut_from_config(merge_lut_dict, max_label)

    return luts


def apply_multi_scope_luts(
    mask: np.ndarray,
    luts: Dict[str, Optional[np.ndarray]]
) -> Dict[str, np.ndarray]:
    """
    Apply multiple scope LUTs to a mask, generating multiple merged masks.

    Args:
        mask: Original mask array
        luts: Dict[scope_name, lut] - lut being None indicates no merging

    Returns:
        Dict[scope_name, merged_mask]
    """
    results = {}
    for scope_name, lut in luts.items():
        if lut is None:
            # No merging, keep original labels
            results[scope_name] = mask.copy()
        else:
            results[scope_name] = apply_lut(mask, lut)
    return results


def get_scope_output_dir(base_output_dir: str, scope_name: str) -> str:
    """
    Get output directory for a specific scope.

    Args:
        base_output_dir: Base output directory
        scope_name: Scope name

    Returns:
        Output directory path for the scope
    """
    return os.path.join(base_output_dir, scope_name)


def get_scope_num_classes(config: Dict, scope_name: str) -> int:
    """
    Get the number of classes for a specific scope

    Args:
        config: Configuration dictionary
        scope_name: Scope name

    Returns:
        Number of classes (including background)
    """
    merge_scopes = get_all_merge_scopes(config)
    scope_config = merge_scopes.get(scope_name, {})
    return scope_config.get("num_merged_classes", 0)


# ============== General Preprocessing Pipeline ==============

def _safe_process_single_case(args) -> dict:
    """
    Wrapper for single case processing with error handling
    Deletes raw file on failure and marks for restart
    """
    case_id, image_path, label_path, output_dir, target_size, scopes_config, save_debug_nii, compression_level, target_orientation, raw_input_dir = args

    try:
        result = _process_single_case_impl(
            case_id, image_path, label_path, output_dir, target_size,
            scopes_config, save_debug_nii, compression_level, target_orientation
        )
        return result
    except Exception as e:
        # Processing failed, delete raw file for regeneration on next run
        delete_raw_case(raw_input_dir, case_id)
        return {
            "case_id": case_id,
            "status": "failed_deleted",
            "error": str(e),
            "needs_restart": True
        }


def _process_single_case_impl(
    case_id: str,
    image_path: str,
    label_path: str,
    output_dir: str,
    target_size: Tuple[int, int, int],
    scopes_config: dict,
    save_debug_nii: bool,
    compression_level: int,
    target_orientation: str
) -> dict:
    """
    Actual implementation for processing a single case: processes image + all scope labels simultaneously
    More efficient by completing one case in a single pass
    """
    # target_size is [D, H, W] = [z, y, x], convert to (x, y, z) order
    target_shape_xyz = (target_size[2], target_size[1], target_size[0])

    # Load original image and reorient to target orientation
    img_nii = nib.load(image_path)
    img_nii = reorient_nifti(img_nii, target_orientation)
    img_data = img_nii.get_fdata().astype(np.float32)  # xyz order
    original_spacing = tuple(abs(s) for s in img_nii.header.get_zooms()[:3])
    original_shape = img_data.shape

    # Load label and reorient to target orientation
    label_nii = nib.load(label_path)
    label_nii = reorient_nifti(label_nii, target_orientation)
    label_data = label_nii.get_fdata().astype(np.uint8)  # xyz order

    # Resize image (using xyz-ordered target_shape)
    zoom_factors = [t / s for t, s in zip(target_shape_xyz, original_shape)]
    img_resized = zoom(img_data, zoom_factors, order=1, mode='constant')

    # Calculate new spacing (xyz order)
    new_spacing = compute_new_spacing(original_shape, target_shape_xyz, original_spacing)

    # Save image blosc2 (int16)
    img_output_dir = os.path.join(output_dir, "imagesTr")
    os.makedirs(img_output_dir, exist_ok=True)
    img_output_path = os.path.join(img_output_dir, f"{case_id}.b2nd")
    write_blosc2_file(img_output_path, img_resized.astype(np.int16), compression_level)

    # Process label for each scope
    label_results = {}
    for scope_name, scope_info in scopes_config.items():
        merge_lut = scope_info.get("merge_lut")
        num_classes = scope_info.get("num_classes", 0)

        # Apply class merging
        if merge_lut is not None:
            lut = create_lut_from_config(merge_lut)
            label_merged = apply_lut(label_data, lut)
        else:
            label_merged = label_data.copy()

        # Resize label
        label_resized = zoom(label_merged, zoom_factors, order=0, mode='constant')

        # Save label blosc2
        label_output_dir = os.path.join(output_dir, f"labelsTr_{scope_name}")
        os.makedirs(label_output_dir, exist_ok=True)
        label_id = case_id[:-5] if case_id.endswith('_0000') else case_id
        label_output_path = os.path.join(label_output_dir, f"{label_id}.b2nd")
        write_blosc2_file(label_output_path, label_resized.astype(np.uint8), compression_level)

        # Compute class_locations and save pkl
        all_labels = list(range(1, num_classes)) if num_classes > 0 else None
        class_locations = compute_class_locations(label_resized, all_labels)

        # Build properties
        properties = {
            "shape": label_resized.shape,
            "spacing": new_spacing,
            "class_locations": class_locations,
            "original_shape": original_shape,
            "original_spacing": original_spacing,
        }

        # Save pkl file
        pkl_output_path = os.path.join(label_output_dir, f"{label_id}.pkl")
        save_pkl(pkl_output_path, properties)

        label_results[scope_name] = {
            "blosc2_path": label_output_path,
            "pkl_path": pkl_output_path
        }

    # Optional: Save debug nii.gz
    debug_paths = {}
    if save_debug_nii:
        debug_dir = os.path.join(output_dir, "debug_nii")
        os.makedirs(debug_dir, exist_ok=True)

        # LPS orientation affine: x(L) negative, y(P) negative, z(S) positive
        new_affine = np.diag([-new_spacing[0], -new_spacing[1], new_spacing[2], 1.0])

        # Save image (int16)
        debug_img_path = os.path.join(debug_dir, f"{case_id}.nii.gz")
        nib.save(nib.Nifti1Image(img_resized.astype(np.int16), new_affine), debug_img_path)
        debug_paths["debug_img_path"] = debug_img_path

        # Save label for each scope
        for scope_name, scope_info in scopes_config.items():
            merge_lut = scope_info.get("merge_lut")
            if merge_lut is not None:
                lut = create_lut_from_config(merge_lut)
                label_merged = apply_lut(label_data, lut)
            else:
                label_merged = label_data.copy()
            label_resized = zoom(label_merged, zoom_factors, order=0, mode='constant')

            label_id = case_id[:-5] if case_id.endswith('_0000') else case_id
            debug_label_path = os.path.join(debug_dir, f"{label_id}_label_{scope_name}.nii.gz")
            nib.save(nib.Nifti1Image(label_resized.astype(np.uint8), new_affine), debug_label_path)
            debug_paths[f"debug_label_{scope_name}"] = debug_label_path

    return {
        "case_id": case_id,
        "status": "success",
        "img_blosc2_path": img_output_path,
        "label_results": label_results,
        "original_shape": original_shape,
        "new_shape": target_shape_xyz,
        "original_spacing": original_spacing,
        "new_spacing": new_spacing,
        **debug_paths
    }


def resize_dataset_to_preprocessed(
    raw_input_dir: str,
    preprocessed_output_dir: str,
    config: dict,
    scopes: List[str] = None,
    target_size: Tuple[int, int, int] = (192, 256, 256),
    target_orientation: str = "LPS",
    num_workers: int = 20,
    save_debug_nii: bool = True,
    debug_nii_sample_size: int = 10,
    case_ids: List[str] = None,
    compression_level: int = 8,
    split: str = "train"
) -> List[dict]:
    """
    General resize + blosc2 compression pipeline

    Args:
        raw_input_dir: nnUNet_raw input directory
        preprocessed_output_dir: nnUNet_preprocessed output directory
        config: Dataset configuration
        scopes: List of scopes to process, None means all
        target_size: Target size (D, H, W)
        target_orientation: Target orientation (default "LPS")
        num_workers: Number of parallel processes
        save_debug_nii: Whether to save debug nii.gz
        debug_nii_sample_size: Random sample size for saving debug nii (default 10)
        case_ids: Specific cases to process, None means all
        compression_level: blosc2 compression level
        split: "train" or "val" (validation set)

    Returns:
        List of processing results
    """
    import random
    import shutil

    # Copy dataset.json from raw to preprocessed (required by nnUNet training)
    os.makedirs(preprocessed_output_dir, exist_ok=True)
    src_dataset_json = os.path.join(raw_input_dir, "dataset.json")
    dst_dataset_json = os.path.join(preprocessed_output_dir, "dataset.json")
    if os.path.exists(src_dataset_json) and not os.path.exists(dst_dataset_json):
        shutil.copy(src_dataset_json, dst_dataset_json)
        print(f"Copied dataset.json to {preprocessed_output_dir}")

    # Default: process all scopes
    if scopes is None:
        scopes = list(config.get("merge_scopes", {}).keys())

    # Build scopes_config
    scopes_config = {}
    for scope in scopes:
        scope_info = config.get("merge_scopes", {}).get(scope, {})
        scopes_config[scope] = {
            "merge_lut": scope_info.get("merge_lut"),
            "num_classes": scope_info.get("num_merged_classes", 0)
        }

    print(f"Scopes: {scopes}")
    print(f"Target size: {target_size}")
    print(f"Target orientation: {target_orientation}")
    print(f"Input dir: {raw_input_dir}")
    print(f"Output dir: {preprocessed_output_dir}")
    print(f"Split: {split}")
    print(f"Save debug nii: {save_debug_nii} (sample size: {debug_nii_sample_size})")

    # Select input directory based on split, but output unified to imagesTr and labelsTr_*
    # This keeps train and val data in the same directory, distinguished by CSV (similar to abdominal dataset iid/ood)
    if split == "val":
        images_subdir = "imagesVal"
        labels_subdir = "labelsVal"
    else:  # train
        images_subdir = "imagesTr"
        labels_subdir = "labelsTr"

    # Output unified to imagesTr and labelsTr_* (regardless of train or val)
    output_images_subdir = "imagesTr"
    output_labels_prefix = "labelsTr"

    images_dir = os.path.join(raw_input_dir, images_subdir)
    labels_dir = os.path.join(raw_input_dir, labels_subdir)

    if not os.path.exists(images_dir):
        print(f"Error: Images dir not found: {images_dir}")
        return []

    if not os.path.exists(labels_dir):
        print(f"Error: Labels dir not found: {labels_dir}")
        return []

    # Get all image files
    all_image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith('.nii.gz')
    ])

    print(f"Found {len(all_image_files)} image files")

    # First collect all valid cases
    valid_cases = []
    for img_file in all_image_files:
        case_id = img_file.replace('.nii.gz', '')

        # If case_ids specified, only process those
        if case_ids is not None and case_id not in case_ids:
            continue

        # Label filename: remove trailing _0000 suffix (only at end, not all)
        if case_id.endswith('_0000'):
            label_id = case_id[:-5]  # Remove last 5 characters "_0000"
        else:
            label_id = case_id
        label_file = label_id + '.nii.gz'

        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            valid_cases.append((case_id, img_path, label_path, label_id))

    # Get set of existing files (avoid repeated os.path.exists calls)
    existing_img_files = set()
    img_output_dir = os.path.join(preprocessed_output_dir, output_images_subdir)
    if os.path.exists(img_output_dir):
        existing_img_files = set(os.listdir(img_output_dir))

    existing_label_files = {}
    existing_pkl_files = {}
    for scope_name in scopes_config.keys():
        label_output_dir = os.path.join(preprocessed_output_dir, f"{output_labels_prefix}_{scope_name}")
        if os.path.exists(label_output_dir):
            files = os.listdir(label_output_dir)
            existing_label_files[scope_name] = set(f for f in files if f.endswith('.b2nd'))
            existing_pkl_files[scope_name] = set(f for f in files if f.endswith('.pkl'))
        else:
            existing_label_files[scope_name] = set()
            existing_pkl_files[scope_name] = set()

    # Filter already processed cases
    pending_cases = []
    for case_id, img_path, label_path, label_id in valid_cases:
        img_b2nd = f"{case_id}.b2nd"
        if img_b2nd not in existing_img_files:
            pending_cases.append((case_id, img_path, label_path))
            continue
        # Check label and pkl for all scopes
        all_exist = True
        for scope_name in scopes_config.keys():
            label_b2nd = f"{label_id}.b2nd"
            label_pkl = f"{label_id}.pkl"
            if label_b2nd not in existing_label_files[scope_name] or label_pkl not in existing_pkl_files[scope_name]:
                all_exist = False
                break
        if not all_exist:
            pending_cases.append((case_id, img_path, label_path))

    skip_count = len(valid_cases) - len(pending_cases)
    print(f"Found {len(valid_cases)} valid cases, {skip_count} already processed, {len(pending_cases)} to process")

    results = []
    if pending_cases:
        # Randomly select cases for debug nii (choose from first 50 cases to see results earlier)
        debug_nii_cases = set()
        if save_debug_nii:
            case_ids_list = [c[0] for c in pending_cases]
            debug_nii_cases = select_debug_nii_samples(case_ids_list, debug_nii_sample_size)

        # Build tasks (add raw_input_dir for deletion on failure)
        tasks = []
        for case_id, img_path, label_path in pending_cases:
            case_save_debug = case_id in debug_nii_cases
            tasks.append((case_id, img_path, label_path, preprocessed_output_dir,
                         target_size, scopes_config, case_save_debug, compression_level,
                         target_orientation, raw_input_dir))

        # Parallel processing (use Pool instead of ProcessPoolExecutor for more accurate progress bar)
        from multiprocessing import Pool
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(tasks), desc="Processing (image + labels)") as pbar:
                for result in pool.imap_unordered(_safe_process_single_case, tasks):
                    results.append(result)
                    pbar.update(1)

        success_count = sum(1 for r in results if r["status"] == "success")
        failed_deleted = [r for r in results if r.get("status") == "failed_deleted"]
        print(f"\nComplete: Success={success_count}, Total={len(tasks)}")

        # If there are failed and deleted cases, prompt user to rerun
        if failed_deleted:
            print(f"\n[WARNING] {len(failed_deleted)} cases failed and raw files deleted:")
            for r in failed_deleted[:10]:  # Show at most 10
                print(f"  - {r['case_id']}: {r.get('error', 'unknown error')}")
            if len(failed_deleted) > 10:
                print(f"  ... and {len(failed_deleted) - 10} more")
            print("\n[ACTION] Please re-run the pipeline to regenerate these cases.")

    # Generate dataset.json and CSV for each scope (regardless of newly processed cases)
    print("\nGenerating dataset.json and CSV for each scope...")
    for scope_name, scope_info in scopes_config.items():
        generate_scope_dataset_json(
            preprocessed_output_dir=preprocessed_output_dir,
            scope_name=scope_name,
            num_classes=scope_info.get("num_classes", 0),
            config=config
        )
        generate_scope_seg_csv(
            preprocessed_output_dir=preprocessed_output_dir,
            scope_name=scope_name,
            split=split,
            raw_input_dir=raw_input_dir
        )

    # Generate dataset_fingerprint.json (required for nnUNet training)
    generate_dataset_fingerprint(preprocessed_output_dir)

    return results
def generate_dataset_fingerprint(preprocessed_output_dir: str) -> str:
    """
    Generate dataset_fingerprint.json (required for nnUNet training)

    Args:
        preprocessed_output_dir: Preprocessed output directory

    Returns:
        Path of the generated file
    """
    output_path = os.path.join(preprocessed_output_dir, "dataset_fingerprint.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({}, f)

    print(f"  Generated {output_path}")
    return output_path


def generate_scope_dataset_json(
    preprocessed_output_dir: str,
    scope_name: str,
    num_classes: int,
    config: dict
) -> str:
    """
    Generate scope-specific dataset.json

    Args:
        preprocessed_output_dir: Preprocessed output directory
        scope_name: Scope name (e.g., seg_basic, seg_full)
        num_classes: Number of classes
        config: Dataset configuration

    Returns:
        Path of the generated dataset.json
    """
    # Get merged_label_names
    merge_scope_config = config.get("merge_scopes", {}).get(scope_name, {})
    label_names = merge_scope_config.get("merged_label_names", {})

    # Build labels dictionary (nnUNet format: "background": 0, "organ1": 1, ...)
    labels = {}
    for idx_str, name in label_names.items():
        labels[name] = int(idx_str)

    # If no label_names, generate default ones
    if not labels:
        labels = {"background": 0}
        for i in range(1, num_classes):
            labels[f"class_{i}"] = i

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": len(os.listdir(os.path.join(preprocessed_output_dir, "imagesTr"))) if os.path.exists(os.path.join(preprocessed_output_dir, "imagesTr")) else 0,
        "file_ending": ".b2nd",
        "dataset_name": config.get("dataset_name", ""),
        "scope": scope_name,
        "num_classes": num_classes
    }

    output_path = os.path.join(preprocessed_output_dir, f"dataset_{scope_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print(f"  Generated {output_path} (num_classes={num_classes})")
    return output_path


def generate_scope_seg_csv(
    preprocessed_output_dir: str,
    scope_name: str,
    include_paths: bool = True,
    split: str = "train",
    raw_input_dir: str = None
) -> str:
    """
    Generate scope-specific segmentation CSV

    All data are under imagesTr and labelsTr_*, distinguished by CSV for train/val (similar to abdominal iid/ood)

    CSV column names match nnUNet blosc2 dataset class:
    - series_id: Sample ID
    - blosc2_path: Image path
    - seg_blosc2_path: Segmentation label path

    Args:
        preprocessed_output_dir: Preprocessed output directory
        scope_name: Scope name
        include_paths: Whether to include full paths (default True)
        split: "train" or "val" - specifies which split CSV to generate
        raw_input_dir: Raw data directory, used to determine which cases belong to this split

    Returns:
        Path of the generated CSV
    """
    # CSV name determined by split
    csv_name = f"dataset_{split}_{scope_name}.csv"

    # All preprocessed data are under imagesTr and labelsTr_* (regardless of original train/val)
    images_dir = os.path.join(preprocessed_output_dir, "imagesTr")
    labels_dir = os.path.join(preprocessed_output_dir, f"labelsTr_{scope_name}")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"  Warning: images or labels dir not found for {scope_name}")
        return ""

    # Get case list for this split: read from raw data directory
    valid_case_ids = set()
    if raw_input_dir:
        # Read case list for this split from raw directory
        raw_images_subdir = "imagesVal" if split == "val" else "imagesTr"
        raw_images_dir = os.path.join(raw_input_dir, raw_images_subdir)
        if os.path.exists(raw_images_dir):
            for f in os.listdir(raw_images_dir):
                if f.endswith('.nii.gz'):
                    case_id = f.replace('.nii.gz', '')
                    valid_case_ids.add(case_id)

    # If raw_input_dir not provided, include all cases (backward compatibility)
    if not valid_case_ids:
        all_b2nd_files = [f for f in os.listdir(images_dir) if f.endswith('.b2nd')]
        valid_case_ids = set(f.replace('.b2nd', '') for f in all_b2nd_files)

    # Generate CSV records
    records = []
    for case_id in sorted(valid_case_ids):
        img_file = f"{case_id}.b2nd"
        img_path = os.path.join(images_dir, img_file)

        # Check if preprocessed image exists
        if not os.path.exists(img_path):
            continue

        # Label filename: remove _0000 suffix
        label_id = case_id[:-5] if case_id.endswith('_0000') else case_id
        label_file = f"{label_id}.b2nd"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # Use column names expected by nnUNet blosc2 dataset class
        record = {"series_id": case_id}

        if include_paths:
            record["blosc2_path"] = img_path
            record["seg_blosc2_path"] = label_path

        records.append(record)

    df = pd.DataFrame(records)
    output_csv = os.path.join(preprocessed_output_dir, csv_name)
    df.to_csv(output_csv, index=False)
    print(f"  Generated {output_csv} ({len(df)} samples)")

    return output_csv


def generate_iid_ood_scope_csvs(
    preprocessed_output_dir: str,
    scopes: List[str],
    split_dir: str,
    include_paths: bool = True
) -> Dict[str, List[str]]:
    """
    Generate IID/OOD split CSV files for each scope

    Generated files:
    - iid_train_{scope}.csv
    - iid_test_{scope}.csv  
    - ood_train_{scope}.csv
    - ood_test_{scope}.csv

    Args:
        preprocessed_output_dir: Preprocessed output directory
        scopes: List of scopes (e.g., ['seg_basic', 'seg_basic_lesion', 'seg_full'])
        split_dir: Directory containing IID_train.csv, IID_test.csv, OOD_train.csv, OOD_test.csv
        include_paths: Whether to include full paths

    Returns:
        Dict[split_name, List[csv_path]] - List of generated CSV paths for each split
    """
    split_files = {
        "iid_train": os.path.join(split_dir, "IID_train.csv"),
        "iid_test": os.path.join(split_dir, "IID_test.csv"),
        "ood_train": os.path.join(split_dir, "OOD_train.csv"),
        "ood_test": os.path.join(split_dir, "OOD_test.csv"),
    }

    # Check if split files exist
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found: {split_file}")

    print("\n" + "=" * 50)
    print("Generating IID/OOD Scope CSVs")
    print("=" * 50)
    print(f"Preprocessed dir: {preprocessed_output_dir}")
    print(f"Split dir: {split_dir}")
    print(f"Scopes: {scopes}")

    results = {}
    images_dir = os.path.join(preprocessed_output_dir, "imagesTr")

    # Get all preprocessed case IDs
    if not os.path.exists(images_dir):
        print(f"Error: Images dir not found: {images_dir}")
        return results

    all_case_ids = set()
    for f in os.listdir(images_dir):
        if f.endswith('.b2nd'):
            # Case ID format: BDMAP_xxxxx_0000.b2nd -> BDMAP_xxxxx
            case_id = f.replace('.b2nd', '')
            if case_id.endswith('_0000'):
                case_id = case_id[:-5]
            all_case_ids.add(case_id)

    print(f"Found {len(all_case_ids)} preprocessed cases")

    # Generate CSV for each split
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            results[split_name] = []
            continue

        # Load case IDs for this split
        split_ids = load_split_ids(split_file)
        valid_ids = all_case_ids & split_ids
        print(f"\n{split_name}: {len(valid_ids)}/{len(split_ids)} cases available")

        results[split_name] = []

        for scope in scopes:
            labels_dir = os.path.join(preprocessed_output_dir, f"labelsTr_{scope}")
            if not os.path.exists(labels_dir):
                print(f"  Warning: Labels dir not found for {scope}: {labels_dir}")
                continue

            # Generate CSV
            records = []
            for case_id in sorted(valid_ids):
                # Image path: {case_id}_0000.b2nd
                img_file = f"{case_id}_0000.b2nd"
                img_path = os.path.join(images_dir, img_file)

                if not os.path.exists(img_path):
                    continue

                # Label path: {case_id}.b2nd
                label_file = f"{case_id}.b2nd"
                label_path = os.path.join(labels_dir, label_file)

                if not os.path.exists(label_path):
                    continue

                record = {"series_id": f"{case_id}_0000"}
                if include_paths:
                    record["blosc2_path"] = img_path
                    record["seg_blosc2_path"] = label_path
                records.append(record)

            if records:
                csv_name = f"{split_name}_{scope}.csv"
                output_csv = os.path.join(preprocessed_output_dir, csv_name)
                pd.DataFrame(records).to_csv(output_csv, index=False)
                print(f"  Generated {csv_name} ({len(records)} samples)")
                results[split_name].append(output_csv)

    return results



def run_debug_mode(
    raw_input_dir: str,
    debug_output_dir: str,
    config: dict,
    num_cases: int = 10,
    scopes: List[str] = None,
    target_size: Tuple[int, int, int] = (192, 256, 256),
    target_orientation: str = "LPS",
    num_workers: int = 10,
    seed: int = 42
) -> List[dict]:
    """
    Debug mode: Randomly select N cases for resize + blosc2 + save debug nii.gz

    Args:
        raw_input_dir: nnUNet_raw input directory
        debug_output_dir: debug output directory
        config: dataset configuration
        num_cases: number of randomly selected cases
        scopes: list of scopes to process
        target_size: target size
        target_orientation: target orientation (default "LPS")
        num_workers: number of parallel processes
        seed: random seed

    Returns:
        List of processing results
    """
    import random

    if scopes is None:
        scopes = list(config.get("merge_scopes", {}).keys())

    print("=" * 50)
    print("DEBUG MODE")
    print("=" * 50)
    print(f"Num cases: {num_cases}")
    print(f"Scopes: {scopes}")
    print(f"Target size: {target_size}")
    print(f"Target orientation: {target_orientation}")
    print(f"Input: {raw_input_dir}")
    print(f"Output: {debug_output_dir}")

    # Get all cases
    images_dir = os.path.join(raw_input_dir, "imagesTr")
    all_cases = sorted([
        f.replace('.nii.gz', '')
        for f in os.listdir(images_dir)
        if f.endswith('.nii.gz')
    ])

    # Random selection
    random.seed(seed)
    selected_cases = random.sample(all_cases, min(num_cases, len(all_cases)))
    print(f"\nSelected cases: {selected_cases}")

    # Process
    results = resize_dataset_to_preprocessed(
        raw_input_dir=raw_input_dir,
        preprocessed_output_dir=debug_output_dir,
        config=config,
        scopes=scopes,
        target_size=target_size,
        target_orientation=target_orientation,
        num_workers=num_workers,
        save_debug_nii=True,
        debug_nii_sample_size=num_cases,  # In debug mode, save nii for all cases
        case_ids=selected_cases
    )

    # Save results
    results_path = os.path.join(debug_output_dir, "debug_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Serialize results
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, tuple):
                sr[k] = list(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            else:
                sr[k] = v
        serializable_results.append(sr)

    with open(results_path, 'w') as f:
        json.dump({
            "total_cases": len(all_cases),
            "selected_cases": selected_cases,
            "scopes": scopes,
            "results": serializable_results
        }, f, indent=2)

    print(f"\nDebug results saved to: {results_path}")
    print("Done!")

    return results

