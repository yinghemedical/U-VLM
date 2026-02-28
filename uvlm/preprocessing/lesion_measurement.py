"""
Lesion size measurement following WHO standard

Based on RadGPT Algorithm 2: WHO-based Lesion Size Measurement
- Longest diameter: Maximum lesion diameter across any axial plane
- Perpendicular diameter: Diameter perpendicular to the longest diameter

Reference:
    RadGPT Algorithm 2: WHO-based Tumors Size Measurement
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from skimage import measure
from skimage.transform import rotate
from math import atan2, degrees
from typing import Dict, List, Optional, Tuple, Any
import nibabel as nib


def resample_image(
    image: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1, 1, 1),
    order: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample image to target spacing (isotropic 1mm by default).

    Args:
        image: 3D numpy array
        original_spacing: Original voxel spacing (x, y, z)
        target_spacing: Target voxel spacing
        order: Interpolation order (0=nearest, 1=linear)

    Returns:
        Resampled image and resize factor
    """
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    resampled_image = ndimage.zoom(image, resize_factor, order=order)
    return resampled_image, resize_factor


def measure_diameter(binary_image: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Measures the diameter of an arbitrary shape in a binary image and returns the diameter
    along with the two extreme points that define this diameter.

    Parameters:
    binary_image (numpy.ndarray): 2D binary array where the shape is represented by 1s.

    Returns:
    tuple: The diameter of the shape and the coordinates of the two extreme points.
    """
    contours = measure.find_contours(binary_image, 0.5)

    if not contours:
        raise ValueError("No contours found in the binary image")

    contour = contours[0]
    distances = pdist(contour)
    distance_matrix = squareform(distances)
    max_distance_idx = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)
    point1 = contour[max_distance_idx[0]]
    point2 = contour[max_distance_idx[1]]
    max_distance = distance_matrix[max_distance_idx]

    return max_distance, point1, point2


def measure_vertical_span(binary_image: np.ndarray) -> int:
    """
    Measure the vertical span of the shape in a 2D binary image.

    Args:
        binary_image: 2D binary array

    Returns:
        Vertical span in pixels
    """
    y_coords, x_coords = np.where(binary_image == 1)
    vertical_span = np.max(y_coords) - np.min(y_coords)
    return vertical_span


def analyze_tumor_components(
    tumor_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    ct_data: np.ndarray = None,
    erode: int = 3,
    max_tumors: int = None
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze all lesion connected components in a 3D mask.

    Implements RadGPT Algorithm 2 (WHO standard measurement).
    For GT, no volume threshold is used.

    Args:
        tumor_mask: 3D binary lesion mask
        spacing: Voxel spacing (x, y, z) in mm
        ct_data: Optional CT data for HU measurement
        erode: Binary erosion size to denoise (0 to disable)
        max_tumors: Maximum number of lesions to analyze (None for all)

    Returns:
        Dictionary mapping lesion index (1-based) to measurements:
        {
            1: {
                'volume_mm3': float,
                'longest_diameter_mm': float,
                'perpendicular_diameter_mm': float,
                'center_voxel': (x, y, z),
                'slice': int,
                'mean_hu': float (if ct_data provided),
                'std_hu': float (if ct_data provided),
            },
            2: {...},
            ...
        }
    """
    resampled_mask, resize_factor = resample_image(
        tumor_mask, spacing, target_spacing=(1, 1, 1), order=0
    )
    resampled_mask = resampled_mask.astype(bool)

    if ct_data is not None:
        resampled_ct, _ = resample_image(ct_data, spacing, target_spacing=(1, 1, 1), order=1)
    else:
        resampled_ct = None

    structure = np.ones((3, 3, 3), dtype=int)
    labeled_array, num_features = ndimage.label(resampled_mask, structure=structure)

    if num_features == 0:
        return {}

    sizes = ndimage.sum(resampled_mask, labeled_array, range(1, num_features + 1))
    sorted_indices = np.argsort(sizes)[::-1]

    outputs = {}
    included = 0

    for n, idx in enumerate(sorted_indices):
        if max_tumors is not None and included >= max_tumors:
            break

        label = idx + 1
        component_mask = (labeled_array == label)

        if erode > 0:
            eroded = ndimage.binary_erosion(
                component_mask,
                structure=np.ones((erode, erode, erode)),
                iterations=1
            )
            if eroded.sum() == 0:
                continue

        volume_mm3 = component_mask.sum()

        max_longest_diameter = 0
        longest_points = None
        longest_diameter_slice = 0

        for z in range(component_mask.shape[2]):
            slice_mask = component_mask[:, :, z]
            if np.any(slice_mask):
                try:
                    diam, point1, point2 = measure_diameter(slice_mask)
                    if diam > max_longest_diameter:
                        max_longest_diameter = diam
                        longest_points = (point1, point2)
                        longest_diameter_slice = z
                except ValueError:
                    continue

        longest_diameter = max_longest_diameter

        if longest_points is not None:
            z = longest_diameter_slice
            slice_mask = component_mask[:, :, z]
            point1, point2 = longest_points
            angle = degrees(atan2(point2[0] - point1[0], point2[1] - point1[1]))

            rotated_image = rotate(
                slice_mask, angle, resize=True, order=0,
                preserve_range=True, mode='constant', cval=0
            )
            assert np.array_equal(rotated_image, rotated_image.astype(bool))

            perpendicular_diameter = measure_vertical_span(rotated_image)
        else:
            perpendicular_diameter = 0

        component_indices = np.where(component_mask)
        x = component_indices[0].max() - component_indices[0].min() + 1
        center_x = int(component_indices[0].min() + x / 2)

        y = component_indices[1].max() - component_indices[1].min() + 1
        center_y = int(component_indices[1].min() + y / 2)
        center_y = component_mask.shape[1] - center_y

        center_z = int(np.mean(component_indices[2]))

        mean_hu = None
        std_hu = None
        if resampled_ct is not None:
            msk = np.where(component_mask > 0.5, 1, 0)
            msk = ndimage.binary_erosion(
                msk,
                structure=np.ones((1, 1, 1)),
                iterations=1
            )
            segmented = resampled_ct * msk
            mean_hu = segmented.sum() / msk.sum() if msk.sum() > 0 else None
            std_hu = segmented[msk != 0].std() if msk.sum() > 0 else None
            if mean_hu is not None:
                mean_hu = float(np.round(mean_hu, 1))
            if std_hu is not None:
                std_hu = float(np.round(std_hu, 1))

        included += 1
        outputs[included] = {
            'volume_mm3': float(volume_mm3),
            'longest_diameter_mm': float(int(longest_diameter)),
            'perpendicular_diameter_mm': float(int(perpendicular_diameter)),
            'center_voxel': (center_x, center_y, center_z),
            'slice': np.round(longest_diameter_slice / resize_factor[-1], 1),
            'mean_hu': mean_hu,
            'std_hu': std_hu,
        }

    return outputs


def compute_tumor_size_who(
    tumor_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    erode: int = 3
) -> Optional[float]:
    """
    Compute the largest lesion size using WHO standard.

    Returns the longest diameter of the largest lesion in mm.

    Args:
        tumor_mask: 3D binary lesion mask
        spacing: Voxel spacing (x, y, z) in mm
        erode: Binary erosion size to denoise

    Returns:
        Longest diameter of the largest lesion in mm, or None if no lesion
    """
    tumors = analyze_tumor_components(
        tumor_mask, spacing, erode=erode, max_tumors=1
    )

    if not tumors:
        return None

    return tumors[1]['longest_diameter_mm']


def categorize_case_by_tumor_size(
    largest_diameter_mm: float,
    threshold_mm: float = 20
) -> str:
    """
    Categorize a case by lesion size.

    Args:
        largest_diameter_mm: Largest lesion diameter in mm
        threshold_mm: Size threshold (default 20mm = 2cm)

    Returns:
        'small' if ≤threshold, 'large' if >threshold, 'negative' if 0
    """
    if largest_diameter_mm <= 0:
        return 'negative'
    elif largest_diameter_mm <= threshold_mm:
        return 'small'
    else:
        return 'large'
