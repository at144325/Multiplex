"""Watershed-based nucleus segmentation from hematoxylin channel."""

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, binary_opening, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from config.defaults import (
    GAUSSIAN_SIGMA, MIN_NUCLEUS_AREA, MAX_NUCLEUS_AREA,
    WATERSHED_MIN_DISTANCE, MORPHOLOGY_DISK_RADIUS,
)


def segment_nuclei(
    hematoxylin: np.ndarray,
    gaussian_sigma: float = GAUSSIAN_SIGMA,
    min_area: int = MIN_NUCLEUS_AREA,
    max_area: int = MAX_NUCLEUS_AREA,
    min_distance: int = WATERSHED_MIN_DISTANCE,
    morph_radius: int = MORPHOLOGY_DISK_RADIUS,
) -> np.ndarray:
    """
    Segment nuclei from hematoxylin channel using watershed.

    Parameters
    ----------
    hematoxylin : (H, W) float array — higher values = more hematoxylin

    Returns
    -------
    labels : (H, W) int array — each nucleus has a unique label, 0 = background
    """
    # 1. Smooth
    smoothed = gaussian(hematoxylin, sigma=gaussian_sigma)

    # 2. Otsu threshold
    thresh = threshold_otsu(smoothed)
    binary = smoothed > thresh

    # 3. Morphological opening to remove small debris
    selem = disk(morph_radius)
    binary = binary_opening(binary, selem)

    # 4. Remove small objects below min area
    binary = remove_small_objects(binary, min_size=min_area)

    # 5. Distance transform for watershed markers
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary.astype(int),
    )
    marker_mask = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(marker_mask)

    # 6. Watershed
    labels = watershed(-distance, markers, mask=binary)

    # 7. Filter by area
    labels = _filter_by_area(labels, min_area, max_area)

    return labels


def _filter_by_area(labels: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Remove labeled regions outside the area range."""
    props_ids = np.unique(labels)
    for region_id in props_ids:
        if region_id == 0:
            continue
        area = np.sum(labels == region_id)
        if area < min_area or area > max_area:
            labels[labels == region_id] = 0
    # Re-label to keep IDs contiguous
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(unique_labels, start=1):
        new_labels[labels == old_id] = new_id
    return new_labels
