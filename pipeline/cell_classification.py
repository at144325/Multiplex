"""Classify cells as MTAP-positive/negative and SDMA-positive/negative."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, dilation
from skimage.filters import threshold_otsu
from config.defaults import CYTOPLASM_DILATION_RADIUS, MIN_CYTOPLASM_PIXELS


def compute_cytoplasmic_rings(
    labels: np.ndarray,
    dilation_radius: int = CYTOPLASM_DILATION_RADIUS,
    min_pixels: int = MIN_CYTOPLASM_PIXELS,
) -> dict[int, np.ndarray]:
    """
    For each nucleus, compute the cytoplasmic ring mask (dilated - all nuclei).

    Returns dict mapping label_id -> (N, 2) array of (row, col) pixel coordinates,
    or empty array if ring is too small.
    """
    all_nuclei_mask = labels > 0
    selem = disk(dilation_radius)
    rings = {}

    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids != 0]

    for label_id in unique_ids:
        nucleus_mask = labels == label_id
        dilated = dilation(nucleus_mask, selem)
        ring_mask = dilated & ~all_nuclei_mask
        coords = np.argwhere(ring_mask)
        if len(coords) >= min_pixels:
            rings[label_id] = coords
        else:
            rings[label_id] = np.empty((0, 2), dtype=int)

    return rings


def classify_mtap(
    labels: np.ndarray,
    red_channel: np.ndarray,
    dilation_radius: int = CYTOPLASM_DILATION_RADIUS,
    min_cytoplasm_pixels: int = MIN_CYTOPLASM_PIXELS,
    threshold: float | None = None,
):
    """
    Classify cells as MTAP-positive or MTAP-negative based on red
    (cytoplasmic) intensity.

    Parameters
    ----------
    labels : nucleus label image
    red_channel : deconvolved red channel (float, higher = more red)
    threshold : if None, auto-computed via Otsu on per-cell intensities

    Returns
    -------
    mtap_positive : set of label IDs that are MTAP-positive
    mtap_negative : set of label IDs that are MTAP-negative
    cell_red_intensities : dict label_id -> mean red intensity in cytoplasmic ring
    auto_threshold : the threshold used
    """
    rings = compute_cytoplasmic_rings(labels, dilation_radius, min_cytoplasm_pixels)

    cell_red_intensities = {}
    for label_id, coords in rings.items():
        if len(coords) == 0:
            cell_red_intensities[label_id] = 0.0
        else:
            cell_red_intensities[label_id] = float(
                np.mean(red_channel[coords[:, 0], coords[:, 1]])
            )

    # Auto-threshold via Otsu if not provided
    intensities = np.array(list(cell_red_intensities.values()))
    if threshold is None:
        if len(intensities) > 1 and np.std(intensities) > 1e-6:
            threshold = float(threshold_otsu(intensities))
        else:
            threshold = float(np.mean(intensities)) if len(intensities) > 0 else 0.0

    mtap_positive = set()
    mtap_negative = set()
    for label_id, intensity in cell_red_intensities.items():
        if intensity >= threshold:
            mtap_positive.add(label_id)
        else:
            mtap_negative.add(label_id)

    return mtap_positive, mtap_negative, cell_red_intensities, threshold


def classify_sdma(
    labels: np.ndarray,
    brown_channel: np.ndarray,
    mtap_negative: set,
    threshold: float | None = None,
):
    """
    Among MTAP-negative cells, classify as SDMA-positive or negative
    based on nuclear brown/DAB intensity.

    Returns
    -------
    sdma_positive : set of label IDs
    sdma_negative : set of label IDs
    cell_brown_intensities : dict label_id -> mean brown intensity in nucleus
    auto_threshold : the threshold used
    """
    cell_brown_intensities = {}
    for label_id in mtap_negative:
        nucleus_mask = labels == label_id
        pixels = brown_channel[nucleus_mask]
        cell_brown_intensities[label_id] = float(np.mean(pixels)) if len(pixels) > 0 else 0.0

    intensities = np.array(list(cell_brown_intensities.values()))
    if threshold is None:
        if len(intensities) > 1 and np.std(intensities) > 1e-6:
            threshold = float(threshold_otsu(intensities))
        else:
            threshold = float(np.mean(intensities)) if len(intensities) > 0 else 0.0

    sdma_positive = set()
    sdma_negative = set()
    for label_id, intensity in cell_brown_intensities.items():
        if intensity >= threshold:
            sdma_positive.add(label_id)
        else:
            sdma_negative.add(label_id)

    return sdma_positive, sdma_negative, cell_brown_intensities, threshold
