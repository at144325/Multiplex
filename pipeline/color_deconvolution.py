"""Color deconvolution to separate hematoxylin, red (MTAP), and brown/DAB (SDMA)."""

from __future__ import annotations

import numpy as np
import cv2
from skimage.color import separate_stains
from config.defaults import (
    STAIN_MATRIX,
    HSV_RED_LOWER1, HSV_RED_UPPER1, HSV_RED_LOWER2, HSV_RED_UPPER2,
    HSV_BROWN_LOWER, HSV_BROWN_UPPER,
)


def _normalize_stain_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize stain matrix so each vector has unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_inverse_matrix(stain_matrix: np.ndarray) -> np.ndarray:
    """Compute the inverse of the stain matrix for deconvolution."""
    normed = _normalize_stain_matrix(stain_matrix)
    return np.linalg.inv(normed)


def deconvolve(image_rgb: np.ndarray, stain_matrix: np.ndarray | None = None):
    """
    Separate a 3-stain image into individual channels.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 array
    stain_matrix : 3x3 array, rows = stain vectors. Uses default if None.

    Returns
    -------
    hematoxylin, red, brown : each (H, W) float64, higher = more stain
    """
    if stain_matrix is None:
        stain_matrix = STAIN_MATRIX

    inv_matrix = build_inverse_matrix(stain_matrix)
    # separate_stains expects the *inverse* of the stain matrix
    stains = separate_stains(image_rgb, inv_matrix)

    hematoxylin = stains[:, :, 0]
    red = stains[:, :, 1]
    brown = stains[:, :, 2]

    # Clip negatives (artefact of deconvolution)
    hematoxylin = np.clip(hematoxylin, 0, None)
    red = np.clip(red, 0, None)
    brown = np.clip(brown, 0, None)

    return hematoxylin, red, brown


def hsv_fallback(image_rgb: np.ndarray):
    """
    HSV-based fallback for separating red and brown when deconvolution fails.

    Returns
    -------
    red_mask, brown_mask : (H, W) bool arrays
    red_intensity, brown_intensity : (H, W) float64
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Red spans hue wrap-around
    red_mask1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
    red_mask2 = cv2.inRange(hsv, HSV_RED_LOWER2, HSV_RED_UPPER2)
    red_mask = (red_mask1 | red_mask2).astype(bool)

    brown_mask = cv2.inRange(hsv, HSV_BROWN_LOWER, HSV_BROWN_UPPER).astype(bool)

    # Use saturation channel as proxy for intensity
    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    red_intensity = sat * red_mask
    brown_intensity = sat * brown_mask

    return red_mask, brown_mask, red_intensity, brown_intensity
