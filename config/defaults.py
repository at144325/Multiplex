"""Default stain vectors and segmentation parameters."""

import numpy as np

# --- Stain Vectors (OD space, unit-normalized) ---
# Each row is [R, G, B] in optical density space

HEMATOXYLIN_VECTOR = np.array([0.650, 0.704, 0.286])
# AEC-like red chromogen (default — unknown red type)
RED_AEC_VECTOR = np.array([0.2743, 0.6796, 0.6803])
DAB_VECTOR = np.array([0.268, 0.570, 0.776])

# 3x3 stain matrix: rows = stain vectors
STAIN_MATRIX = np.array([
    HEMATOXYLIN_VECTOR,
    RED_AEC_VECTOR,
    DAB_VECTOR,
])

# --- HSV Fallback Ranges (0-180 hue for OpenCV) ---
HSV_RED_LOWER1 = np.array([0, 50, 50])
HSV_RED_UPPER1 = np.array([8, 255, 255])
HSV_RED_LOWER2 = np.array([172, 50, 50])
HSV_RED_UPPER2 = np.array([180, 255, 255])
HSV_BROWN_LOWER = np.array([8, 50, 30])
HSV_BROWN_UPPER = np.array([20, 255, 200])

# --- Nucleus Segmentation Parameters ---
GAUSSIAN_SIGMA = 2.0
MIN_NUCLEUS_AREA = 80       # pixels
MAX_NUCLEUS_AREA = 5000     # pixels
WATERSHED_MIN_DISTANCE = 8  # pixels between local maxima
MORPHOLOGY_DISK_RADIUS = 3  # for opening operation

# --- Cell Classification Parameters ---
CYTOPLASM_DILATION_RADIUS = 12   # pixels to expand nucleus for cytoplasmic ring
MIN_CYTOPLASM_PIXELS = 20        # minimum ring pixels to score a cell

# --- Display ---
DOWNSCALE_DISPLAY = 0.25    # factor for displaying large images in Streamlit
