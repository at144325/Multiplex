"""Export and import full analysis state as a single .npz bundle."""

from __future__ import annotations

import json
from io import BytesIO

import numpy as np


def save_analysis(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    hematoxylin: np.ndarray,
    red_channel: np.ndarray,
    brown_channel: np.ndarray,
    cell_red_intensities: dict[int, float],
    cell_brown_intensities: dict[int, float],
    mtap_threshold: float,
    sdma_threshold: float,
    step: str,
    source_filename: str,
    parameters: dict,
) -> bytes:
    """Serialize full analysis state into a compressed .npz bundle.

    Returns the raw bytes of the .npz file.
    """
    metadata = {
        "source_filename": source_filename,
        "image_shape": list(image_rgb.shape[:2]),
        "step": step,
        "mtap_threshold": mtap_threshold,
        "sdma_threshold": sdma_threshold,
        "cell_red_intensities": {str(k): v for k, v in cell_red_intensities.items()},
        "cell_brown_intensities": {str(k): v for k, v in cell_brown_intensities.items()},
        "parameters": parameters,
    }
    meta_bytes = json.dumps(metadata).encode("utf-8")

    buf = BytesIO()
    np.savez_compressed(
        buf,
        image_rgb=image_rgb,
        labels=labels,
        hematoxylin=hematoxylin,
        red_channel=red_channel,
        brown_channel=brown_channel,
        metadata=np.void(meta_bytes),
    )
    return buf.getvalue()


def load_analysis(data: bytes) -> dict:
    """Deserialize a .npz bundle back into session-state–ready dict.

    Returns a dict with all keys needed to populate ``st.session_state``.
    """
    buf = BytesIO(data)
    npz = np.load(buf, allow_pickle=False)

    image_rgb = npz["image_rgb"]
    labels = npz["labels"]
    hematoxylin = npz["hematoxylin"]
    red_channel = npz["red_channel"]
    brown_channel = npz["brown_channel"]

    meta_bytes = bytes(npz["metadata"])
    metadata = json.loads(meta_bytes.decode("utf-8"))

    # Reconstruct intensity dicts with int keys
    cell_red_intensities = {int(k): v for k, v in metadata["cell_red_intensities"].items()}
    cell_brown_intensities = {int(k): v for k, v in metadata["cell_brown_intensities"].items()}

    mtap_threshold = metadata["mtap_threshold"]
    sdma_threshold = metadata["sdma_threshold"]

    # Re-derive classification sets from intensities + thresholds
    mtap_positive = {lid for lid, v in cell_red_intensities.items() if v >= mtap_threshold}
    mtap_negative = {lid for lid, v in cell_red_intensities.items() if v < mtap_threshold}
    sdma_positive = {lid for lid, v in cell_brown_intensities.items() if v >= sdma_threshold}
    sdma_negative = {lid for lid, v in cell_brown_intensities.items() if v < sdma_threshold}

    return {
        "image_rgb": image_rgb,
        "labels": labels,
        "hematoxylin": hematoxylin,
        "red_channel": red_channel,
        "brown_channel": brown_channel,
        "cell_red_intensities": cell_red_intensities,
        "cell_brown_intensities": cell_brown_intensities,
        "mtap_threshold": mtap_threshold,
        "sdma_threshold": sdma_threshold,
        "mtap_positive": mtap_positive,
        "mtap_negative": mtap_negative,
        "sdma_positive": sdma_positive,
        "sdma_negative": sdma_negative,
        "step": metadata["step"],
        "source_filename": metadata["source_filename"],
        "parameters": metadata.get("parameters", {}),
    }
