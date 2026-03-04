"""Generate annotated overlay images and histograms."""

from __future__ import annotations

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO


def _find_contours(labels: np.ndarray, label_ids: set) -> list:
    """Find contours for a set of label IDs."""
    mask = np.isin(labels, list(label_ids)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _per_label_contours(labels: np.ndarray, label_ids: set) -> list[tuple[int, list]]:
    """Find contours per individual label."""
    results = []
    for lid in label_ids:
        mask = (labels == lid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results.append((lid, contours))
    return results


def mtap_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    mtap_positive: set,
    mtap_negative: set,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draw MTAP classification overlay.
    Green contours + fill = MTAP-negative, Red = MTAP-positive.
    """
    overlay = image_rgb.copy()
    canvas = image_rgb.copy()

    # Semi-transparent fills
    for lid, contours in _per_label_contours(labels, mtap_negative):
        cv2.drawContours(canvas, contours, -1, (0, 200, 0), -1)  # green fill
    for lid, contours in _per_label_contours(labels, mtap_positive):
        cv2.drawContours(canvas, contours, -1, (200, 0, 0), -1)  # red fill

    overlay = cv2.addWeighted(canvas, alpha, overlay, 1 - alpha, 0)

    # Contour outlines
    neg_contours = _find_contours(labels, mtap_negative)
    pos_contours = _find_contours(labels, mtap_positive)
    cv2.drawContours(overlay, neg_contours, -1, (0, 220, 0), 1)
    cv2.drawContours(overlay, pos_contours, -1, (220, 0, 0), 1)

    return overlay


def sdma_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    sdma_positive: set,
    sdma_negative: set,
    mtap_positive: set,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Draw SDMA classification overlay.
    Yellow fill = SDMA-positive, Blue fill = SDMA-negative, Grey = MTAP-positive (dimmed).
    """
    overlay = image_rgb.copy()
    canvas = image_rgb.copy()

    for lid, contours in _per_label_contours(labels, mtap_positive):
        cv2.drawContours(canvas, contours, -1, (150, 150, 150), -1)  # grey
    for lid, contours in _per_label_contours(labels, sdma_negative):
        cv2.drawContours(canvas, contours, -1, (80, 80, 220), -1)  # blue
    for lid, contours in _per_label_contours(labels, sdma_positive):
        cv2.drawContours(canvas, contours, -1, (220, 220, 0), -1)  # yellow

    overlay = cv2.addWeighted(canvas, alpha, overlay, 1 - alpha, 0)

    # Outlines
    for group, color in [
        (sdma_positive, (220, 220, 0)),
        (sdma_negative, (80, 80, 220)),
        (mtap_positive, (150, 150, 150)),
    ]:
        contours = _find_contours(labels, group)
        cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay


def intensity_histogram(
    intensities: dict[int, float],
    threshold: float,
    title: str = "Intensity Distribution",
    xlabel: str = "Mean intensity",
    pos_label: str = "Positive",
    neg_label: str = "Negative",
) -> BytesIO:
    """
    Plot histogram of per-cell intensities with threshold line.
    Returns PNG as BytesIO.
    """
    vals = np.array(list(intensities.values()))
    if len(vals) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No cells to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(vals, bins=50, color="#888888", edgecolor="white", alpha=0.8)
        ax.axvline(threshold, color="red", linewidth=2, linestyle="--", label=f"Threshold = {threshold:.4f}")
        n_pos = int(np.sum(vals >= threshold))
        n_neg = int(np.sum(vals < threshold))
        ax.set_title(f"{title}  ({neg_label}: {n_neg}, {pos_label}: {n_pos})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cell count")
        ax.legend()

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def sdma_slide_figure(
    overlay_rgb: np.ndarray,
    source_filename: str,
    n_mtap_pos: int,
    n_mtap_neg: int,
    n_sdma_pos: int,
    n_sdma_neg: int,
    mtap_threshold: float,
    sdma_threshold: float,
) -> BytesIO:
    """Create a presentation-ready slide figure with overlay + stats.

    Returns PNG as BytesIO.
    """
    fig, (ax_img, ax_txt) = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3, 2]}
    )

    # Left panel: SDMA overlay
    ax_img.imshow(overlay_rgb)
    ax_img.axis("off")
    ax_img.set_title("SDMA Overlay", fontsize=14, fontweight="bold")

    # Right panel: stats
    ax_txt.axis("off")

    pct = (n_sdma_pos / n_mtap_neg * 100) if n_mtap_neg > 0 else 0.0
    n_total = n_mtap_pos + n_mtap_neg

    # Headline
    ax_txt.text(
        0.5, 0.88, f"{pct:.1f}% SDMA-positive",
        ha="center", va="top", fontsize=28, fontweight="bold",
        transform=ax_txt.transAxes,
    )
    ax_txt.text(
        0.5, 0.80, "(among MTAP-negative cells)",
        ha="center", va="top", fontsize=13, color="grey",
        transform=ax_txt.transAxes,
    )

    # Cell count breakdown (monospace block)
    breakdown = (
        f"Total nuclei:        {n_total:>6d}\n"
        f"  MTAP-positive:     {n_mtap_pos:>6d}\n"
        f"  MTAP-negative:     {n_mtap_neg:>6d}\n"
        f"    SDMA-positive:   {n_sdma_pos:>6d}  ({pct:.1f}%)\n"
        f"    SDMA-negative:   {n_sdma_neg:>6d}  ({100 - pct:.1f}%)"
    )
    ax_txt.text(
        0.5, 0.65, breakdown,
        ha="center", va="top", fontsize=12, family="monospace",
        transform=ax_txt.transAxes,
    )

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DCDC00", edgecolor="black", label="SDMA+ (MTAP−)"),
        Patch(facecolor="#5050DC", edgecolor="black", label="SDMA− (MTAP−)"),
        Patch(facecolor="#969696", edgecolor="black", label="MTAP+ (excluded)"),
    ]
    ax_txt.legend(
        handles=legend_elements, loc="center", fontsize=11,
        bbox_to_anchor=(0.5, 0.22), frameon=True,
        fancybox=True, shadow=True,
    )

    # Title and footnote
    fig.suptitle(source_filename, fontsize=11, y=0.97, color="grey")
    fig.text(
        0.5, 0.02,
        f"MTAP threshold: {mtap_threshold:.4f}  |  SDMA threshold: {sdma_threshold:.4f}",
        ha="center", fontsize=9, color="grey",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def channel_preview(channel: np.ndarray, title: str) -> BytesIO:
    """Render a single deconvolved channel as a grayscale PNG."""
    fig, ax = plt.subplots(figsize=(4, 4))
    vmax = np.percentile(channel, 99.5) if channel.max() > 0 else 1
    ax.imshow(channel, cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf
