"""
IHC Double Staining Quantification App
Quantify SDMA-positive cells among MTAP-negative cells in brain tumor tissue.
"""

import streamlit as st
import numpy as np
import tifffile
from PIL import Image
from io import BytesIO
import cv2

from config.defaults import (
    CYTOPLASM_DILATION_RADIUS,
    MIN_NUCLEUS_AREA,
    MAX_NUCLEUS_AREA,
    GAUSSIAN_SIGMA,
    WATERSHED_MIN_DISTANCE,
    MORPHOLOGY_DISK_RADIUS,
    MIN_CYTOPLASM_PIXELS,
    DOWNSCALE_DISPLAY,
)
from pipeline.color_deconvolution import deconvolve, hsv_fallback
from pipeline.nucleus_segmentation import segment_nuclei
from pipeline.cell_classification import classify_mtap, classify_sdma
from pipeline.visualization import (
    mtap_overlay,
    sdma_overlay,
    intensity_histogram,
    channel_preview,
    sdma_slide_figure,
)
from pipeline.export import save_analysis, load_analysis

st.set_page_config(
    page_title="IHC Double Staining Quantification",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────
for key, default in [
    ("step", "upload"),
    ("image_rgb", None),
    ("hematoxylin", None),
    ("red_channel", None),
    ("brown_channel", None),
    ("labels", None),
    ("mtap_positive", None),
    ("mtap_negative", None),
    ("cell_red_intensities", None),
    ("mtap_threshold", None),
    ("sdma_positive", None),
    ("sdma_negative", None),
    ("cell_brown_intensities", None),
    ("sdma_threshold", None),
    ("source_filename", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def downscale_for_display(img: np.ndarray, factor: float = DOWNSCALE_DISPLAY) -> np.ndarray:
    """Downscale image for Streamlit display."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    if new_h < 1 or new_w < 1:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_png_bytes(img: np.ndarray) -> bytes:
    """Convert RGB array to PNG bytes for download."""
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ── Apply pending imported parameters before widgets are created ──────
if "_pending_params" in st.session_state:
    params = st.session_state.pop("_pending_params")
    if "gaussian_sigma" in params:
        st.session_state.seg_sigma = params["gaussian_sigma"]
    if "min_area" in params:
        st.session_state.seg_min_area = params["min_area"]
    if "max_area" in params:
        st.session_state.seg_max_area = params["max_area"]
    if "min_distance" in params:
        st.session_state.seg_min_dist = params["min_distance"]
    if "dilation_radius" in params:
        st.session_state.mtap_dilation = params["dilation_radius"]
    if "min_cytoplasm_pixels" in params:
        st.session_state.mtap_min_cyto = params["min_cytoplasm_pixels"]

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("Parameters")

seg_expander = st.sidebar.expander("Segmentation", expanded=False)
with seg_expander:
    p_gaussian_sigma = st.slider("Gaussian sigma", 0.5, 5.0, GAUSSIAN_SIGMA, 0.5, key="seg_sigma")
    p_min_area = st.slider("Min nucleus area (px)", 20, 500, MIN_NUCLEUS_AREA, 10, key="seg_min_area")
    p_max_area = st.slider("Max nucleus area (px)", 1000, 20000, MAX_NUCLEUS_AREA, 500, key="seg_max_area")
    p_min_distance = st.slider("Watershed min distance", 3, 20, WATERSHED_MIN_DISTANCE, 1, key="seg_min_dist")

mtap_expander = st.sidebar.expander("MTAP Classification", expanded=False)
with mtap_expander:
    p_dilation = st.slider("Cytoplasm ring width (px)", 5, 30, CYTOPLASM_DILATION_RADIUS, 1, key="mtap_dilation")
    p_min_cyto = st.slider("Min cytoplasm pixels", 5, 100, MIN_CYTOPLASM_PIXELS, 5, key="mtap_min_cyto")

# ── Title ─────────────────────────────────────────────────────────────
st.title("IHC Double Staining Quantification")
st.caption("SDMA-positive cells among MTAP-negative cells — Red (MTAP) + Brown/DAB (SDMA) + Blue (Hematoxylin)")


# ══════════════════════════════════════════════════════════════════════
# STEP: Upload
# ══════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "Upload a microscopy image or exported analysis (.npz)",
    type=["tif", "tiff", "png", "jpg", "jpeg", "npz"],
)

if uploaded is not None and st.session_state.image_rgb is None:
    raw = uploaded.read()

    # ── Import from .npz bundle ──────────────────────────────────────
    if uploaded.name.lower().endswith(".npz"):
        with st.status("Importing analysis...", expanded=True) as status:
            st.write("Loading exported analysis bundle...")
            state = load_analysis(raw)
            for key, value in state.items():
                if key != "parameters":
                    st.session_state[key] = value
            # Store parameters for restoration on next rerun (before widgets)
            params = state.get("parameters", {})
            if params:
                st.session_state._pending_params = params
            n_nuclei = int(state["labels"].max())
            st.write(f"Restored **{n_nuclei}** nuclei from {state.get('source_filename', 'unknown')}")
            status.update(label="Import complete!", state="complete")
        st.rerun()

    # ── Normal image upload ──────────────────────────────────────────
    else:
        with st.status("Loading and processing image...", expanded=True) as status:
            st.write("Reading image file...")
            st.session_state.source_filename = uploaded.name
            if uploaded.name.lower().endswith((".tif", ".tiff")):
                img = tifffile.imread(BytesIO(raw))
            else:
                img = np.array(Image.open(BytesIO(raw)))

            # Handle various formats
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]  # drop alpha

            if img.dtype != np.uint8:
                if img.max() > 255:
                    img = (img / img.max() * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            st.session_state.image_rgb = img
            st.write(f"Image shape: {img.shape[1]}×{img.shape[0]} px")

            # Deconvolution
            st.write("Running color deconvolution...")
            h_ch, r_ch, b_ch = deconvolve(img)
            st.session_state.hematoxylin = h_ch
            st.session_state.red_channel = r_ch
            st.session_state.brown_channel = b_ch

            # Segmentation
            st.write("Segmenting nuclei...")
            labels = segment_nuclei(
                h_ch,
                gaussian_sigma=p_gaussian_sigma,
                min_area=p_min_area,
                max_area=p_max_area,
                min_distance=p_min_distance,
            )
            st.session_state.labels = labels
            n_nuclei = int(labels.max())
            st.write(f"Detected **{n_nuclei}** nuclei")

            # MTAP classification
            st.write("Classifying MTAP...")
            mtap_pos, mtap_neg, red_int, mtap_thr = classify_mtap(
                labels, r_ch,
                dilation_radius=p_dilation,
                min_cytoplasm_pixels=p_min_cyto,
            )
            st.session_state.mtap_positive = mtap_pos
            st.session_state.mtap_negative = mtap_neg
            st.session_state.cell_red_intensities = red_int
            st.session_state.mtap_threshold = mtap_thr

            status.update(label="Processing complete!", state="complete")
            st.session_state.step = "mtap_review"

# ══════════════════════════════════════════════════════════════════════
# STEP 1: MTAP Review
# ══════════════════════════════════════════════════════════════════════
if st.session_state.step in ("mtap_review", "sdma_results") and st.session_state.image_rgb is not None:
    st.divider()
    st.header("Step 1 — MTAP Classification Review")

    img = st.session_state.image_rgb
    labels = st.session_state.labels
    red_int = st.session_state.cell_red_intensities

    # Threshold slider
    intensities_arr = np.array(list(red_int.values()))
    i_min = float(intensities_arr.min()) if len(intensities_arr) > 0 else 0.0
    i_max = float(intensities_arr.max()) if len(intensities_arr) > 0 else 1.0
    mtap_thr = st.slider(
        "MTAP threshold (red cytoplasmic intensity)",
        min_value=i_min,
        max_value=i_max,
        value=st.session_state.mtap_threshold,
        key="mtap_thr_slider",
    )

    # Re-classify if threshold changed
    if mtap_thr != st.session_state.mtap_threshold:
        st.session_state.mtap_threshold = mtap_thr
        mtap_pos = {lid for lid, v in red_int.items() if v >= mtap_thr}
        mtap_neg = {lid for lid, v in red_int.items() if v < mtap_thr}
        st.session_state.mtap_positive = mtap_pos
        st.session_state.mtap_negative = mtap_neg
        # Reset SDMA if threshold changed
        st.session_state.sdma_positive = None
        st.session_state.sdma_negative = None
        st.session_state.step = "mtap_review"

    mtap_pos = st.session_state.mtap_positive
    mtap_neg = st.session_state.mtap_negative

    # Metrics
    n_total = len(mtap_pos) + len(mtap_neg)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total nuclei", n_total)
    col_m2.metric("MTAP-positive", len(mtap_pos))
    col_m3.metric("MTAP-negative", len(mtap_neg))

    # Side-by-side: original vs overlay
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(downscale_for_display(img), use_container_width=True)
    with col2:
        st.subheader("MTAP Overlay")
        overlay = mtap_overlay(img, labels, mtap_pos, mtap_neg)
        st.image(downscale_for_display(overlay), use_container_width=True)
        st.caption("Green = MTAP-negative, Red = MTAP-positive")

    # Histogram
    hist_buf = intensity_histogram(
        red_int, mtap_thr,
        title="Red Cytoplasmic Intensity",
        xlabel="Mean red intensity in cytoplasmic ring",
        pos_label="MTAP+",
        neg_label="MTAP−",
    )
    st.image(hist_buf, use_container_width=True)

    # QC: deconvolved channels
    with st.expander("QC: Deconvolved Channels", expanded=False):
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            st.image(
                channel_preview(st.session_state.hematoxylin, "Hematoxylin"),
                use_container_width=True,
            )
        with qc2:
            st.image(
                channel_preview(st.session_state.red_channel, "Red (MTAP)"),
                use_container_width=True,
            )
        with qc3:
            st.image(
                channel_preview(st.session_state.brown_channel, "Brown/DAB (SDMA)"),
                use_container_width=True,
            )

    # Proceed button
    if st.button("Proceed to SDMA Analysis →", type="primary"):
        with st.status("Classifying SDMA...", expanded=True) as status:
            st.write("Measuring brown/DAB intensity in MTAP-negative nuclei...")
            sdma_pos, sdma_neg, brown_int, sdma_thr = classify_sdma(
                labels,
                st.session_state.brown_channel,
                mtap_neg,
            )
            st.session_state.sdma_positive = sdma_pos
            st.session_state.sdma_negative = sdma_neg
            st.session_state.cell_brown_intensities = brown_int
            st.session_state.sdma_threshold = sdma_thr
            st.session_state.step = "sdma_results"
            status.update(label="SDMA classification complete!", state="complete")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# STEP 2: SDMA Results
# ══════════════════════════════════════════════════════════════════════
if st.session_state.step == "sdma_results" and st.session_state.sdma_positive is not None:
    st.divider()
    st.header("Step 2 — SDMA Quantification Results")

    img = st.session_state.image_rgb
    labels = st.session_state.labels
    brown_int = st.session_state.cell_brown_intensities
    mtap_pos = st.session_state.mtap_positive
    mtap_neg = st.session_state.mtap_negative

    # Threshold slider
    intensities_arr = np.array(list(brown_int.values()))
    b_min = float(intensities_arr.min()) if len(intensities_arr) > 0 else 0.0
    b_max = float(intensities_arr.max()) if len(intensities_arr) > 0 else 1.0
    sdma_thr = st.slider(
        "SDMA threshold (brown nuclear intensity)",
        min_value=b_min,
        max_value=b_max,
        value=st.session_state.sdma_threshold,
        key="sdma_thr_slider",
    )

    # Re-classify if threshold changed
    if sdma_thr != st.session_state.sdma_threshold:
        st.session_state.sdma_threshold = sdma_thr
        sdma_pos = {lid for lid, v in brown_int.items() if v >= sdma_thr}
        sdma_neg = {lid for lid, v in brown_int.items() if v < sdma_thr}
        st.session_state.sdma_positive = sdma_pos
        st.session_state.sdma_negative = sdma_neg

    sdma_pos = st.session_state.sdma_positive
    sdma_neg = st.session_state.sdma_negative

    # Key metric
    n_mtap_neg = len(mtap_neg)
    n_sdma_pos = len(sdma_pos)
    pct = (n_sdma_pos / n_mtap_neg * 100) if n_mtap_neg > 0 else 0.0

    st.markdown(
        f"### {pct:.1f}% of MTAP-negative cells are SDMA-positive"
    )

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("MTAP-negative", n_mtap_neg)
    col_s2.metric("SDMA-positive", n_sdma_pos)
    col_s3.metric("SDMA-negative", len(sdma_neg))
    col_s4.metric("MTAP-positive (excluded)", len(mtap_pos))

    # Side-by-side overlay
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(downscale_for_display(img), use_container_width=True)
    with col2:
        st.subheader("SDMA Overlay")
        sdma_ov = sdma_overlay(img, labels, sdma_pos, sdma_neg, mtap_pos)
        st.image(downscale_for_display(sdma_ov), use_container_width=True)
        st.caption("Yellow = SDMA+ (among MTAP−), Blue = SDMA− (among MTAP−), Grey = MTAP+")

    # Histogram
    hist_buf = intensity_histogram(
        brown_int, sdma_thr,
        title="Brown/DAB Nuclear Intensity (MTAP-negative cells)",
        xlabel="Mean brown intensity in nucleus",
        pos_label="SDMA+",
        neg_label="SDMA−",
    )
    st.image(hist_buf, use_container_width=True)

    # Summary table
    with st.expander("Summary Statistics", expanded=True):
        st.markdown(f"""
| Metric | Count | Percentage |
|--------|------:|----------:|
| Total nuclei detected | {len(mtap_pos) + len(mtap_neg)} | — |
| MTAP-positive | {len(mtap_pos)} | {len(mtap_pos)/(len(mtap_pos)+len(mtap_neg))*100:.1f}% |
| MTAP-negative | {n_mtap_neg} | {n_mtap_neg/(len(mtap_pos)+len(mtap_neg))*100:.1f}% |
| SDMA-positive (of MTAP−) | {n_sdma_pos} | {pct:.1f}% |
| SDMA-negative (of MTAP−) | {len(sdma_neg)} | {100-pct:.1f}% |
""")

    # Downloads
    st.divider()
    st.subheader("Downloads")
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    source_fname = st.session_state.source_filename or "analysis"
    base_name = source_fname.rsplit(".", 1)[0] if "." in source_fname else source_fname

    with dl_col1:
        npz_bytes = save_analysis(
            image_rgb=img,
            labels=labels,
            hematoxylin=st.session_state.hematoxylin,
            red_channel=st.session_state.red_channel,
            brown_channel=st.session_state.brown_channel,
            cell_red_intensities=st.session_state.cell_red_intensities,
            cell_brown_intensities=brown_int,
            mtap_threshold=st.session_state.mtap_threshold,
            sdma_threshold=st.session_state.sdma_threshold,
            step=st.session_state.step,
            source_filename=source_fname,
            parameters={
                "gaussian_sigma": st.session_state.get("seg_sigma", GAUSSIAN_SIGMA),
                "min_area": st.session_state.get("seg_min_area", MIN_NUCLEUS_AREA),
                "max_area": st.session_state.get("seg_max_area", MAX_NUCLEUS_AREA),
                "min_distance": st.session_state.get("seg_min_dist", WATERSHED_MIN_DISTANCE),
                "dilation_radius": st.session_state.get("mtap_dilation", CYTOPLASM_DILATION_RADIUS),
                "min_cytoplasm_pixels": st.session_state.get("mtap_min_cyto", MIN_CYTOPLASM_PIXELS),
            },
        )
        st.download_button(
            "Export analysis (.npz)",
            data=npz_bytes,
            file_name=f"{base_name}_analysis.npz",
            mime="application/octet-stream",
        )

    with dl_col2:
        slide_buf = sdma_slide_figure(
            overlay_rgb=sdma_ov,
            source_filename=source_fname,
            n_mtap_pos=len(mtap_pos),
            n_mtap_neg=n_mtap_neg,
            n_sdma_pos=n_sdma_pos,
            n_sdma_neg=len(sdma_neg),
            mtap_threshold=st.session_state.mtap_threshold,
            sdma_threshold=st.session_state.sdma_threshold,
        )
        st.download_button(
            "Download slide figure (.png)",
            data=slide_buf.getvalue(),
            file_name=f"{base_name}_slide_figure.png",
            mime="image/png",
        )

    with dl_col3:
        mtap_ov = mtap_overlay(img, labels, mtap_pos, mtap_neg)
        st.download_button(
            "MTAP overlay (.png)",
            data=image_to_png_bytes(mtap_ov),
            file_name=f"{base_name}_mtap_overlay.png",
            mime="image/png",
        )

    with dl_col4:
        st.download_button(
            "SDMA overlay (.png)",
            data=image_to_png_bytes(sdma_ov),
            file_name=f"{base_name}_sdma_overlay.png",
            mime="image/png",
        )

    # Reset button
    st.divider()
    if st.button("Start over with new image"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

elif st.session_state.image_rgb is None:
    st.info("Upload a TIF image to begin analysis.")
