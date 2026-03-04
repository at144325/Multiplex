# IHC Double Staining Quantification

A Streamlit app for quantifying **SDMA-positive cells among MTAP-negative cells** in IHC (immunohistochemistry) double-stained brain tumor tissue sections.

## What It Does

This app takes a microscopy image stained with:
- **Hematoxylin** (blue) — nuclear counterstain
- **Red chromogen** (AEC-like) — MTAP marker
- **Brown/DAB chromogen** — SDMA marker

It then:
1. **Deconvolves** the stains using color deconvolution to separate channels
2. **Segments** nuclei via watershed on the hematoxylin channel
3. **Classifies MTAP** status by measuring red intensity in a cytoplasmic ring around each nucleus
4. **Classifies SDMA** status by measuring brown/DAB intensity within MTAP-negative nuclei
5. Provides **interactive threshold sliders** to adjust classification in real-time with overlay previews and histograms

### Key Features
- Interactive threshold adjustment with live overlay updates
- Export/import full analysis as `.npz` bundles for collaboration
- Presentation-ready slide figures with overlay + quantification stats
- QC channel previews for verifying deconvolution quality
- Downloadable overlay PNGs (MTAP and SDMA)

## Setup

### Prerequisites
- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/anchi/Multiplex.git
cd Multiplex

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

### Basic Workflow

1. **Upload** a microscopy TIF/PNG/JPEG image
2. **Review MTAP classification** — adjust the red intensity threshold using the slider; cells above threshold are MTAP-positive (red overlay), below are MTAP-negative (green overlay)
3. **Proceed to SDMA analysis** — the app measures brown/DAB intensity in MTAP-negative nuclei
4. **Review SDMA results** — adjust the brown intensity threshold; SDMA-positive cells appear yellow, SDMA-negative appear blue, MTAP-positive (excluded) appear grey
5. **Download results** — overlay PNGs, slide figures, or full analysis exports

### Export & Import

- **Export analysis (.npz)**: Saves the complete analysis state (image, segmentation, intensities, thresholds, parameters) as a single compressed file
- **Import**: Upload a `.npz` file to restore a previous analysis with full interactive functionality (threshold sliders, overlays, histograms all work)
- Share `.npz` files with collaborators so they can interactively review and adjust your analysis

### Slide Figure

The "Download slide figure" button generates a 16:9 presentation-ready PNG with:
- SDMA overlay image on the left
- Headline metric, cell count breakdown, and color legend on the right
- Source filename and threshold values annotated

## Parameters

All parameters are adjustable via the sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Gaussian sigma | 2.0 | Smoothing for nucleus detection |
| Min nucleus area | 80 px | Minimum nucleus size |
| Max nucleus area | 5000 px | Maximum nucleus size |
| Watershed min distance | 8 px | Minimum distance between nuclei |
| Cytoplasm ring width | 12 px | Dilation radius for cytoplasmic ring |
| Min cytoplasm pixels | 20 | Minimum ring pixels to score a cell |

## Project Structure

```
Multiplex/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── config/
│   └── defaults.py                 # Stain vectors & default parameters
├── pipeline/
│   ├── color_deconvolution.py      # Stain separation
│   ├── nucleus_segmentation.py     # Watershed-based nuclear segmentation
│   ├── cell_classification.py      # MTAP & SDMA classification (Otsu thresholds)
│   ├── visualization.py            # Overlays, histograms, slide figures
│   └── export.py                   # Analysis export/import (.npz bundles)
└── .streamlit/
    └── config.toml                 # Streamlit theme configuration
```
