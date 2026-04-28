"""
Blood Cell Detector — Streamlit Web Application
================================================
Interactive web app for detecting and classifying cells in peripheral blood
smear images using a YOLO-based object detection model.

Detects 7 cell types:
  RBC, Platelets, Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil
"""

from __future__ import annotations

import io
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
TEST_IMAGES_DIR = ROOT / "test_images"

CLASS_NAMES = {
    0: "RBC",
    1: "Platelets",
    2: "Neutrophil",
    3: "Lymphocyte",
    4: "Monocyte",
    5: "Eosinophil",
    6: "Basophil",
}

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# Colors in BGR (for OpenCV drawing) — visually distinct per category
COLORS_BGR = {
    "RBC":        (60, 60, 220),      # Deep red
    "Platelets":  (50, 190, 80),      # Green
    "Neutrophil": (230, 140, 30),     # Bright blue
    "Lymphocyte": (210, 90, 180),     # Purple-blue
    "Monocyte":   (40, 200, 220),     # Cyan-yellow
    "Eosinophil": (30, 120, 240),     # Orange
    "Basophil":   (180, 60, 200),     # Magenta
}

# Corresponding RGB hex for UI badges
COLORS_HEX = {
    "RBC":        "#DC3C3C",
    "Platelets":  "#50BE32",
    "Neutrophil": "#1E8CE6",
    "Lymphocyte": "#B45AD2",
    "Monocyte":   "#DCC828",
    "Eosinophil": "#F0781E",
    "Basophil":   "#C83CB4",
}

# Category grouping for summary
CATEGORY_MAP = {
    "RBC":        "Red Blood Cells",
    "Platelets":  "Platelets",
    "Neutrophil": "White Blood Cells",
    "Lymphocyte": "White Blood Cells",
    "Monocyte":   "White Blood Cells",
    "Eosinophil": "White Blood Cells",
    "Basophil":   "White Blood Cells",
}


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the YOLO model once and cache it across sessions."""
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    return model


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def draw_detections(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    names: dict[int, str],
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image. Returns a copy."""
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.45, 1

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, classes, confidences):
        name = names[int(cls_id)]
        color = COLORS_BGR.get(name, (200, 200, 200))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        if show_labels:
            label = name
            if show_conf:
                label = f"{name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
            ly = y1 - 4
            if ly - th - 4 < 0:
                ly = y2 + th + 6
            # Background rectangle for text
            cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
            cv2.putText(out, label, (x1 + 3, ly - 1), font, fs,
                        (255, 255, 255), ft, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for premium look
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Hero header ---- */
.hero-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(230, 57, 70, 0.25);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #E63946, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #a0aec0;
    line-height: 1.6;
}

/* ---- Stat cards ---- */
.stat-card {
    background: linear-gradient(145deg, #1A1D29, #21253a);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.stat-label {
    font-size: 0.82rem;
    color: #8892a4;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ---- Cell badge ---- */
.cell-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 3px 2px;
    border: 1px solid rgba(255,255,255,0.08);
}
.cell-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

/* ---- Results panel ---- */
.results-panel {
    background: linear-gradient(145deg, #1A1D29, #1f2336);
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ---- Sidebar styling ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117, #151923);
}

/* ---- Button styling ---- */
.stDownloadButton > button {
    background: linear-gradient(135deg, #E63946, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(230, 57, 70, 0.4) !important;
}

/* ---- Divider ---- */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(230,57,70,0.3), transparent);
    margin: 1.5rem 0;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: #4a5568;
    font-size: 0.8rem;
    margin-top: 2rem;
}
.footer a {
    color: #E63946;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 Blood Cell Detector</div>
    <div class="hero-subtitle">
        AI-powered detection and classification of cells in peripheral blood smear images.<br>
        Upload a microscopy image to identify <strong>Red Blood Cells</strong>, <strong>Platelets</strong>,
        and <strong>5 White Blood Cell subtypes</strong> — powered by YOLOv11.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence score to keep a detection. Lower values find more cells but increase false positives.",
    )

    iou_threshold = st.slider(
        "IoU threshold (NMS)",
        min_value=0.1,
        max_value=0.95,
        value=0.7,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression. Higher values allow more overlapping boxes.",
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🎨 Display Options")

    show_labels = st.checkbox("Show class labels", value=True)
    show_confidence = st.checkbox("Show confidence scores", value=True)
    line_width = st.slider("Box line width", 1, 5, 2)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Class Legend")
    for name, hex_color in COLORS_HEX.items():
        category = "WBC" if name in WBC_SUBTYPES else name
        st.markdown(
            f'<span class="cell-badge" style="background: {hex_color}18;">'
            f'<span class="cell-dot" style="background: {hex_color};"></span>'
            f'{name}</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">'
        'Built with <a href="https://streamlit.io">Streamlit</a> &amp; '
        '<a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO</a><br>'
        '⚠️ Research use only — not for clinical diagnosis'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

tab_upload, tab_sample = st.tabs(["📤 Upload Image", "🖼️ Sample Images"])

input_image: Image.Image | None = None
image_name: str = ""

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a blood smear image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Supported formats: JPG, PNG, BMP, TIFF",
    )
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
        image_name = uploaded.name

with tab_sample:
    sample_images = sorted(TEST_IMAGES_DIR.glob("*")) if TEST_IMAGES_DIR.exists() else []
    sample_images = [p for p in sample_images if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    if sample_images:
        cols = st.columns(min(len(sample_images), 3))
        for idx, img_path in enumerate(sample_images):
            col = cols[idx % 3]
            with col:
                thumb = Image.open(img_path).convert("RGB")
                st.image(thumb, caption=img_path.stem.replace("_", " ").title(), use_container_width=True,)
                if st.button(f"Use this image", key=f"sample_{idx}", use_container_width=True,):
                    input_image = thumb
                    image_name = img_path.name
    else:
        st.info("No sample images found in the `test_images/` directory.")


# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------

if input_image is not None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Convert PIL -> numpy BGR for OpenCV
    img_rgb = np.array(input_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Load model
    with st.spinner("Loading model..."):
        model = load_model()

    # Save temp file for YOLO (it expects a path or numpy array)
    with st.spinner("Running detection..."):
        results = model.predict(
            source=img_rgb,
            conf=confidence,
            iou=iou_threshold,
            imgsz=640,
            device="cpu",
            save=False,
            verbose=False,
        )

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    names = result.names

    # Count detections
    counts = Counter(names[int(c)] for c in classes)
    total = len(boxes)
    wbc_total = sum(v for k, v in counts.items() if k in WBC_SUBTYPES)
    rbc_total = counts.get("RBC", 0)
    plt_total = counts.get("Platelets", 0)

    # Draw annotated image
    annotated_bgr = draw_detections(
        img_bgr, boxes, classes, confs, names,
        show_labels=show_labels,
        show_conf=show_confidence,
        line_width=line_width,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # --- Summary statistics ---
    st.markdown("### 📊 Detection Results")

    stat_cols = st.columns(4)
    stats = [
        (str(total), "Total Cells", "#E63946"),
        (str(rbc_total), "Red Blood Cells", COLORS_HEX["RBC"]),
        (str(wbc_total), "White Blood Cells", COLORS_HEX["Neutrophil"]),
        (str(plt_total), "Platelets", COLORS_HEX["Platelets"]),
    ]
    for col, (number, label, color) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-number" style="color: {color};">{number}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # --- Image comparison ---
    col_orig, col_det = st.columns(2)

    with col_orig:
        st.markdown("**Original Image**")
        st.image(input_image, use_container_width=True,)

    with col_det:
        st.markdown("**Detected Cells**")
        st.image(annotated_rgb, use_container_width=True,)

    # --- Detailed breakdown ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    detail_col1, detail_col2 = st.columns([1, 1])

    with detail_col1:
        st.markdown("#### 🧬 Cell Type Breakdown")
        if counts:
            # Build a nice breakdown
            for name in CLASS_NAMES.values():
                count = counts.get(name, 0)
                if count > 0:
                    hex_c = COLORS_HEX.get(name, "#888")
                    pct = count / total * 100 if total > 0 else 0
                    st.markdown(
                        f'<div style="display:flex; align-items:center; gap:10px; '
                        f'margin-bottom:8px; padding:8px 12px; '
                        f'background: {hex_c}12; border-radius:8px; '
                        f'border-left: 3px solid {hex_c};">'
                        f'<span class="cell-dot" style="background:{hex_c};"></span>'
                        f'<span style="flex:1; font-weight:500;">{name}</span>'
                        f'<span style="font-weight:700; color:{hex_c};">{count}</span>'
                        f'<span style="color:#8892a4; font-size:0.85rem;">({pct:.1f}%)</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No cells detected. Try lowering the confidence threshold.")

    with detail_col2:
        st.markdown("#### 🔬 WBC Differential")
        if wbc_total > 0:
            st.markdown(
                '<p style="color:#8892a4; font-size:0.85rem; margin-bottom:12px;">'
                'Relative proportions of white blood cell subtypes</p>',
                unsafe_allow_html=True,
            )
            for name in ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]:
                count = counts.get(name, 0)
                pct = count / wbc_total * 100 if wbc_total > 0 else 0
                hex_c = COLORS_HEX[name]
                # Progress bar style
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="display:flex; justify-content:space-between; margin-bottom:3px;">'
                    f'<span style="font-size:0.9rem;">{name}</span>'
                    f'<span style="font-weight:600; color:{hex_c};">{pct:.1f}%</span>'
                    f'</div>'
                    f'<div style="height:8px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden;">'
                    f'<div style="height:100%; width:{pct}%; background:{hex_c}; border-radius:4px; '
                    f'transition: width 0.5s ease;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No white blood cells detected in this image.")

    # --- Download button ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Encode annotated image to bytes for download
    annotated_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)

    download_name = Path(image_name).stem + "_detected.jpg"
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=buf.getvalue(),
        file_name=download_name,
        mime="image/jpeg",
        use_container_width=True,
    )

else:
    # No image selected — show placeholder
    st.markdown(
        '<div style="text-align:center; padding:4rem 2rem; '
        'background: linear-gradient(145deg, #1A1D29, #1f2336); '
        'border-radius: 16px; border: 2px dashed rgba(230,57,70,0.3);">'
        '<p style="font-size:3rem; margin-bottom:0.5rem;">🔬</p>'
        '<p style="font-size:1.2rem; color:#a0aec0; font-weight:500;">Upload a blood smear image to get started</p>'
        '<p style="color:#4a5568; font-size:0.9rem;">or try one of the sample images above</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# --- Footer ---
st.markdown(
    '<div class="footer">'
    '🔬 Blood Cell Detector &mdash; YOLO26 object detection model<br>'
    'Detects: RBC &bull; Platelets &bull; Neutrophil &bull; Lymphocyte &bull; '
    'Monocyte &bull; Eosinophil &bull; Basophil<br>'
    '<em>For research and educational purposes only</em>'
    '</div>',
    unsafe_allow_html=True,
)
