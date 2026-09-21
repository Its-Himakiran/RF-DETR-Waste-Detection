"""
RF-DETR: Multi-Class Waste Detection using a Transformer Framework Based on DINOv2 with Real-World Field Validation
Streamlit Production Application with Precision UI/UX & Real Model Inference
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from PIL import Image

# Import detection engine directly from core/detector.py
from core.detector import (
    ModelSpec,
    DetectionResult,
    WasteDetector,
    build_registry,
)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RF-DETR: Multi-Class Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- COMPATIBILITY HELPERS (PREVENTS STREAMLIT DEPRECATION WARNINGS) ---
def render_button(label: str, **kwargs: Any) -> bool:
    """Renders button using modern width parameter to eliminate terminal warnings."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except (TypeError, ValueError):
        return st.button(label, use_container_width=True, **kwargs)


def render_image(image: Any, caption: str | None = None, **kwargs: Any) -> None:
    """Renders image using modern width parameter to eliminate terminal warnings."""
    try:
        st.image(image, caption=caption, width="stretch", **kwargs)
    except (TypeError, ValueError):
        st.image(image, caption=caption, use_container_width=True, **kwargs)


# --- EXACT THEME STYLING, ANIMATIONS & EFFECTS ---
st.markdown(
    """
    <style>
        /* Base page styling matching reference dark theme */
        [data-testid="stAppViewContainer"] {
            background-color: #070c14 !important;
            color: #e2e8f0;
        }
        [data-testid="stSidebar"] {
            background-color: #04080e !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }

        /* Hero Banner Gradient & Glow */
        .hero-card {
            background: linear-gradient(135deg, #092c25 0%, #0d3830 50%, #0b2729 100%);
            border: 1px solid rgba(0, 229, 153, 0.35);
            border-radius: 14px;
            padding: 26px 32px;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            animation: fadeIn 0.7s ease-out;
        }
        .hero-title {
            color: #ffffff;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
            line-height: 1.35;
        }
        .hero-title span.recycle-icon {
            color: #2ed573;
            font-size: 1.7rem;
            filter: drop-shadow(0 0 8px rgba(46, 213, 115, 0.6));
        }
        .hero-subtitle {
            color: #8da4b8;
            font-size: 0.95rem;
            font-weight: 400;
            margin-bottom: 16px;
        }

        /* Top Feature Badges */
        .badge-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }
        .badge-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            font-weight: 600;
            padding: 5px 13px;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #d1dbe5;
            transition: all 0.25s ease;
        }
        .badge-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-green { background-color: #00e599; box-shadow: 0 0 6px #00e599; }
        .dot-pink { background-color: #ff477e; box-shadow: 0 0 6px #ff477e; }
        .dot-yellow { background-color: #feca57; box-shadow: 0 0 6px #feca57; }

        /* Sidebar Styling */
        .sidebar-heading {
            font-size: 0.95rem;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .val-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
            margin-top: 6px;
        }
        .val-table th {
            color: #6d8498;
            font-weight: 600;
            text-align: left;
            padding: 6px 4px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .val-table td {
            padding: 7px 4px;
            color: #c9d6e4;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .val-highlight {
            color: #00e599;
            font-weight: 700;
        }
        .sidebar-footer {
            font-size: 0.72rem;
            color: #5c7285;
            line-height: 1.45;
            margin-top: 10px;
        }

        /* High-Visibility Animated Detect Button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #00e599 0%, #00a884 100%) !important;
            color: #04140d !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            padding: 12px 28px !important;
            border-radius: 10px !important;
            border: 1px solid #38ef7d !important;
            box-shadow: 0 0 20px rgba(0, 229, 153, 0.35) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            letter-spacing: 0.02em !important;
            width: 100% !important;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: 0 0 28px rgba(0, 229, 153, 0.6) !important;
            color: #000000 !important;
        }
        div.stButton > button:first-child:active {
            transform: translateY(1px) !important;
        }

        /* Model Comparison Column Headers */
        .col-tag {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #7b93a7;
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .tag-taco { color: #feca57; }
        .tag-mw { color: #54a0ff; }

        /* Class Pill Badges */
        .obj-badge {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 0.76rem;
            color: #d2dbe5;
            margin-right: 6px;
            margin-bottom: 6px;
            display: inline-block;
        }
        .obj-badge b {
            color: #00e599;
        }

        /* Subtle Fade In Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOAD MODELS ONCE USING STREAMLIT CACHE ---
@st.cache_resource(show_spinner=False)
def load_cached_detectors() -> Tuple[Dict[str, WasteDetector], Dict[str, ModelSpec]]:
    models_dir = Path(__file__).resolve().parent / "models"
    registry = build_registry(models_dir)
    detectors: Dict[str, WasteDetector] = {}

    try:
        detectors["taco"] = WasteDetector(registry["taco"])
    except Exception as err:
        st.sidebar.warning(f"TACO Model offline: {err}")

    try:
        detectors["multiwaste"] = WasteDetector(registry["multiwaste"])
    except Exception as err:
        st.sidebar.warning(f"MultipleWaste Model offline: {err}")

    return detectors, registry


detectors, registry = load_cached_detectors()

# --- SESSION STATE INITIALIZATION ---
if "staged_images" not in st.session_state:
    st.session_state.staged_images = {}
if "detection_store" not in st.session_state:
    st.session_state.detection_store = {}
if "inference_complete" not in st.session_state:
    st.session_state.inference_complete = False

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown('<div class="sidebar-heading">⚙️ Configuration</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Detection model",
        options=["Both (Comparison)", "TACO RF-DETR", "MultipleWaste RF-DETR"],
        index=0,
    )

    conf_slider = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.50,
        step=0.05,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">📊 Validation results</div>', unsafe_allow_html=True)

    # Validated Metrics
    st.markdown(
        """
        <table class="val-table">
            <thead>
                <tr>
                    <th>MODEL</th>
                    <th>CLS</th>
                    <th>mAP@50</th>
                    <th>mAP@50:95</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>TACO v3</b></td>
                    <td>10</td>
                    <td class="val-highlight">52.5</td>
                    <td class="val-highlight">41.9</td>
                </tr>
                <tr>
                    <td><b>MultipleWaste v3</b></td>
                    <td>24</td>
                    <td class="val-highlight">79.4</td>
                    <td class="val-highlight">69.9</td>
                </tr>
            </tbody>
        </table>
        <div class="sidebar-footer">
            Both models: RF-DETR Medium, resolution 704.<br>
            Backbone: DINOv2-Medium. Fine-tuned separately; datasets never merged.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading">🛠️ Training Hyperparameters</div>', unsafe_allow_html=True)

    # Training Specifications
    st.markdown(
        """
        <table class="val-table">
            <thead>
                <tr>
                    <th>HYPERPARAMETER</th>
                    <th>VALUE</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Model</b></td>
                    <td>RF-DETR-Medium</td>
                </tr>
                <tr>
                    <td><b>Framework</b></td>
                    <td>PyTorch</td>
                </tr>
                <tr>
                    <td><b>Backbone</b></td>
                    <td>DINOv2-medium (self-supervised ViT)</td>
                </tr>
                <tr>
                    <td><b>Optimiser</b></td>
                    <td>AdamW</td>
                </tr>
                <tr>
                    <td><b>Learning rate</b></td>
                    <td>1×10⁻⁴, cosine schedule</td>
                </tr>
                <tr>
                    <td><b>Warm-up</b></td>
                    <td>2 epochs</td>
                </tr>
                <tr>
                    <td><b>Weight decay</b></td>
                    <td>1×10⁻⁴</td>
                </tr>
                <tr>
                    <td><b>Precision</b></td>
                    <td>bf16-mixed</td>
                </tr>
                <tr>
                    <td><b>GPU</b></td>
                    <td>2 × NVIDIA T4 (Kaggle)</td>
                </tr>
                <tr>
                    <td><b>Checkpoint selection</b></td>
                    <td>Best EMA validation mAP@50:95</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

# --- HERO HEADER BANNER ---
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            <span class="recycle-icon">♻️</span>
            <span>RF-DETR: Multi-Class Waste Detection using a Transformer Framework Based on DINOv2 with Real-World Field Validation</span>
        </div>
        <div class="hero-subtitle">
            RF-DETR-Medium · DINOv2 backbone · TACO v3 vs Multi-Class Waste v3
        </div>
        <div class="badge-list">
            <span class="badge-item"><span class="dot dot-green"></span>mAP@50:95 up to 69.9 (mAP@50: 79.4)</span>
            <span class="badge-item"><span class="dot dot-pink"></span>RF-DETR-Medium</span>
            <span class="badge-item"><span class="dot dot-yellow"></span>Batch + ZIP upload</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- FILE UPLOADER & DETECT ACTION ---
upload_col, detect_col = st.columns([3.8, 1.2], gap="large")

with upload_col:
    files = st.file_uploader(
        "Upload images or .zip",
        type=["jpg", "jpeg", "png", "bmp", "zip"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

# Unpack and stage images in session memory
new_images: Dict[str, Image.Image] = {}
if files:
    for f in files:
        if f.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(f) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) and not name.startswith("__MACOSX"):
                            content = zf.read(name)
                            new_images[Path(name).name] = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                st.error(f"Failed to read archive {f.name}: {e}")
        else:
            new_images[f.name] = Image.open(f).convert("RGB")

    st.session_state.staged_images = new_images

with detect_col:
    detect_btn = render_button("⚡ Detect Waste")

# --- EXECUTE DETECTION ON DEMAND ---
if detect_btn:
    if not st.session_state.staged_images:
        st.warning("Please upload at least one image or ZIP file first.")
    elif not detectors:
        st.error("No detector models could be loaded. Verify .pth checkpoints exist in the models/ directory.")
    else:
        st.session_state.detection_store.clear()
        total_count = len(st.session_state.staged_images)

        status_box = st.status(f"Running RF-DETR inference on {total_count} image(s)...", expanded=True)
        bar = st.progress(0)

        for i, (name, img) in enumerate(st.session_state.staged_images.items()):
            record: Dict[str, Any] = {"raw": img}

            # Run TACO RF-DETR
            if model_choice in ("Both (Comparison)", "TACO RF-DETR") and "taco" in detectors:
                record["taco"] = detectors["taco"].predict(img, threshold=conf_slider)

            # Run MultipleWaste RF-DETR
            if model_choice in ("Both (Comparison)", "MultipleWaste RF-DETR") and "multiwaste" in detectors:
                record["multiwaste"] = detectors["multiwaste"].predict(img, threshold=conf_slider)

            st.session_state.detection_store[name] = record
            bar.progress((i + 1) / total_count)

        st.session_state.inference_complete = True
        status_box.update(label="Inference finished successfully!", state="complete", expanded=False)

# --- RESULTS RENDERING ---
if st.session_state.inference_complete and st.session_state.detection_store:
    st.markdown(
        f"<div style='color: #4cd137; font-size: 0.92rem; font-weight: 600; margin-top: 14px; margin-bottom: 12px;'>"
        f"• Processing {len(st.session_state.detection_store)} image(s) · threshold {conf_slider:.2f}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 12px;'>• Results</div>", unsafe_allow_html=True)

    for img_filename, output in st.session_state.detection_store.items():
        with st.expander(f"📁 {img_filename}", expanded=True):
            if model_choice == "Both (Comparison)":
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown('<div class="col-tag tag-taco">● TACO RF-DETR · TACO V3 - 10 CLASSES (ORIGINAL TAXONOMY)</div>', unsafe_allow_html=True)
                    if "taco" in output:
                        taco_data: DetectionResult = output["taco"]
                        render_image(taco_data.annotated)
                        st.caption(
                            f"Detections: **{taco_data.count}** | Latency: **{taco_data.inference_ms:.1f}ms** | Mean Confidence: **{taco_data.mean_confidence:.2f}**"
                        )
                        if taco_data.class_counts:
                            pills = "".join([f'<span class="obj-badge">{k}: <b>{v}</b></span>' for k, v in taco_data.class_counts.items()])
                            st.markdown(pills, unsafe_allow_html=True)
                        else:
                            st.caption("No instances above confidence threshold.")

                with c2:
                    st.markdown('<div class="col-tag tag-mw">● MULTIPLEWASTE RF-DETR · MULTI-CLASS WASTE V3 - 24 MATERIAL CLASSES</div>', unsafe_allow_html=True)
                    if "multiwaste" in output:
                        mw_data: DetectionResult = output["multiwaste"]
                        render_image(mw_data.annotated)
                        st.caption(
                            f"Detections: **{mw_data.count}** | Latency: **{mw_data.inference_ms:.1f}ms** | Mean Confidence: **{mw_data.mean_confidence:.2f}**"
                        )
                        if mw_data.class_counts:
                            pills = "".join([f'<span class="obj-badge">{k}: <b>{v}</b></span>' for k, v in mw_data.class_counts.items()])
                            st.markdown(pills, unsafe_allow_html=True)
                        else:
                            st.caption("No instances above confidence threshold.")

            elif model_choice == "TACO RF-DETR" and "taco" in output:
                st.markdown('<div class="col-tag tag-taco">● TACO RF-DETR · TACO V3 - 10 CLASSES</div>', unsafe_allow_html=True)
                taco_data: DetectionResult = output["taco"]
                render_image(taco_data.annotated)
                st.caption(f"Detections: **{taco_data.count}** | Latency: **{taco_data.inference_ms:.1f}ms**")
                if taco_data.class_counts:
                    pills = "".join([f'<span class="obj-badge">{k}: <b>{v}</b></span>' for k, v in taco_data.class_counts.items()])
                    st.markdown(pills, unsafe_allow_html=True)

            elif model_choice == "MultipleWaste RF-DETR" and "multiwaste" in output:
                st.markdown('<div class="col-tag tag-mw">● MULTIPLEWASTE RF-DETR · 24 MATERIAL CLASSES</div>', unsafe_allow_html=True)
                mw_data: DetectionResult = output["multiwaste"]
                render_image(mw_data.annotated)
                st.caption(f"Detections: **{mw_data.count}** | Latency: **{mw_data.inference_ms:.1f}ms**")
                if mw_data.class_counts:
                    pills = "".join([f'<span class="obj-badge">{k}: <b>{v}</b></span>' for k, v in mw_data.class_counts.items()])
                    st.markdown(pills, unsafe_allow_html=True)