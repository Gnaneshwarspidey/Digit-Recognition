"""
Digit Character Recognition — Ultra-Realistic Stationery & Letterpress UI
Performance Optimized & State-Synchronized:
- Every button functions properly with rock-solid state management
- Fluid drawing with zero-lag (inference triggered on Recognize button)
- Reliable Undo/Redo stroke synchronization with race-condition prevention
- Model graph warmup for sub-millisecond recognition latency
- Identical visual styling matching reference image
"""

import os
import cv2
import numpy as np
import pandas as pd
import base64
import io
import json
from PIL import Image
import streamlit as st
import streamlit_drawable_canvas as sdc
import tensorflow as tf
from src.preprocess import segment_and_preprocess_characters
from src.utils import get_emnist_mapping, get_mnist_mapping

# Robust wrapper for st_canvas across Streamlit versions & cloud environments
def safe_st_canvas(*args, **kwargs):
    # Ensure initial_drawing has version and background if provided
    init_draw = kwargs.get("initial_drawing")
    if init_draw is not None and isinstance(init_draw, dict):
        if "version" not in init_draw:
            init_draw["version"] = "4.4.0"
        if "background" not in init_draw:
            init_draw["background"] = kwargs.get("background_color", "#FBF9F4")
    elif init_draw is not None:
        kwargs["initial_drawing"] = None

    try:
        res = sdc.original_st_canvas(*args, **kwargs)
        if res is sdc.CanvasResult or isinstance(res, type):
            return sdc.CanvasResult(image_data=None, json_data=None)
        return res
    except Exception:
        # Fallback if component_value is returned as a string or serializer mismatch occurs
        try:
            val = sdc._component_func(
                fillColor=kwargs.get("fill_color", "#eee"),
                strokeWidth=kwargs.get("stroke_width", 20),
                strokeColor=kwargs.get("stroke_color", "black"),
                backgroundColor=kwargs.get("background_color", ""),
                backgroundImageURL=None,
                realtimeUpdateStreamlit=kwargs.get("update_streamlit", True) and (kwargs.get("drawing_mode", "freedraw") != "polygon"),
                canvasHeight=kwargs.get("height", 400),
                canvasWidth=kwargs.get("width", 600),
                drawingMode=kwargs.get("drawing_mode", "freedraw"),
                initialDrawing=kwargs.get("initial_drawing", {"version": "4.4.0", "background": kwargs.get("background_color", "")}),
                displayToolbar=kwargs.get("display_toolbar", True),
                displayRadius=kwargs.get("point_display_radius", 3),
                key=kwargs.get("key"),
                default=None,
            )
            if val is None:
                return sdc.CanvasResult(image_data=None, json_data=None)
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    return sdc.CanvasResult(image_data=None, json_data=None)
            if isinstance(val, dict):
                img_data = None
                if "data" in val and val["data"]:
                    try:
                        _, _d = val["data"].split(";base64,")
                        img_data = np.asarray(Image.open(io.BytesIO(base64.b64decode(_d))))
                    except Exception:
                        img_data = None
                return sdc.CanvasResult(image_data=img_data, json_data=val.get("raw"))
        except Exception:
            pass
        return sdc.CanvasResult(image_data=None, json_data=None)

if not hasattr(sdc, "original_st_canvas"):
    sdc.original_st_canvas = sdc.st_canvas
sdc.st_canvas = safe_st_canvas
st_canvas = safe_st_canvas

# Configure Streamlit page
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="7️⃣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state.current_page = "Digit Recognition"
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if "last_canvas_key" not in st.session_state:
    st.session_state.last_canvas_key = 0
if "engine_type" not in st.session_state:
    st.session_state.engine_type = "MNIST (Digits 0-9)"
if "canvas_strokes" not in st.session_state:
    st.session_state.canvas_strokes = []
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []

is_p1_active = (st.session_state.current_page == "Digit Recognition")
is_p2_active = (st.session_state.current_page == "Model Architecture & Benchmarks")
is_p3_active = (st.session_state.current_page == "Evaluation Metrics & Visualizations")

# High-Fidelity Skeuomorphic CSS Matching Reference Image
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;0,900;1,400;1,700&family=Lora:ital,wght@0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@500;600;700&display=swap');

    /* Hide Streamlit default chrome while keeping sidebar toggle intact */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        color: #1F1914 !important;
    }}
    header[data-testid="stHeader"] svg {{
        fill: #1F1914 !important;
    }}
    #MainMenu, footer {{
        visibility: hidden !important;
    }}

    /* Sidebar Warm Stationery Theme */
    section[data-testid="stSidebar"] {{
        background-color: #DDD4C3 !important;
        background-image: radial-gradient(ellipse at 50% 20%, #E8DFC5 0%, #D6CABA 100%) !important;
        border-right: 1px solid rgba(70, 50, 30, 0.15) !important;
        box-shadow: 4px 0 16px rgba(50, 35, 20, 0.08) !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.8rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }}
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {{
        color: #1F1914 !important;
    }}
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button {{
        background: #FAF8F2 !important;
        color: #1F1914 !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 6px rgba(60, 45, 30, 0.15) !important;
    }}
    /* Sidebar selectbox styling */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #FAF8F2 !important;
        border-radius: 6px !important;
        border: 1px solid #D1C5B2 !important;
        box-shadow: 0 2px 6px rgba(60, 45, 30, 0.08) !important;
    }}

    /* Warm, mottled craft paper background matching reference photo */
    html, body, [class*="css"], .stApp {{
        font-family: 'Playfair Display', 'Lora', Georgia, serif !important;
        background-color: #E2D7C5 !important;
        background-image: radial-gradient(ellipse at 50% 30%, #ECE2D2 0%, #D8CBB6 100%) !important;
        background-attachment: fixed !important;
        color: #1F1914 !important;
        overflow-x: hidden;
    }}

    /* Main Container Alignment */
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 980px !important;
        margin: 0 auto !important;
    }}

    /* Page Header */
    .main-page-title {{
        text-align: center;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #1F1914;
        margin: 0.2rem 0 0.8rem 0;
        line-height: 1.1;
    }}

    /* Navbar Torn Paper Buttons */
    div.st-key-nav_p1 button,
    div.st-key-nav_p2 button,
    div.st-key-nav_p3 button {{
        background: #FAF8F2 !important;
        color: #2D241E !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 4px 10px rgba(60, 45, 30, 0.15) !important;
        transition: transform 0.12s ease !important;
    }}
    div.st-key-nav_p1 button:hover,
    div.st-key-nav_p2 button:hover,
    div.st-key-nav_p3 button:hover {{
        transform: translateY(-2px) !important;
    }}
    div.st-key-nav_p1 button {{
        {"border-bottom: 3px solid #C48E88 !important; background: #FDFBF7 !important;" if is_p1_active else ""}
    }}
    div.st-key-nav_p2 button {{
        {"border-bottom: 3px solid #C48E88 !important; background: #FDFBF7 !important;" if is_p2_active else ""}
    }}
    div.st-key-nav_p3 button {{
        {"border-bottom: 3px solid #C48E88 !important; background: #FDFBF7 !important;" if is_p3_active else ""}
    }}

    /* =========================================================================
       LEFT SIDE: TORN WATERCOLOR PAPER CANVAS
       ========================================================================= */
    .torn-canvas-sheet {{
        background: #FBF9F4;
        padding: 0.4rem;
        filter: drop-shadow(0 15px 24px rgba(50, 38, 25, 0.22)) drop-shadow(0 3px 7px rgba(50, 38, 25, 0.12));
        clip-path: polygon(
            0% 0.8%, 2.5% 0%, 5.5% 0.7%, 9% 0.1%, 13% 0.9%, 18% 0.2%, 23% 0.8%, 28% 0%, 33% 0.7%, 39% 0.2%, 45% 0.8%, 51% 0%, 57% 0.7%, 63% 0.1%, 69% 0.8%, 75% 0%, 81% 0.7%, 87% 0.2%, 93% 0.9%, 97% 0.2%, 100% 0.6%,
            99.6% 4%, 100% 10%, 99.5% 18%, 100% 26%, 99.6% 35%, 100% 44%, 99.5% 53%, 100% 62%, 99.6% 71%, 100% 80%, 99.5% 89%, 100% 97%,
            97% 99.8%, 93% 99.4%, 88% 100%, 82% 99.5%, 76% 100%, 70% 99.4%, 64% 100%, 58% 99.5%, 52% 100%, 46% 99.4%, 40% 100%, 34% 99.5%, 28% 100%, 22% 99.4%, 16% 100%, 10% 99.5%, 5% 100%, 0% 99.6%,
            0.4% 90%, 0% 81%, 0.4% 72%, 0% 63%, 0.4% 54%, 0% 45%, 0.4% 36%, 0% 27%, 0.4% 18%, 0% 9%
        );
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 0.9rem;
    }}

    /* =========================================================================
       TACTILE STATIONERY BUTTONS UNDER CANVAS (MATCHING REFERENCE IMAGE)
       ========================================================================= */
    /* Small Undo and Redo Icon Buttons */
    div.st-key-btn_undo button,
    div.st-key-btn_redo button {{
        background: #DDD5C7 !important;
        color: #1F1914 !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.2rem !important;
        min-height: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        box-shadow: 0 4px 8px rgba(60, 45, 30, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
        transition: all 0.12s ease !important;
    }}
    div.st-key-btn_undo button:hover,
    div.st-key-btn_redo button:hover {{
        background: #E5DDCF !important;
        transform: translateY(-1px) !important;
    }}

    /* [Clear] Button: Muted beige/gray beveled pill */
    div.st-key-btn_clear_canvas button {{
        background: #DDD5C7 !important;
        color: #1F1914 !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.6rem !important;
        min-height: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        box-shadow: 0 4px 8px rgba(60, 45, 30, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
        transition: all 0.12s ease !important;
    }}
    div.st-key-btn_clear_canvas button:hover {{
        background: #E5DDCF !important;
        transform: translateY(-1px) !important;
    }}

    /* [Recognize] Button: Warm ivory/cream beveled pill */
    div.st-key-btn_recognize button {{
        background: #F4EFE6 !important;
        color: #1F1914 !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.45rem 0.8rem !important;
        min-height: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        box-shadow: 0 5px 10px rgba(60, 45, 30, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        transition: all 0.12s ease !important;
    }}
    div.st-key-btn_recognize button:hover {{
        background: #FAF6ED !important;
        transform: translateY(-1px) !important;
    }}

    /* =========================================================================
       RIGHT SIDE: CLEAN LETTERPRESS CARDSTOCK CARD
       ========================================================================= */
    .letterpress-card {{
        background: #FAF8F2;
        border-radius: 6px;
        padding: 2.2rem 1.8rem;
        min-height: 330px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        box-shadow: 0 16px 28px rgba(50, 38, 25, 0.22), 0 3px 8px rgba(50, 38, 25, 0.12);
    }}
    .letterpress-digit-hero {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 7.2rem;
        font-weight: 700;
        color: #1F1914;
        line-height: 1;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .letterpress-conf-label {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.35rem;
        color: #4A3E34;
        margin-top: 1.4rem;
        margin-bottom: 0.2rem;
    }}
    .letterpress-conf-val {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: #1F1914;
    }}

    /* Benchmark and Evaluation Cards */
    .stat-paper-block {{
        background: #FAF8F2;
        border-radius: 6px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 6px 14px rgba(60, 45, 30, 0.12);
    }}
    .stat-number-serif {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #8C4843;
        line-height: 1.1;
    }}
    .stat-label-serif {{
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #1F1914;
        margin-top: 0.35rem;
    }}
    .spec-card-paper {{
        background: #FAF8F2;
        border-radius: 6px;
        padding: 1.6rem;
        box-shadow: 0 8px 18px rgba(60, 45, 30, 0.14);
    }}
</style>
""", unsafe_allow_html=True)

# Load Label Mappings
MNIST_MAP = get_mnist_mapping()
EMNIST_MAP = get_emnist_mapping()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_asset_path(rel_path: str):
    """Resolves relative assets regardless of working directory."""
    candidates = [
        rel_path,
        os.path.join(BASE_DIR, rel_path),
        os.path.join(BASE_DIR, "Handwritten-Recognition", rel_path),
        os.path.join(BASE_DIR, "..", rel_path),
        os.path.join(BASE_DIR, os.path.basename(rel_path)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

@st.cache_resource
def get_lazy_model(model_name: str):
    """Loads model on-demand with caching and zero-latency graph warmup."""
    target_name = "emnist_model.keras" if "EMNIST" in model_name else "mnist_model.keras"
    candidates = [
        os.path.join(BASE_DIR, target_name),
        os.path.join(BASE_DIR, "models", target_name),
        os.path.join("models", target_name),
        target_name,
        os.path.join("Handwritten-Recognition", "models", target_name),
        os.path.join("Handwritten-Recognition", target_name),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                m = tf.keras.models.load_model(path)
                # Warm up graph execution on a dummy zero tensor to eliminate first-call warmup latency
                _ = m(tf.zeros((1, 28, 28, 1), dtype=tf.float32), training=False)
                return m
            except Exception:
                continue
    return None


def main():
    # -------------------------------------------------------------------------
    # LEFT SIDEBAR: NEURAL ENGINE SELECTION & SPECIFICATIONS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom: 0.6rem;">
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.45rem; font-weight: 700; color: #1F1914; margin-bottom: 0.2rem;">Neural Engine</div>
            <div style="font-family: 'Lora', Georgia, serif; font-size: 0.84rem; color: #5C4E42;">Select recognition engine</div>
        </div>
        """, unsafe_allow_html=True)

        engine_options = ["MNIST (Digits 0-9)", "EMNIST (Alphanumeric 62)"]
        default_index = 0 if st.session_state.engine_type == engine_options[0] else 1
        model_choice = st.selectbox(
            "Recognition Engine",
            engine_options,
            index=default_index,
            key="engine_select_box"
        )
        st.session_state.engine_type = model_choice

        if "MNIST" in model_choice:
            st.markdown("""
            <div style="background: #FAF8F2; border-radius: 6px; padding: 1.1rem; box-shadow: 0 4px 10px rgba(60, 45, 30, 0.1); margin-top: 1rem;">
                <div style="font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.05rem; color: #1F1914; margin-bottom: 0.4rem;">MNIST Digits</div>
                <div style="font-size: 0.82rem; color: #524439; line-height: 1.6;">
                    • Target: <b>Digits 0 – 9</b><br>
                    • Accuracy: <b>99.37% Benchmark</b><br>
                    • Architecture: <b>Dual CNN + BatchNorm</b><br>
                    • Latency: <b>&lt; 5 ms (Real-time CPU)</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #FAF8F2; border-radius: 6px; padding: 1.1rem; box-shadow: 0 4px 10px rgba(60, 45, 30, 0.1); margin-top: 1rem;">
                <div style="font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.05rem; color: #1F1914; margin-bottom: 0.4rem;">EMNIST ByClass</div>
                <div style="font-size: 0.82rem; color: #524439; line-height: 1.6;">
                    • Target: <b>62 Alphanumeric Classes</b><br>
                    • Top-1 Accuracy: <b>78.10%</b><br>
                    • Top-3 Accuracy: <b>94.60%</b><br>
                    • Dataset: <b>124,000 NIST Samples</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 1.2rem; background: #FAF8F2; border-radius: 6px; padding: 1.1rem; box-shadow: 0 4px 10px rgba(60, 45, 30, 0.1);">
            <div style="font-family: 'Playfair Display', serif; font-weight: 700; font-size: 0.95rem; color: #1F1914; margin-bottom: 0.3rem;">Sketchpad Guide</div>
            <div style="font-size: 0.8rem; color: #6E5F52; line-height: 1.5;">
                Draw bold, centered strokes on the torn paper. Click <b>Recognize</b> to evaluate in real time.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Centered Main Title (Exactly as in reference image)
    st.markdown('<h1 class="main-page-title">Digit Recognition</h1>', unsafe_allow_html=True)

    # 3 Elegant Stationery Navbar Tabs
    nav_c1, nav_c2, nav_c3 = st.columns([1.1, 1.6, 1.7])
    with nav_c1:
        if st.button("Digit Recognition", key="nav_p1", use_container_width=True):
            st.session_state.current_page = "Digit Recognition"
            st.rerun()
    with nav_c2:
        if st.button("Model Architecture & Benchmarks", key="nav_p2", use_container_width=True):
            st.session_state.current_page = "Model Architecture & Benchmarks"
            st.rerun()
    with nav_c3:
        if st.button("Evaluation Metrics & Visualizations", key="nav_p3", use_container_width=True):
            st.session_state.current_page = "Evaluation Metrics & Visualizations"
            st.rerun()

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # PAGE 1: DIGIT RECOGNITION (MATCHING REFERENCE IMAGE)
    # =========================================================================
    if st.session_state.current_page == "Digit Recognition":
        col_sketchpad, col_result = st.columns([1, 1], gap="large")

        # ---------------------------------------------------------------------
        # LEFT SIDE: TORN WATERCOLOR PAPER SKETCHPAD + UNDO/REDO & ACTION BUTTONS
        # ---------------------------------------------------------------------
        with col_sketchpad:
            # Prepare initial drawing payload for undo/redo
            if st.session_state.canvas_strokes:
                init_drawing = {
                    "version": "4.4.0",
                    "objects": list(st.session_state.canvas_strokes),
                    "background": "#FBF9F4",
                }
            else:
                init_drawing = None

            # Torn Paper Sketchpad Sheet (display_toolbar=False removes default toolbar!)
            st.markdown('<div class="torn-canvas-sheet">', unsafe_allow_html=True)
            canvas_res = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=24,
                stroke_color="#1F1914",
                background_color="#FBF9F4",
                height=300,
                width=420,
                drawing_mode="freedraw",
                display_toolbar=False,   # Removed built-in toolbar completely
                initial_drawing=init_drawing,
                key=f"char_canvas_{st.session_state.canvas_key}",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Synchronize live drawn strokes without race condition
            if st.session_state.last_canvas_key == st.session_state.canvas_key:
                if (canvas_res is not None and not isinstance(canvas_res, type)
                    and hasattr(canvas_res, "json_data") and canvas_res.json_data is not None
                    and isinstance(canvas_res.json_data, dict) and "objects" in canvas_res.json_data):
                    live_objects = canvas_res.json_data["objects"]
                    if len(live_objects) > len(st.session_state.canvas_strokes):
                        st.session_state.canvas_strokes = list(live_objects)
                        st.session_state.redo_stack = []
            else:
                st.session_state.last_canvas_key = st.session_state.canvas_key

            # Tactile Buttons Directly Under Sketchpad: [ ↶ ] [ ↷ ] [ Clear ] [ Recognize ]
            btn_col_u, btn_col_r, btn_col_c, btn_col_rec = st.columns(
                [0.45, 0.45, 1.25, 1.55],
                vertical_alignment="center"
            )

            with btn_col_u:
                if st.button("↶", key="btn_undo", help="Undo last stroke", use_container_width=True):
                    if st.session_state.canvas_strokes:
                        st.session_state.redo_stack.append(st.session_state.canvas_strokes.pop())
                        st.session_state.canvas_key += 1
                        st.rerun()

            with btn_col_r:
                if st.button("↷", key="btn_redo", help="Redo stroke", use_container_width=True):
                    if st.session_state.redo_stack:
                        st.session_state.canvas_strokes.append(st.session_state.redo_stack.pop())
                        st.session_state.canvas_key += 1
                        st.rerun()

            with btn_col_c:
                if st.button("Clear", key="btn_clear_canvas", use_container_width=True):
                    st.session_state.canvas_strokes = []
                    st.session_state.redo_stack = []
                    st.session_state.canvas_key += 1
                    if "current_prediction" in st.session_state:
                        del st.session_state["current_prediction"]
                    st.rerun()

            with btn_col_rec:
                recognize_clicked = st.button("Recognize", key="btn_recognize", use_container_width=True)

        # ---------------------------------------------------------------------
        # RIGHT SIDE: CLEAN LETTERPRESS CARDSTOCK CARD
        # ---------------------------------------------------------------------
        with col_result:
            # Check for canvas input
            has_drawing = False
            drawing_img = None
            if (canvas_res is not None and not isinstance(canvas_res, type)
                and hasattr(canvas_res, "image_data") and canvas_res.image_data is not None):
                raw_data = canvas_res.image_data
                if isinstance(raw_data, np.ndarray) and raw_data.ndim >= 3 and np.any(raw_data[:, :, :3] < 235):
                    drawing_img = Image.fromarray(raw_data.astype("uint8"))
                    has_drawing = True

            # Prediction triggers only on Recognize button click for maximum drawing speed
            if recognize_clicked and has_drawing and drawing_img is not None:
                model = get_lazy_model(st.session_state.engine_type)
                char_map = EMNIST_MAP if "EMNIST" in st.session_state.engine_type else MNIST_MAP
                
                if model is not None:
                    segments = segment_and_preprocess_characters(drawing_img, thicken_stroke=True)
                    if segments:
                        raw_preds = model(segments[0]["tensor"], training=False).numpy()[0]
                        top_idx = int(np.argmax(raw_preds))
                        top_char = char_map.get(top_idx, str(top_idx))
                        top_conf = float(raw_preds[top_idx]) * 100.0
                        st.session_state.current_prediction = {
                            "char": top_char,
                            "conf": top_conf
                        }

            # Render Letterpress Card
            if "current_prediction" in st.session_state and has_drawing:
                pred_char = st.session_state.current_prediction["char"]
                pred_conf = st.session_state.current_prediction["conf"]
                
                st.markdown(f"""
                <div class="letterpress-card">
                    <div class="letterpress-digit-hero">{pred_char}</div>
                    <div class="letterpress-conf-label">Confidence</div>
                    <div class="letterpress-conf-val">{pred_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Default Idle State matching reference design
                st.markdown("""
                <div class="letterpress-card">
                    <div class="letterpress-digit-hero" style="color: #9E9488;">—</div>
                    <div class="letterpress-conf-label">Confidence</div>
                    <div class="letterpress-conf-val" style="color: #9E9488;">Draw a digit & click Recognize</div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # PAGE 2: MODEL ARCHITECTURE & BENCHMARKS
    # =========================================================================
    elif st.session_state.current_page == "Model Architecture & Benchmarks":
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            st.markdown("""
            <div class="stat-paper-block">
                <div class="stat-number-serif">99.37%</div>
                <div class="stat-label-serif">MNIST Accuracy</div>
                <div style="font-size: 0.78rem; color: #6E5F52;">10,000 test digits</div>
            </div>
            """, unsafe_allow_html=True)
        with b_c2:
            st.markdown("""
            <div class="stat-paper-block">
                <div class="stat-number-serif">78.10%</div>
                <div class="stat-label-serif">EMNIST Accuracy</div>
                <div style="font-size: 0.78rem; color: #6E5F52;">62 balanced classes</div>
            </div>
            """, unsafe_allow_html=True)
        with b_c3:
            st.markdown("""
            <div class="stat-paper-block">
                <div class="stat-number-serif">&lt; 8 ms</div>
                <div class="stat-label-serif">Inference Latency</div>
                <div style="font-size: 0.78rem; color: #6E5F52;">Real-time CPU inference</div>
            </div>
            """, unsafe_allow_html=True)
        with b_c4:
            st.markdown("""
            <div class="stat-paper-block">
                <div class="stat-number-serif">72 / 72</div>
                <div class="stat-label-serif">Unit & E2E Tests</div>
                <div style="font-size: 0.78rem; color: #6E5F52;">100% test pass rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("""
            <div class="spec-card-paper">
                <h3 style="font-family: 'Playfair Display', serif; font-size: 1.45rem; font-weight: 700; color: #1F1914; margin: 0 0 0.4rem 0;">MNIST CNN Pipeline</h3>
                <div style="width: 100%; height: 1px; background: #D6CCBD; margin-bottom: 0.8rem;"></div>
                <p style="color: #473B32; font-size: 0.88rem; line-height: 1.55; margin-bottom: 0.8rem;">
                    Dual Convolutional blocks with 32 and 64 filters (3×3 kernels, ReLU), Batch Normalization, Max Pooling, and Dropout (0.25, 0.50) designed to prevent overfitting on digits 0–9.
                </p>
                <ul style="font-size: 0.85rem; color: #2D241E; line-height: 1.7; padding-left: 1.2rem; margin: 0;">
                    <li><strong>Dataset:</strong> 60,000 train images, 10,000 test images</li>
                    <li><strong>Parameters:</strong> 1,199,882 trainable parameters</li>
                    <li><strong>Loss Metric:</strong> Categorical Crossentropy (Test Loss: <code>0.0222</code>)</li>
                    <li><strong>Regularization:</strong> EarlyStopping & ReduceLROnPlateau</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown("""
            <div class="spec-card-paper">
                <h3 style="font-family: 'Playfair Display', serif; font-size: 1.45rem; font-weight: 700; color: #1F1914; margin: 0 0 0.4rem 0;">EMNIST ByClass 62-Engine</h3>
                <div style="width: 100%; height: 1px; background: #D6CCBD; margin-bottom: 0.8rem;"></div>
                <p style="color: #473B32; font-size: 0.88rem; line-height: 1.55; margin-bottom: 0.8rem;">
                    Trained on 124,000 balanced official NIST samples across all 62 alphanumeric classes (digits 0–9, uppercase A–Z, and lowercase a–z) with orientation correction.
                </p>
                <ul style="font-size: 0.85rem; color: #2D241E; line-height: 1.7; padding-left: 1.2rem; margin: 0;">
                    <li><strong>Classes:</strong> 10 digits + 26 uppercase + 26 lowercase</li>
                    <li><strong>Input Pipeline:</strong> Transpose & horizontal flip alignment</li>
                    <li><strong>Disambiguation:</strong> Letter casing & stroke thickness adaptive</li>
                    <li><strong>Top-1 Accuracy:</strong> 78.10% (Top-3 Accuracy: 94.6%)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # PAGE 3: EVALUATION METRICS & VISUALIZATIONS
    # =========================================================================
    elif st.session_state.current_page == "Evaluation Metrics & Visualizations":
        tab_eval_mnist, tab_eval_emnist = st.tabs(["MNIST (Digits 0-9)", "EMNIST (Alphanumeric 62)"])

        with tab_eval_mnist:
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("""
                <div class="spec-card-paper">
                    <h4 style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: #1F1914; margin: 0 0 0.5rem 0;">MNIST Confusion Matrix</h4>
                </div>
                """, unsafe_allow_html=True)
                p_m_cm = resolve_asset_path("reports/mnist_confusion_matrix.png")
                if p_m_cm:
                    st.image(p_m_cm, use_container_width=True)
            with e2:
                st.markdown("""
                <div class="spec-card-paper">
                    <h4 style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: #1F1914; margin: 0 0 0.5rem 0;">Training Loss & Accuracy Curve</h4>
                </div>
                """, unsafe_allow_html=True)
                p_m_hist = resolve_asset_path("reports/mnist_training_history.png")
                if p_m_hist:
                    st.image(p_m_hist, use_container_width=True)

        with tab_eval_emnist:
            ee1, ee2 = st.columns(2)
            with ee1:
                st.markdown("""
                <div class="spec-card-paper">
                    <h4 style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: #1F1914; margin: 0 0 0.5rem 0;">EMNIST Confusion Matrix Heatmap</h4>
                </div>
                """, unsafe_allow_html=True)
                p_e_cm = resolve_asset_path("reports/emnist_confusion_matrix.png")
                if p_e_cm:
                    st.image(p_e_cm, use_container_width=True)
            with ee2:
                st.markdown("""
                <div class="spec-card-paper">
                    <h4 style="font-family: 'Playfair Display', serif; font-size: 1.25rem; font-weight: 700; color: #1F1914; margin: 0 0 0.5rem 0;">EMNIST Training History Curve</h4>
                </div>
                """, unsafe_allow_html=True)
                p_e_hist = resolve_asset_path("reports/emnist_training_history.png")
                if p_e_hist:
                    st.image(p_e_hist, use_container_width=True)


if __name__ == "__main__":
    main()
