"""
Streamlit Application for Handwritten Digit and Character Recognition

Features:
- Sidebar model selector: "MNIST (Digits 0-9)" or "EMNIST (Digits + Letters)"
- Interactive drawing canvas (streamlit-drawable-canvas) with white background & black stroke
- Image upload option (PNG, JPG, JPEG)
- Advanced MNIST/EMNIST-grade preprocessing:
    1. Grayscale conversion with RGBA alpha flattening
    2. Automatic color inversion (bright stroke on black background)
    3. Otsu automatic thresholding to remove antialiasing noise
    4. Morphological dilation for realistic stroke thickening
    5. Non-zero bounding box detection and tight cropping
    6. Aspect-ratio preserving scale into 20x20 inner box
    7. 28x28 black canvas padding
    8. Center-of-Mass translation to (14, 14) (standard MNIST centering)
    9. Normalization to [0.0, 1.0] and reshaping to (1, 28, 28, 1)
- Model loading with @st.cache_resource
- Real-time Top 1 and Top 3 predictions with confidence bars
- Live visual inspection of preprocessed 28x28 tensor
"""

import os
import string
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Handwritten Digit & Character Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean modern UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .pred-card {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .pred-char {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .pred-conf {
        font-size: 1.25rem;
        font-weight: 500;
        opacity: 0.95;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Build EMNIST ByClass (62 classes) label mapping
def get_emnist_mapping():
    mapping = {}
    for i in range(10):
        mapping[i] = str(i)
    for i, char in enumerate(string.ascii_uppercase):
        mapping[10 + i] = char
    for i, char in enumerate(string.ascii_lowercase):
        mapping[36 + i] = char
    return mapping


EMNIST_MAPPING = get_emnist_mapping()
MNIST_MAPPING = {i: str(i) for i in range(10)}


@st.cache_resource
def load_trained_model(model_path):
    """Load and cache Keras model."""
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None


def preprocess_image(image: Image.Image, thicken_stroke: bool = True) -> tuple[np.ndarray, Image.Image]:
    """
    Advanced MNIST/EMNIST standard preprocessing pipeline:
    1. Grayscale Conversion: Handles alpha compositing over white background.
    2. Color Inversion: Ensures character strokes are bright (255) on dark (0) background.
    3. Thresholding: Applies Otsu's binarization to remove anti-aliasing noise and stray artifacts.
    4. Stroke Thickening: Morphological dilation to compensate for thin canvas strokes.
    5. Bounding Box Cropping: Finds the tight non-zero bounding box around the digit/character.
    6. Aspect-Ratio Scaling: Scales the character to fit within a 20x20 box (standard MNIST convention).
    7. 28x28 Canvas Padding: Embeds the 20x20 character onto a 28x28 black canvas.
    8. Center of Mass Translation: Shifts the 28x28 image so the center of mass aligns at (14, 14).
    9. Normalization & Reshaping: Scales pixel intensities to [0.0, 1.0] and reshapes to (1, 28, 28, 1).
    """
    # 1. Convert to grayscale numpy array with alpha channel composite
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            # Composite over white background to avoid transparent black artifacts
            canvas_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(canvas_bg, image)
        gray = np.array(image.convert("L"), dtype=np.uint8)
    else:
        gray = np.array(image, dtype=np.uint8)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # 2. Invert colors if background is light (mean > 127)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    # 3. Apply Otsu's thresholding to get sharp binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Apply morphological dilation to ensure stroke width matches MNIST training distribution
    if thicken_stroke:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 5. Find bounding box of the non-zero character pixels
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Empty drawing fallback
        empty_canvas = np.zeros((28, 28), dtype=np.float32)
        return empty_canvas.reshape(1, 28, 28, 1), Image.fromarray((empty_canvas * 255).astype(np.uint8))

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y + h, x:x + w]

    # 6. Fit cropped character into a 20x20 bounding box (preserving aspect ratio)
    if w > h:
        new_w = 20
        new_h = max(int(round((h * 20.0) / w)), 1)
    else:
        new_h = 20
        new_w = max(int(round((w * 20.0) / h)), 1)

    resized_inner = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 7. Pad the 20x20 character onto a 28x28 black canvas
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    canvas_28x28 = cv2.copyMakeBorder(
        resized_inner, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    # 8. Center of Mass adjustment (shifts digit so mass center is exactly at (14, 14))
    cy, cx = center_of_mass(canvas_28x28)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        # Apply affine transformation matrix for sub-pixel shift
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas_28x28 = cv2.warpAffine(
            canvas_28x28,
            M,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # 9. Normalize to [0.0, 1.0] and reshape to (1, 28, 28, 1)
    normalized_tensor = canvas_28x28.astype("float32") / 255.0
    tensor = normalized_tensor.reshape(1, 28, 28, 1)
    preview_img = Image.fromarray(canvas_28x28)

    return tensor, preview_img


def main():
    # Header
    st.markdown('<div class="main-title">✍️ Handwritten Digit & Character Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Interactive AI classifier powered by TensorFlow CNNs with MNIST-grade image normalization and centering.</div>',
        unsafe_allow_html=True
    )

    # Sidebar: Model Selection
    st.sidebar.header("⚙️ Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ("MNIST (Digits 0-9)", "EMNIST (Digits + Letters)")
    )

    if model_choice == "MNIST (Digits 0-9)":
        model_filename = "mnist_model.keras"
        label_map = MNIST_MAPPING
        st.sidebar.info("📌 **Model**: MNIST CNN\n\n🎯 **Targets**: Digits `0` – `9` (10 classes)")
    else:
        model_filename = "emnist_model.keras"
        label_map = EMNIST_MAPPING
        st.sidebar.info("📌 **Model**: EMNIST ByClass CNN\n\n🎯 **Targets**: Digits `0-9`, Uppercase `A-Z`, Lowercase `a-z` (62 classes)")

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), model_filename)
    model = load_trained_model(model_path)

    if model is not None:
        st.sidebar.success(f"✅ Loaded `{model_filename}`")
    else:
        st.sidebar.warning(f"⚠️ `{model_filename}` not found. Please run `python {model_filename.replace('_model.keras', '_train.py')}`.")

    st.sidebar.divider()
    st.sidebar.header("🖌️ Preprocessing Options")
    stroke_width = st.sidebar.slider("Brush Stroke Width", min_value=10, max_value=36, value=20, step=2)
    thicken_stroke = st.sidebar.checkbox("Enhance / Thicken Thin Strokes", value=True, help="Applies morphological dilation to ensure thin drawn lines match MNIST stroke thickness.")

    # Two Column Layout
    col_input, col_pred = st.columns([1.1, 1], gap="large")

    input_mode = col_input.radio(
        "Choose Input Method:",
        ("🖌️ Draw on Canvas", "📁 Upload Image"),
        horizontal=True
    )

    input_image = None
    has_drawing = False

    with col_input:
        if input_mode == "🖌️ Draw on Canvas":
            st.write("**Draw inside the canvas below:**")

            if "canvas_key" not in st.session_state:
                st.session_state.canvas_key = 0

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state.canvas_key}",
            )

            c_btn1, c_btn2 = st.columns([1, 1])
            if c_btn1.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state.canvas_key += 1
                st.rerun()

            if canvas_result.image_data is not None:
                raw_rgba = canvas_result.image_data
                rgb_channels = raw_rgba[:, :, :3]
                if np.any(rgb_channels < 240):  # Detect dark user drawing
                    input_image = Image.fromarray(raw_rgba.astype("uint8"))
                    has_drawing = True

        else:
            uploaded_file = st.file_uploader(
                "Upload an image containing a single digit or character",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                has_drawing = True
                st.image(input_image, caption="Uploaded Image", width=220)

    # Right Column: Preprocessing & Predictions
    with col_pred:
        st.subheader("🔍 Prediction Results")

        if not has_drawing or input_image is None:
            st.info("👈 Draw a digit or character on the canvas, or upload an image to view predictions.")
        else:
            # 1. Preprocess image
            processed_tensor, preview_28x28 = preprocess_image(input_image, thicken_stroke=thicken_stroke)

            # Preprocessed Visual Inspection
            with st.expander("🔬 Preprocessed 28x28 Tensor Inspection", expanded=True):
                p_col1, p_col2 = st.columns([1, 2])
                p_col1.image(preview_28x28, caption="28x28 Center-of-Mass", width=110)
                p_col2.markdown(f"""
                - **Dimensions**: `{processed_tensor.shape}`
                - **Value Range**: `[{processed_tensor.min():.2f}, {processed_tensor.max():.2f}]`
                - **Centering**: Center of Mass shifted to `(14, 14)`
                - **Cropping & Aspect**: 20x20 scaled with square padding
                """)

            # 2. Predict with Model
            if model is None:
                st.error(f"Model file `{model_filename}` is not available. Please train it first.")
            else:
                preds = model.predict(processed_tensor, verbose=0)[0]

                # Top 1 Prediction
                top1_idx = int(np.argmax(preds))
                top1_char = label_map.get(top1_idx, str(top1_idx))
                top1_conf = float(preds[top1_idx]) * 100

                # Main Prediction Banner
                st.markdown(f"""
                <div class="pred-card">
                    <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">Top Prediction</div>
                    <div class="pred-char">{top1_char}</div>
                    <div class="pred-conf">Confidence: <strong>{top1_conf:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

                # Top 3 Predictions Bar Chart
                st.markdown("#### 📊 Top 3 Predictions")
                top3_indices = np.argsort(preds)[::-1][:3]

                for rank, idx in enumerate(top3_indices, start=1):
                    char = label_map.get(int(idx), str(idx))
                    conf = float(preds[idx]) * 100
                    st.markdown(f"**#{rank} &nbsp; Predicted: `{char}` &nbsp; ({conf:.1f}%)**")
                    st.progress(min(max(conf / 100.0, 0.0), 1.0))


if __name__ == "__main__":
    main()
