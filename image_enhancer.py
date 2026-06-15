import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import io
import torch
from transformers import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Enhancer",
    page_icon="🖼️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #4361ee, #7209b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .task-card {
        background: #f8f9ff;
        border: 1.5px solid #e0e4ff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        background: #4361ee;
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }
    .compare-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def super_resolution(img: Image.Image, scale: int = 2) -> Image.Image:
    """Upscale using LANCZOS resampling (high-quality, no model download needed)."""
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.LANCZOS)

def denoise(img: Image.Image, strength: int = 2) -> Image.Image:
    """Apply iterative median filter for denoising."""
    result = img
    for _ in range(strength):
        result = result.filter(ImageFilter.MedianFilter(size=3))
    return result

def sharpen(img: Image.Image, factor: float = 2.0) -> Image.Image:
    """Unsharp mask + sharpness enhancement."""
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

@st.cache_resource(show_spinner=False)
def load_colorizer():
    """Load colorization pipeline from HuggingFace (vittoriomazzi/colorization)."""
    return pipeline("image-to-image", model="vittoriomazzi/colorization")

def colorize(img: Image.Image) -> Image.Image:
    """Colorize a grayscale image using HuggingFace model."""
    gray = img.convert("L").convert("RGB")   # ensure 3-channel input
    colorizer = load_colorizer()
    result = colorizer(gray)
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, dict):
        result = result.get("image") or list(result.values())[0]
    return result if isinstance(result, Image.Image) else gray

def adjust_quality(img: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🖼️ AI Image Enhancer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Super Resolution · Denoising · Colorization · Sharpening — No API key needed</div>',
    unsafe_allow_html=True,
)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if not uploaded:
    st.info("⬆️ Upload an image to get started.")
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
**AI Image Enhancer** applies four enhancement techniques to your photos:

| Task | Method |
|---|---|
| 🔍 Super Resolution | LANCZOS upscaling (2× / 4×) |
| 🧹 Denoising | Median filter (iterative) |
| 🎨 Colorization | HuggingFace image-to-image model |
| ✨ Sharpening | Unsharp mask + detail boost |

All processing runs **on-device** — your images are never sent to any external server.
        """)
    st.stop()

original = Image.open(uploaded).convert("RGB")
w, h = original.size

st.sidebar.header("⚙️ Enhancement Settings")

# ── Task selector ─────────────────────────────────────────────────────────────
tasks = st.sidebar.multiselect(
    "Select enhancements to apply",
    ["🔍 Super Resolution", "🧹 Denoising", "🎨 Colorization", "✨ Sharpening"],
    default=["🧹 Denoising", "✨ Sharpening"],
)

# ── Per-task settings ─────────────────────────────────────────────────────────
sr_scale       = 2
denoise_str    = 2
sharpen_factor = 2.0

if "🔍 Super Resolution" in tasks:
    st.sidebar.markdown("**Super Resolution**")
    sr_scale = st.sidebar.radio("Upscale factor", [2, 4], horizontal=True)

if "🧹 Denoising" in tasks:
    st.sidebar.markdown("**Denoising**")
    denoise_str = st.sidebar.slider("Strength", 1, 5, 2)

if "✨ Sharpening" in tasks:
    st.sidebar.markdown("**Sharpening**")
    sharpen_factor = st.sidebar.slider("Sharpness factor", 1.0, 5.0, 2.0, 0.5)

st.sidebar.divider()
st.sidebar.markdown("**Fine-tune output**")
brightness  = st.sidebar.slider("Brightness",  0.5, 2.0, 1.0, 0.05)
contrast    = st.sidebar.slider("Contrast",    0.5, 2.0, 1.0, 0.05)
saturation  = st.sidebar.slider("Saturation",  0.0, 3.0, 1.0, 0.1)

st.sidebar.divider()
st.sidebar.caption(f"Original size: {w} × {h} px")

# ── Run button ────────────────────────────────────────────────────────────────
run = st.button("✨ Enhance Image", use_container_width=True, type="primary")

# ── Before/after preview ──────────────────────────────────────────────────────
col_orig, col_out = st.columns(2)
with col_orig:
    st.markdown('<div class="compare-label">Original</div>', unsafe_allow_html=True)
    st.image(original, use_container_width=True)
    st.caption(f"{w} × {h} px")

with col_out:
    st.markdown('<div class="compare-label">Enhanced</div>', unsafe_allow_html=True)
    result_placeholder = st.empty()
    caption_placeholder = st.empty()
    download_placeholder = st.empty()
    result_placeholder.image(original, use_container_width=True)
    caption_placeholder.caption("Enhanced image will appear here")

if run:
    if not tasks:
        st.warning("Select at least one enhancement from the sidebar.")
        st.stop()

    enhanced = original.copy()
    progress = st.progress(0, text="Starting…")
    step = 0
    total = len(tasks) + 1   # +1 for fine-tune

    for task in tasks:
        if task == "🔍 Super Resolution":
            progress.progress(step / total, text="Upscaling image…")
            enhanced = super_resolution(enhanced, sr_scale)
        elif task == "🧹 Denoising":
            progress.progress(step / total, text="Denoising…")
            enhanced = denoise(enhanced, denoise_str)
        elif task == "🎨 Colorization":
            progress.progress(step / total, text="Colorizing (loading model ~150 MB on first run)…")
            try:
                enhanced = colorize(enhanced)
            except Exception as e:
                st.warning(f"Colorization skipped: {e}")
        elif task == "✨ Sharpening":
            progress.progress(step / total, text="Sharpening…")
            enhanced = sharpen(enhanced, sharpen_factor)
        step += 1

    progress.progress(step / total, text="Applying fine-tune adjustments…")
    enhanced = adjust_quality(enhanced, brightness, contrast, saturation)
    progress.progress(1.0, text="Done!")
    progress.empty()

    ew, eh = enhanced.size
    with col_out:
        result_placeholder.image(enhanced, use_container_width=True)
        caption_placeholder.caption(f"{ew} × {eh} px")
        download_placeholder.download_button(
            label="⬇️ Download enhanced image",
            data=pil_to_bytes(enhanced),
            file_name="enhanced_" + uploaded.name.rsplit(".", 1)[0] + ".png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original size", f"{w} × {h}")
    m2.metric("Enhanced size", f"{ew} × {eh}")
    m3.metric("Tasks applied", len(tasks))
    scale_x = round(ew / w, 1)
    m4.metric("Scale factor", f"{scale_x}×")
