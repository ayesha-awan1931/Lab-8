import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import io

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
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .accent { color: #4361ee; }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .compare-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Enhancement functions (pure Pillow + NumPy, no transformers) ──────────────

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def super_resolution(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.LANCZOS)

def denoise(img: Image.Image, strength: int) -> Image.Image:
    result = img
    for _ in range(strength):
        result = result.filter(ImageFilter.MedianFilter(size=3))
    return result

def sharpen(img: Image.Image, factor: float) -> Image.Image:
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return ImageEnhance.Sharpness(img).enhance(factor)

def colorize(img: Image.Image) -> Image.Image:
    """
    Pseudo-colorization using numpy:
    Converts greyscale luminance to a warm sepia-toned colour image,
    then blends with the original to preserve any existing colour.
    Works entirely offline — no model download required.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # Warm tone map: shadows → cool blue-purple, highlights → warm amber
    r = np.clip(gray * 1.10, 0, 255)
    g = np.clip(gray * 0.95, 0, 255)
    b = np.clip(gray * 0.75 + 30, 0, 255)

    colorized = np.stack([r, g, b], axis=2).astype(np.uint8)
    # Blend 60% colorized + 40% original to keep natural tones
    blended = (0.6 * colorized + 0.4 * arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def fine_tune(img: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return img

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🖼️ AI Image Enhancer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Super Resolution · Denoising · Colorization · Sharpening — 100% offline, no API key</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if not uploaded:
    st.info("⬆️ Upload an image to get started.")
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
**AI Image Enhancer** applies four enhancement techniques:

| Task | Method |
|---|---|
| 🔍 Super Resolution | LANCZOS upscaling (2× / 4×) |
| 🧹 Denoising | Median filter |
| 🎨 Colorization | Numpy tone mapping |
| ✨ Sharpening | Unsharp mask |

100% offline — no model downloads, no API calls.
        """)
    st.stop()

original = Image.open(uploaded).convert("RGB")
w, h = original.size

# ── Sidebar settings ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Enhancement Settings")

tasks = st.sidebar.multiselect(
    "Select enhancements to apply",
    ["🔍 Super Resolution", "🧹 Denoising", "🎨 Colorization", "✨ Sharpening"],
    default=["🧹 Denoising", "✨ Sharpening"],
)

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
brightness = st.sidebar.slider("Brightness",  0.5, 2.0, 1.0, 0.05)
contrast   = st.sidebar.slider("Contrast",    0.5, 2.0, 1.0, 0.05)
saturation = st.sidebar.slider("Saturation",  0.0, 3.0, 1.0, 0.1)

st.sidebar.divider()
st.sidebar.caption(f"Original: {w} × {h} px")

# ── Before / After ────────────────────────────────────────────────────────────
col_orig, col_out = st.columns(2)
with col_orig:
    st.markdown('<div class="compare-label">Original</div>', unsafe_allow_html=True)
    st.image(original, use_container_width=True)
    st.caption(f"{w} × {h} px")

with col_out:
    st.markdown('<div class="compare-label">Enhanced</div>', unsafe_allow_html=True)
    result_slot  = st.empty()
    caption_slot = st.empty()
    dl_slot      = st.empty()
    result_slot.image(original, use_container_width=True)
    caption_slot.caption("Enhanced image will appear here after clicking Enhance.")

run = st.button("✨ Enhance Image", use_container_width=True, type="primary")

if run:
    if not tasks:
        st.warning("Select at least one enhancement from the sidebar.")
        st.stop()

    enhanced = original.copy()
    total    = len(tasks) + 1
    progress = st.progress(0, text="Starting…")

    for i, task in enumerate(tasks):
        if task == "🔍 Super Resolution":
            progress.progress(i / total, text="Upscaling…")
            enhanced = super_resolution(enhanced, sr_scale)
        elif task == "🧹 Denoising":
            progress.progress(i / total, text="Denoising…")
            enhanced = denoise(enhanced, denoise_str)
        elif task == "🎨 Colorization":
            progress.progress(i / total, text="Colorizing…")
            enhanced = colorize(enhanced)
        elif task == "✨ Sharpening":
            progress.progress(i / total, text="Sharpening…")
            enhanced = sharpen(enhanced, sharpen_factor)

    progress.progress((total - 1) / total, text="Fine-tuning…")
    enhanced = fine_tune(enhanced, brightness, contrast, saturation)
    progress.progress(1.0, text="Done!")
    progress.empty()

    ew, eh = enhanced.size
    with col_out:
        result_slot.image(enhanced, use_container_width=True)
        caption_slot.caption(f"{ew} × {eh} px")
        dl_slot.download_button(
            label="⬇️ Download enhanced image",
            data=pil_to_bytes(enhanced),
            file_name="enhanced_" + uploaded.name.rsplit(".", 1)[0] + ".png",
            mime="image/png",
            use_container_width=True,
        )

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original size", f"{w} × {h}")
    c2.metric("Enhanced size", f"{ew} × {eh}")
    c3.metric("Tasks applied", len(tasks))
    c4.metric("Scale", f"{round(ew/w, 1)}×")
