import streamlit as st
from transformers import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Language Translator",
    page_icon="🌐",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.1rem;
        color: #1a1a2e;
        margin-top: 1rem;
        line-height: 1.7;
    }
    .model-tag {
        font-size: 0.75rem;
        color: #999;
        text-align: right;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Language pairs (Helsinki-NLP opus-mt) ─────────────────────────────────────
LANGUAGE_PAIRS = {
    "English → French":     "Helsinki-NLP/opus-mt-en-fr",
    "English → Spanish":    "Helsinki-NLP/opus-mt-en-es",
    "English → German":     "Helsinki-NLP/opus-mt-en-de",
    "English → Italian":    "Helsinki-NLP/opus-mt-en-it",
    "English → Portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "English → Dutch":      "Helsinki-NLP/opus-mt-en-nl",
    "English → Russian":    "Helsinki-NLP/opus-mt-en-ru",
    "English → Arabic":     "Helsinki-NLP/opus-mt-en-ar",
    "English → Chinese":    "Helsinki-NLP/opus-mt-en-zh",
    "English → Hindi":      "Helsinki-NLP/opus-mt-en-hi",
    "English → Urdu":       "Helsinki-NLP/opus-mt-en-ur",
    "English → Turkish":    "Helsinki-NLP/opus-mt-en-tr",
    "English → Polish":     "Helsinki-NLP/opus-mt-en-pl",
    "French → English":     "Helsinki-NLP/opus-mt-fr-en",
    "Spanish → English":    "Helsinki-NLP/opus-mt-es-en",
    "German → English":     "Helsinki-NLP/opus-mt-de-en",
    "Italian → English":    "Helsinki-NLP/opus-mt-it-en",
    "Portuguese → English": "Helsinki-NLP/opus-mt-pt-en",
    "Dutch → English":      "Helsinki-NLP/opus-mt-nl-en",
    "Russian → English":    "Helsinki-NLP/opus-mt-ru-en",
    "Arabic → English":     "Helsinki-NLP/opus-mt-ar-en",
    "Chinese → English":    "Helsinki-NLP/opus-mt-zh-en",
    "Hindi → English":      "Helsinki-NLP/opus-mt-hi-en",
    "Urdu → English":       "Helsinki-NLP/opus-mt-ur-en",
    "Turkish → English":    "Helsinki-NLP/opus-mt-tr-en",
}

@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    """Load translation pipeline once and cache it."""
    return pipeline("translation", model=model_name)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌐 Language Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Helsinki-NLP · Opus-MT · HuggingFace · No API key needed</div>',
    unsafe_allow_html=True,
)

pair_choice = st.selectbox("Select translation direction", list(LANGUAGE_PAIRS.keys()))
model_name  = LANGUAGE_PAIRS[pair_choice]

source_text = st.text_area(
    "Enter text to translate",
    placeholder="Type or paste your text here…",
    height=160,
    max_chars=1000,
)
st.caption(f"{len(source_text)} / 1000 characters")

if st.button("Translate ✨", use_container_width=True, type="primary"):
    if not source_text.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner(f"Loading `{model_name}` and translating…"):
            try:
                translator = load_pipeline(model_name)
                output = translator(
                    source_text.strip(),
                    max_length=512,
                    clean_up_tokenization_spaces=True,
                )
                result = output[0]["translation_text"]
                st.markdown(
                    f'<div class="result-box">{result}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="model-tag">Model: <code>{model_name}</code></div>',
                    unsafe_allow_html=True,
                )
                st.code(result, language=None)   # easy copy
            except Exception as e:
                st.error(f"Translation error: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
Translate between **25 language pairs** using free
[Helsinki-NLP](https://huggingface.co/Helsinki-NLP) Opus-MT models
from HuggingFace — no API key, no cost.

**Stack**
- 🤗 `transformers` pipeline API
- 🚀 Streamlit UI
- ☁️ Streamlit Cloud deployment

**Tips**
- First run downloads the model (~300 MB); subsequent runs use the cache.
- Keep text under 1000 characters for best speed.
    """)
    st.divider()
    st.caption(f"Active model: `{model_name}`")
