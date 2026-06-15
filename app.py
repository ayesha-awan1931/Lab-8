import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Language Translator",
    page_icon="🌐",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
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
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.05rem;
        color: #1a1a2e;
        margin-top: 1rem;
    }
    .model-badge {
        font-size: 0.75rem;
        color: #888;
        text-align: right;
        margin-top: 0.4rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Language pairs supported by Helsinki-NLP/opus-mt-* ───────────────────────
LANGUAGE_PAIRS = {
    "English → French":       ("en", "fr"),
    "English → Spanish":      ("en", "es"),
    "English → German":       ("en", "de"),
    "English → Italian":      ("en", "it"),
    "English → Portuguese":   ("en", "pt"),
    "English → Dutch":        ("en", "nl"),
    "English → Russian":      ("en", "ru"),
    "English → Arabic":       ("en", "ar"),
    "English → Chinese":      ("en", "zh"),
    "English → Japanese":     ("en", "jap"),
    "English → Hindi":        ("en", "hi"),
    "English → Urdu":         ("en", "ur"),
    "English → Turkish":      ("en", "tr"),
    "English → Polish":       ("en", "pl"),
    "English → Korean":       ("en", "ko"),
    "French → English":       ("fr", "en"),
    "Spanish → English":      ("es", "en"),
    "German → English":       ("de", "en"),
    "Italian → English":      ("it", "en"),
    "Portuguese → English":   ("pt", "en"),
    "Dutch → English":        ("nl", "en"),
    "Russian → English":      ("ru", "en"),
    "Arabic → English":       ("ar", "en"),
    "Chinese → English":      ("zh", "en"),
    "Hindi → English":        ("hi", "en"),
    "Urdu → English":         ("ur", "en"),
    "Turkish → English":      ("tr", "en"),
}

def get_model_name(src: str, tgt: str) -> str:
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load tokenizer + TF model; cached so it only runs once per pair."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate(text: str, tokenizer, model) -> str:
    inputs = tokenizer(
        text,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=512,
    )
    translated = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌐 Language Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by Helsinki-NLP / Opus-MT models · TensorFlow · HuggingFace</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1])
with col1:
    pair_choice = st.selectbox("Translation direction", list(LANGUAGE_PAIRS.keys()))
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    beam_size = st.selectbox("Beam size", [1, 2, 4, 8], index=2)

src_code, tgt_code = LANGUAGE_PAIRS[pair_choice]
model_name = get_model_name(src_code, tgt_code)

source_text = st.text_area(
    "Source text",
    placeholder="Type or paste your text here…",
    height=160,
    max_chars=1000,
)

char_count = len(source_text)
st.caption(f"{char_count} / 1000 characters")

translate_clicked = st.button("Translate ✨", use_container_width=True, type="primary")

if translate_clicked:
    if not source_text.strip():
        st.warning("Please enter some text to translate.")
    else:
        with st.spinner(f"Loading model `{model_name}` and translating…"):
            try:
                tokenizer, model = load_model(model_name)
                result = translate(source_text.strip(), tokenizer, model)
                st.markdown(
                    f'<div class="result-box">{result}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="model-badge">Model: <code>{model_name}</code></div>',
                    unsafe_allow_html=True,
                )
                # Copy helper
                st.code(result, language=None)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.info(
                    "Some language pairs (e.g. English → Japanese/Korean) may need "
                    "slightly different model slugs. Check "
                    "https://huggingface.co/Helsinki-NLP for the exact repo name."
                )

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
This app uses **free, open-source** machine-translation models from
[Helsinki-NLP](https://huggingface.co/Helsinki-NLP) on HuggingFace —
no API key required.

**Stack**
- 🤗 `transformers` (HuggingFace)
- 🧠 TensorFlow backend
- 🚀 Streamlit UI

**How it works**
1. Select a language pair.
2. Paste your text.
3. Click **Translate** — the model is downloaded once and cached locally.

**Beam size**
Higher beam sizes give better quality but are slower.
`4` is a good default.
    """)
    st.divider()
    st.caption("Models are cached after first download — subsequent translations are fast.")
