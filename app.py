"""
Hinglish Sentiment Engine: deployable Streamlit app.
"""
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

HF_REPO = "v-krishna07/hinglish-models"
GITHUB_URL = "https://github.com/v-krishna07/polyglots-shorthand-sentiment"
ROOT_DIR = Path(__file__).resolve().parent
MAX_LEN = 128
LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
ICONS = {"Positive": "✅", "Negative": "🚩", "Neutral": "⚖️"}

# Lock deployment to the most stable, performant FP16 ONNX model
MODEL_FOLDER = "model_1.0/hinglish_onnx_fp16"
MODEL_FILE = "model_optimized.onnx"
TOKENIZER_FOLDER = "model_1.0/hinglish_onnx_fp16"

EXAMPLES = {
    "Order cancel": "bhai order cancel krdo please, urgent meeting h",
    "Praise": "Bohot badhiya service",
    "Sarcasm (emoji)": "Bohot badhiya service 😒",
    "Not delivered": "item delivered bol rha h but mila hi nhi",
}

# Emoji / pictograph ranges
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]")

# --------------------------------------------------------------------------
# Model loading (cached: one copy shared by all visitors)
# --------------------------------------------------------------------------
def _resolve(folder, required_file, patterns):
    local = ROOT_DIR / folder
    if (local / required_file).exists():
        return local
    snapshot = snapshot_download(repo_id=HF_REPO, allow_patterns=patterns)
    return Path(snapshot) / folder

@st.cache_resource(show_spinner=False)
def load_engine():
    model_dir = _resolve(MODEL_FOLDER, MODEL_FILE, [f"{MODEL_FOLDER}/*"])
    tok_dir = _resolve(TOKENIZER_FOLDER, "tokenizer.json", [f"{TOKENIZER_FOLDER}/*.json"])
    
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), fix_mistral_regex=True)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    
    # Hardware-Agnostic Setup
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    
    session = ort.InferenceSession(str(model_dir / MODEL_FILE), sess_options=opts, providers=providers)
    return tokenizer, session

def predict(tokenizer, session, texts):
    """Return an (n, 3) array of class probabilities."""
    cleaned = [t.replace("\uFE0F", "") for t in texts]
    enc = tokenizer(cleaned, return_tensors="np", truncation=True, max_length=MAX_LEN, padding=True)
    wanted = {i.name for i in session.get_inputs()}
    feed = {k: v.astype(np.int64) for k, v in enc.items() if k in wanted}
    logits = session.run(None, feed)[0].astype(np.float32)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# --------------------------------------------------------------------------
# UI Setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Hinglish Sentiment AI", page_icon="🔥", layout="centered")

st.title("🔥 Hinglish Sentiment Engine")
st.markdown(
    "Sentiment analysis for **Romanized, code-mixed Hindi-English** with typos, shorthand and emojis. "
    "Emojis are treated as signal, not noise."
)

with st.sidebar:
    st.header("Deployment Details")
    st.markdown("**Active Engine:** FP16 ONNX Optimized")
    st.markdown("**Latency Profile:** Sub-3ms (GPU) / ~30ms (CPU)")
    st.caption("Running on Streamlit Community Cloud (Edge CPU Fallback)")
    st.markdown("---")
    st.markdown(f"[Code on GitHub]({GITHUB_URL})  \n[Models on Hugging Face](https://huggingface.co/{HF_REPO})")

try:
    with st.spinner("Initializing ONNX Engine (First run downloads weights)..."):
        tokenizer, session = load_engine()
except Exception as exc:
    st.error("Engine failed to initialize. Host memory limit exceeded.")
    st.exception(exc)
    st.stop()

provider = session.get_providers()[0].replace("ExecutionProvider", "")
tab_single, tab_batch = st.tabs(["Single message", "Batch"])

# ---- single message ----
with tab_single:
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for i, (col, (label, sample)) in enumerate(zip(cols, EXAMPLES.items())):
        col.button(
            label,
            key=f"example_{i}",
            on_click=lambda s=sample: st.session_state.update(single_text=s),
        )

    text = st.text_area(
        "Type your message here:",
        key="single_text",
        height=110,
        placeholder="Bhai kya solid update diya hai team ne, dil jeet liya ❤️",
    )

    if st.button("Analyze sentiment", type="primary", key="analyze"):
        if not text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            t0 = time.perf_counter()
            probs = predict(tokenizer, session, [text])[0]
            latency_ms = (time.perf_counter() - t0) * 1000
            label = LABELS[int(probs.argmax())]

            st.markdown(f"### {ICONS[label]} :{COLORS[label]}[{label}]")
            st.progress(float(probs.max()), text=f"Confidence: {probs.max() * 100:.1f}%")
            st.bar_chart(pd.DataFrame({"probability": probs}, index=LABELS))
            st.caption(f"⚡ {latency_ms:.1f} ms · {provider} · includes tokenization and inference")

            stripped = EMOJI_RE.sub("", text).strip()
            if stripped and stripped != text.strip():
                p2 = predict(tokenizer, session, [stripped])[0]
                l2 = LABELS[int(p2.argmax())]
                if l2 != label:
                    st.info(f"🔄 **The emoji changed the result.** Without it, this reads as "
                            f"**{l2}** ({p2.max() * 100:.1f}%).")
                else:
                    st.caption(f"Without the emoji the result is the same: {l2} ({p2.max() * 100:.1f}%).")

# ---- batch ----
with tab_batch:
    batch_text = st.text_area(
        "One message per line (up to 200 lines):",
        key="batch_text",
        height=170,
        placeholder="kya kr rhe ho\nBohot badhiya service 😒\norder abhi tak nhi aaya",
    )
    if st.button("Analyze all", key="analyze_batch"):
        lines = [ln.strip() for ln in batch_text.splitlines() if ln.strip()][:200]
        if not lines:
            st.warning("Please enter at least one message.")
        else:
            t0 = time.perf_counter()
            chunks = [predict(tokenizer, session, lines[i:i + 32]) for i in range(0, len(lines), 32)]
            probs = np.concatenate(chunks)
            total_ms = (time.perf_counter() - t0) * 1000
            df = pd.DataFrame(
                {
                    "text": lines,
                    "sentiment": [LABELS[int(i)] for i in probs.argmax(1)],
                    "confidence": np.round(probs.max(1), 3),
                }
            )
            st.dataframe(df)
            st.caption(f"⚡ {len(lines)} messages in {total_ms:.0f} ms "
                       f"({total_ms / len(lines):.1f} ms each) · {provider}")
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")