import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydantic import BaseModel
from transformers import AutoTokenizer

# Force CUDA dlls if on Windows/NVIDIA, ignore otherwise
try:
    ort.preload_dlls()
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Model location: use local folders if present, otherwise download from the Hub
# ---------------------------------------------------------------------------
HF_REPO = "v-krishna07/hinglish-models"

# This file lives in model_1.0/, the repo/project root is one level up
ROOT_DIR = Path(__file__).resolve().parent.parent

# Choose with:  MODEL_VARIANT=fast uvicorn app:app --port 8000
VARIANTS = {
    # newer model, optimized fp16 ONNX (default)
    "fp16": {
        "folder": "model_1.0/hinglish_onnx_fp16",
        "file": "model_optimized.onnx",
        "tokenizer_folder": "model_1.0/hinglish_onnx_fp16",
    },
    # newer model, plain ONNX (has no tokenizer files, so borrow the fp16 one)
    "onnx": {
        "folder": "model_1.0/hinglish_onnx_model",
        "file": "model.onnx",
        "tokenizer_folder": "model_1.0/hinglish_onnx_fp16",
    },
    # older model, faster
    "fast": {
        "folder": "model_-1.0",
        "file": "model_optimized.onnx",
        "tokenizer_folder": "model_-1.0",
    },
}

VARIANT = os.environ.get("MODEL_VARIANT", "fp16")
if VARIANT not in VARIANTS:
    raise ValueError(f"MODEL_VARIANT must be one of {list(VARIANTS)}, got '{VARIANT}'")
CFG = VARIANTS[VARIANT]


def resolve(folder: str, required_file: str, patterns: list[str]) -> str:
    """Return a local path for `folder`. Download from the Hub if it isn't on disk."""
    local = ROOT_DIR / folder
    if (local / required_file).exists():
        print(f"Using local files: {local}")
        return str(local)

    print(f"'{folder}' not found locally, downloading from {HF_REPO} ...")
    snapshot = snapshot_download(repo_id=HF_REPO, allow_patterns=patterns)
    return str(Path(snapshot) / folder)


model_dir = resolve(CFG["folder"], CFG["file"], [f"{CFG['folder']}/*"])
tokenizer_dir = resolve(
    CFG["tokenizer_folder"], "tokenizer.json", [f"{CFG['tokenizer_folder']}/*.json"]
)

# ---------------------------------------------------------------------------
app = FastAPI(title="Hinglish Sentiment API", description="Sub-3ms ONNX Inference")

# Hardware detection (falls back to CPU on Mac, uses CUDA on Colab/Production)
available_providers = ort.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    PROVIDER = "CUDAExecutionProvider"
    USE_IO_BINDING = True
    DEVICE = "cuda"
else:
    PROVIDER = "CPUExecutionProvider"
    USE_IO_BINDING = False
    DEVICE = "cpu"

print(f"Booting Engine with: {PROVIDER} | variant: {VARIANT}")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = ORTModelForSequenceClassification.from_pretrained(
    model_dir,
    file_name=CFG["file"],
    provider=PROVIDER,
    use_io_binding=USE_IO_BINDING,
)
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


class ChatRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict_sentiment(request: ChatRequest):
    start = time.perf_counter()

    clean_text = request.text.replace("\uFE0F", "")

    if USE_IO_BINDING:
        # Strict padding for GPU IO-Binding
        inputs = tokenizer(
            clean_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128
        )
    else:
        # Dynamic padding for CPU efficiency (MacBook)
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128)

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))

    latency = (time.perf_counter() - start) * 1000

    return {
        "text": request.text,
        "sentiment": LABEL_MAP[pred_idx],
        "confidence": float(probs[pred_idx]),
        "latency_ms": round(latency, 2),
        "device_used": DEVICE,
        "model_variant": VARIANT,
    }