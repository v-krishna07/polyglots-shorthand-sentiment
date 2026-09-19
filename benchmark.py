"""
Benchmark and evaluation harness for the Hinglish sentiment models.

Run from the project root. Models are read from the local folders if present,
otherwise they are downloaded from the Hugging Face Hub.

    python3 benchmark.py                          # latency + probes, all variants
    python3 benchmark.py --data test.csv          # + accuracy / macro-F1 / error dump
    python3 benchmark.py --variants fast onnx     # subset of variants
    python3 benchmark.py --data test.csv --fertility   # + tokenizer fertility vs baselines

test.csv needs the columns:  text,label
(label = 0/1/2 or Negative/Neutral/Positive)

Outputs: benchmark_results.json, and errors_<variant>.csv when --data is given.
Every number in the whitepaper's TBD cells can be taken from these outputs.
"""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

try:
    ort.preload_dlls()  # helps CUDA on Windows, ignored elsewhere
except AttributeError:
    pass

HF_REPO = "v-krishna07/hinglish-models"
ROOT_DIR = Path(__file__).resolve().parent
MAX_LEN = 128
LABELS = ["Negative", "Neutral", "Positive"]

VARIANTS = {
    "fp16": {  # newer model, optimized fp16 ONNX
        "folder": "model_1.0/hinglish_onnx_fp16",
        "file": "model_optimized.onnx",
        "tokenizer_folder": "model_1.0/hinglish_onnx_fp16",
    },
    "onnx": {  # newer model, plain ONNX (borrows the fp16 folder's tokenizer)
        "folder": "model_1.0/hinglish_onnx_model",
        "file": "model.onnx",
        "tokenizer_folder": "model_1.0/hinglish_onnx_fp16",
    },
    "fast": {  # older model, latency-optimized
        "folder": "model_-1.0",
        "file": "model_optimized.onnx",
        "tokenizer_folder": "model_-1.0",
    },
}

BASELINE_TOKENIZERS = ["distilbert-base-multilingual-cased", "xlm-roberta-base"]

if "CUDAExecutionProvider" in ort.get_available_providers():
    PROVIDER, USE_IO_BINDING, DEVICE = "CUDAExecutionProvider", True, "cuda"
else:
    PROVIDER, USE_IO_BINDING, DEVICE = "CPUExecutionProvider", False, "cpu"

# --------------------------------------------------------------------------
# Probe sets (behaviour checks taken from the problem statement)
# --------------------------------------------------------------------------
PROBES = [
    "bhai order cancel krdo please, urgent meeting h",
    "item delivered bol rha h but mila hi nhi",
    "aap busy ho?",
    "Bohot badhiya service",
    "Bohot badhiya service 😒",
    "kya kr rhe ho",
]
VARIANT_GROUPS = [  # spelling-drift variants: predictions should agree
    ["kya kar rahe ho", "kya kr rhe ho", "kya krre ho"],
    ["bahut badhiya service", "bohot badhiya service", "bhot badhiya servis", "bht badhiya service"],
]
EMOJI_PAIRS = [  # same words, emoji should flip polarity
    ("Bohot badhiya service", "Bohot badhiya service 😒"),
    ("Mast kaam kiya team ne", "Mast kaam kiya team ne 🙄"),
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def resolve(folder, required_file, patterns):
    local = ROOT_DIR / folder
    if (local / required_file).exists():
        return local
    print(f"  '{folder}' not found locally, downloading from {HF_REPO} ...")
    snap = snapshot_download(repo_id=HF_REPO, allow_patterns=patterns)
    return Path(snap) / folder


def load_variant(variant):
    cfg = VARIANTS[variant]
    model_dir = resolve(cfg["folder"], cfg["file"], [f"{cfg['folder']}/*"])
    tok_dir = resolve(cfg["tokenizer_folder"], "tokenizer.json", [f"{cfg['tokenizer_folder']}/*.json"])
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), file_name=cfg["file"], provider=PROVIDER, use_io_binding=USE_IO_BINDING
    )
    return tok, model, model_dir / cfg["file"]


def count_params(onnx_path):
    m = onnx.load(str(onnx_path))
    return int(sum(int(np.prod(t.dims)) for t in m.graph.initializer))


# --------------------------------------------------------------------------
# Inference helpers (mirror the serving path in model_1.0/app.py)
# --------------------------------------------------------------------------
def clean(text):
    return text.replace("\uFE0F", "")


def encode(tok, texts):
    enc = tok(
        [clean(t) for t in texts],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length" if USE_IO_BINDING else True,
    )
    return {k: v.to(DEVICE) for k, v in enc.items()}


@torch.no_grad()
def predict_probs(tok, model, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        logits = model(**encode(tok, texts[i:i + batch_size])).logits
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


@torch.no_grad()
def bench_latency(tok, model, texts, batch_size, runs, warmup=20):
    batch = [texts[i % len(texts)] for i in range(batch_size)]
    for _ in range(warmup):
        model(**encode(tok, batch))
    tok_ms, model_ms = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        inputs = encode(tok, batch)
        t1 = time.perf_counter()
        logits = model(**inputs).logits
        torch.softmax(logits, dim=-1).cpu().numpy()  # .cpu() forces GPU sync
        t2 = time.perf_counter()
        tok_ms.append((t1 - t0) * 1000)
        model_ms.append((t2 - t1) * 1000)
    total = np.array(tok_ms) + np.array(model_ms)
    return {
        "batch_size": batch_size,
        "tokenize_ms_mean": round(float(np.mean(tok_ms)), 3),
        "model_ms_mean": round(float(np.mean(model_ms)), 3),
        "total_ms_p50": round(float(np.percentile(total, 50)), 3),
        "total_ms_p95": round(float(np.percentile(total, 95)), 3),
        "total_ms_p99": round(float(np.percentile(total, 99)), 3),
        "throughput_samples_per_s": round(batch_size / (float(np.mean(total)) / 1000), 1),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def label_to_id(v):
    v = str(v).strip()
    if v.isdigit():
        return int(v)
    return {"negative": 0, "neutral": 1, "positive": 2}[v.lower()]


def load_dataset(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "text" not in rows[0] or "label" not in rows[0]:
        raise SystemExit("CSV must have the columns: text,label")
    return [r["text"] for r in rows], [label_to_id(r["label"]) for r in rows]


def compute_metrics(y_true, y_pred, n=len(LABELS)):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "accuracy": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class_f1": {LABELS[i]: round(float(f), 4) for i, f in enumerate(f1s)},
        "confusion_matrix": cm.tolist(),
    }


def run_probes(tok, model):
    def pred(texts):
        p = predict_probs(tok, model, texts)
        return [(LABELS[int(i)], float(p[k][int(i)])) for k, i in enumerate(p.argmax(1))]

    probes = [{"text": t, "sentiment": s, "confidence": round(c, 4)} for t, (s, c) in zip(PROBES, pred(PROBES))]

    groups = []
    for g in VARIANT_GROUPS:
        preds = pred(g)
        groups.append({"variants": g, "predictions": [s for s, _ in preds], "consistent": len({s for s, _ in preds}) == 1})

    flips = []
    for base, emo in EMOJI_PAIRS:
        (b, _), (e, _) = pred([base, emo])
        flips.append({"base": base, "with_emoji": emo, "base_pred": b, "emoji_pred": e, "changed": b != e})
    return {"probes": probes, "variant_groups": groups, "emoji_pairs": flips}


def fertility(tok, texts):
    ids = tok([clean(t) for t in texts], add_special_tokens=False)["input_ids"]
    words = sum(len(t.split()) for t in texts)
    return round(sum(len(i) for i in ids) / max(words, 1), 3)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--data", help="CSV with columns text,label")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--fertility", action="store_true", help="compare tokens/word against baseline tokenizers")
    args = ap.parse_args()

    eval_texts, eval_labels = load_dataset(args.data) if args.data else (None, None)
    latency_texts = eval_texts[:500] if eval_texts else PROBES
    fert_texts = eval_texts if eval_texts else PROBES + [t for g in VARIANT_GROUPS for t in g]

    results = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "provider": PROVIDER,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "onnxruntime": ort.__version__,
            "torch": torch.__version__,
        },
        "variants": {},
    }
    print("Hardware:", json.dumps(results["hardware"], indent=2))

    for variant in args.variants:
        print(f"\n=== {variant} ===")
        tok, model, onnx_path = load_variant(variant)
        entry = {
            "onnx_file": str(onnx_path),
            "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 1),
            "params_millions": round(count_params(onnx_path) / 1e6, 1),
        }
        print(f"  size: {entry['onnx_size_mb']} MB | params: ~{entry['params_millions']} M")

        entry["latency"] = []
        for bs in args.batch_sizes:
            print(f"  latency, batch size {bs} ...")
            entry["latency"].append(bench_latency(tok, model, latency_texts, bs, args.runs))

        entry.update(run_probes(tok, model))

        if eval_texts:
            print(f"  evaluating on {len(eval_texts)} examples ...")
            probs = predict_probs(tok, model, eval_texts)
            preds = probs.argmax(1)
            entry["eval"] = compute_metrics(eval_labels, preds.tolist())
            with open(f"errors_{variant}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["text", "true", "pred", "confidence"])
                for t, y, p, pr in zip(eval_texts, eval_labels, preds, probs):
                    if y != p:
                        w.writerow([t, LABELS[y], LABELS[int(p)], round(float(pr[int(p)]), 4)])

        if args.fertility:
            entry["tokens_per_word"] = fertility(tok, fert_texts)

        results["variants"][variant] = entry
        del model

    if args.fertility:
        results["baseline_tokens_per_word"] = {}
        for name in BASELINE_TOKENIZERS:
            results["baseline_tokens_per_word"][name] = fertility(AutoTokenizer.from_pretrained(name), fert_texts)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- markdown tables, ready to paste into the whitepaper ----
    print("\n\n### Size and latency\n")
    print("| Variant | Params (M) | Size (MB) | Batch | p50 ms | p95 ms | p99 ms | Samples/s |")
    print("|---|---|---|---|---|---|---|---|")
    for v, e in results["variants"].items():
        for L in e["latency"]:
            print(f"| {v} | {e['params_millions']} | {e['onnx_size_mb']} | {L['batch_size']} | "
                  f"{L['total_ms_p50']} | {L['total_ms_p95']} | {L['total_ms_p99']} | {L['throughput_samples_per_s']} |")

    if eval_texts:
        print("\n### Quality\n")
        print("| Variant | Accuracy | Macro-F1 | F1 Neg | F1 Neu | F1 Pos |")
        print("|---|---|---|---|---|---|")
        for v, e in results["variants"].items():
            m = e["eval"]
            f = m["per_class_f1"]
            print(f"| {v} | {m['accuracy']} | {m['macro_f1']} | {f['Negative']} | {f['Neutral']} | {f['Positive']} |")

    if args.fertility:
        print("\n### Tokens per word (lower is better)\n")
        print("| Tokenizer | Tokens/word |")
        print("|---|---|")
        for v, e in results["variants"].items():
            print(f"| ours ({v}) | {e['tokens_per_word']} |")
        for n, val in results["baseline_tokens_per_word"].items():
            print(f"| {n} | {val} |")

    print("\nSaved benchmark_results.json")


if __name__ == "__main__":
    main()