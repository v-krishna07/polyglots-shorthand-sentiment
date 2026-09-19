
# Polyglot's Shorthand: Code-Mixed Sentiment Analysis

A sentiment classifier for **Romanized Hinglish** text (tweets, YouTube comments, casual chat) that stays under a **500M parameter** budget and targets **low-latency inference**. Text in this domain is noisy by nature — code-mixed spelling, dropped vowels, elongated words, and heavy emoji use — so most off-the-shelf tokenizers and sentiment models degrade quickly on it. This project builds a pipeline specifically tuned for that noise.

## Pipeline Overview

**1. Data fusion**
Two Hugging Face datasets are merged into a single 3-class (`negative` / `neutral` / `positive`) corpus:
- [`Abhishek4896/hindi-english-code-mixed-tweets-sentiment`](https://huggingface.co/datasets/Abhishek4896/hindi-english-code-mixed-tweets-sentiment) — Hinglish tweets
- [`shae2977/hinglish-youtube-sentiments-dataset`](https://huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset) — Hinglish YouTube comments

Both are normalized to a common `{text, label}` schema, concatenated, and shuffled (`seed=42`), giving ~3.7k labeled examples.

**2. Emoji-aware vocabulary**
Emojis are extracted from the corpus with the `emoji` library, normalized (skin-tone modifiers, variation selectors, and ZWJ sequences stripped so visually identical emojis collapse to one token), and the top-50 most frequent ones are added as explicit vocabulary tokens rather than being dropped or fragmented by the tokenizer.

**3. Base model**
[`l3cube-pune/hing-roberta`](https://huggingface.co/l3cube-pune/hing-roberta) (an XLM-RoBERTa model pretrained on Hindi-English code-mixed text) is used as the backbone. Its tokenizer is extended with the top emoji tokens and the model's embedding matrix is resized (250,006 × 768) so the new tokens get trained embeddings instead of falling back to `[UNK]`.

**4. Synthetic noise augmentation**
To make the model robust to how people actually type Hinglish, a custom augmenter randomly applies one of three corruptions to words in each training example:
- `drop_vowels` — mimics shorthand like "krdo" for "kardo"
- `elongate` — mimics emphasis like "sooo good"
- `qwerty_swap` — mimics fat-finger typos using a QWERTY adjacency map

This roughly doubles the training set (3,319 → 6,638 rows) with a noisy variant of each example.

**5. Training**
A custom PyTorch loop (not `Trainer`) handles:
- Class-weighted `CrossEntropyLoss` (to offset class imbalance) with label smoothing (0.1)
- `AdamW` with weight decay excluded on biases/LayerNorm
- Cosine LR schedule with 10% warmup
- Mixed-precision training (`autocast` + `GradScaler`) with gradient clipping
- Early stopping on validation accuracy (patience = 5)

**6. Semi-supervised pseudo-labeling (Phase 2)**
The best Phase-1 checkpoint is used to label an additional unlabeled dataset ([`Yugrathee28/Hinglish-dataset`](https://huggingface.co/datasets/Yugrathee28/Hinglish-dataset)). Only predictions with **>95% softmax confidence** are kept and folded back into the training set for a second training pass, growing the effective training data from 6,638 → 7,109 rows.

**7. Deployment optimization**
The final PyTorch model is exported to ONNX and optimized with `optimum.onnxruntime`:
- Operator fusion at optimization level **O4**
- **FP16** weight quantization
- Fixed 128-token padding so the CUDA execution provider can reuse memory buffers via IO binding

**8. Benchmarking & sanity inference**
Latency is measured over hundreds of warmed-up passes on GPU (CUDA/TensorRT execution providers), and the final ONNX engine is sanity-checked against real-world Hinglish sentences with slang and emojis.

## Results

| Metric | Phase 1 (augmented data only) | Phase 2 (+ pseudo-labeled data) |
|---|---|---|
| Best validation accuracy | 79.95% | **81.30%** |
| Best validation macro-F1 | 0.7947 | **0.8076** |

| Latency (FP16 ONNX, GPU, batch=1, 128 tokens) | Value |
|---|---|
| Average | ~3.3 ms |
| P50 | ~2.6 ms |
| P75 | ~4.0 ms |
| P99 | ~8.2 ms |

Sample predictions from the optimized engine:

| Input | Prediction |
|---|---|
| "bhai sach bata raha hu, product ekdum fire hai 🔥 maza aa gaya!" | Positive (93.8%) |
| "Paisa barbaad ho gaya yaar, bilkul ghatiya customer support hai 😡 scam pura" | Negative (91.3%) |
| "Kal match kitne baje start hoga? Mujhe time verify karna tha bas." | Neutral (96.6%) |
| "Bhai kya solid update diya hai team ne, dil jeet liya ❤️" | Positive (94.2%) |

## Repository Structure

```
.
├── inter_iit_csai_ps_4.ipynb   # End-to-end pipeline: data → training → ONNX export → benchmarking
└── README.md
```

## Running It

The whole pipeline lives in a single Colab-first notebook.

1. Open the notebook via the **Open in Colab** badge above (or run it locally with a Jupyter/JupyterLab install).
2. Use a **GPU runtime** — training uses CUDA + AMP, and the export/benchmark cells require a CUDA execution provider.
3. Run cells top to bottom. Key dependencies pulled in along the way:
   ```
   pip install datasets transformers emoji scikit-learn tqdm optimum[onnxruntime-gpu]
   ```
4. Training checkpoints save to `./hinglish_model_checkpoint`; the optimized inference engine saves to `./hinglish_onnx_fp16`.

## Status

- [x] Data prep, cleaning, and emoji vocabulary extraction
- [x] Tokenizer/model setup (HingRoBERTa + resized embeddings)
- [x] Phonetic/typo augmentation for robustness
- [x] Phase 1 supervised training
- [x] Phase 2 semi-supervised training with pseudo-labels
- [x] ONNX export with FP16 quantization
- [x] Latency benchmarking and real-world inference sanity checks
- [ ] Formal test-set evaluation / held-out leaderboard submission
- [ ] Package the ONNX engine behind a lightweight inference API
