# Polyglot's Shorthand — Hinglish Sentiment Engine

> **Emoji-aware sentiment analysis for Romanized, code-mixed Hinglish, built for Inter IIT CSAI Problem Statement 4.**

**Author:** Krishna · Solo Project  
**Problem Statement:** 4 — *The Polyglot's Shorthand*  
**Status:** Final competition implementation

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://polyglots-shorthand-sentiment.streamlit.app)
[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](https://huggingface.co/v-krishna07/hinglish-models)
[![Runtime](https://img.shields.io/badge/Runtime-ONNX%20Runtime-informational)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## Live Demo

**Try the deployed model:**  
https://polyglots-shorthand-sentiment.streamlit.app

The application exposes the final sentiment engine through a Streamlit interface and runs the deployed inference stack on CPU. The observed deployed latency is approximately **15 ms per request**, while the optimized FP16 ONNX model was benchmarked at **2.58 ms p50** on an NVIDIA T4 under the project's GPU benchmark setup.

---

## 1. What problem does this solve?

A large amount of everyday Indian online communication is not written in formal English or native Hindi script. It is often:

- Romanized Hindi
- English-Hindi code-mixing
- heavily abbreviated
- phonetically spelled
- inconsistent across users
- rich in emojis

For example:

```text
bhai order cancel krdo please
```

and

```text
kya kr rhe ho
```

can be perfectly understandable to a human while creating difficult tokenization and representation problems for a conventional NLP pipeline.

The goal of this project is therefore not simply to classify clean Hindi or English text. It is to build a **fast sentiment engine specifically adapted to the shorthand style of Romanized Hinglish**.

The model predicts three classes:

| Label | Meaning |
|---|---|
| Negative | negative sentiment |
| Neutral | neutral / non-polar sentiment |
| Positive | positive sentiment |

---

## 2. Key idea

The system combines four ideas:

1. **Hinglish-specific pretrained representation**
2. **Explicit emoji support**
3. **Robustness to spelling and phonetic variation**
4. **Pseudo-labeling to expand the effective training set**

The final model is then exported to **ONNX Runtime** for low-latency inference.

### High-level pipeline

```text
Romanized Hinglish + emojis
            │
            ▼
   Unicode / text normalization
            │
            ▼
 Existing Hing-RoBERTa tokenizer
       + 50 frequent emojis
            │
            ▼
      Hing-RoBERTa encoder
            │
            ▼
   3-class classification head
   Negative / Neutral / Positive
            │
            ▼
      ONNX Runtime inference
            │
       ┌────┴────┐
       ▼         ▼
      CPU      CUDA GPU
```

---

## 3. Why emojis are treated as signal

Consider:

```text
Bohot badhiya service
```

versus:

```text
Bohot badhiya service 😒
```

The words are almost identical, but the pragmatic meaning can change substantially.

Instead of deleting emojis as noise, the training pipeline identifies the **50 most frequent emojis in the training corpus** and adds them to the existing tokenizer vocabulary.

The tokenizer is not trained from scratch. The project starts from the tokenizer supplied with:

```text
l3cube-pune/hing-roberta
```

and extends that vocabulary with the frequent emojis.

The embedding matrix is resized accordingly.

---

## 4. Data and training

### Labeled data

The initial supervised dataset is formed by combining:

- Hindi-English code-mixed Twitter sentiment data
- Hinglish YouTube sentiment data

The project started with approximately **6.5k labeled examples**.

After the second training phase and pseudo-labeling process, the effective training pool reached approximately **7.2k examples** according to the final project run.

### Train/validation split

The notebook uses a **stratified 90/10 split** with a fixed random seed of 42.

### Text normalization

The preprocessing step removes Unicode variation selectors such as `U+FE0F` so visually equivalent emoji representations are handled consistently.

The project deliberately preserves the semantic emoji itself.

---

## 5. Phase 1 — robustness first

The first training phase is designed to make the model less sensitive to the spelling variation common in Romanized Hinglish.

The augmentation pipeline introduces realistic variations such as:

- vowel dropping
- shorthand spellings
- character-level / QWERTY-style noise

This is important because:

```text
kya kar rahe ho
kya kr rhe ho
kya krre ho
```

can represent the same underlying expression.

The objective is to prevent the model from learning that every spelling variant is a completely unrelated phrase.

### Training configuration

The implementation in `inter_iit_csai_ps_4.ipynb` uses:

| Parameter | Value |
|---|---:|
| Backbone | `l3cube-pune/hing-roberta` |
| Number of classes | 3 |
| Maximum sequence length | 128 |
| Training batch size | 32 |
| Validation batch size | 32 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | 0.01 for decayed parameters |
| Label smoothing | 0.1 |
| Scheduler | Cosine schedule with 10% warmup |
| Gradient clipping | 1.0 |
| Early stopping patience | 5 epochs |
| Phase training budget | up to 20 epochs |

The notebook is the authoritative source for the implementation configuration.

---

## 6. Phase 2 — confidence-filtered pseudo-labeling

The project then uses the trained Phase 1 model on an additional unlabeled Hinglish corpus:

```text
Yugrathee28/Hinglish-dataset
```

For each example, the model produces a probability distribution over the three sentiment classes.

Only predictions satisfying:

```text
max(class probability) > 0.95
```

are accepted as pseudo-labels.

Those high-confidence examples are then combined with the original augmented training set and used for the second training phase.

This is best described as **confidence-filtered self-training / pseudo-labeling**, rather than claiming a separate teacher and student architecture.

### Why this helps

The labeled dataset is relatively small. Pseudo-labeling allows the model to expose itself to additional linguistic patterns while applying a conservative confidence threshold to reduce the amount of noisy synthetic supervision.

---

## 7. Model architecture

The final classifier is based on `l3cube-pune/hing-roberta`.

The model is configured for three-way classification:

```text
Input text
   ↓
Hing-RoBERTa tokenizer
   ↓
Embedding layer
   ├── original vocabulary
   └── + 50 frequent emoji tokens
   ↓
Transformer encoder
   ↓
Sequence classification head
   ↓
Negative / Neutral / Positive
```

The final model contains approximately **278 million parameters**, keeping it below the project's 500M-parameter constraint.

---

## 8. ONNX optimization and deployment

After training, the PyTorch checkpoint is exported to ONNX.

The project creates:

- a reference ONNX model
- an optimized FP16 ONNX model

The optimization pipeline uses Hugging Face Optimum's ONNX Runtime tooling and `O4` optimization.

The production GPU path supports CUDA execution and IO binding.

### Deployment stack

```text
Streamlit UI
     │
     ▼
FastAPI inference layer
     │
     ▼
Tokenizer
     │
     ▼
ONNX Runtime
     │
 ┌───┴────┐
 ▼        ▼
CPU     CUDA
```

The public Streamlit deployment currently operates on CPU, where the observed latency is approximately **15 ms/request**.

The benchmarked GPU configuration is substantially faster.

---

## 9. Final results

### Model quality

| Metric | Final result |
|---|---:|
| Parameters | ~278M |
| Validation accuracy | >92% |
| Macro-F1 | ~0.84 |

### GPU latency

The final optimized FP16 ONNX model was benchmarked on an NVIDIA Tesla T4.

| Metric | Result |
|---|---:|
| p50 latency | **2.58 ms** |
| Average latency | **3.32 ms** |
| p75 latency | **3.95 ms** |
| p99 latency | **8.24 ms** |

The project's latency measurement includes the inference path used by the benchmark, including tokenization, model execution, and probability generation as implemented by the benchmark.

### Deployed CPU latency

The public Streamlit deployment currently runs on CPU and has been observed at approximately:

**~15 ms/request**

Latency is hardware- and environment-dependent, so these values should not be treated as universal guarantees.

---

## 10. Robustness probes

The project includes probes for the types of variation that motivated the architecture.

### Phonetic variation

```text
kya kar rahe ho
kya kr rhe ho
kya krre ho
```

### Shorthand variation

```text
bahut badhiya service
bht badhiya servis
```

### Emoji context

```text
Bohot badhiya service
Bohot badhiya service 😒
```

### Code-mixed input

```text
aap busy ho?
```

These are **robustness probes**, not a substitute for a separately curated statistical robustness benchmark.

---

## 11. Repository structure

```text
.
├── README.md
├── WHITEPAPER.md
├── requirements.txt
├── benchmark.py
├── upload_models.py
├── inter_iit_csai_ps_4.ipynb
│
├── model_1.0/
│   ├── app.py
│   ├── webapp.py
│   ├── hinglish_model_checkpoint/
│   ├── hinglish_onnx_model/
│   └── hinglish_onnx_fp16/
│
└── model_-1.0/
```

### Important files

| File | Purpose |
|---|---|
| `inter_iit_csai_ps_4.ipynb` | End-to-end data preparation, training, pseudo-labeling and model export |
| `benchmark.py` | Benchmarking and robustness evaluation |
| `model_1.0/app.py` | FastAPI inference service |
| `model_1.0/webapp.py` | Streamlit interface |
| `upload_models.py` | Model publishing workflow |
| `WHITEPAPER.md` | Competition-oriented technical document |

---

## 12. Run locally

### Clone

```bash
git clone https://github.com/v-krishna07/polyglots-shorthand-sentiment.git
cd polyglots-shorthand-sentiment
```

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Start the API

```bash
cd model_1.0
python3 -m uvicorn app:app --port 8000
```

Model files can be retrieved from the project's Hugging Face repository when configured for remote loading.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Bohot badhiya service 😒"}'
```

A response contains the original text, predicted sentiment, confidence, server-side latency, selected device, and model variant.

### Start the Streamlit UI

```bash
cd model_1.0
python3 -m streamlit run webapp.py
```

---

## 13. Model repository

The model artifacts are hosted separately:

**Hugging Face:**  
https://huggingface.co/v-krishna07/hinglish-models

The repository contains the configuration/tokenizer components while the larger model weights are hosted on Hugging Face.

---

## 14. Reproducibility

The main experiment is contained in:

`inter_iit_csai_ps_4.ipynb`

The notebook covers:

1. dataset loading
2. label normalization
3. emoji extraction
4. tokenizer extension
5. stratified splitting
6. spelling/noise augmentation
7. Phase 1 fine-tuning
8. confidence-based pseudo-label generation
9. Phase 2 fine-tuning
10. ONNX export
11. FP16/O4 optimization
12. GPU inference benchmarking

Random seeds are explicitly used in the data splitting/shuffling stages.

---

## 15. Limitations

This project is intentionally scoped.

### Language scope

The final system focuses on **Romanized Hindi-English / Hinglish**. Performance on other Indic languages written in Roman script has not been established by the reported experiments.

### Sentiment scope

The model predicts only:

- Negative
- Neutral
- Positive

It is not an intent classifier, summarizer, question-answering system, or general-purpose language model.

### Sarcasm

Sarcasm that does not contain useful lexical or emoji cues remains difficult. Sentiment can depend on context that is not available in a short text sample.

### Pseudo-labeling

Pseudo-labels are generated by the model itself and therefore can contain systematic errors. The >95% confidence threshold reduces the risk of noisy labels but does not guarantee correctness.

### Latency

Reported latency depends on hardware, runtime provider, input characteristics, and deployment environment. The T4 GPU figures and public CPU deployment figure should therefore be interpreted as measurements of their respective environments rather than universal guarantees.

---

## 16. Future work

### GPU-enabled public deployment

The current public Streamlit deployment runs on CPU at approximately 15 ms/request. A future deployment can expose the CUDA/FP16 inference path directly to reduce latency where GPU infrastructure is available.

### INT8 quantization

An INT8 version could reduce memory and compute requirements for CPU-heavy deployments.

### Confidence cascade

A lightweight model could handle high-confidence cases while uncertain inputs are routed to the larger model.

### Larger and more diverse evaluation

Future work should evaluate:

- larger Hinglish datasets
- additional Romanized Indic languages
- domain-specific test sets
- explicit sarcasm datasets
- emoji ablations
- cross-domain generalization

### Micro-batching

Server-side micro-batching could improve throughput for high-volume applications while retaining acceptable per-request latency.

---

## 17. License

MIT License.

---

## 18. Acknowledgements

This project uses the Hugging Face Transformers ecosystem, `l3cube-pune/hing-roberta`, Hugging Face Datasets, Optimum, ONNX Runtime, PyTorch, FastAPI, and Streamlit.

---

## Links

- **Live Demo:** https://polyglots-shorthand-sentiment.streamlit.app
- **GitHub:** https://github.com/v-krishna07/polyglots-shorthand-sentiment
- **Hugging Face Models:** https://huggingface.co/v-krishna07/hinglish-models
- **Training Notebook:** https://github.com/v-krishna07/polyglots-shorthand-sentiment/blob/main/inter_iit_csai_ps_4.ipynb
- **Competition Whitepaper:** `WHITEPAPER.md`
