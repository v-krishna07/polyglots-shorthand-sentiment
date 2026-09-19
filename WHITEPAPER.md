# Technical Whitepaper

# Polyglot's Shorthand: An Emoji-Aware Sentiment Engine for Romanized Hinglish

**Problem Statement:** 4 — *The Polyglot's Shorthand*  
**Author:** Krishna  
**Team:** Solo  
**Competition:** Inter IIT CSAI  
**Date:** September 2026  
**Repository:** https://github.com/v-krishna07/polyglots-shorthand-sentiment  
**Live deployment:** https://polyglots-shorthand-sentiment.streamlit.app

---

## Abstract

Romanized Hinglish occupies an unusual position in natural-language processing. It is neither standard English nor standard Hindi: users frequently combine English and Hindi within the same sentence, write Hindi phonetically in Latin characters, shorten words aggressively, and use emojis as part of the intended meaning.

This creates a practical sentiment-analysis problem. A model can understand the underlying language yet remain brittle to the spelling patterns through which that language is actually typed.

This work presents an end-to-end sentiment engine designed specifically for this setting. The system starts from the pretrained `l3cube-pune/hing-roberta` model, extends its existing tokenizer with the 50 most frequent emojis found in the project corpus, and fine-tunes the resulting three-class classifier on Hinglish sentiment data. Robustness to Romanized spelling variation is encouraged through programmatic text augmentation. A second training phase uses confidence-filtered pseudo-labeling on an unlabeled Hinglish corpus, accepting only predictions whose maximum class probability exceeds 0.95.

The final model contains approximately 278 million parameters and achieves greater than 92% validation accuracy with approximately 0.84 Macro-F1. For deployment, the trained model is exported to ONNX and optimized with ONNX Runtime tooling. The final FP16 model records 2.58 ms p50 latency and 8.24 ms p99 latency on an NVIDIA Tesla T4 under the project's benchmark configuration. The public Streamlit deployment currently runs on CPU and has been observed at approximately 15 ms per request.

The work demonstrates a complete path from domain-specific data preparation and robustness engineering to optimized inference for Romanized Hinglish sentiment classification.

---

# 1. Introduction

## 1.1 Problem

A large amount of conversational text in India is written in Latin characters even when the underlying language is Hindi. Users also freely mix English words into Hindi grammatical structures.

Examples include:

```text
bhai order cancel krdo please
```

```text
kya kr rhe ho
```

```text
bahut badhiya service
```

The same linguistic content can appear with multiple spellings:

```text
kya kar rahe ho
kya kr rhe ho
kya krre ho
```

For a human reader these variations can be closely related. For a tokenizer, they can produce substantially different token sequences.

The problem becomes more difficult when emojis are included:

```text
Bohot badhiya service
```

versus:

```text
Bohot badhiya service 😒
```

The lexical content is nearly identical, but the second expression can communicate dissatisfaction or sarcasm.

The system therefore targets the actual shorthand distribution rather than an idealized clean-language distribution.

---

# 2. Objectives

The project was designed around four objectives:

1. Build a sentiment classifier specifically adapted to Romanized Hinglish.
2. Improve robustness to phonetic and shorthand spelling variation.
3. preserve emoji information instead of treating emojis as disposable noise.
4. Produce an inference stack fast enough for interactive use.

The classifier predicts:

- **Negative**
- **Neutral**
- **Positive**

The final model remains below the project's 500M-parameter deployment constraint.

---

# 3. Data

## 3.1 Supervised data

The initial labeled corpus combines two public Hinglish sentiment sources:

- `Abhishek4896/hindi-english-code-mixed-tweets-sentiment`
- `shae2977/hinglish-youtube-sentiments-dataset`

The notebook normalizes both datasets into a common schema:

```text
text
label
```

with:

```text
negative → 0
neutral  → 1
positive → 2
```

The initial project corpus contained approximately **6.5k labeled examples**.

The datasets are shuffled with a fixed seed of 42 before the stratified split.

## 3.2 Validation split

A stratified 90/10 train-validation split is used.

This preserves the approximate class proportions between the training and validation subsets.

## 3.3 Additional unlabeled data

For Phase 2, the project loads:

```text
Yugrathee28/Hinglish-dataset
```

The Phase 1 classifier is applied to this corpus. Only examples with maximum predicted probability above 0.95 are retained as pseudo-labeled examples.

The resulting effective training pool reached approximately **7.2k examples** in the final project run.

---

# 4. Emoji-aware preprocessing

## 4.1 Why emoji handling matters

Emojis can carry sentiment that is absent from the words themselves.

A conventional preprocessing pipeline may remove emojis as punctuation or irrelevant Unicode characters. That behaviour is undesirable for this task.

The project instead:

1. detects emojis in the corpus,
2. normalizes their Unicode representation,
3. counts their frequency,
4. selects the top 50,
5. adds those emojis to the tokenizer vocabulary.

## 4.2 Unicode normalization

The implementation removes the variation selector `U+FE0F` to reduce representation differences such as emoji sequences that contain an invisible variation-selector character.

This normalization is applied consistently during training/inference preprocessing.

The intent is not to remove emoji meaning; it is to make the representation more stable.

---

# 5. Tokenization and model selection

## 5.1 Backbone

The project uses:

```text
l3cube-pune/hing-roberta
```

The existing tokenizer is loaded rather than training an entirely new tokenizer from scratch.

The tokenizer is extended using:

```python
tokenizer.add_tokens(top_emojis)
```

The model embedding matrix is then resized to accommodate the additional tokens.

This preserves the pretrained linguistic representation while explicitly giving frequent emojis dedicated vocabulary entries.

## 5.2 Sequence length

The maximum input sequence length is:

```text
128 tokens
```

This provides a bounded computational cost and also allows the same fixed input-length policy to be used in the optimized inference path.

---

# 6. Robustness through phonetic-noise augmentation

Romanized Hinglish does not have a single canonical spelling.

Users routinely omit vowels, shorten words, or make keyboard-neighbour substitutions.

The Phase 1 training pipeline therefore augments training examples with spelling/noise variations.

The main motivation is to expose the classifier to transformations such as:

```text
samajh → smjh
```

and other shorthand/keyboard variations.

The desired behaviour is not necessarily token-level identity. Instead, the objective is to make semantically similar shorthand expressions produce stable sentiment predictions.

---

# 7. Training methodology

## 7.1 Phase 1

The first phase trains the three-class classifier on the labeled, augmented dataset.

The notebook uses:

| Configuration | Value |
|---|---:|
| Backbone | `l3cube-pune/hing-roberta` |
| Classes | 3 |
| Max sequence length | 128 |
| Train batch size | 32 |
| Validation batch size | 32 |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Scheduler | Cosine decay |
| Warmup | 10% of training steps |
| Gradient clipping | 1.0 |
| Early stopping patience | 5 |
| Maximum epochs per phase | 20 |

Class-weighted cross entropy is used to reduce the effect of class imbalance.

The loss includes label smoothing of 0.1.

Mixed-precision CUDA training is used when a CUDA device is available.

## 7.2 Model selection

Validation accuracy is monitored after each epoch.

The best checkpoint is saved when validation accuracy improves.

Training can terminate early after five consecutive non-improving validation evaluations.

---

# 8. Phase 2: confidence-filtered pseudo-labeling

## 8.1 Motivation

The labeled corpus is relatively small compared with the capacity of a transformer encoder.

Rather than assigning labels to every unlabeled example, the project uses a conservative pseudo-labeling strategy.

The Phase 1 model is applied to the additional unlabeled Hinglish corpus.

For each sample:

```text
p = softmax(logits)
```

and:

```text
confidence = max(p)
```

The example is accepted only if:

```text
confidence > 0.95
```

The predicted class becomes the pseudo-label.

## 8.2 Second training phase

The pseudo-labeled examples are concatenated with the augmented supervised training data.

The combined dataset is shuffled and used for the second fine-tuning phase.

This is more accurately described as **confidence-filtered self-training** than as a strict teacher-student architecture because the same model lineage is used to generate the pseudo-labels and continue training.

## 8.3 Risk control

Pseudo-labeling can amplify model errors.

The 0.95 threshold is therefore used as a conservative filter. It does not make pseudo-labels guaranteed-correct; it simply limits the training set to examples for which the current model expresses high confidence.

---

# 9. Inference architecture

The production pipeline is:

```text
                Raw Romanized Hinglish
                         │
                         ▼
              Unicode normalization
                         │
                         ▼
            Hing-RoBERTa tokenizer
                 + 50 emoji tokens
                         │
                         ▼
                 Transformer encoder
                         │
                         ▼
              3-class classification
                         │
                         ▼
                ONNX Runtime engine
                    /          \
                   /            \
                CPU             CUDA
```

The application layer exposes the model through FastAPI and provides a Streamlit interface for interactive use.

---

# 10. Model export and optimization

## 10.1 ONNX export

The trained PyTorch checkpoint is exported using Hugging Face Optimum's ONNX Runtime integration.

This separates model training from the optimized inference graph.

## 10.2 O4 optimization

The project applies:

```text
AutoOptimizationConfig.O4()
```

through ONNX Runtime's optimization tooling.

This produces an optimized computational graph intended for accelerated inference.

## 10.3 FP16 model

A final FP16 optimized model is produced for GPU inference.

The repository therefore maintains multiple deployable artifacts, including:

- PyTorch checkpoint
- reference ONNX model
- optimized FP16 ONNX model
- an older latency-oriented model variant

---

# 11. Evaluation

## 11.1 Classification quality

The final model achieves:

| Metric | Result |
|---|---:|
| Parameters | ~278M |
| Validation accuracy | >92% |
| Macro-F1 | ~0.84 |

Macro-F1 is particularly useful here because it weights the three sentiment classes equally rather than allowing the largest class to dominate the metric.

## 11.2 GPU latency

The final optimized FP16 ONNX model was benchmarked on an NVIDIA Tesla T4.

| Metric | Result |
|---|---:|
| Average latency | 3.32 ms |
| p50 latency | 2.58 ms |
| p75 latency | 3.95 ms |
| p99 latency | 8.24 ms |

The project's benchmark measures the server-side inference path used by the implementation, including preprocessing/tokenization and output probability generation.

## 11.3 Public deployment

The public Streamlit deployment runs on CPU.

Observed deployment latency:

```text
≈ 15 ms/request
```

This figure is environment-specific and should not be interpreted as a hardware-independent guarantee.

---

# 12. Robustness evaluation probes

The repository contains targeted probes corresponding to the problem statement.

### Phonetic drift

```text
kya kar rahe ho
kya kr rhe ho
kya krre ho
```

### Shorthand spelling

```text
bahut badhiya service
bht badhiya servis
```

### Emoji-dependent context

```text
Bohot badhiya service
Bohot badhiya service 😒
```

### Code mixing

```text
aap busy ho?
```

These probes are designed to reveal whether the model reacts sensibly to spelling variation, shorthand, emoji context and intra-sentence English words.

They should be considered qualitative robustness probes rather than a replacement for a large independently annotated robustness benchmark.

---

# 13. Engineering design decisions

## 13.1 Why start from a pretrained Hinglish model?

Training a transformer representation from scratch would be impractical for the project's dataset scale.

Starting from `hing-roberta` provides an existing language representation suited to the target domain, after which the model can specialize for sentiment.

## 13.2 Why add emojis to the vocabulary?

Frequent emojis occur often enough to justify explicit representation.

Adding the top 50 emojis gives the model dedicated token entries rather than relying entirely on fragmented sub-token representations.

## 13.3 Why augment spelling?

The input language is inherently noisy.

A model trained only on canonical-looking spellings can learn correlations that fail when a user writes the same expression differently.

Noise augmentation makes this variation part of training rather than an unseen distribution shift.

## 13.4 Why pseudo-label?

The project has access to additional unlabeled Hinglish text.

High-confidence pseudo-labeling allows some of that linguistic diversity to be incorporated without requiring manual annotation of every example.

## 13.5 Why ONNX?

Training-time PyTorch is not necessarily the most efficient production inference runtime.

ONNX Runtime provides a deployment-oriented execution path and enables graph optimization and GPU-specific execution strategies.

---

# 14. Deployment architecture

The project is organized as:

```text
                  Streamlit
                     │
                     ▼
                  FastAPI
                     │
                     ▼
               Model loader
                     │
                     ▼
              ONNX Runtime
                /       \
              CPU       CUDA
```

Model artifacts are hosted through the project's Hugging Face model repository, allowing deployment systems to download the model rather than storing all large weight files directly in Git.

---

# 15. Limitations

## 15.1 Language coverage

The reported evaluation is focused on Romanized Hindi-English.

It does not establish equivalent performance for every Indian language written in Roman script.

## 15.2 Three-class formulation

The model distinguishes only:

```text
Negative / Neutral / Positive
```

It does not independently classify intent, emotion categories, sarcasm, toxicity, topic, or user intent.

## 15.3 Sarcasm

Sarcasm can require context that is absent from a short message.

An emoji or lexical cue can make some sarcastic expressions easier, but sarcasm without such cues remains a difficult case.

## 15.4 Pseudo-label noise

A confidence score is not a correctness guarantee.

The pseudo-labeling phase can therefore inherit biases or mistakes from Phase 1.

## 15.5 Benchmark portability

Latency depends on:

- hardware
- execution provider
- CPU/GPU configuration
- model variant
- input length
- runtime environment

The T4 benchmark and Streamlit CPU measurement should therefore be reported with their hardware context.

---

# 16. Future work

## 16.1 GPU-enabled public deployment

The current public Streamlit service runs on CPU.

A future deployment can expose the optimized CUDA/FP16 inference path directly. This would align the public service more closely with the project's low-latency GPU benchmark.

## 16.2 INT8 CPU deployment

INT8 quantization could reduce memory consumption and improve CPU efficiency, making the model more attractive for inexpensive edge or server deployments.

## 16.3 Confidence-based model cascade

A two-stage inference system could use a smaller model for easy examples and route uncertain predictions to the larger model.

This could reduce average compute while preserving accuracy on difficult examples.

## 16.4 Broader language evaluation

Future experiments should evaluate:

- Romanized Marathi
- Romanized Bengali
- Romanized Punjabi
- Romanized Gujarati
- other code-mixed Indian-language distributions

without assuming that Hinglish performance transfers automatically.

## 16.5 Dedicated emoji ablation

A controlled experiment comparing:

```text
full model
```

against:

```text
same model without emoji tokens
```

would quantify the independent contribution of explicit emoji vocabulary.

## 16.6 Larger robustness benchmark

A future benchmark should include independently labeled examples for:

- spelling variation
- phonetic variation
- code switching
- emoji polarity shifts
- sarcasm
- negation
- domain transfer

---

# 17. Reproducibility

The complete experiment is contained in:

```text
inter_iit_csai_ps_4.ipynb
```

The notebook covers:

1. dataset loading
2. schema normalization
3. emoji frequency analysis
4. tokenizer extension
5. train/validation splitting
6. augmentation
7. Phase 1 training
8. pseudo-label generation
9. Phase 2 training
10. model checkpointing
11. ONNX export
12. O4 optimization
13. FP16 model creation
14. GPU latency benchmarking

The repository also contains `benchmark.py` for repeatable evaluation of the deployed model variants.

---

# 18. Reproducibility notes

The notebook uses fixed random seeds in the dataset split/shuffling process.

The validation set is separated before pseudo-labeling, and the pseudo-labeling process is applied to the additional unlabeled corpus rather than replacing the original supervised labels.

For exact reproduction, the notebook, dependency versions, model artifacts and dataset revisions should be retained together because public datasets and software packages can change over time.

---

# 19. Ethical and practical considerations

Sentiment classification is probabilistic.

A prediction should therefore be treated as an automated signal rather than a definitive statement about a person's feelings or intent.

The model is designed for aggregate or assistive applications such as:

- feedback analysis
- routing
- monitoring
- exploratory analytics

It should not be treated as a complete interpretation of a user's intent, especially for sarcasm, ambiguous language, or culturally specific expressions.

---

# 20. Conclusion

This project approaches Romanized Hinglish sentiment analysis as an engineering problem shaped by the way users actually type.

Instead of assuming clean spelling and removing emojis, the pipeline explicitly addresses:

- Romanized language variation
- code mixing
- shorthand spelling
- emoji information
- limited labeled data
- low-latency deployment

The final system combines a pretrained Hinglish transformer, explicit emoji vocabulary extension, spelling/noise augmentation, confidence-filtered pseudo-labeling, and optimized ONNX inference.

The resulting model contains approximately 278M parameters, achieves over 92% validation accuracy and approximately 0.84 Macro-F1, and reaches 2.58 ms p50 latency on an NVIDIA T4 in the project's optimized GPU benchmark. The deployed CPU service currently operates at approximately 15 ms/request.

The main remaining opportunity is to bring the optimized GPU path into the public deployment while expanding evaluation beyond the current Hinglish-focused distribution.

---

# References

1. **L3Cube-Pune — Hing-RoBERTa**  
   https://huggingface.co/l3cube-pune/hing-roberta

2. **Hugging Face Transformers**  
   https://github.com/huggingface/transformers

3. **Hugging Face Datasets**  
   https://huggingface.co/docs/datasets/

4. **Hugging Face Optimum**  
   https://huggingface.co/docs/optimum/

5. **ONNX Runtime**  
   https://onnxruntime.ai/

6. **PyTorch**  
   https://pytorch.org/

7. **FastAPI**  
   https://fastapi.tiangolo.com/

8. **Streamlit**  
   https://streamlit.io/

9. **Project repository**  
   https://github.com/v-krishna07/polyglots-shorthand-sentiment

10. **Project model repository**  
    https://huggingface.co/v-krishna07/hinglish-models

11. **Project live deployment**  
    https://polyglots-shorthand-sentiment.streamlit.app
