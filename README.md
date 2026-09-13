# Polyglot's Shorthand: Code-Mixed Sentiment Analysis

This repository contains my solution for the "Polyglot's Shorthand" sentiment analysis challenge, focusing on Romanized Hinglish text. The goal is to classify text as positive, negative, or neutral while staying under a 500M parameter limit and optimizing for fast inference.

## Approach

Standard tokenizers usually struggle with Hinglish (e.g., breaking "krdo" into random pieces) and often ignore emojis. To handle this, the project is structured in two main parts:

* **Custom Tokenizer:** I trained a Byte-Level BPE tokenizer from scratch on Hinglish data. It includes a normalization step to clean up invisible unicode characters and specifically keeps frequent emojis intact so they can be processed as actual features.
* **Model Architecture:** I am using `distilbert-base-multilingual-cased` as the base model because it's lightweight. I resized its embedding layer to fit the custom BPE tokenizer so it can leverage the pre-trained weights while understanding the new Hinglish vocabulary.

## Current Progress

* [x] **Data Prep & Tokenizer:** Cleaned the text, extracted top emojis, and trained the custom BPE tokenizer.
* [ ] **Model Setup:** Connect DistilBERT, resize embeddings, and build the PyTorch training loop.
* [ ] **Training:** Fine-tune the model with data augmentation to handle common typos and spelling variations.
* [ ] **Optimization:** Convert the final model to ONNX and apply int8 quantization to hit the latency target.
