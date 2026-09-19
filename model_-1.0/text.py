import time
import torch
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# 1. Path to your exported model folder (adjust if in a different directory)
MODEL_DIR = "./hinglish_onnx_fp16"

print("Loading tokenizer and ONNX model on CPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
onnx_model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    file_name="model_optimized.onnx",
    provider="CPUExecutionProvider"
)
print("Model loaded successfully!\n")

def test_live_sentiment(text: str):
    start = time.perf_counter()
    
    # 2. Tokenize input string (kept on CPU)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 3. Graph execution without gradient tracking
    with torch.no_grad():
        logits = onnx_model(**inputs).logits
        
    # 4. Softmax conversion to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
    pred_idx = np.argmax(probs)
    
    latency_ms = (time.perf_counter() - start) * 1000
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    print("── LIVE PREDICTION ──")
    print(f"Input Text:  {text}")
    print(f"Prediction:  {label_map[pred_idx]}")
    print(f"Confidence:  {probs[pred_idx] * 100:.2f}%")
    print(f"Latency:     {latency_ms:.2f} ms\n")

if __name__ == "__main__":
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("bhai sachme kya mast project banaya hai tune 🔥")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")
    test_live_sentiment("mera refund abhi tak nahi aaya 😡 scam hai ye!")