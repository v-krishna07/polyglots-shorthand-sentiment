"""
Upload both Hinglish sentiment models to a single Hugging Face repo.

Run from the project root (the folder that contains model_1.0 and model_-1.0):

    export HF_TOKEN="your_new_write_token"
    python3 upload_models.py

Resulting repo layout:
    v-krishna07/hinglish-models
    ├── README.md                (model card)
    ├── model_-1.0/              older model, faster (optimized ONNX)
    └── model_1.0/               newer model, more accurate
        ├── hinglish_model_checkpoint/   PyTorch (safetensors)
        ├── hinglish_onnx_model/         ONNX
        └── hinglish_onnx_fp16/          ONNX fp16 optimized
"""

import os
import sys
from getpass import getpass

from huggingface_hub import HfApi

USERNAME = "v-krishna07"
REPO_NAME = "hinglish-models"
REPO_ID = f"{USERNAME}/{REPO_NAME}"
PRIVATE = False  # set True to keep the repo private

# (local folder, folder name inside the repo)
MODEL_FOLDERS = [
    ("model_-1.0", "model_-1.0"),
    ("model_1.0", "model_1.0"),
]

# Only code files are skipped; every .json, .onnx and .safetensors is uploaded.
IGNORE = ["*.py", "__pycache__/*", ".DS_Store", "*.ipynb_checkpoints*"]

MODEL_CARD = f"""---
language:
- hi
- en
library_name: transformers
tags:
- sentiment-analysis
- hinglish
- onnx
---

# Hinglish sentiment models

Two Hinglish (Hindi-English code-mixed) sentiment models.

| Folder | Description |
|---|---|
| `model_1.0/` | Newer model with better accuracy. Contains `hinglish_model_checkpoint` (PyTorch), `hinglish_onnx_model` (ONNX) and `hinglish_onnx_fp16` (optimized fp16 ONNX). |
| `model_-1.0/` | Older model with better speed (optimized ONNX). |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "{REPO_ID}"
sub = "model_1.0/hinglish_model_checkpoint"
tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=sub)
```

For the ONNX folders, use `optimum`:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    repo, subfolder="model_-1.0", file_name="model_optimized.onnx"
)
```
"""


def main():
    token = os.environ.get("HF_TOKEN") or getpass("Paste your Hugging Face write token: ")
    api = HfApi(token=token)

    # Fail early if the token is wrong
    who = api.whoami()["name"]
    print(f"Logged in as: {who}")
    if who != USERNAME:
        print(f"Warning: token belongs to '{who}', but USERNAME is '{USERNAME}'.")
        sys.exit(1)

    # Check local folders exist
    for local, _ in MODEL_FOLDERS:
        if not os.path.isdir(local):
            print(f"Folder not found: {local}. Run this from the project root.")
            sys.exit(1)

    api.create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{REPO_ID}")

    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add model card",
    )

    for local, in_repo in MODEL_FOLDERS:
        print(f"\nUploading {local}/ -> {REPO_ID}/{in_repo}/  (large files, this can take a while)")
        api.upload_folder(
            folder_path=local,
            path_in_repo=in_repo,
            repo_id=REPO_ID,
            repo_type="model",
            ignore_patterns=IGNORE,
            commit_message=f"Upload {in_repo}",
        )
        print(f"Done: {in_repo}")

    print(f"\nAll uploaded: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()