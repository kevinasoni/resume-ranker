# utils.py
import os
import re

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def clean_text(text: str) -> str:
    # Basic cleanup to reduce noise before vectorization
    text = text.replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
