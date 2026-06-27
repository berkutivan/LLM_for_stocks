from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def load_finbert(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return tokenizer, model, device, labels
