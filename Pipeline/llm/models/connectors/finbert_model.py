"""Низкоуровневая обёртка над Hugging Face FinBERT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "ProsusAI/finbert"
DEFAULT_MODEL_DIR = ROOT / "weights" / "finbert"

LABEL_ALIASES = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "label_0": "positive",
    "label_1": "negative",
    "label_2": "neutral",
}


def resolve_model_path(model: str | Path | None = None) -> str:
    if model is None:
        return str(DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else DEFAULT_MODEL_ID)
    path = Path(model)
    return str(path if path.exists() else model)


def normalize_label(raw_label: str) -> str:
    key = raw_label.lower().strip()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    if "pos" in key:
        return "positive"
    if "neg" in key:
        return "negative"
    return "neutral"


class FinBERTPipeline:
    """Ленивая загрузка FinBERT и инференс через transformers pipeline."""

    def __init__(
        self,
        model: str | Path | None = None,
        device: str | int | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = resolve_model_path(model)
        self.device = device
        self.max_length = max_length
        self._pipe: Any | None = None

    @property
    def pipe(self) -> Any:
        if self._pipe is None:
            if self.device is None:
                self.device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_path,
                tokenizer=self.model_path,
                device=self.device,
                top_k=None,
            )
        return self._pipe

    def _normalize_outputs(self, outputs: Any) -> list[dict[str, Any]]:
        if not outputs:
            return []
        if isinstance(outputs[0], dict):
            return outputs
        if outputs and isinstance(outputs[0], list):
            return outputs[0]
        return outputs

    def predict_one(self, text: str) -> dict[str, Any]:
        outputs = self._normalize_outputs(
            self.pipe(
                text,
                truncation=True,
                max_length=self.max_length,
            )
        )
        probs = {
            normalize_label(item["label"]): float(item["score"]) for item in outputs
        }
        label = max(probs, key=probs.get)
        return {
            "text": text,
            "label": label,
            "score": probs[label],
            "probabilities": probs,
        }

    def predict_many(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []
        if len(texts) == 1:
            return [self.predict_one(texts[0])]

        batch_outputs = self.pipe(
            texts,
            truncation=True,
            max_length=self.max_length,
        )
        results: list[dict[str, Any]] = []
        for text, outputs in zip(texts, batch_outputs):
            normalized = self._normalize_outputs(outputs)
            probs = {
                normalize_label(item["label"]): float(item["score"])
                for item in normalized
            }
            label = max(probs, key=probs.get)
            results.append(
                {
                    "text": text,
                    "label": label,
                    "score": probs[label],
                    "probabilities": probs,
                }
            )
        return results


def download_model(target_dir: str | Path | None = None) -> Path:
    """Скачивает веса ProsusAI/finbert в локальную папку репозитория."""
    from huggingface_hub import snapshot_download

    destination = Path(target_dir) if target_dir is not None else DEFAULT_MODEL_DIR
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DEFAULT_MODEL_ID,
        local_dir=str(destination),
    )
    # Прогреваем tokenizer/model, чтобы убедиться, что файлы читаются.
    AutoTokenizer.from_pretrained(str(destination))
    AutoModelForSequenceClassification.from_pretrained(str(destination))
    return destination
