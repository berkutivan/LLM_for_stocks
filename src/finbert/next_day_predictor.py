from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.context.windows import build_news_context, build_price_context
from src.finbert.model import load_finbert
from src.prices.align import PRODUCTS

PRODUCT_LABELS = {
    "urea": "мочевина (urea)",
    "dap": "DAP (dap)",
    "mop": "MOP (mop)",
}


def _load_finbert_prompt(project_root: Path) -> str:
    return (project_root / "prompts" / "finbert.md").read_text(encoding="utf-8")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _score_from_probs(label_probs: dict[str, float]) -> float:
    pos = label_probs.get("positive", 0.0)
    neg = label_probs.get("negative", 0.0)
    return max(-1.0, min(1.0, pos - neg))


def _direction_from_score(score: float, flat_threshold: float) -> str:
    if abs(score) <= flat_threshold:
        return "flat"
    return "up" if score > 0 else "down"


class FinBERTNextDayPredictor:
    """Next-day direction via FinBERT on the same context blocks as streak variant v3."""

    def __init__(
        self,
        config: dict[str, Any],
        retriever: Any,
        news_df: pd.DataFrame,
        prices_path: str,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.news_df = news_df.sort_values("date").reset_index(drop=True)
        self.prices_path = prices_path
        self.prompt_template = _load_finbert_prompt(Path(config["project_root"]))

        fb = config.get("finbert", {})
        self.model_name = fb.get("model_name", "ProsusAI/finbert")
        self.max_length = fb.get("max_length", 512)
        self.max_rag_chars = fb.get("max_rag_chars", 2000)
        self.flat_score_threshold = fb.get("flat_score_threshold", 0.1)

        ctx = config.get("context", {})
        self.window_weeks = ctx.get("window_weeks", 4)
        self.window_days = ctx.get("window_days", 28)

        self.tokenizer, self.model, self.device, self.labels = load_finbert(self.model_name)

    def _build_v3_context(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        category: str | None,
        precedents: list[str],
    ) -> str:
        parts = [
            f"Дата новости: {news_date.strftime('%Y-%m-%d')}",
            f"Источник: {source}",
            f"Заголовок: {headline}",
        ]
        if category:
            parts.extend(["", f"## Категория новости\n{category}"])
        parts.extend(
            [
                "",
                "## RAG-прецеденты (source + headline → серии после новости)",
                _truncate("\n\n".join(precedents), self.max_rag_chars)
                if precedents
                else "Нет прецедентов.",
                "",
                "## Окно цен до новости",
                build_price_context(self.prices_path, news_date, self.window_weeks),
                "",
                "## Окно прошлых новостей",
                build_news_context(self.news_df, news_date, self.window_days),
            ]
        )
        return "\n".join(parts)

    def _build_product_text(
        self,
        product: str,
        v3_context: str,
    ) -> str:
        prompt = self.prompt_template.format(product=PRODUCT_LABELS[product])
        return f"{prompt}\n\n{v3_context}\n\nTarget product: {PRODUCT_LABELS[product]}"

    def _count_tokens(self, texts: list[str]) -> int:
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        return sum(len(ids) for ids in encoded["input_ids"])

    def _infer_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: list[dict[str, Any]] = []
        for row in probs:
            label_probs = {self.labels[i]: float(row[i]) for i in range(len(self.labels))}
            best_label = max(label_probs, key=label_probs.get)
            score = _score_from_probs(label_probs)
            results.append(
                {
                    "label": best_label,
                    "score": score,
                    "probs": label_probs,
                }
            )
        return results

    def predict_with_meta(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        variant: int = 3,
        category: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        del variant
        t0 = time.perf_counter()
        precedents = self.retriever.retrieve(source, headline)
        v3_context = self._build_v3_context(source, headline, news_date, category, precedents)
        texts = [self._build_product_text(product, v3_context) for product in PRODUCTS]
        token_count = self._count_tokens(texts)
        inferences = self._infer_batch(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result: dict[str, Any] = {"variant": 3, "model": "finbert"}
        reasoning_parts: list[str] = []
        for product, inf in zip(PRODUCTS, inferences):
            direction = _direction_from_score(inf["score"], self.flat_score_threshold)
            result[f"dir_{product}"] = direction
            result[f"weeks_{product}"] = 0
            result[f"days_{product}"] = 0
            reasoning_parts.append(
                f"{product}: {direction} ({inf['label']}, score={inf['score']:+.2f})"
            )
        result["reasoning"] = (
            "FinBERT next-day на контексте v3: " + "; ".join(reasoning_parts) + "."
        )

        usage = {
            "prompt_tokens": token_count,
            "completion_tokens": 0,
            "total_tokens": token_count,
        }
        return result, usage, elapsed_ms

    def predict(
        self,
        source: str,
        headline: str,
        news_date: pd.Timestamp,
        category: str | None = None,
    ) -> dict[str, Any]:
        result, _, _ = self.predict_with_meta(source, headline, news_date, category=category)
        return result
