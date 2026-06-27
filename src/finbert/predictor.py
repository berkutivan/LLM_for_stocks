from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.finbert.model import load_finbert
from src.rules.hints import analyze_headline, format_hints

PRODUCTS = ["urea", "dap", "mop"]
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


def _infer_category_from_hints(source: str, headline: str) -> str:
    hints = analyze_headline(source, headline)
    if hints.likely_prochee:
        return "Прочее"
    if hints.suggested_category:
        return hints.suggested_category
    return "Спрос/агрорынок"


class FinBERTNewsPredictor:
    def __init__(self, config: dict[str, Any], retriever: Any) -> None:
        self.config = config
        self.retriever = retriever
        self.prompt_template = _load_finbert_prompt(Path(config["project_root"]))
        fb = config.get("finbert", {})
        self.model_name = fb.get("model_name", "ProsusAI/finbert")
        self.max_length = fb.get("max_length", 512)
        self.max_rag_chars = fb.get("max_rag_chars", 2000)
        self.tokenizer, self.model, self.device, self.labels = load_finbert(self.model_name)

    def _build_product_text(
        self,
        source: str,
        headline: str,
        product: str,
        precedents: list[str],
        hints_text: str,
    ) -> str:
        prompt = self.prompt_template.format(product=PRODUCT_LABELS[product])
        rag_block = _truncate("\n\n".join(precedents), self.max_rag_chars) if precedents else "Нет прецедентов."
        return (
            f"{prompt}\n\n"
            f"Source: {source}\n"
            f"Headline: {headline}\n\n"
            f"Pre-analysis hints:\n{hints_text}\n\n"
            f"Historical precedents (RAG):\n{rag_block}\n\n"
            f"Target product: {PRODUCT_LABELS[product]}"
        )

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

        results = []
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

    def predict(self, source: str, headline: str) -> dict[str, Any]:
        hints = analyze_headline(source, headline)
        hints_text = format_hints(hints)
        precedents = self.retriever.retrieve(
            source,
            headline,
            suggested_category=hints.suggested_category,
        )

        texts = [
            self._build_product_text(source, headline, product, precedents, hints_text)
            for product in PRODUCTS
        ]
        inferences = self._infer_batch(texts)

        scores = {product: inf["score"] for product, inf in zip(PRODUCTS, inferences)}
        if hints.likely_prochee:
            scores = {p: 0.0 for p in PRODUCTS}

        importance = max(abs(scores[p]) for p in PRODUCTS)
        if hints.likely_prochee:
            importance = min(importance, 0.1)

        reasoning_parts = [
            f"{PRODUCTS[i]}: {inferences[i]['label']} ({inferences[i]['score']:+.2f})"
            for i in range(len(PRODUCTS))
        ]
        reasoning = (
            "FinBERT по каждому удобрению с RAG-прецедентами: "
            + "; ".join(reasoning_parts)
            + "."
        )

        return {
            "category": _infer_category_from_hints(source, headline),
            "reasoning": reasoning,
            "importance": importance,
            "score_urea": scores["urea"],
            "score_dap": scores["dap"],
            "score_mop": scores["mop"],
        }

    def predict_batch(
        self,
        items: list[tuple[str, str]],
        max_workers: int = 10,
        retries: int = 2,
    ) -> list[dict[str, Any]]:
        del max_workers, retries
        if not items:
            return []

        texts: list[str] = []
        item_meta: list[tuple[str, str, Any, list[str], str]] = []

        for source, headline in items:
            hints = analyze_headline(source, headline)
            hints_text = format_hints(hints)
            precedents = self.retriever.retrieve(
                source,
                headline,
                suggested_category=hints.suggested_category,
            )
            item_meta.append((source, headline, hints, precedents, hints_text))
            for product in PRODUCTS:
                texts.append(
                    self._build_product_text(source, headline, product, precedents, hints_text)
                )

        inferences = self._infer_batch(texts)
        results: list[dict[str, Any]] = []

        for i, (source, headline, hints, _precedents, _hints_text) in enumerate(item_meta):
            offset = i * len(PRODUCTS)
            product_inferences = inferences[offset : offset + len(PRODUCTS)]
            scores = {
                product: product_inferences[j]["score"]
                for j, product in enumerate(PRODUCTS)
            }
            if hints.likely_prochee:
                scores = {p: 0.0 for p in PRODUCTS}

            importance = max(abs(scores[p]) for p in PRODUCTS)
            if hints.likely_prochee:
                importance = min(importance, 0.1)

            reasoning_parts = [
                f"{PRODUCTS[j]}: {product_inferences[j]['label']} ({product_inferences[j]['score']:+.2f})"
                for j in range(len(PRODUCTS))
            ]
            reasoning = (
                "FinBERT по каждому удобрению с RAG-прецедентами: "
                + "; ".join(reasoning_parts)
                + "."
            )
            results.append(
                {
                    "category": _infer_category_from_hints(source, headline),
                    "reasoning": reasoning,
                    "importance": importance,
                    "score_urea": scores["urea"],
                    "score_dap": scores["dap"],
                    "score_mop": scores["mop"],
                }
            )
        return results
