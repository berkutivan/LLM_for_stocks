"""Pydantic-схемы structured output для FinBERT."""

from typing import Literal

from pydantic import BaseModel, Field


class FinBERTSentiment(BaseModel):
    """Базовый результат сентимент-анализа FinBERT."""

    label: Literal["positive", "negative", "neutral"] = Field(
        description="Предсказанный класс сентимента"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Уверенность модели в предсказанном классе",
    )


class FinBERTSentimentDetailed(FinBERTSentiment):
    """Расширенный результат с вероятностями по всем классам."""

    probabilities: dict[str, float] = Field(
        description="Вероятности для positive, negative и neutral"
    )


class FinBERTBatchItem(BaseModel):
    """Элемент батч-ответа с исходным текстом."""

    text: str = Field(description="Исходный входной текст")
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=0.0, le=1.0)


class FinBERTBatchResult(BaseModel):
    """Structured output для пакетной обработки текстов."""

    items: list[FinBERTBatchItem]
