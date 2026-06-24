"""OpenAI-подобный клиент для FinBERT с structured output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from .finbert_model import FinBERTPipeline
from .schemas import FinBERTBatchItem, FinBERTBatchResult, FinBERTSentiment, FinBERTSentimentDetailed

T = TypeVar("T", bound=BaseModel)


def _to_schema(raw: dict[str, Any], response_format: type[T]) -> T:
    if response_format is FinBERTSentimentDetailed:
        return response_format(
            label=raw["label"],
            score=raw["score"],
            probabilities=raw["probabilities"],
        )
    if response_format is FinBERTBatchResult:
        return response_format(
            items=[
                FinBERTBatchItem(
                    text=item["text"],
                    label=item["label"],
                    score=item["score"],
                )
                for item in raw["items"]
            ]
        )
    return response_format(label=raw["label"], score=raw["score"])


@dataclass
class ParsedMessage(Generic[T]):
    parsed: T
    raw: dict[str, Any]


@dataclass
class ParsedChoice(Generic[T]):
    message: ParsedMessage[T]
    index: int = 0


@dataclass
class ChatCompletion(Generic[T]):
    choices: list[ParsedChoice[T]]
    model: str
    usage: dict[str, int]


@dataclass
class ParseResponse(Generic[T]):
    """Аналог openai.responses.parse()."""

    output_parsed: T
    model: str
    raw: dict[str, Any]


class ChatCompletions:
    def __init__(self, client: "FinBERT") -> None:
        self._client = client

    def parse(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: type[T] = FinBERTSentiment,
        model: str | None = None,
    ) -> ChatCompletion[T]:
        text = self._client._extract_user_text(messages)
        raw = self._client._engine.predict_one(text)
        parsed = _to_schema(raw, response_format)
        return ChatCompletion(
            choices=[ParsedChoice(message=ParsedMessage(parsed=parsed, raw=raw))],
            model=model or self._client.model_name,
            usage={"prompt_tokens": len(text.split()), "completion_tokens": 0},
        )


class Responses:
    def __init__(self, client: "FinBERT") -> None:
        self._client = client

    def parse(
        self,
        *,
        input: str | list[str],
        text_format: type[T] = FinBERTSentiment,
        model: str | None = None,
    ) -> ParseResponse[T] | list[ParseResponse[T]]:
        if isinstance(input, list):
            if text_format is FinBERTBatchResult:
                raw_items = self._client._engine.predict_many(input)
                parsed = _to_schema({"items": raw_items}, FinBERTBatchResult)
                return ParseResponse(
                    output_parsed=parsed,
                    model=model or self._client.model_name,
                    raw={"items": raw_items},
                )
            return self.parse_many(input=input, text_format=text_format, model=model)

        raw = self._client._engine.predict_one(input)
        parsed = _to_schema(raw, text_format)
        return ParseResponse(
            output_parsed=parsed,
            model=model or self._client.model_name,
            raw=raw,
        )

    def parse_many(
        self,
        *,
        input: list[str],
        text_format: type[T] = FinBERTSentiment,
        model: str | None = None,
    ) -> list[ParseResponse[T]]:
        raw_items = self._client._engine.predict_many(input)
        return [
            ParseResponse(
                output_parsed=_to_schema(raw, text_format),
                model=model or self._client.model_name,
                raw=raw,
            )
            for raw in raw_items
        ]


class Chat:
    def __init__(self, client: "FinBERT") -> None:
        self.completions = ChatCompletions(client)


class FinBERT:
    """
    Клиент FinBERT в стиле OpenAI SDK.

    Примеры:

        from llm.models.connectors import FinBERT, FinBERTSentiment

        client = FinBERT()
        result = client.responses.parse(
            input="Profits rose sharply this quarter.",
            text_format=FinBERTSentiment,
        )
        print(result.output_parsed.label, result.output_parsed.score)

        completion = client.chat.completions.parse(
            messages=[{"role": "user", "content": "Markets fell on weak demand."}],
            response_format=FinBERTSentiment,
        )
        print(completion.choices[0].message.parsed.label)
    """

    def __init__(
        self,
        model: str | Path | None = None,
        device: str | int | None = None,
        max_length: int = 512,
    ) -> None:
        self._engine = FinBERTPipeline(model=model, device=device, max_length=max_length)
        self.model_name = self._engine.model_path
        self.chat = Chat(self)
        self.responses = Responses(self)

    @staticmethod
    def _extract_user_text(messages: list[dict[str, str]]) -> str:
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise ValueError("messages must contain at least one user message")
        return user_messages[-1]
