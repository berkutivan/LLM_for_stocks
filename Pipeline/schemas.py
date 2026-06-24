import math
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

AnalysisMode = Literal["standard", "hybrid"]

NewsCategory = Literal[
    "Аммиак",
    "Фосфаты",
    "Калий",
    "Энергоносители",
    "Логистика",
    "Макро/валюта",
    "Спрос/агрорынок",
    "Прочее",
]


class Ticker(BaseModel):
    Increase_price: float = Field(description="Increase price")
    Decrease_price: float = Field(description="Decrease price")
    Volatility_price: float = Field(description="No change price")

    @model_validator(mode="after")
    def validate_components_sum_to_one(self) -> Self:
        total = self.Increase_price + self.Decrease_price + self.Volatility_price
        if not math.isclose(total, 1.0):
            raise ValueError(
                f"Сумма компонентов должна быть равна 1, получено {total}"
            )
        return self


class Ticker_from_news(BaseModel):
    urea_ticker: Ticker = Field(description="Urea ticker")
    dap_ticker: Ticker = Field(description="Dap ticker")
    mop_ticker: Ticker = Field(description="Mop ticker")


class ModelLog(BaseModel):
    node: str = Field(description="Pipeline node name")
    model: str = Field(description="Model identifier")
    io: Literal["input", "output"] = Field(description="Log direction")
    content: str = Field(description="Input or output payload")


class GraphState(BaseModel):
    # Input data
    data: str = Field(description="Date of news batch (YYYY-MM-DD)")
    news: dict[str, str] = Field(description="All today's news: source-headline -> description")
    analysis_mode: AnalysisMode = Field(
        default="standard",
        description="standard — LLM only; hybrid — FinBERT×3 on ticker + standard prioritizer",
    )

    # Counting data
    classified_news: list[NewsCategory] = Field(
        default_factory=list,
        description="Class per news item (same order as news keys)",
    )
    prioritased_news: list[int] = Field(
        default_factory=list,
        description="Priority per news item from 0 to 100",
    )
    ticker_from_news: list[Ticker_from_news] = Field(
        default_factory=list,
        description="Ticker per news item",
    )

    # Technical data
    token_use: dict[str, int] = Field(
        default_factory=dict,
        description="Token use: model -> tokens",
    )
    time_use: dict[str, float] = Field(
        default_factory=dict,
        description="Time use: node -> seconds",
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description="Error messages",
    )
    model_logs: list[ModelLog] = Field(
        default_factory=list,
        description="Per-model input/output logs (hybrid mode)",
    )

    @classmethod
    def create(
        cls,
        data: str,
        news: dict[str, str],
        analysis_mode: AnalysisMode = "standard",
    ) -> Self:
        """Initial state contains only date and news."""
        return cls(data=data, news=news, analysis_mode=analysis_mode)
