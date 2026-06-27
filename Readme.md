# LLM for stocks — предсказание цен удобрений по новостям

Пайплайн: **категоризация (prioritizer)** → **RAG-прецеденты** → **прогноз движения** (LLM v3 / FinBERT next-day) с метриками на test 30%.

Подробное описание, метрики и инструкции: [RESEARCH.md](RESEARCH.md), слайды: [Presentation.md](Presentation.md).

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # OPENROUTER_API_KEY, OPENROUTER_MODEL
```

```bash
python -m src.cli build-rag      # RAG на train 70%
python -m src.cli predict        # LangGraph + FinBERT, test split
python -m src.cli eval-days      # метрики и графики days/direction
```

## Данные

| Файл | Описание |
|------|----------|
| `market_news.csv` | Новости 2020–2026 |
| `new_prices.csv` | Недельные котировки urea / dap / mop |

## Результаты

Артефакты экспериментов (CSV, JSON, графики): папка [`Results/`](Results/).

Старый hybrid-пайплайн (`Pipeline/`, `Experiments_with_pipeline/`, `Results/hybrid_comparison/`) удалён — актуальная версия в `src/`.
