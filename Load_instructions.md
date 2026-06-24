# Инструкция по запуску

Репозиторий содержит гибридный пайплайн генерации тикеров (FinBERT + LLM), тесты гибридного режима и оценку торговой стратегии Калмана с `gain = 0.5`.

## Структура

| Папка / файл | Назначение |
|---|---|
| `Pipeline/` | LangGraph-пайплайн: классификация, приоритизация, гибридные тикеры |
| `Experiments_with_pipeline/` | Пакетная генерация предсказаний и оценка стратегии |
| `tests/` | Pytest-тесты пайплайна (в т.ч. один пример новости в гибридном режиме) |
| `Results/hybrid_comparison/` | Сохранённые результаты последнего прогона |
| `market_news.csv` | Новости с разметкой категорий |
| `right_strategy.csv` | Эталонная стратегия для бенчмарка |
| `new_prices.csv` | Недельные цены urea / dap / mop |
| `examples.csv` | Few-shot примеры для prioritizer |

RAG-сравнение (`run_rag_hybrid_comparison.py`, `Results/hybrid_rag_comparison/`) в репозиторий не включено.

## Требования

- Python 3.11+
- API-ключ OpenAI или OpenRouter (для LLM-вызовов)
- ~2 ГБ для весов FinBERT (скачиваются отдельно)

## Установка

```powershell
cd LLM_for_stocks
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r Pipeline/requirements.txt
pip install -r requirements-ml.txt
```

Скопируйте `env.example` в `.env` и укажите ключ:

```powershell
copy env.example .env
```

Минимально нужен один из ключей:

- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

Скачайте FinBERT (один раз):

```powershell
python Pipeline/llm/models/download_finbert.py
```

## 1. Гибридный пайплайн на одном дне

Пример: все новости за `2020-01-10` из `market_news.csv`.

```powershell
python Pipeline/run_example.py
```

Гибридный режим с одной новостью и логами моделей (нужен API-ключ):

```powershell
python tests/run_hybrid_test.py
```

Или через pytest:

```powershell
python -m pytest tests/test_pipeline.py::TestHybridPipeline::test_hybrid_single_news_model_logs -s -v
```

Тест использует дату `2020-01-10` и первую новость дня; проверяет 10 model logs (2 prioritizer + 8 ticker/FinBERT).

## 2. Пакетная генерация тикеров (hybrid)

```powershell
python Experiments_with_pipeline/predict_tickers.py --analysis-mode hybrid
```

Результат по умолчанию: `Results/market_to_ticker_hybrid.csv`.

## 3. Полный прогон: генерация + сравнение + стратегия (gain = 0.5)

```powershell
python Experiments_with_pipeline/run_hybrid_comparison.py
```

Скрипт:

1. Перегенерирует предсказания в `Results/hybrid_comparison/market_to_ticker_hybrid_new.csv`
2. Сравнивает категории с `market_news.csv` и с baseline `Results/market_to_ticker_hybrid.csv`
3. Считает корреляции priority ↔ цена
4. Оценивает стратегию Калмана с `gain = 0.5`

Повторная оценка без вызовов LLM (если CSV уже есть):

```powershell
python Experiments_with_pipeline/run_hybrid_comparison.py --output-dir Results/hybrid_comparison --skip-regenerate
```

Только стратегия на готовом CSV:

```powershell
python -c "from pathlib import Path; import sys; sys.path.insert(0,'Experiments_with_pipeline'); from run_hybrid_comparison import _run_strategy_eval; _run_strategy_eval('custom', Path('Results/market_to_ticker_hybrid.csv'), Path('Results/hybrid_comparison'))"
```

## 4. Сохранённые результаты

Каталог `Results/hybrid_comparison/`:

| Файл | Содержание |
|---|---|
| `comparison_summary.json` | Сводка: категории, корреляции, PnL |
| `evaluation_kalman_hybrid_new_summary.json` | Стратегия на новых предсказаниях, gain=0.5 |
| `evaluation_kalman_hybrid_baseline_summary.json` | Стратегия на baseline, gain=0.5 |
| `category_comparison_new_vs_actual_summary.json` | Accuracy категорий vs разметка |
| `market_to_ticker_hybrid_new.csv` | Сгенерированные тикеры |

Ключевые метрики последнего прогона (gain = 0.5):

- Категории (new vs actual): **96.22%** accuracy
- Стратегия new: **+76.17%** суммарная compound-доходность, profit **76 174**
- Стратегия baseline: **+85.65%**, profit **85 652**

## Переменные окружения

| Переменная | Описание |
|---|---|
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | Ключ API |
| `OPENAI_MODEL` / `OPENROUTER_MODEL` | Модель по умолчанию |
| `OPENAI_MODEL_PRIORITIZER` | Модель для prioritizer |
| `OPENAI_MODEL_TICKER` | Модель для ticker-агента |

Подробнее — в `env.example`.
