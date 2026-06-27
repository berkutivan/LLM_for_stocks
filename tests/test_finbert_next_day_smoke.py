from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.finbert.next_day_predictor import (
    FinBERTNextDayPredictor,
    _direction_from_score,
)
from src.prices.align import PRODUCTS


class TestDirectionFromScore:
    def test_up(self):
        assert _direction_from_score(0.5, 0.1) == "up"

    def test_down(self):
        assert _direction_from_score(-0.5, 0.1) == "down"

    def test_flat(self):
        assert _direction_from_score(0.05, 0.1) == "flat"


class TestFinBERTNextDayContext:
    @patch("src.finbert.next_day_predictor.load_finbert")
    def test_v3_context_blocks(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), "cpu", ["positive", "negative", "neutral"])

        news_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01", "2024-04-20"]),
                "source": ["Argus", "Reuters"],
                "headline": ["Old news", "Target headline"],
                "category": ["Аммиак", "Фосфаты"],
            }
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = ["precedent fact text"]

        config = {
            "project_root": ".",
            "context": {"window_weeks": 4, "window_days": 28},
            "finbert": {"flat_score_threshold": 0.1},
        }

        predictor = FinBERTNextDayPredictor(
            config,
            retriever,
            news_df,
            "new_prices.csv",
        )
        ctx = predictor._build_v3_context(
            "Reuters",
            "Target headline",
            pd.Timestamp("2024-04-24"),
            "Аммиак",
            ["precedent fact text"],
        )

        assert "2024-04-24" in ctx
        assert "Reuters" in ctx
        assert "Target headline" in ctx
        assert "## Категория новости" in ctx
        assert "Аммиак" in ctx
        assert "## RAG-прецеденты" in ctx
        assert "precedent fact text" in ctx
        assert "## Окно цен до новости" in ctx
        assert "## Окно прошлых новостей" in ctx
        assert "Old news" in ctx

    @patch("src.finbert.next_day_predictor.load_finbert")
    def test_predict_outputs_direction_only(self, mock_load):
        tokenizer = MagicMock()

        def tokenizer_side_effect(*_args, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                import torch

                return {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                }
            return {"input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]}

        tokenizer.side_effect = tokenizer_side_effect
        model = MagicMock()
        mock_load.return_value = (tokenizer, model, "cpu", ["positive", "negative", "neutral"])

        import torch

        model_out = MagicMock()
        model_out.logits = torch.tensor(
            [
                [0.1, 0.1, 0.8],
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )
        model.return_value = model_out

        news_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-04-01"]),
                "source": ["Argus"],
                "headline": ["Gas up"],
                "category": ["Аммиак"],
            }
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = []

        config = {
            "project_root": ".",
            "context": {"window_weeks": 4, "window_days": 28},
            "finbert": {"flat_score_threshold": 0.1},
        }

        predictor = FinBERTNextDayPredictor(config, retriever, news_df, "new_prices.csv")
        result, usage, elapsed = predictor.predict_with_meta(
            "Argus",
            "Gas up",
            pd.Timestamp("2024-05-01"),
            category="Аммиак",
        )

        assert result["model"] == "finbert"
        assert result["dir_urea"] == "flat"
        assert result["dir_dap"] == "up"
        assert result["dir_mop"] == "down"
        for product in PRODUCTS:
            assert result[f"days_{product}"] == 0
            assert result[f"weeks_{product}"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["prompt_tokens"] > 0
        assert elapsed >= 0
