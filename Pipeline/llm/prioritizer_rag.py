"""Top-k RAG для prioritizer: перебор examples.csv по пересечению токенов."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_PATH = ROOT / "examples.csv"
DEFAULT_TOP_K = 5

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _overlap_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


@lru_cache(maxsize=1)
def _load_examples(path: str) -> tuple[pd.DataFrame, ...]:
    df = pd.read_csv(path)
    required = {"source", "headline", "urea_abs_pct", "dap_abs_pct", "mop_abs_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return (df,)


def retrieve_prioritizer_examples(
    *,
    source: str,
    headline: str,
    examples_path: Path | str = DEFAULT_EXAMPLES_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    path = Path(examples_path)
    if not path.exists():
        return []

    df = _load_examples(str(path.resolve()))[0]
    query = f"{source} {headline}"
    query_tokens = _tokenize(query)

    scored: list[tuple[float, int]] = []
    for idx, row in df.iterrows():
        doc = f"{row['source']} {row['headline']}"
        score = _overlap_score(query_tokens, _tokenize(doc))
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda item: (-item[0], item[1]))
    top_indices = [idx for _, idx in scored[:top_k]]

    if len(top_indices) < top_k:
        seen = set(top_indices)
        for idx in range(len(df)):
            if idx not in seen:
                top_indices.append(idx)
            if len(top_indices) >= top_k:
                break

    results: list[dict[str, object]] = []
    for rank, idx in enumerate(top_indices[:top_k], start=1):
        row = df.iloc[idx]
        results.append(
            {
                "rank": rank,
                "source": row["source"],
                "headline": row["headline"],
                "urea_abs_pct": float(row["urea_abs_pct"]),
                "dap_abs_pct": float(row["dap_abs_pct"]),
                "mop_abs_pct": float(row["mop_abs_pct"]),
            }
        )
    return results


def format_prioritizer_examples(examples: list[dict[str, object]]) -> str:
    if not examples:
        return "Нет исторических примеров."

    lines: list[str] = []
    for ex in examples:
        lines.append(
            f"{ex['rank']}. Источник: {ex['source']} | «{ex['headline']}» → "
            f"|Δ%| urea={100 * float(ex['urea_abs_pct']):.2f}%, "
            f"dap={100 * float(ex['dap_abs_pct']):.2f}%, "
            f"mop={100 * float(ex['mop_abs_pct']):.2f}%"
        )
    return "\n".join(lines)
