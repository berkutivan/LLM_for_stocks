from __future__ import annotations

from pathlib import Path

import numpy as np

from src.rag.store import FactVectorStore, HistoricalFact


def _mmr_select(
    candidate_indices: list[int],
    candidate_scores: list[float],
    doc_embeddings: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    if not candidate_indices:
        return []
    selected: list[int] = []
    remaining = candidate_indices.copy()
    remaining_scores = {idx: score for idx, score in zip(candidate_indices, candidate_scores)}

    while remaining and len(selected) < top_k:
        best_idx = None
        best_mmr = -1e9
        for idx in remaining:
            relevance = remaining_scores[idx]
            if not selected:
                redundancy = 0.0
            else:
                sims = [float(doc_embeddings[idx] @ doc_embeddings[s]) for s in selected]
                redundancy = max(sims)
            mmr = lambda_mult * relevance - (1 - lambda_mult) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def _format_fact_for_prompt(fact: HistoricalFact, rank: int) -> str:
    return (
        f"### Прецедент {rank}\n"
        f"Источник: {fact.source} | Заголовок: {fact.headline}\n"
        f"После новости: {fact.fact_text}"
    )


class HistoricalFactRetriever:
    def __init__(
        self,
        facts_path: Path,
        index_dir: Path,
        embedding_model: str,
        top_k: int,
        retrieve_candidates: int = 20,
    ) -> None:
        self.facts_path = facts_path
        self.top_k = top_k
        self.retrieve_candidates = retrieve_candidates
        self.store = FactVectorStore(index_dir, embedding_model)

    def initialize(self, force_rebuild: bool = False) -> None:
        self.store.build_or_load(self.facts_path, force_rebuild=force_rebuild)

    def retrieve(self, source: str, headline: str) -> list[str]:
        query = f"{source} | {headline}"
        candidates = self.store.dense_search(query, self.retrieve_candidates)
        if not candidates:
            return []

        boosted: list[tuple[int, float]] = []
        for idx, score in candidates:
            fact = self.store.facts[idx]
            boost = score + (0.05 if fact.source == source else 0.0)
            boosted.append((idx, boost))

        boosted.sort(key=lambda x: x[1], reverse=True)
        cand_indices = [idx for idx, _ in boosted]
        cand_scores = [score for _, score in boosted]

        if self.store.doc_embeddings is None:
            return []

        selected = _mmr_select(
            cand_indices,
            cand_scores,
            self.store.doc_embeddings,
            self.top_k,
        )

        return [
            _format_fact_for_prompt(self.store.facts[idx], rank)
            for rank, idx in enumerate(selected, start=1)
        ]
