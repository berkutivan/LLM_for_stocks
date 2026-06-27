from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class HistoricalFact:
    source: str
    headline: str
    date_first_seen: str
    streaks: dict
    fact_text: str
    index_text: str


class FactVectorStore:
    def __init__(self, index_dir: Path, embedding_model: str) -> None:
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model
        self.facts: list[HistoricalFact] = []
        self.doc_embeddings: np.ndarray | None = None
        self._model: SentenceTransformer | None = None
        self._encode_lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _meta_path(self, facts_path: Path) -> Path:
        import hashlib

        digest = hashlib.md5(str(facts_path.resolve()).encode()).hexdigest()[:8]
        return self.index_dir / f"facts_index_{digest}.pkl"

    @staticmethod
    def load_facts(facts_path: Path) -> list[HistoricalFact]:
        facts: list[HistoricalFact] = []
        with open(facts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                facts.append(
                    HistoricalFact(
                        source=raw["source"],
                        headline=raw["headline"],
                        date_first_seen=raw["date_first_seen"],
                        streaks=raw["streaks"],
                        fact_text=raw["fact_text"],
                        index_text=raw["index_text"],
                    )
                )
        return facts

    def build_or_load(self, facts_path: Path, force_rebuild: bool = False) -> None:
        import pickle

        self.index_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._meta_path(facts_path)
        facts_mtime = facts_path.stat().st_mtime

        if not force_rebuild and meta_path.exists():
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            if (
                data.get("facts_mtime") == facts_mtime
                and data.get("embedding_model") == self.embedding_model_name
            ):
                self.facts = data["facts"]
                self.doc_embeddings = data["doc_embeddings"]
                return

        self.facts = self.load_facts(facts_path)
        texts = [fact.index_text for fact in self.facts]
        self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "facts_mtime": facts_mtime,
                    "embedding_model": self.embedding_model_name,
                    "facts": self.facts,
                    "doc_embeddings": self.doc_embeddings,
                },
                f,
            )

    def encode_query(self, query: str) -> np.ndarray:
        with self._encode_lock:
            return self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    def dense_search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        if self.doc_embeddings is None or not self.facts:
            return []
        q = self.encode_query(query)
        scores = self.doc_embeddings @ q
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_idx]
