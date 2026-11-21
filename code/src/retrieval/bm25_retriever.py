"""Sparse BM25 retrieval over chunked corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.config import BM25_DIR, CHUNKS_JSONL, SNIPPET_CHARS


@dataclass
class RawHit:
    doc_id: str
    chunk_id: str
    score: float
    snippet: str


class BM25Retriever:
    def __init__(self, index_dir: Path = BM25_DIR):
        import bm25s

        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_dir}. "
                "Run: python scripts/build_indexes.py --stage bm25"
            )
        self.retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)
        mapping = json.loads((self.index_dir / "id_map.json").read_text(encoding="utf-8"))
        self.chunk_ids: list[str] = mapping["chunk_ids"]
        self.doc_ids: list[str] = mapping["doc_ids"]
        self._chunk_text_cache: dict[str, str] = {}
        self._load_chunk_texts()

    def _load_chunk_texts(self) -> None:
        if not CHUNKS_JSONL.exists():
            return
        with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self._chunk_text_cache[row["chunk_id"]] = row["text"]

    def search(self, query: str, top_k: int = 5) -> list[RawHit]:
        import bm25s

        tokens = bm25s.tokenize([query], stopwords="en")
        # Retrieve more chunks then collapse to unique docs
        fetch_k = min(len(self.chunk_ids), max(top_k * 8, top_k))
        results, scores = self.retriever.retrieve(tokens, k=fetch_k)
        hits: list[RawHit] = []
        seen_docs: set[str] = set()
        for idx, score in zip(results[0], scores[0]):
            idx_i = int(idx)
            doc_id = self.doc_ids[idx_i]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            chunk_id = self.chunk_ids[idx_i]
            text = self._chunk_text_cache.get(chunk_id, "")
            snippet = text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")
            hits.append(
                RawHit(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=float(score),
                    snippet=snippet,
                )
            )
            if len(hits) >= top_k:
                break
        return hits
