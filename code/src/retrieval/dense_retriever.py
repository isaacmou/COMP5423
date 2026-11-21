"""Dense retrieval with BGE encoder + FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config import CHUNKS_JSONL, DENSE_IDS, DENSE_INDEX, DENSE_MODEL_NAME, SNIPPET_CHARS
from src.retrieval.bm25_retriever import RawHit


class DenseRetriever:
    def __init__(
        self,
        index_path: Path = DENSE_INDEX,
        ids_path: Path = DENSE_IDS,
        model_name: Optional[str] = None,
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        if not self.index_path.exists() or not self.ids_path.exists():
            raise FileNotFoundError(
                f"Dense index missing ({self.index_path}). "
                "Run: python scripts/build_indexes.py --stage dense"
            )
        meta = json.loads(self.ids_path.read_text(encoding="utf-8"))
        self.chunk_ids: list[str] = meta["chunk_ids"]
        self.doc_ids: list[str] = meta["doc_ids"]
        self.model_name = model_name or meta.get("model") or DENSE_MODEL_NAME
        from src.retrieval.faiss_io import read_faiss_index

        self.index = read_faiss_index(self.index_path)
        self.model = SentenceTransformer(self.model_name)
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
        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        fetch_k = min(self.index.ntotal, max(top_k * 8, top_k))
        scores, indices = self.index.search(emb, fetch_k)
        hits: list[RawHit] = []
        seen: set[str] = set()
        for idx, score in zip(indices[0], scores[0]):
            if int(idx) < 0:
                continue
            doc_id = self.doc_ids[int(idx)]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            chunk_id = self.chunk_ids[int(idx)]
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
