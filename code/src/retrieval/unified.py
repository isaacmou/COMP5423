"""Unified retrieval facade: one search entrypoint with internal methods/fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

from src.config import DEFAULT_TOP_K
from src.retrieval.bm25_retriever import BM25Retriever, RawHit

MethodName = Literal["auto", "bm25", "dense", "instruct"]


@dataclass
class SearchHit:
    doc_id: str
    score: float
    snippet: str
    method: str
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RawHit]],
    method_names: list[str],
    top_k: int,
    k: int = 60,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    best: dict[str, RawHit] = {}
    sources: dict[str, set[str]] = {}
    for method, hits in zip(method_names, ranked_lists):
        for rank, hit in enumerate(hits):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(hit.doc_id, set()).add(method)
            prev = best.get(hit.doc_id)
            if prev is None or hit.score > prev.score:
                best[hit.doc_id] = hit
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out: list[SearchHit] = []
    for doc_id, fused in ordered:
        h = best[doc_id]
        methods = "+".join(sorted(sources[doc_id]))
        out.append(
            SearchHit(
                doc_id=doc_id,
                score=float(fused),
                snippet=h.snippet,
                method=f"rrf({methods})",
                chunk_id=h.chunk_id,
            )
        )
    return out


class UnifiedRetriever:
    """Exposes all three retrieval categories behind a single interface."""

    def __init__(
        self,
        enable_bm25: bool = True,
        enable_dense: bool = True,
        enable_instruct: bool = True,
        lazy: bool = True,
    ):
        self.enable_bm25 = enable_bm25
        self.enable_dense = enable_dense
        self.enable_instruct = enable_instruct
        self._bm25: Optional[BM25Retriever] = None
        self._dense = None
        self._instruct = None
        self._lazy = lazy
        if not lazy:
            self._ensure_loaded()

    def available_methods(self) -> list[str]:
        methods = []
        if self.enable_bm25:
            methods.append("bm25")
        if self.enable_dense:
            methods.append("dense")
        if self.enable_instruct:
            methods.append("instruct")
        return methods

    def _ensure_loaded(self) -> None:
        if self.enable_bm25 and self._bm25 is None:
            try:
                self._bm25 = BM25Retriever()
            except FileNotFoundError as exc:
                print(f"Warning: BM25 unavailable: {exc}")
                self.enable_bm25 = False
        if self.enable_dense and self._dense is None:
            try:
                from src.retrieval.dense_retriever import DenseRetriever

                self._dense = DenseRetriever()
            except FileNotFoundError as exc:
                print(f"Warning: dense retriever unavailable: {exc}")
                self.enable_dense = False
        if self.enable_instruct and self._instruct is None:
            try:
                from src.retrieval.instruct_retriever import InstructRetriever

                self._instruct = InstructRetriever()
            except FileNotFoundError as exc:
                print(f"Warning: instruct retriever unavailable: {exc}")
                self.enable_instruct = False
        if not any([self.enable_bm25, self.enable_dense, self.enable_instruct]):
            raise RuntimeError(
                "No retrieval backends available. Build indexes with "
                "`python scripts/build_indexes.py`."
            )

    def search(
        self,
        query: str,
        method: MethodName = "auto",
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchHit]:
        self._ensure_loaded()
        method = (method or "auto").lower()  # type: ignore[assignment]

        # If a requested backend is missing, fall back to auto fusion.
        if method == "bm25" and (not self.enable_bm25 or self._bm25 is None):
            method = "auto"  # type: ignore[assignment]
        elif method == "dense" and (not self.enable_dense or self._dense is None):
            method = "auto"  # type: ignore[assignment]
        elif method == "instruct" and (
            not self.enable_instruct or self._instruct is None
        ):
            method = "auto"  # type: ignore[assignment]

        if method == "bm25":
            return [
                SearchHit(
                    doc_id=h.doc_id,
                    score=h.score,
                    snippet=h.snippet,
                    method="bm25",
                    chunk_id=h.chunk_id,
                )
                for h in self._bm25.search(query, top_k=top_k)  # type: ignore[union-attr]
            ]
        if method == "dense":
            return [
                SearchHit(
                    doc_id=h.doc_id,
                    score=h.score,
                    snippet=h.snippet,
                    method="dense",
                    chunk_id=h.chunk_id,
                )
                for h in self._dense.search(query, top_k=top_k)  # type: ignore[union-attr]
            ]
        if method == "instruct":
            return [
                SearchHit(
                    doc_id=h.doc_id,
                    score=h.score,
                    snippet=h.snippet,
                    method="instruct",
                    chunk_id=h.chunk_id,
                )
                for h in self._instruct.search(query, top_k=top_k)  # type: ignore[union-attr]
            ]

        # auto: RRF fusion of whatever is enabled
        lists: list[list[RawHit]] = []
        names: list[str] = []
        fetch = max(top_k * 3, top_k)
        if self.enable_bm25 and self._bm25 is not None:
            lists.append(self._bm25.search(query, top_k=fetch))
            names.append("bm25")
        if self.enable_dense and self._dense is not None:
            lists.append(self._dense.search(query, top_k=fetch))
            names.append("dense")
        if self.enable_instruct and self._instruct is not None:
            lists.append(self._instruct.search(query, top_k=fetch))
            names.append("instruct")
        if not lists:
            raise RuntimeError("No retrieval backends enabled")
        if len(lists) == 1:
            return [
                SearchHit(
                    doc_id=h.doc_id,
                    score=h.score,
                    snippet=h.snippet,
                    method=names[0],
                    chunk_id=h.chunk_id,
                )
                for h in lists[0][:top_k]
            ]
        return reciprocal_rank_fusion(lists, names, top_k=top_k)
