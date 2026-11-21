"""Unified search tool wrapper."""

from __future__ import annotations

import json
from typing import Any

from src.config import DEFAULT_TOP_K
from src.retrieval.unified import UnifiedRetriever


def run_search(
    retriever: UnifiedRetriever,
    query: str,
    method: str = "auto",
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    hits = retriever.search(query=query, method=method, top_k=top_k)  # type: ignore[arg-type]
    return {
        "query": query,
        "method": method,
        "results": [h.to_dict() for h in hits],
    }


def format_search_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
