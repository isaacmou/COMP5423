"""Open a document by id and return capped full text."""

from __future__ import annotations

import json
from typing import Any

from src.config import VISIT_MAX_CHARS
from src.corpus.loader import DocStore


def run_visit(store: DocStore, doc_id: str, max_chars: int = VISIT_MAX_CHARS) -> dict[str, Any]:
    doc = store.get(doc_id)
    if doc is None:
        return {"doc_id": doc_id, "error": "document not found"}
    text = doc.text
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n...[truncated]..."
    return {
        "doc_id": doc.doc_id,
        "url": doc.url,
        "truncated": truncated,
        "text": text,
    }


def format_visit_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
