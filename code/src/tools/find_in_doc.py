"""Find matching snippets inside a document."""

from __future__ import annotations

import json
import re
from typing import Any

from src.config import FIND_CONTEXT_CHARS, FIND_MAX_MATCHES
from src.corpus.loader import DocStore


def run_find_in_doc(
    store: DocStore,
    doc_id: str,
    pattern: str,
    max_matches: int = FIND_MAX_MATCHES,
    context_chars: int = FIND_CONTEXT_CHARS,
) -> dict[str, Any]:
    doc = store.get(doc_id)
    if doc is None:
        return {"doc_id": doc_id, "error": "document not found"}
    text = doc.text
    matches: list[dict[str, Any]] = []
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # fall back to literal substring search
        literal = pattern.lower()
        start = 0
        lower = text.lower()
        while len(matches) < max_matches:
            idx = lower.find(literal, start)
            if idx < 0:
                break
            left = max(0, idx - context_chars)
            right = min(len(text), idx + len(pattern) + context_chars)
            matches.append(
                {
                    "start": idx,
                    "snippet": text[left:right],
                }
            )
            start = idx + max(1, len(pattern))
        return {
            "doc_id": doc_id,
            "pattern": pattern,
            "mode": "literal",
            "matches": matches,
            "match_count": len(matches),
        }

    for m in regex.finditer(text):
        left = max(0, m.start() - context_chars)
        right = min(len(text), m.end() + context_chars)
        matches.append({"start": m.start(), "snippet": text[left:right]})
        if len(matches) >= max_matches:
            break
    return {
        "doc_id": doc_id,
        "pattern": pattern,
        "mode": "regex",
        "matches": matches,
        "match_count": len(matches),
    }


def format_find_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
