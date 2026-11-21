"""Tool schemas and execution router for the agent."""

from __future__ import annotations

import json
from typing import Any, Optional

from src.config import DEFAULT_TOP_K
from src.corpus.loader import DocStore
from src.retrieval.unified import UnifiedRetriever
from src.tools.find_in_doc import format_find_result, run_find_in_doc
from src.tools.search import format_search_result, run_search
from src.tools.visit import format_visit_result, run_visit

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Unified retrieval over the fixed document collection. "
                "Returns ranked document IDs, scores, and text snippets. "
                "method: auto (RRF fusion of BM25+dense+instruct), bm25, dense, or instruct."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["auto", "bm25", "dense", "instruct"],
                        "description": "Internal retrieval method selection",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": (
                "Open a document by doc_id and return its (possibly truncated) full text "
                "and source URL metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document identifier from search results",
                    }
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_in_doc",
            "description": (
                "Search for a regex or literal pattern inside a known document and "
                "return matching snippets with local context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "pattern": {
                        "type": "string",
                        "description": "Regex or literal text to find",
                    },
                },
                "required": ["doc_id", "pattern"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(
        self,
        retriever: UnifiedRetriever,
        store: DocStore,
    ):
        self.retriever = retriever
        self.store = store
        self.call_count = 0
        self.last_search_payload: Optional[dict[str, Any]] = None

    def execute(self, name: str, arguments: dict[str, Any] | str) -> str:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}
        self.call_count += 1
        if name == "search":
            query = str(arguments.get("query", "")).strip()
            method = str(arguments.get("method", "auto") or "auto")
            top_k = int(arguments.get("top_k") or DEFAULT_TOP_K)
            payload = run_search(self.retriever, query=query, method=method, top_k=top_k)
            self.last_search_payload = payload
            return format_search_result(payload)
        if name == "visit":
            doc_id = str(arguments.get("doc_id", "")).strip()
            return format_visit_result(run_visit(self.store, doc_id))
        if name == "find_in_doc":
            doc_id = str(arguments.get("doc_id", "")).strip()
            pattern = str(arguments.get("pattern", "")).strip()
            return format_find_result(
                run_find_in_doc(self.store, doc_id=doc_id, pattern=pattern)
            )
        return json.dumps({"error": f"unknown tool: {name}"})
