"""Context management: prune, dedupe, and compress tool outputs before LLM calls."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.state import AgentState
from src.config import MAX_EVIDENCE_ITEMS, MAX_TOOL_RESULT_CHARS


def truncate_text(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]..."


def compress_tool_result(name: str, content: str, state: AgentState) -> str:
    """Shrink tool payloads and update structured state."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return truncate_text(content)

    if name == "search" and isinstance(data, dict):
        query = data.get("query", "")
        if query and query not in state.search_queries:
            state.search_queries.append(query)
            state.active_subqueries.append(query)
        results = data.get("results") or []
        if not results:
            state.weak_search_streak += 1
        else:
            state.weak_search_streak = 0
        compact = []
        seen_docs: set[str] = set()
        for r in results:
            doc_id = str(r.get("doc_id", ""))
            if not doc_id or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            snippet = str(r.get("snippet", ""))[:400]
            score = float(r.get("score") or 0.0)
            state.add_evidence(
                doc_id=doc_id,
                snippet=snippet,
                source="search",
                score=score,
                max_items=MAX_EVIDENCE_ITEMS,
            )
            compact.append(
                {
                    "doc_id": doc_id,
                    "score": round(score, 4),
                    "method": r.get("method"),
                    "snippet": snippet,
                }
            )
        return truncate_text(
            json.dumps(
                {
                    "query": query,
                    "method": data.get("method"),
                    "results": compact,
                },
                ensure_ascii=False,
            )
        )

    if name == "visit" and isinstance(data, dict):
        doc_id = str(data.get("doc_id", ""))
        if data.get("error"):
            return json.dumps(data, ensure_ascii=False)
        state.mark_visited(doc_id)
        text = str(data.get("text", ""))
        # Keep a short evidence excerpt
        excerpt = re.sub(r"\s+", " ", text).strip()[:500]
        state.add_evidence(
            doc_id=doc_id,
            snippet=excerpt,
            source="visit",
            score=1.0,
            max_items=MAX_EVIDENCE_ITEMS,
        )
        return truncate_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "url": data.get("url", ""),
                    "truncated": data.get("truncated", False),
                    "text": text,
                },
                ensure_ascii=False,
            )
        )

    if name == "find_in_doc" and isinstance(data, dict):
        doc_id = str(data.get("doc_id", ""))
        matches = data.get("matches") or []
        for m in matches[:5]:
            state.add_evidence(
                doc_id=doc_id,
                snippet=str(m.get("snippet", ""))[:400],
                source="find_in_doc",
                score=0.8,
                max_items=MAX_EVIDENCE_ITEMS,
            )
        return truncate_text(json.dumps(data, ensure_ascii=False))

    return truncate_text(content)


def build_messages_for_llm(
    system_prompt: str,
    question: str,
    trajectory: list[dict[str, Any]],
    state: AgentState,
    max_history_tool_msgs: int = 8,
) -> list[dict[str, Any]]:
    """
    Construct a pruned message list for the next LLM call.

    Keeps system + user question, a compact state summary, and a window of
    recent tool interactions instead of dumping the entire history.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {
            "role": "system",
            "content": (
                "Internal search state (not conversation history):\n"
                + json.dumps(state.summary_dict(), ensure_ascii=False)
            ),
        },
    ]

    # Collect recent assistant/tool pairs from trajectory (skip initial system/user)
    recent: list[dict[str, Any]] = []
    for msg in trajectory:
        role = msg.get("role")
        if role in {"assistant", "tool"}:
            recent.append(msg)
    # Keep only the last N tool-related messages
    if len(recent) > max_history_tool_msgs:
        recent = recent[-max_history_tool_msgs:]

    for msg in recent:
        # Strip usage from what we send to the API
        clean = {k: v for k, v in msg.items() if k != "usage"}
        if clean.get("role") == "tool" and isinstance(clean.get("content"), str):
            clean["content"] = truncate_text(clean["content"], MAX_TOOL_RESULT_CHARS)
        messages.append(clean)
    return messages
