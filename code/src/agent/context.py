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


def _history_turns(
    trajectory: list[dict[str, Any]],
    question: str,
) -> list[list[dict[str, Any]]]:
    """
    Group trajectory into API-safe turns.

    Each turn is either:
    - assistant (+ all matching tool results for its tool_calls), or
    - a follow-up user nudge (not the original question).

    Orphan tool messages (no preceding assistant tool_calls) are dropped.
    """
    turns: list[list[dict[str, Any]]] = []
    i = 0
    n = len(trajectory)
    while i < n:
        msg = trajectory[i]
        role = msg.get("role")
        if role == "assistant":
            turn = [msg]
            i += 1
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                expected = {
                    tc.get("id") for tc in tool_calls if isinstance(tc, dict) and tc.get("id")
                }
                while i < n and trajectory[i].get("role") == "tool" and expected:
                    tool_msg = trajectory[i]
                    tid = tool_msg.get("tool_call_id")
                    if tid in expected:
                        turn.append(tool_msg)
                        expected.discard(tid)
                    i += 1
                # Skip leftover orphan tools before the next assistant/user
                while i < n and trajectory[i].get("role") == "tool":
                    i += 1
            turns.append(turn)
        elif role == "user" and (msg.get("content") or "") != question:
            turns.append([msg])
            i += 1
        elif role == "tool":
            # Orphan tool at the start of a window — drop
            i += 1
        else:
            i += 1
    return turns


def _prune_turns(
    turns: list[list[dict[str, Any]]],
    max_msgs: int,
) -> list[dict[str, Any]]:
    """Keep newest turns without splitting an assistant/tool group."""
    if max_msgs <= 0:
        return []
    kept_rev: list[list[dict[str, Any]]] = []
    total = 0
    for turn in reversed(turns):
        turn_len = len(turn)
        if kept_rev and total + turn_len > max_msgs:
            break
        kept_rev.append(turn)
        total += turn_len
    flat: list[dict[str, Any]] = []
    for turn in reversed(kept_rev):
        flat.extend(turn)
    return flat


def build_messages_for_llm(
    system_prompt: str,
    question: str,
    trajectory: list[dict[str, Any]],
    state: AgentState,
    max_history_tool_msgs: int = 12,
) -> list[dict[str, Any]]:
    """
    Construct a pruned message list for the next LLM call.

    Keeps system + user question, a compact state summary, and a window of
    recent tool interactions instead of dumping the entire history.

    Pruning never splits an assistant message from its tool results, which
    would make SiliconFlow/OpenAI reject the request.
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

    recent = _prune_turns(_history_turns(trajectory, question), max_history_tool_msgs)

    for msg in recent:
        # Strip usage from what we send to the API
        clean = {k: v for k, v in msg.items() if k != "usage"}
        if clean.get("role") == "tool" and isinstance(clean.get("content"), str):
            clean["content"] = truncate_text(clean["content"], MAX_TOOL_RESULT_CHARS)
        messages.append(clean)
    return messages
