"""Helpers for validating LLM answers and recovering leaked tool-call markup."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

# DeepSeek / SiliconFlow sometimes dumps this markup into `content`
# instead of returning structured OpenAI tool_calls.
# Markers use ASCII `|` or fullwidth `｜`.
_P = r"[|｜]"
_DSML_INVOKE_RE = re.compile(
    rf"<{_P}?DSML{_P}?invoke\s+name=\"([^\"]+)\">\s*"
    rf"<{_P}?DSML{_P}?parameter\s+name=\"arguments\"[^>]*>\s*(\{{.*?\}})\s*"
    rf"</{_P}?DSML{_P}?parameter>\s*</{_P}?DSML{_P}?invoke>",
    re.DOTALL | re.IGNORECASE,
)

_DSML_MARK_RE = re.compile(rf"<{_P}?/?DSML{_P}?[^>]*>", re.IGNORECASE)

_PLANNING_MARKERS = (
    "decomposition",
    "let me search",
    "let me look",
    "i need to search",
    "i will search",
    "search for candidates",
    "clues listed",
    "tool_calls",
    "｜dsml｜",
    "|dsml|",
)


def parse_dsml_tool_calls(content: Optional[str]) -> list[dict[str, Any]]:
    """Extract tool calls from DSML XML leaked into assistant content."""
    if not content:
        return []
    calls: list[dict[str, Any]] = []
    for match in _DSML_INVOKE_RE.finditer(content):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        try:
            json.loads(raw_args)
        except json.JSONDecodeError:
            continue
        calls.append(
            {
                "id": f"dsml_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": raw_args},
            }
        )
    return calls


def looks_like_tool_markup(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if _DSML_MARK_RE.search(text):
        return True
    if "tool_calls" in lower and ("invoke" in lower or "parameter" in lower):
        return True
    return False


def is_plausible_final_answer(
    text: str,
    *,
    min_tool_calls: int = 1,
    tool_calls_made: int = 0,
) -> bool:
    """Reject planning prose / leaked markup masquerading as the answer."""
    answer = (text or "").strip()
    if not answer:
        return False
    if looks_like_tool_markup(answer):
        return False
    lower = answer.lower()
    if any(marker in lower for marker in _PLANNING_MARKERS):
        return False
    # Final answers for this task are short entities / phrases.
    if len(answer) > 160:
        return False
    if answer.count("\n") >= 2:
        return False
    if tool_calls_made < min_tool_calls:
        # Without evidence, only accept a very short candidate (rare).
        if len(answer) > 64:
            return False
        return False
    return True


def normalize_final_answer(text: str) -> str:
    answer = (text or "").strip()
    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    # Drop accidental trailing markup if any slipped through.
    answer = _DSML_MARK_RE.sub("", answer).strip()
    return answer
