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
    "based on",
    "according to",
    "best match",
    "evidence gathered",
    "the answer is",
    "final answer",
)

# Refusal / meta text is never a BrowseComp final span.
_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "insufficient",
    "cannot identify",
    "can't identify",
    "unable to",
    "not enough evidence",
    "cannot determine",
    "can't determine",
)

_CORP_SUFFIX_RE = re.compile(
    r"(?:,?\s+)?\b(?:ltd\.?|limited|inc\.?|incorporated|llc|corp\.?|corporation|co\.?|plc)\s*$",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_DATE_DMY_RE = re.compile(
    r"^(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<month>[A-Za-z]+)\s+"
    r"(?P<year>\d{4})$",
    re.IGNORECASE,
)
_DATE_MDY_RE = re.compile(
    r"^(?P<month>[A-Za-z]+)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+"
    r"(?P<year>\d{4})$",
    re.IGNORECASE,
)

_PROSE_LEAD_RE = re.compile(
    r"^(?:based on(?: the)?(?: evidence)?(?: gathered)?[,:]?\s*|"
    r"according to(?: the)?(?: evidence)?[,:]?\s*|"
    r"the (?:best )?match is\s*|"
    r"the answer is\s*|"
    r"final answer\s*[:\-]?\s*|"
    r"answer\s*[:\-]?\s*)",
    re.IGNORECASE,
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


def _compact_long_date(answer: str) -> str:
    text = answer.strip()
    for pattern in (_DATE_DMY_RE, _DATE_MDY_RE):
        m = pattern.match(text)
        if not m:
            continue
        month_key = m.group("month").lower()
        month = _MONTHS.get(month_key)
        if not month:
            continue
        day = int(m.group("day"))
        year = int(m.group("year"))
        if not (1 <= day <= 31 and 1000 <= year <= 2100):
            continue
        return f"{day:02d}/{month:02d}/{year}"
    return answer


def _strip_corp_suffix(answer: str) -> str:
    cleaned = _CORP_SUFFIX_RE.sub("", answer).strip().rstrip(",.")
    # Keep suffix if stripping would leave a single short token.
    if len(cleaned.split()) >= 2:
        return cleaned
    return answer.strip()


def normalize_final_answer(text: str) -> str:
    """Polish model output into a short EM-friendly answer span."""
    answer = (text or "").strip()
    if not answer:
        return ""
    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    answer = _DSML_MARK_RE.sub("", answer).strip()
    answer = answer.replace("**", "").strip()
    answer = _PROSE_LEAD_RE.sub("", answer).strip()
    # If a long sentence remains, keep the last short clause after "is".
    if " is " in answer.lower() and len(answer) > 40:
        parts = re.split(r"\bis\b", answer, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[1].strip():
            answer = parts[1].strip().rstrip(".")
    # Whole-answer wrapping quotes
    if len(answer) >= 2 and (
        (answer[0] == answer[-1] == '"')
        or (answer[0] == answer[-1] == "'")
    ):
        answer = answer[1:-1].strip()
    answer = _strip_corp_suffix(answer)
    answer = _compact_long_date(answer)
    return answer.strip()


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
    if "**" in answer:
        return False
    lower = answer.lower()
    if any(marker in lower for marker in _PLANNING_MARKERS):
        return False
    if any(marker in lower for marker in _REFUSAL_MARKERS):
        return False
    # Allow paper titles / longer phrases; still reject essay-length dumps.
    # (80 was too tight: gold titles ~82 chars were discarded as empty.)
    if len(answer) > 160:
        return False
    if answer.count("\n") >= 1:
        return False
    if answer.count(".") >= 2:
        return False
    if tool_calls_made < min_tool_calls:
        if len(answer) > 64:
            return False
        return False
    return True


def answer_grounded_in_evidence(answer: str, evidence_texts: list[str]) -> bool:
    """
    Require the normalized answer (or all significant tokens) to appear in
    visited evidence text so the model cannot invent ungrounded names.
    """
    from src.utils_em import normalize_answer

    norm = normalize_answer(answer)
    if not norm:
        return False
    blob = normalize_answer("\n".join(evidence_texts or []))
    if not blob:
        return False
    if norm in blob:
        return True
    # Multi-token names: require every token length>=3 to appear
    tokens = [t for t in norm.split() if len(t) >= 3]
    if len(tokens) >= 2 and all(t in blob for t in tokens):
        return True
    # Short numeric / percent answers: substring match on raw-ish blob
    if len(norm) <= 8 and norm in blob:
        return True
    return False
