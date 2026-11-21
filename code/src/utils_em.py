"""Normalized exact-match helper aligned with the course scoring note."""

from __future__ import annotations

import re

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)
