"""Structured agent state separate from raw conversation history (Feature B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvidenceItem:
    doc_id: str
    snippet: str
    source: str  # search|visit|find_in_doc
    score: float = 0.0


@dataclass
class AgentState:
    question: str
    visited_docs: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    active_subqueries: list[str] = field(default_factory=list)
    candidate_answer: Optional[str] = None
    selected_skill: Optional[str] = None
    search_queries: list[str] = field(default_factory=list)
    weak_search_streak: int = 0
    step: int = 0

    def mark_visited(self, doc_id: str) -> None:
        if doc_id and doc_id not in self.visited_docs:
            self.visited_docs.append(doc_id)

    def add_evidence(
        self,
        doc_id: str,
        snippet: str,
        source: str,
        score: float = 0.0,
        max_items: int = 12,
    ) -> None:
        # Deduplicate near-identical snippets for the same doc
        key = (doc_id, snippet[:160])
        existing = {(e.doc_id, e.snippet[:160]) for e in self.evidence}
        if key in existing:
            return
        self.evidence.append(
            EvidenceItem(doc_id=doc_id, snippet=snippet, source=source, score=score)
        )
        if len(self.evidence) > max_items:
            # Keep highest-score / most recent
            self.evidence = sorted(
                self.evidence, key=lambda e: e.score, reverse=True
            )[:max_items]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "selected_skill": self.selected_skill,
            "visited_docs": self.visited_docs[-20:],
            "active_subqueries": self.active_subqueries[-10:],
            "search_queries": self.search_queries[-10:],
            "candidate_answer": self.candidate_answer,
            "evidence": [
                {
                    "doc_id": e.doc_id,
                    "source": e.source,
                    "score": e.score,
                    "snippet": e.snippet[:240],
                }
                for e in self.evidence[:8]
            ],
            "step": self.step,
            "weak_search_streak": self.weak_search_streak,
        }
