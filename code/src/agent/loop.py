"""Iterative tool-using search agent (Basic + Feature A + Feature B)."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.agent.context import build_messages_for_llm, compress_tool_result
from src.agent.response_utils import (
    is_plausible_final_answer,
    normalize_final_answer,
    parse_dsml_tool_calls,
)
from src.agent.skills_loader import (
    Skill,
    auto_select_skill,
    catalog_text,
    discover_skills,
    load_skill_body,
    maybe_upgrade_skill,
)
from src.agent.state import AgentState
from src.config import MAX_AGENT_STEPS, SKILLS_DIR
from src.corpus.loader import DocStore
from src.llm.siliconflow import SiliconFlowClient
from src.retrieval.unified import UnifiedRetriever
from src.tools.registry import TOOL_DEFINITIONS, ToolExecutor

ProgressCallback = Callable[[str], None]

# Keep tool-call budget efficient for relative scoring.
_MAX_SEARCHES_PER_STEP = 2
_MAX_AUTO_VISITS = 1


BASE_SYSTEM = """You are a skill-augmented search agent answering questions over a FIXED local document collection.
You must use tools to gather evidence. Do not invent documents or URLs.

Available tools:
- search(query, method?, top_k?): unified retrieval (BM25 / dense / instruct / auto fusion). Returns doc_ids, scores, snippets.
- visit(doc_id): open a document's text.
- find_in_doc(doc_id, pattern): locate text inside a document.

Workflow:
1) Decide whether a skill is useful from the skill catalog (names+descriptions only).
2) Call tools iteratively via the API tool-calling interface: search, then visit promising docs, reformulate queries based on evidence.
3) When confident, stop calling tools and reply with ONLY the final short answer string (no explanation).

Rules:
- Prefer method="auto" unless recovery requires switching methods.
- Avoid repeating the exact same search query.
- After a useful search, visit the single best unvisited doc_id before searching again.
- At most two search calls in one turn; never issue many parallel searches.
- Prefer visiting a few high-value documents over many shallow searches.
- Search one distinctive atomic clue at a time (rare phrase, year+venue, quoted span), not a long paraphrase of the whole question.
- If the question contains a quoted phrase, search that quote early.
- Do not commit to a person/name until at least two independent constraints are supported by visited text.
- If evidence conflicts or a candidate fails a constraint, discard it and search a different clue instead of more visits of the same cluster.
- For percentage/number answers, use find_in_doc with the question's target phrase (e.g. heard/familiar/knew) before committing; do not pick a random % from the same article.
- For date answers, prefer the compact numeric form that appears in documents (DD/MM/YYYY). Use find_in_doc for day/month/year digits when needed.
- For rare quoted spans / proper-name clues, prefer method="bm25" once for an exact phrase hit.
- Keep answers concise and match the expected answer form (entity, date, number, short phrase). No markdown, no company suffixes like Ltd/Inc unless essential.
- Cross-check the candidate against every clue in the question before stopping.
- Never write DSML/XML tool markup in your message content. Use structured tool calls only.
- Do not output planning essays, clue lists, or "let me search" prose as the final answer.
"""

_CONTINUE_NUDGE = (
    "That was not a valid final answer. Use the structured tool interface to "
    "search/visit documents (do not write XML/DSML). When done, reply with ONLY "
    "the short final answer string."
)

_VISIT_NUDGE = (
    "You already have unvisited search hits. Call visit (or find_in_doc) on the "
    "top doc_id before another search. Then answer with ONLY the short final "
    "answer when evidence is enough."
)

_QUOTE_RE = re.compile(r'"([^"]{8,})"')


@dataclass
class AgentResult:
    answer: str
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    total_tokens: int = 0
    selected_skill: Optional[str] = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    state: Optional[AgentState] = None


class SearchAgent:
    def __init__(
        self,
        llm: Optional[SiliconFlowClient] = None,
        retriever: Optional[UnifiedRetriever] = None,
        store: Optional[DocStore] = None,
        skills: Optional[list[Skill]] = None,
        max_steps: int = MAX_AGENT_STEPS,
        enable_dense: bool = True,
        enable_instruct: bool = True,
        enable_bm25: bool = True,
    ):
        self.llm = llm or SiliconFlowClient()
        self.retriever = retriever or UnifiedRetriever(
            enable_bm25=enable_bm25,
            enable_dense=enable_dense,
            enable_instruct=enable_instruct,
            lazy=True,
        )
        self.store = store or DocStore()
        self.tools = ToolExecutor(self.retriever, self.store)
        self.skills = skills if skills is not None else discover_skills(SKILLS_DIR)
        self.max_steps = max_steps
        self._skills_disabled = False

    def _system_prompt(self, skill_body: Optional[str], skill_name: Optional[str]) -> str:
        parts = [
            BASE_SYSTEM,
            "",
            "Skill catalog (load full instructions only when selected):",
            catalog_text(self.skills),
        ]
        if skill_name and skill_body:
            parts.extend(
                [
                    "",
                    f"Loaded skill `{skill_name}` instructions:",
                    skill_body,
                ]
            )
        return "\n".join(parts)

    @staticmethod
    def _tool_name(tc: dict[str, Any]) -> str:
        return str((tc.get("function") or {}).get("name") or "")

    @staticmethod
    def _tool_args(tc: dict[str, Any]) -> dict[str, Any]:
        raw = (tc.get("function") or {}).get("arguments") or "{}"
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _normalize_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        state: AgentState,
        progress: ProgressCallback,
    ) -> list[dict[str, Any]]:
        """Cap parallel searches and drop duplicate/near-duplicate queries."""
        kept: list[dict[str, Any]] = []
        search_count = 0
        seen_queries = {q.strip().lower() for q in state.search_queries if q}
        for tc in tool_calls:
            name = self._tool_name(tc)
            if name == "search":
                query = str(self._tool_args(tc).get("query", "")).strip()
                key = query.lower()
                if not query:
                    continue
                if key in seen_queries:
                    progress(f"Skipping duplicate search: {query[:80]}")
                    continue
                if search_count >= _MAX_SEARCHES_PER_STEP:
                    progress("Capping searches this step; dropping extra search")
                    continue
                seen_queries.add(key)
                search_count += 1
                kept.append(tc)
            elif name in {"visit", "find_in_doc"}:
                kept.append(tc)
            else:
                kept.append(tc)
        return kept

    def _make_visit_call(self, doc_id: str) -> dict[str, Any]:
        return {
            "id": f"auto_visit_{uuid.uuid4().hex[:10]}",
            "type": "function",
            "function": {
                "name": "visit",
                "arguments": json.dumps({"doc_id": doc_id}),
            },
        }

    def _unvisited_from_last_search(self, state: AgentState) -> list[str]:
        payload = self.tools.last_search_payload or {}
        results = payload.get("results") or []
        out: list[str] = []
        for row in results:
            doc_id = str(row.get("doc_id") or "").strip()
            if doc_id and doc_id not in state.visited_docs and doc_id not in out:
                out.append(doc_id)
        if out:
            return out
        # Fall back to evidence docs that were never opened.
        for item in state.evidence:
            if item.doc_id and item.doc_id not in state.visited_docs and item.doc_id not in out:
                out.append(item.doc_id)
        return out

    def _append_auto_visits_after_search(
        self,
        executed: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        state: AgentState,
        progress: ProgressCallback,
    ) -> None:
        names = [self._tool_name(tc) for tc in executed]
        if "search" not in names:
            return
        if "visit" in names or "find_in_doc" in names:
            return
        doc_ids = self._unvisited_from_last_search(state)[:_MAX_AUTO_VISITS]
        if not doc_ids:
            return
        auto_calls = [self._make_visit_call(doc_id) for doc_id in doc_ids]
        progress(f"Auto-visiting {len(auto_calls)} doc(s) after search")
        if trajectory and trajectory[-1].get("role") == "assistant":
            existing = list(trajectory[-1].get("tool_calls") or [])
            existing.extend(auto_calls)
            trajectory[-1]["tool_calls"] = existing
        for tc in auto_calls:
            self._run_one_tool(tc, trajectory, state, progress)

    def _run_one_tool(
        self,
        tc: dict[str, Any],
        trajectory: list[dict[str, Any]],
        state: AgentState,
        progress: ProgressCallback,
    ) -> None:
        name = self._tool_name(tc)
        raw_args = (tc.get("function") or {}).get("arguments") or "{}"
        progress(f"Tool call: {name}({str(raw_args)[:120]})")
        args = self._tool_args(tc)
        raw_result = self.tools.execute(name, args)
        compressed = compress_tool_result(name, raw_result, state)
        trajectory.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": name,
                "content": compressed,
            }
        )

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        state: AgentState,
        progress: ProgressCallback,
    ) -> bool:
        """Execute tool calls. Returns True if at least one tool ran."""
        normalized = self._normalize_tool_calls(tool_calls, state, progress)
        if not normalized:
            # Ensure we do not leave an assistant tool_calls message without tools.
            if trajectory and trajectory[-1].get("role") == "assistant":
                trajectory[-1].pop("tool_calls", None)
            return False
        if trajectory and trajectory[-1].get("role") == "assistant":
            trajectory[-1]["tool_calls"] = normalized
        before = self.tools.call_count
        for tc in normalized:
            self._run_one_tool(tc, trajectory, state, progress)
        self._append_auto_visits_after_search(normalized, trajectory, state, progress)
        return self.tools.call_count > before

    def _has_unvisited_hits(self, state: AgentState) -> bool:
        return bool(self._unvisited_from_last_search(state))

    @staticmethod
    def _extract_quoted_clue(question: str) -> Optional[str]:
        match = _QUOTE_RE.search(question or "")
        if not match:
            return None
        clue = match.group(1).strip()
        return clue if len(clue) >= 8 else None

    def _make_search_call(
        self, query: str, *, method: str = "auto"
    ) -> dict[str, Any]:
        return {
            "id": f"auto_search_{uuid.uuid4().hex[:10]}",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": json.dumps(
                    {"query": query, "method": method, "top_k": 7}
                ),
            },
        }

    def _bootstrap_quoted_search(
        self,
        question: str,
        tool_calls: list[dict[str, Any]],
        step: int,
        progress: ProgressCallback,
    ) -> list[dict[str, Any]]:
        """On step 1, ensure a distinctive quoted span is searched if present."""
        if step != 1:
            return tool_calls
        clue = self._extract_quoted_clue(question)
        if not clue:
            return tool_calls
        clue_l = clue.lower()
        for tc in tool_calls:
            if self._tool_name(tc) != "search":
                continue
            q = str(self._tool_args(tc).get("query", "")).lower()
            if clue_l in q or q in clue_l:
                return tool_calls
        progress(f"Bootstrap BM25 search for quoted clue: {clue[:80]}")
        # Rare quoted spans benefit from exact lexical match once.
        return [self._make_search_call(clue, method="bm25")] + list(tool_calls)

    def _force_final_answer(
        self,
        *,
        system_prompt: str,
        question: str,
        trajectory: list[dict[str, Any]],
        state: AgentState,
        progress: ProgressCallback,
    ) -> str:
        evidence_blob = json.dumps(state.summary_dict(), ensure_ascii=False)
        messages = build_messages_for_llm(
            system_prompt=system_prompt
            + "\n\nYou must now output ONLY the final short answer with no tool calls."
            + "\nUse the evidence below; if unsure, still give the best short guess.\n"
            + evidence_blob,
            question=question,
            trajectory=trajectory,
            state=state,
        )
        assistant = self.llm.chat(messages, tools=None)
        answer = normalize_final_answer(assistant.get("content") or "")
        if not is_plausible_final_answer(
            answer,
            min_tool_calls=0,
            tool_calls_made=max(self.tools.call_count, 1),
        ):
            recovered = parse_dsml_tool_calls(assistant.get("content"))
            if recovered:
                progress("Final turn still had DSML; executing recovered tools once")
                traj_assistant = {
                    "role": "assistant",
                    "content": assistant.get("content"),
                    "tool_calls": recovered,
                    "usage": assistant.get("usage"),
                }
                trajectory.append(traj_assistant)
                self._execute_tool_calls(recovered, trajectory, state, progress)
                messages = build_messages_for_llm(
                    system_prompt=system_prompt
                    + "\n\nOutput ONLY the final short answer now.\n"
                    + json.dumps(state.summary_dict(), ensure_ascii=False),
                    question=question,
                    trajectory=trajectory,
                    state=state,
                )
                assistant = self.llm.chat(messages, tools=None)
                answer = normalize_final_answer(assistant.get("content") or "")
        if not is_plausible_final_answer(
            answer,
            min_tool_calls=0,
            tool_calls_made=max(self.tools.call_count, 1),
        ):
            answer = self._fallback_from_evidence(state, progress)
        trajectory.append(
            {
                "role": "assistant",
                "content": answer,
                "usage": assistant.get("usage"),
            }
        )
        return answer

    def _fallback_from_evidence(
        self, state: AgentState, progress: ProgressCallback
    ) -> str:
        """Last-resort short answer from evidence when the model returns empty/junk."""
        if not state.evidence:
            return ""
        snippets = "\n".join(
            f"- doc {e.doc_id} ({e.source}, score={e.score:.3f}): {e.snippet[:280]}"
            for e in state.evidence[:6]
        )
        prompt = (
            "Question:\n"
            f"{state.question}\n\n"
            "Evidence snippets:\n"
            f"{snippets}\n\n"
            "Reply with ONLY the short final answer string (entity/date/number/phrase). "
            "No explanation."
        )
        progress("Fallback: deriving answer from AgentState evidence")
        assistant = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "Extract the best short answer from evidence only.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        answer = normalize_final_answer(assistant.get("content") or "")
        if is_plausible_final_answer(
            answer,
            min_tool_calls=0,
            tool_calls_made=max(self.tools.call_count, 1),
        ):
            return answer
        return ""

    def run(
        self,
        question: str,
        on_progress: Optional[ProgressCallback] = None,
        force_skill: Optional[str] = None,
        disable_skills: bool = False,
    ) -> AgentResult:
        def progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Reset per-question token/tool counters on shared clients if reused
        start_tokens = self.llm.total_tokens
        self.tools.call_count = 0

        state = AgentState(question=question)
        self._skills_disabled = disable_skills
        if disable_skills:
            selected = None
        elif force_skill is not None:
            selected = force_skill or None
        else:
            selected = auto_select_skill(question, state, self.skills)
        # Validate forced/auto name
        if selected and selected not in {s.name for s in self.skills}:
            selected = None
        state.selected_skill = selected
        skill_body = load_skill_body(self.skills, selected) if selected else None
        if selected:
            progress(f"Selected skill: {selected}")
        else:
            progress("No skill selected")

        system_prompt = self._system_prompt(skill_body, selected)
        trajectory: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        answer = ""
        for step in range(1, self.max_steps + 1):
            state.step = step
            # Mid-loop skill upgrade (recovery / constraint-verify)
            if not self._skills_disabled:
                upgraded = maybe_upgrade_skill(state, self.skills)
                if upgraded and upgraded != state.selected_skill:
                    state.selected_skill = upgraded
                    skill_body = load_skill_body(self.skills, upgraded)
                    system_prompt = self._system_prompt(skill_body, upgraded)
                    trajectory[0] = {"role": "system", "content": system_prompt}
                    progress(f"Switched skill to {upgraded}")

            messages = build_messages_for_llm(
                system_prompt=system_prompt,
                question=question,
                trajectory=trajectory,
                state=state,
            )
            progress(f"LLM step {step}/{self.max_steps}")
            # Force tool use until we have some evidence; then allow a short answer.
            choice = "required" if self.tools.call_count == 0 else "auto"
            try:
                assistant = self.llm.chat(
                    messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice=choice,
                )
            except Exception:
                # Some backends reject tool_choice=required; fall back to auto.
                if choice != "auto":
                    progress("tool_choice=required unsupported; falling back to auto")
                    assistant = self.llm.chat(messages, tools=TOOL_DEFINITIONS)
                else:
                    raise
            traj_assistant: dict[str, Any] = {
                "role": "assistant",
                "content": assistant.get("content"),
                "usage": assistant.get("usage"),
            }
            tool_calls = list(assistant.get("tool_calls") or [])
            if not tool_calls:
                recovered = parse_dsml_tool_calls(assistant.get("content"))
                if recovered:
                    progress(f"Recovered {len(recovered)} tool call(s) from DSML markup")
                    tool_calls = recovered

            tool_calls = self._bootstrap_quoted_search(
                question, tool_calls, step, progress
            )

            if tool_calls:
                traj_assistant["tool_calls"] = tool_calls
            trajectory.append(traj_assistant)

            if tool_calls:
                ran = self._execute_tool_calls(tool_calls, trajectory, state, progress)
                if ran:
                    continue
                # All proposed calls were filtered (e.g. duplicate searches).
                progress("No new tools executed after filtering; nudging model")

            candidate = normalize_final_answer(assistant.get("content") or "")
            if is_plausible_final_answer(
                candidate,
                tool_calls_made=self.tools.call_count,
            ):
                answer = candidate
                state.candidate_answer = answer
                progress(f"Final answer: {answer}")
                break

            progress(
                "Rejected non-answer / missing tools; continuing "
                f"(preview={candidate[:80]!r})"
            )
            nudge = _VISIT_NUDGE if self._has_unvisited_hits(state) else _CONTINUE_NUDGE
            trajectory.append({"role": "user", "content": nudge})
        else:
            # Max steps reached without a plain-text answer: force a final answer
            progress("Max steps reached; requesting final answer")
            answer = self._force_final_answer(
                system_prompt=system_prompt,
                question=question,
                trajectory=trajectory,
                state=state,
                progress=progress,
            )
            state.candidate_answer = answer

        total_tokens = self.llm.total_tokens - start_tokens
        return AgentResult(
            answer=answer,
            trajectory=trajectory,
            tool_calls=self.tools.call_count,
            total_tokens=total_tokens,
            selected_skill=state.selected_skill,
            evidence=[
                {
                    "doc_id": e.doc_id,
                    "source": e.source,
                    "score": e.score,
                    "snippet": e.snippet,
                }
                for e in state.evidence
            ],
            state=state,
        )
