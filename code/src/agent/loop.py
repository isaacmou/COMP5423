"""Iterative tool-using search agent (Basic + Feature A + Feature B)."""

from __future__ import annotations

import json
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
)
from src.agent.state import AgentState
from src.config import MAX_AGENT_STEPS, SKILLS_DIR
from src.corpus.loader import DocStore
from src.llm.siliconflow import SiliconFlowClient
from src.retrieval.unified import UnifiedRetriever
from src.tools.registry import TOOL_DEFINITIONS, ToolExecutor

ProgressCallback = Callable[[str], None]


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
- Prefer visiting a few high-value documents over many shallow searches.
- Keep answers concise and match the expected answer form (entity, date, number, short phrase).
- Never write DSML/XML tool markup in your message content. Use structured tool calls only.
- Do not output planning essays, clue lists, or "let me search" prose as the final answer.
"""

_CONTINUE_NUDGE = (
    "That was not a valid final answer. Use the structured tool interface to "
    "search/visit documents (do not write XML/DSML). When done, reply with ONLY "
    "the short final answer string."
)


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

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        state: AgentState,
        progress: ProgressCallback,
    ) -> None:
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            progress(f"Tool call: {name}({raw_args[:120]})")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
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
            # Mid-loop skill upgrade: switch to recovery if searches keep failing
            if (
                not self._skills_disabled
                and state.weak_search_streak >= 2
                and state.selected_skill != "search-recovery"
                and any(s.name == "search-recovery" for s in self.skills)
            ):
                state.selected_skill = "search-recovery"
                skill_body = load_skill_body(self.skills, "search-recovery")
                system_prompt = self._system_prompt(skill_body, "search-recovery")
                trajectory[0] = {"role": "system", "content": system_prompt}
                progress("Switched skill to search-recovery")

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

            if tool_calls:
                traj_assistant["tool_calls"] = tool_calls
            trajectory.append(traj_assistant)

            if tool_calls:
                self._execute_tool_calls(tool_calls, trajectory, state, progress)
                continue

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
            trajectory.append({"role": "user", "content": _CONTINUE_NUDGE})
        else:
            # Max steps reached without a plain-text answer: force a final answer
            progress("Max steps reached; requesting final answer")
            messages = build_messages_for_llm(
                system_prompt=system_prompt
                + "\n\nYou must now output ONLY the final short answer with no tool calls.",
                question=question,
                trajectory=trajectory,
                state=state,
            )
            # Call without tools to force completion
            assistant = self.llm.chat(messages, tools=None)
            answer = normalize_final_answer(assistant.get("content") or "")
            # If model still dumped markup, strip and keep empty rather than junk.
            if not is_plausible_final_answer(
                answer,
                min_tool_calls=0,
                tool_calls_made=self.tools.call_count,
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
                        + "\n\nOutput ONLY the final short answer now.",
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
                    answer = ""
            trajectory.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "usage": assistant.get("usage"),
                }
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
