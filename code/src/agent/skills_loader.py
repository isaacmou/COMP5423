"""Load Agent Skills (SKILL.md) at runtime and auto-select among them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from src.agent.state import AgentState
from src.config import SKILLS_DIR


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path

    @property
    def catalog_line(self) -> str:
        return f"- {self.name}: {self.description}"


def _parse_skill_md(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"SKILL.md missing YAML frontmatter: {path}")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid SKILL.md frontmatter: {path}")
    meta = yaml.safe_load(parts[1]) or {}
    name = str(meta.get("name", "")).strip()
    description = str(meta.get("description", "")).strip()
    body = parts[2].strip()
    if not name or not description:
        raise ValueError(f"SKILL.md requires name and description: {path}")
    dir_name = path.parent.name
    if name != dir_name:
        # Spec: name must match parent directory
        raise ValueError(
            f"Skill name '{name}' must match directory '{dir_name}' ({path})"
        )
    return Skill(name=name, description=description, body=body, path=path)


def discover_skills(skills_dir: Path = SKILLS_DIR) -> list[Skill]:
    skills: list[Skill] = []
    if not skills_dir.exists():
        return skills
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            skills.append(_parse_skill_md(skill_md))
    return skills


def catalog_text(skills: list[Skill]) -> str:
    if not skills:
        return "(no skills installed)"
    return "\n".join(s.catalog_line for s in skills)


def load_skill_body(skills: list[Skill], name: str) -> Optional[str]:
    for s in skills:
        if s.name == name:
            return s.body
    return None


def auto_select_skill(question: str, state: AgentState, skills: list[Skill]) -> Optional[str]:
    """
    Heuristic + keyword auto-selection.
    Returns a skill name, or None if no skill is needed.
    """
    names = {s.name for s in skills}
    q = question.lower()

    # Prefer recovery when prior searches were weak
    if state.weak_search_streak >= 1 and "search-recovery" in names:
        return "search-recovery"

    # Verification-heavy questions: many enumerated constraints / verify wording
    verify_signals = [
        r"\bverify\b",
        r"\bconfirm\b",
        r"\bsatisf(?:y|ies|ying)\b",
        r"\bfollowing (?:statements|conditions|criteria|details)\b",
        r"\ball of the following\b",
        r"\beach of the following\b",
        r"\bcheck(?:s|ing)? (?:that|whether|against)\b",
    ]
    enum_hits = len(re.findall(r"(?:^|[\n;\s])\d+[\.\)]\s", question))
    if "constraint-verify" in names and (
        any(re.search(p, q) for p in verify_signals) or enum_hits >= 4
    ):
        return "constraint-verify"

    # Multi-clue / multi-constraint questions
    multi_clue_signals = [
        r"\band\b",
        r"\bboth\b",
        r",",
        r"\bwhich\b.*\bwho\b",
        r"\bwho\b.*\bwhich\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bbetween\b",
        r"\bfounded\b",
        r"\blater\b",
        r"\bsame\b",
        r"\bnot\b",
    ]
    clue_hits = sum(1 for p in multi_clue_signals if re.search(p, q))
    word_count = len(q.split())
    if "multi-clue-decompose" in names and (clue_hits >= 2 or word_count >= 28):
        return "multi-clue-decompose"

    # Ambiguous entity / recovery-style phrasing
    if "search-recovery" in names and re.search(
        r"\b(also known|aka|formerly|ambiguous|unclear|might|possibly)\b", q
    ):
        return "search-recovery"

    # Default: multi-clue for long BrowseComp-style questions
    if "multi-clue-decompose" in names and word_count >= 20:
        return "multi-clue-decompose"

    return None


def maybe_upgrade_skill(state: AgentState, skills: list[Skill]) -> Optional[str]:
    """
    Mid-loop skill upgrade based on search progress.

    Only auto-upgrade to search-recovery on weak searches.
    constraint-verify remains available via catalog/auto_select, but is not forced.
    """
    names = {s.name for s in skills}
    if state.weak_search_streak >= 2 and "search-recovery" in names:
        return "search-recovery"
    return None
