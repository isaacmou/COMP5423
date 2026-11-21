---
name: multi-clue-decompose
description: Decompose multi-constraint or multi-hop questions into ordered sub-queries, search each clue separately, then combine evidence. Use when the question stacks several conditions (who/which/when/where together), long BrowseComp-style prompts, or answers that depend on linking two or more facts.
---

# Multi-Clue Decompose

## When to use
Apply this skill when a question contains several interdependent constraints (entities, dates, roles, locations, titles) that are unlikely to match a single keyword search.

## Procedure
1. **List clues.** Mentally extract 2–4 atomic clues from the question (entities, events, attributes). Do not answer yet.
2. **Search the easiest clue first.** Call `search` with a short, concrete subquery (`method="auto"`, `top_k=5`). Prefer named entities and distinctive phrases over the full question.
3. **Visit top documents.** Use `visit` on 1–2 highest-scoring `doc_id`s. If the text is long, use `find_in_doc` with a distinctive keyword.
4. **Bridge to the next clue.** Rewrite the next subquery using facts confirmed in visited text (names, titles, dates). Avoid repeating an identical previous query.
5. **Cross-check.** Confirm the candidate answer satisfies every original constraint before stopping.
6. **Finalize.** When evidence is sufficient, reply with only the short final answer string.

## Anti-patterns
- Dumping the entire original question into every search call.
- Visiting more than three documents for the same subquery without reformulating.
- Ending with a long explanation instead of the answer span.
