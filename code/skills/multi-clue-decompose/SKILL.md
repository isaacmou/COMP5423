---
name: multi-clue-decompose
description: Decompose multi-constraint or multi-hop questions into ordered sub-queries, search each clue separately, then combine evidence. Use when the question stacks several conditions (who/which/when/where together), long BrowseComp-style prompts, or answers that depend on linking two or more facts.
---

# Multi-Clue Decompose

## When to use
Apply this skill when a question contains several interdependent constraints (entities, dates, roles, locations, titles) that are unlikely to match a single keyword search.

## Procedure
1. **List atomic clues.** Mentally extract 2–4 *distinctive* clues (rare phrases, years+venues, quoted titles, unusual attributes). Do not answer yet.
2. **Search one distinctive clue at a time.** Call `search` with a short subquery (`method="auto"`, `top_k=5`). Prefer the rarest phrase, Britannica quote fragment, year+event, or unique attribute — **never** paste a long paraphrase of the whole question. For a rare quoted span, prefer one `method="bm25"` search.
3. **Visit one top document.** Use `visit` on the highest-scoring unvisited `doc_id` before another `search`. If the text is long, use `find_in_doc` for a distinctive keyword, date, or number (for date questions, search day/month/year digits).
4. **Score the candidate against constraints.** After each visit, note which original constraints are supported vs violated. If a named person/entity fails any hard constraint (dates, roles, locations), **discard that candidate** and search a *different* clue — do not keep visiting the same wrong cluster.
5. **Bridge with confirmed facts only.** Rewrite the next subquery using facts that appear in visited text. Avoid repeating an identical previous query.
6. **Ground before finalize.** The final short answer string must appear in visited evidence (or `find_in_doc` hits). Prefer confirming ≥2 independent constraints in visited text.
7. **Finalize.** Reply with ONLY the short final answer string. For dates prefer `DD/MM/YYYY`; for companies omit trailing Ltd/Inc/LLC unless essential; no markdown or explanation prose.

## Anti-patterns
- Dumping the entire original question into every search call.
- Visiting many documents from one bad search without reformulating.
- Committing to a person/name that never appears in visited document text.
- Ending with a long explanation instead of the answer span.
