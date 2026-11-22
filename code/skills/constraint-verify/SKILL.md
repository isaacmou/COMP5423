---
name: constraint-verify
description: Verify a candidate answer against every question constraint using visited documents and find_in_doc before committing. Use when several documents have already been visited, a candidate name/entity is under consideration, or earlier searches risk locking onto the wrong person.
---

# Constraint Verify

## When to use
Use after you have visited multiple documents or formed a candidate answer, especially on multi-constraint identity questions (who/which person).

## Procedure
1. **List constraints.** Extract every hard constraint from the question (birth year window, marriage year, school founding year, quoted phrase, role, location, award, etc.).
2. **Name the candidate.** If you have a candidate person/entity, write it down mentally. If none, search one more distinctive clue first, then return here.
3. **Verify in documents.** For each constraint, use `find_in_doc` on the most relevant visited `doc_id`, or `search` + `visit` a new doc if needed. Prefer literal years, numbers, and unique phrases.
4. **Pass / fail.** A constraint passes only if visited text supports it. If any hard constraint fails, discard the candidate and search a different clue (switch entity cluster).
5. **Commit only when grounded.** Output the short final answer only when (a) the answer string appears in visited evidence and (b) at least two independent constraints are supported.
6. **Finalize.** Reply with ONLY the short final answer string.

## Anti-patterns
- Accepting a famous-looking name from a weakly related page.
- More visits of the same wrong documents instead of a new subquery.
- Final answers that never appear in any visited snippet/text.
