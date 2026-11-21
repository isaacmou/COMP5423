---
name: search-recovery
description: Recover from weak, empty, or off-topic search results by rewriting queries, switching retrieval methods, and inspecting alternate documents. Use when initial searches return irrelevant snippets, repeated low-value hits, ambiguous entities, or when the agent is stuck after unsuccessful retrieval.
---

# Search Recovery

## When to use
Trigger this skill after a weak search (empty results, unrelated snippets, or repeated near-duplicate hits) or when entity names may be misspelled/ambiguous.

## Procedure
1. **Diagnose.** From the last search snippets, note what mismatched (wrong entity type, wrong time period, generic pages).
2. **Rewrite the query.** Produce a new subquery that:
   - drops filler words,
   - uses alternate spellings/aliases if present in prior evidence,
   - adds a disambiguating constraint from the question.
3. **Switch method if needed.**
   - Try `method="bm25"` for rare proper nouns / exact phrases.
   - Try `method="instruct"` for paraphrased natural-language needs.
   - Try `method="dense"` for semantic paraphrases.
   - Fall back to `method="auto"` after a successful rewrite.
4. **Inspect alternates.** `visit` a previously unseen `doc_id` (not already in visited docs). Use `find_in_doc` for a precise keyword.
5. **Stop criterion.** After a successful recovery path yields consistent evidence, output only the short final answer.

## Anti-patterns
- Calling the same query+method pair again unchanged.
- Opening every result without reading whether it addresses the failure mode.
- Infinite recovery loops; after several rewrites, answer with the best-supported candidate.
