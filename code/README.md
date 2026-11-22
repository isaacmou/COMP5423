# COMP5423 Skill-Augmented Search Agent

End-to-end search agent for the COMP5423 project over the BrowseComp-Plus fixed corpus.

**繁體中文詳細說明：** [REPORT_ZH.md](../REPORT_ZH.md)

- **LLM:** DeepSeek-V4-Flash via SiliconFlow (`CS5351/DeepSeek-V4-Flash`)
- **Retrieval (3 categories, one tool):** BM25 (sparse), BGE dense, Qwen3-Embedding (instruction-aware), fused with RRF when `method=auto` (`DEFAULT_TOP_K=7`)
- **Supporting tools:** `visit`, `find_in_doc`
- **Agent:** iterative ReAct loop + structured context management; BM25 quote bootstrap; answer polish (dates / corp suffixes / prose strip); no hard evidence-substring gate
- **Skills:** `multi-clue-decompose`, `search-recovery`, `constraint-verify` (auto-selected at runtime; mid-loop upgrade is recovery-only)
- **UI:** Streamlit

## Directory structure

```
code/
  app.py                 # Streamlit UI
  run_eval.py            # Batch predictions -> test_prediction.jsonl
  requirements.txt
  .env.example
  skills/
    multi-clue-decompose/SKILL.md
    search-recovery/SKILL.md
    constraint-verify/SKILL.md
  scripts/
    build_indexes.py
    run_random_val.py
  src/
    config.py
    llm/siliconflow.py
    corpus/loader.py
    retrieval/           # bm25, dense, instruct, unified
    tools/               # search, visit, find_in_doc
    agent/               # state, context, loop, skills_loader, response_utils
data/
  validation.jsonl       # course validation split (in repo)
  test.jsonl             # course test split (in repo)
  raw/corpus.jsonl       # download from HuggingFace (not in repo)
  indexes/               # built locally (not in repo)
  chunks/                # chunk cache (not in repo)
```

## Environment setup

```bash
cd code
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # then edit SILICONFLOW_API_KEY
```

### LLM API key

Set `SILICONFLOW_API_KEY` in `code/.env` or export it in your shell:

```bash
# PowerShell
$env:SILICONFLOW_API_KEY="sk-..."

# bash
export SILICONFLOW_API_KEY=sk-...
```

Do **not** commit API keys. Model default: `CS5351/DeepSeek-V4-Flash` (SiliconFlow / course gateway).

## Local data

Eval splits ship with the repo:

- `data/validation.jsonl`
- `data/test.jsonl`

Corpus and indexes are large (~14 GB) and are **not** in git. Each teammate downloads the corpus once and builds indexes locally.

1. Download the document collection to `data/raw/corpus.jsonl` (one JSON object per line with `docid`, `text`, and optional `url`) from [Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus).
2. Build indexes (required once per machine):

```bash
cd code
python scripts/build_indexes.py --stage all
```

Stages can be run separately:

```bash
python scripts/build_indexes.py --stage corpus
python scripts/build_indexes.py --stage chunks
python scripts/build_indexes.py --stage bm25
python scripts/build_indexes.py --stage dense
python scripts/build_indexes.py --stage instruct
```

Outputs land under `../data/indexes/` and `../data/chunks/`. Run `--stage instruct` if `faiss_qwen3` is missing (needed for full auto retrieval; use `--bm25-only` on eval if you skip it for now).

## Run the UI

```bash
cd code
streamlit run app.py
```

The UI shows the final answer, selected skill, evidence, tool-call trajectory, and token/tool counts.

## Reproduce `test_prediction.jsonl`

From `code/`:

```bash
python run_eval.py --split test --out ../test_prediction.jsonl
```

Validation with normalized EM:

```bash
python run_eval.py --split val --out val_prediction.jsonl --compute-em
```

Useful flags:

- `--limit N` / `--offset K` for partial runs
- `--bm25-only` to skip dense/instruct backends
- `--api-key sk-...` to pass a key without `.env`

Each output line matches the course schema:

```json
{
  "id": "...",
  "question": "...",
  "answer": "...",
  "trajectory": [...],
  "tool_calls": 0,
  "total_tokens": 0
}
```

`tool_calls` counts agent-issued tool invocations. `total_tokens` sums SiliconFlow `usage.total_tokens` over LLM calls for that question.

## Submission package (group)

Recommended Canvas layout:

```
Group-X.zip
|-- test_prediction.jsonl
|-- code/          # this directory
|-- Group-X-Report.pdf
`-- Group-X-Video.mp4
```

Report / video / signatures are completed by the group (Overleaf template linked in the course PDF).

## Design summary (for report)

1. **Retrieval:** One `search` tool; internal BM25 + BGE + Qwen3 with optional RRF fusion.
2. **Tools:** `visit` opens docs; `find_in_doc` greps inside a doc.
3. **Agent loop:** Multi-step tool use conditioned on previous results (Feature A); auto-visit at most one unvisited hit; step-1 BM25 search for distinctive quoted clues.
4. **Context:** `AgentState` tracks visited docs/evidence/subqueries; tool outputs are deduped/truncated before LLM calls (Feature B).
5. **Skills:** Catalog only in the base prompt; full `SKILL.md` loaded when auto-selected (`constraint-verify` for verification-heavy prompts).
6. **Answer polish:** `normalize_final_answer` rewrites EM-friendly forms (e.g. `DD/MM/YYYY`, strip Ltd/Inc) and rejects explanation prose; acceptance does **not** use a hard evidence-substring gate.
