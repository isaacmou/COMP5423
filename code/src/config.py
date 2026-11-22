"""Project paths and runtime defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# code/ is the package root for running scripts
CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DATA_DIR = Path(os.environ.get("COMP5423_DATA_DIR", PROJECT_ROOT / "data")).resolve()

RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "indexes"
CHUNKS_DIR = DATA_DIR / "chunks"

CORPUS_JSONL = RAW_DIR / "corpus.jsonl"
DOCS_DB = INDEX_DIR / "docs.sqlite"
CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"
CHUNK_META_JSON = CHUNKS_DIR / "chunk_meta.json"

BM25_DIR = INDEX_DIR / "bm25"
DENSE_INDEX = INDEX_DIR / "faiss_bge.index"
DENSE_IDS = INDEX_DIR / "faiss_bge_ids.json"
INSTRUCT_INDEX = INDEX_DIR / "faiss_qwen3.index"
INSTRUCT_IDS = INDEX_DIR / "faiss_qwen3_ids.json"

# Course splits live directly under data/
VAL_JSONL = DATA_DIR / "validation.jsonl"
TEST_JSONL = DATA_DIR / "test.jsonl"

SILICONFLOW_BASE_URL = os.environ.get(
    "SILICONFLOW_BASE_URL", "https://llm.cs.cityu.edu.hk/v1"
)
SILICONFLOW_MODEL = os.environ.get(
    "SILICONFLOW_MODEL", "CS5351/DeepSeek-V4-Flash"
)

# Retrieval / agent defaults (efficiency-oriented)
DEFAULT_TOP_K = 7
SNIPPET_CHARS = 512
VISIT_MAX_CHARS = 8000
FIND_MAX_MATCHES = 8
FIND_CONTEXT_CHARS = 200
MAX_AGENT_STEPS = 10
MAX_EVIDENCE_ITEMS = 12
MAX_TOOL_RESULT_CHARS = 3500
CHUNK_SIZE = 512  # approximate words
CHUNK_OVERLAP = 64

DENSE_MODEL_NAME = os.environ.get("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
INSTRUCT_MODEL_NAME = os.environ.get(
    "INSTRUCT_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)
INSTRUCT_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)

SKILLS_DIR = CODE_ROOT / "skills"


def load_env() -> None:
    """Load .env from code/ or project root if present."""
    for path in (CODE_ROOT / ".env", PROJECT_ROOT / ".env"):
        if path.exists():
            load_dotenv(path, override=False)
            break
    else:
        load_dotenv(override=False)


def ensure_data_dirs() -> None:
    for d in (RAW_DIR, INDEX_DIR, CHUNKS_DIR, BM25_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str:
    load_env()
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not set. Export it or put it in code/.env"
        )
    return key
