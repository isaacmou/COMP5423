"""SQLite document store and chunking helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_JSONL,
    DOCS_DB,
    ensure_data_dirs,
)


@dataclass
class Document:
    doc_id: str
    text: str
    url: str = ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


class DocStore:
    """SQLite-backed document lookup keyed by doc_id."""

    def __init__(self, db_path: Path = DOCS_DB):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Document DB not found: {self.db_path}. "
                "Place corpus at data/raw/corpus.jsonl then run: "
                "python scripts/build_indexes.py --stage corpus"
            )
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def get(self, doc_id: str) -> Optional[Document]:
        row = self._conn.execute(
            "SELECT doc_id, text, url FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return Document(doc_id=row["doc_id"], text=row["text"], url=row["url"] or "")

    def snippet(self, doc_id: str, max_chars: int = 512) -> str:
        doc = self.get(doc_id)
        if doc is None:
            return ""
        text = re.sub(r"\s+", " ", doc.text).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return int(row["n"])

    def close(self) -> None:
        self._conn.close()


def iter_corpus_jsonl(path: Path = CORPUS_JSONL) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("docid") or obj.get("doc_id") or obj.get("id"))
            text = obj.get("text") or obj.get("contents") or ""
            url = obj.get("url") or obj.get("source") or ""
            yield Document(doc_id=doc_id, text=text, url=url)


def build_docs_db(
    corpus_jsonl: Path = CORPUS_JSONL,
    db_path: Path = DOCS_DB,
    expected: Optional[int] = 100195,
) -> int:
    """Materialize corpus.jsonl into a SQLite lookup table."""
    ensure_data_dirs()
    if not corpus_jsonl.exists():
        raise FileNotFoundError(
            f"Missing {corpus_jsonl}. Place the BrowseComp-Plus corpus JSONL there first."
        )
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            url TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)")
    batch: list[tuple[str, str, str]] = []
    n = 0
    for doc in tqdm(iter_corpus_jsonl(corpus_jsonl), desc="Building docs DB"):
        batch.append((doc.doc_id, doc.text, doc.url))
        if len(batch) >= 1000:
            conn.executemany(
                "INSERT OR REPLACE INTO documents(doc_id, text, url) VALUES (?, ?, ?)",
                batch,
            )
            n += len(batch)
            batch.clear()
    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO documents(doc_id, text, url) VALUES (?, ?, ?)",
            batch,
        )
        n += len(batch)
    conn.commit()
    conn.close()
    if expected is not None and n != expected:
        print(f"Warning: expected {expected} docs, wrote {n}")
    return n

